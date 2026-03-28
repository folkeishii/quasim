use cubecl::bytes::Bytes;
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::{CubeDim, Runtime};
use nalgebra::{Complex, DVector};
use rand::distr::{Distribution, weighted::WeightedIndex};

use crate::batched_circuit::{BatchedCircuit, BatchedCircuitOp};
use crate::circuit::HybridCircuit;
use crate::gate::QBits;
use crate::gate_batcher::BatchCommand;
use crate::simulator::RunnableSimulator;
use crate::{
    cart,
    circuit::Circuit,
    expr_dsl::{Expr, Value},
    instruction::Instruction,
    register_file::RegisterFile,
};

mod gpu_kernels;
mod mem_helpers;

const GPU_MAX_TARGET_QUBITS: usize = 3;
const GPU_MAX_BLOCK_SIZE: usize = 1 << GPU_MAX_TARGET_QUBITS;

#[derive(Clone)]
pub struct GpuStateVectorExecutor<R: Runtime> {
    client: ComputeClient<R>,

    state_vector_cache: Bytes,
    state_vector_len: usize,
    state_vector_handle: Handle,
    gate_data_handle: Handle,
    target_data_handle: Handle,
    control_data_handle: Handle,

    batched_circuit: BatchedCircuit,
    pc: usize,
    registers: RegisterFile<Value>,
}

impl<R: Runtime> GpuStateVectorExecutor<R> {
    fn new(circuit: Circuit<HybridCircuit>) -> Self {
        let size = 1 << circuit.n_qubits();
        let registers = RegisterFile::from(circuit.registers());
        let client = R::client(&Default::default());
        let batched_circuit = BatchedCircuit::from_circuit(circuit, GPU_MAX_TARGET_QUBITS);

        let gate_data = batched_circuit.data().gate_data();
        let target_data = batched_circuit.data().target_data();
        let control_data = batched_circuit.data().control_data();

        let gate_data_handle = client.create_from_slice(mem_helpers::bytes_from_complex(gate_data));
        let target_data_handle = client.create_from_slice(u32::as_bytes(target_data));
        let control_data_handle = client.create_from_slice(u32::as_bytes(control_data));

        // Bytes init
        let mut init_state: Vec<f32> = vec![0.0; size * 2]; // two floats for each complex number
        init_state[0] = 1.0; // Sets first complex re = 1.0
        let init_bytes = Bytes::from_elems(init_state);

        let state_vector_handle = client.create_from_slice(&init_bytes);

        Self {
            client,
            state_vector_cache: init_bytes,
            state_vector_len: size,
            state_vector_handle,
            gate_data_handle,
            target_data_handle,
            control_data_handle,
            batched_circuit,
            pc: Default::default(),
            registers,
        }
    }

    /// Run the entire circuit
    pub fn step_all(&mut self) -> &Self {
        while let Some(op) = self.batched_circuit.operation(self.pc) {
            match op {
                BatchedCircuitOp::BatchCommands(commands) => {
                    for command in commands {
                        self.gpu_batch_command(command);
                        self.pc += command.size as usize;
                    }
                }
                BatchedCircuitOp::Instruction(instruction) => {
                    self.apply_instruction(&instruction.clone());
                }
            }
        }

        self
    }

    /// Gets a collapsed result from the current state vector
    pub fn get_collapsed_state(&self) -> usize {
        let probs = self.state_vector().iter().map(|&c| c.norm_sqr());

        let dist = WeightedIndex::new(probs)
            .expect("Failed to create probability distribution. Invalid or empty state vector?");
        let mut rng = rand::rng();

        dist.sample(&mut rng)
    }

    pub fn sync_state_to_cpu(&mut self) {
        self.state_vector_cache = self.client.read_one(self.state_vector_handle.clone());
    }

    /// Get current state of the quantum system
    pub fn state_vector(&self) -> &[Complex<f32>] {
        unsafe { mem_helpers::complex_from_bytes(&self.state_vector_cache) }
    }

    pub fn state_vector_mut(&mut self) -> &mut [Complex<f32>] {
        unsafe { mem_helpers::complex_from_bytes_mut(&mut self.state_vector_cache) }
    }

    fn gpu_batch_command(&self, command: &BatchCommand) {
        let n_amplitudes = self.state_vector_len;
        let data_length = self.batched_circuit.data().len();
        let batch_target_count = command.targets.count();

        let n_superblocks = n_amplitudes >> batch_target_count;

        let cube_dim = CubeDim::new_1d(min(n_superblocks as u32, 128));
        let cube_count =
            cubecl::calculate_cube_count_elemwise(&self.client, n_superblocks, cube_dim);

        unsafe {
            let _ = gpu_kernels::batched_gate2::launch(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<f32>(&self.state_vector_handle, n_amplitudes * 2, 1),
                ArrayArg::from_raw_parts::<u32>(&self.target_data_handle, data_length, 1),
                ArrayArg::from_raw_parts::<u32>(&self.control_data_handle, data_length, 1),
                ArrayArg::from_raw_parts::<f32>(&self.gate_data_handle, data_length * 8, 1),
                ScalarArg::new(command.start_index),
                ScalarArg::new(command.size),
                ScalarArg::new(command.targets.get_bitstring() as u32),
                // ArrayArg::from_raw_parts::<u32>(&block_indices_handle, command.targets.count(), 1),
            );
        }
    }

    fn measure_bit(&mut self, target: usize, reg: &str, bit_pos: usize) {
        let measurement = self.get_collapsed_state();
        let mask = 1 << target;
        let measured_bit = measurement & mask;
        let shifted_measurement = ((measurement >> target) & 1) << bit_pos;
        let register_bit_mask = 1 << bit_pos;

        if let Value::Int(val) = self.registers[reg] {
            let val_cleared = (val as usize) & !register_bit_mask;
            self.registers[reg] = Value::Int((val_cleared | shifted_measurement) as i32)
        } else {
            self.registers[reg] = Value::Int(shifted_measurement as i32)
        }

        // Go through state vector and remove amplitude for all states that do not align with measurement
        for (i, amp) in self.state_vector_mut().iter_mut().enumerate() {
            if (i & mask) != measured_bit {
                *amp = Complex::ZERO;
            }
        }

        // Renormalize state vector
        let norm = self
            .state_vector()
            .iter()
            .map(|x| x.norm_sqr())
            .sum::<f32>()
            .sqrt();
        self.state_vector_mut().iter_mut().for_each(|x| *x /= norm);

        self.pc += 1;
    }

    fn measure_all(&mut self, reg: &str) {
        let measurement = self.get_collapsed_state();

        self.registers[reg] = Value::Int(measurement as i32);

        // Collapse whole state vector
        self.state_vector_mut().fill(cart!(0.0));
        self.state_vector_mut()[measurement] = cart!(1.0);

        self.pc += 1;
    }

    fn jump(&mut self, label_pc: usize) {
        self.pc = label_pc;
    }

    fn jump_if(&mut self, expr: &Expr, label_pc: usize) {
        match expr.eval(&self.registers) {
            Ok(Value::Bool(true)) => self.jump(label_pc),
            Ok(Value::Bool(false)) => self.pc += 1,
            Err(err) => panic!("{}", err),
            _ => panic!(
                "Expression was expected to evaluate to boolean type but got something else."
            ),
        }
    }

    fn assign(&mut self, expr: &Expr, reg: &str) {
        match expr.eval(&self.registers) {
            Ok(value) => self.registers[reg] = value,
            Err(err) => panic!("{}", err),
        }
        self.pc += 1
    }

    fn apply_instruction(&mut self, inst: &Instruction) {
        match inst {
            Instruction::Gate(_gate) => todo!(),
            Instruction::MeasureBit(qbit, (reg, bit_pos)) => self.measure_bit(*qbit, reg, *bit_pos),
            Instruction::MeasureAll(reg) => self.measure_all(reg),
            Instruction::Jump(pc) => self.jump(*pc),
            Instruction::JumpIf(expr, pc) => self.jump_if(expr, *pc),
            Instruction::Assign(expr, reg) => self.assign(expr, reg),
            Instruction::Call(_, _, _) => todo!(),
        }
    }
}

pub struct GpuStateVectorSimulator<R: Runtime> {
    circuit: Circuit<HybridCircuit>,
    _runtime: std::marker::PhantomData<R>,
}

impl<R: Runtime> TryFrom<Circuit<HybridCircuit>> for GpuStateVectorSimulator<R> {
    type Error = GPUSVError;

    fn try_from(value: Circuit<HybridCircuit>) -> Result<Self, Self::Error> {
        Ok(Self {
            circuit: value,
            _runtime: std::marker::PhantomData,
        })
    }
}

impl<R: Runtime> RunnableSimulator for GpuStateVectorSimulator<R> {
    fn run(&self) -> usize {
        let mut exec = GpuStateVectorExecutor::<R>::new(self.circuit.clone());
        exec.step_all();
        exec.sync_state_to_cpu();
        exec.get_collapsed_state()
    }

    fn final_state(&self) -> DVector<Complex<f32>> {
        let mut exec = GpuStateVectorExecutor::<R>::new(self.circuit.clone());
        exec.step_all();
        exec.sync_state_to_cpu();
        DVector::from_row_slice(exec.state_vector())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GPUSVError {}

#[cfg(test)]
mod tests {
    use cubecl::wgpu::WgpuRuntime;

    use crate::{
        circuit::{Circuit, PureCircuit},
        ext::equal_to_matrix_c,
        gpu_sv_simulator::GpuStateVectorSimulator,
        simulator::{BuildSimulator, RunnableSimulator},
        sv_simulator::SVSimulator,
    };

    #[test]
    fn qft_matches_cpu_state_vector() {
        let n_qubits = 4;
        let circuit = Circuit::<PureCircuit>::qft(n_qubits);

        let gpu = GpuStateVectorSimulator::<WgpuRuntime>::build(circuit.clone().into()).unwrap();
        let cpu = SVSimulator::build(circuit).unwrap();

        println!("{}", &gpu.final_state());
        println!("{}", &cpu.final_state());
        assert!(equal_to_matrix_c(
            &gpu.final_state(),
            &cpu.final_state(),
            0.001
        ));
    }
}
