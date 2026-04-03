use cubecl::Runtime;
use nalgebra::{Complex, DVector};

use crate::batched_circuit::{BatchedCircuit, BatchedCircuitOp};
use crate::circuit::{CircuitBehaviour, HybridCircuit};
use crate::expr_dsl::{BitExpr, BoolExpr};
use crate::gate::QBits;
use crate::gpu_sv_simulator::gpu_state_vector::GpuStateVector;
use crate::simulator::RunnableSimulator;
use crate::{
    circuit::Circuit,
    instruction::Instruction,
    register_file::RegisterFile,
};

mod gpu_kernels;
mod gpu_state_vector;
mod mem_helpers;

const GPU_MAX_TARGET_QUBITS: usize = 3;
const GPU_MAX_BLOCK_SIZE: usize = 1 << GPU_MAX_TARGET_QUBITS;

#[derive(Clone)]
pub struct GpuStateVectorExecutor<R: Runtime> {
    gpu_state_vector: GpuStateVector<R>,
    batched_circuit: BatchedCircuit,
    pc: usize,
    registers: RegisterFile,
}

impl<R: Runtime> GpuStateVectorExecutor<R> {
    fn new(circuit: Circuit<HybridCircuit>) -> Self {
        let n_qubits = circuit.n_qubits();
        let registers = RegisterFile::from(circuit.registers());
        let batched_circuit = BatchedCircuit::from_circuit(circuit, GPU_MAX_TARGET_QUBITS);
        let gpu_state = GpuStateVector::<R>::new(n_qubits, &batched_circuit);

        Self {
            gpu_state_vector: gpu_state,
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
                        self.gpu_state_vector.apply_batch_command(command);
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
        self.gpu_state_vector.sample()
    }

    fn measure_bit(&mut self, target: usize, reg: &str, bit_pos: usize) {
        let measurement = self
            .gpu_state_vector
            .measure_bits(QBits::from_bitstring(1 << target));
        let measured_bit = (measurement >> target) & 1;

        self.registers[reg].write_bit(bit_pos, measured_bit).expect("invalid register write");

        self.pc += 1;
    }

    fn measure_all(&mut self, reg: &str) {
        let measurement = self.gpu_state_vector.measure();

        self.registers[reg].write(measurement);

        self.pc += 1;
    }

    fn jump(&mut self, label_pc: usize) {
        self.pc = label_pc;
    }

    fn jump_if(&mut self, expr: &BoolExpr, label_pc: usize) {
        if expr.eval(&self.registers) {
            self.jump(label_pc)
        } else {
            self.pc += 1
        }
    }

    fn assign(&mut self, expr: &BitExpr, reg: &str) {
        let value = expr.eval(&self.registers);
        self.registers[reg].write(value);
        self.pc += 1;
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

impl<B, R: Runtime> TryFrom<Circuit<B>> for GpuStateVectorSimulator<R>
where
    B: CircuitBehaviour,
    Circuit<B>: Into<Circuit<HybridCircuit>>,
{
    type Error = GPUSVError;

    fn try_from(value: Circuit<B>) -> Result<Self, Self::Error> {
        Ok(Self {
            circuit: value.into(),
            _runtime: std::marker::PhantomData,
        })
    }
}

impl<R: Runtime> RunnableSimulator for GpuStateVectorSimulator<R> {
    fn run(&self) -> usize {
        GpuStateVectorExecutor::<R>::new(self.circuit.clone())
            .step_all()
            .get_collapsed_state()
    }

    fn final_state(&self) -> DVector<Complex<f32>> {
        let mut exec = GpuStateVectorExecutor::<R>::new(self.circuit.clone());
        exec.step_all();
        exec.gpu_state_vector.sync_state_to_cpu();
        DVector::from_row_slice(exec.gpu_state_vector.as_slice())
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
        let circuit = Circuit::<PureCircuit>::new_qft(n_qubits);

        let gpu = GpuStateVectorSimulator::<WgpuRuntime>::build(circuit.clone()).unwrap();
        let cpu = SVSimulator::build(circuit).unwrap();

        println!("{}", &gpu.final_state());
        println!("{}", &cpu.final_state());
        assert!(equal_to_matrix_c(
            &gpu.final_state(),
            &cpu.final_state(),
            0.001
        ));
    }

    #[test]
    fn sampling_basis_state_walks_back_to_state_vector() {
        let circuit = Circuit::<PureCircuit>::new(15).x(0).x(7).x(14);
        let gpu = GpuStateVectorSimulator::<WgpuRuntime>::build(circuit).unwrap();

        assert_eq!(gpu.run(), (1 << 14) | (1 << 7) | 1);
    }
}
