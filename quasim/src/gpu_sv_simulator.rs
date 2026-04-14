use cubecl::Runtime;
use nalgebra::Complex;

use crate::circuit::{CircuitBehaviour, HybridCircuit};
use crate::expr_dsl::{BitExpr, BoolExpr};
use crate::gate::QBits;
use crate::gpu_sv_simulator::batched_circuit::{BatchedCircuit, BatchedCircuitOp};
use crate::gpu_sv_simulator::gpu_state_vector::GpuStateVector;
use crate::sampler::Sampler;
use crate::simulator::{Sampleable, Simulator, StoredRegisters};
use crate::{circuit::Circuit, instruction::Instruction, register_file::RegisterFile};

pub mod batched_circuit;
pub mod gate_batcher;
mod gpu_kernels;
mod gpu_state_vector;
mod mem_helpers;

#[derive(Clone)]
pub struct GpuStateVectorSimulator<R: Runtime> {
    gpu_state_vector: GpuStateVector<R>,
    batched_circuit: BatchedCircuit,
    pc: usize,
    registers: RegisterFile,
}

impl<R: Runtime> GpuStateVectorSimulator<R> {
    /// Run the entire circuit on the gpu, don't sync state vector to cpu.
    /// Use function `sync` to explicitly sync state vector to cpu.
    /// Use this function if you don't need access to individual amplitudes in
    /// the state vector after running circuit. For example when sampling.
    pub fn run_without_sync(&mut self) {
        self.reset_without_sync();

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
    }

    /// Resets state vector, register and program counter to initial state.
    /// Doesn't sync the state vector to cpu.
    /// Use function `reset` to explicitly sync state vector to cpu.
    pub fn reset_without_sync(&mut self) {
        self.gpu_state_vector.reset();
        self.registers.reset();
        self.pc = 0;
    }

    /// Explicitly syncs the gpu side state vector to the cpu.
    /// Expensive, avoid unless necessary.
    pub fn sync(&mut self) {
        self.gpu_state_vector.sync_state_to_cpu();
    }

    fn measure_bit(&mut self, target: usize, reg: &str, bit_pos: usize) {
        let measurement = self
            .gpu_state_vector
            .measure_bits(QBits::from_bitstring(1 << target));
        let measured_bit = (measurement >> target) & 1;

        self.registers[reg]
            .write_bit(bit_pos, measured_bit)
            .expect("invalid register write");

        self.pc += 1;
    }

    fn measure_all(&mut self, reg: &str) {
        let measurement = self.gpu_state_vector.measure_all();

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
            Instruction::MeasureBit(qbit, (reg, bit_pos)) => self.measure_bit(*qbit, reg, *bit_pos),
            Instruction::MeasureAll(reg) => self.measure_all(reg),
            Instruction::Jump(pc) => self.jump(*pc),
            Instruction::JumpIf(expr, pc) => self.jump_if(expr, *pc),
            Instruction::Assign(expr, reg) => self.assign(expr, reg),
            Instruction::Gate(_gate) => unreachable!(),
            Instruction::Call(_, _, _) => unreachable!(),
        }
    }
}

impl<B, R: Runtime> TryFrom<Circuit<B>> for GpuStateVectorSimulator<R>
where
    B: CircuitBehaviour,
    Circuit<B>: Into<Circuit<HybridCircuit>>,
{
    type Error = GPUSVError;

    fn try_from(value: Circuit<B>) -> Result<Self, Self::Error> {
        let registers = RegisterFile::from(value.registers());
        let batched_circuit = BatchedCircuit::from(value);
        let gpu_state_vector = GpuStateVector::<R>::new(&batched_circuit);

        Ok(Self {
            gpu_state_vector,
            batched_circuit,
            pc: Default::default(),
            registers,
        })
    }
}

impl<R: Runtime> Simulator for GpuStateVectorSimulator<R> {
    type State = GpuStateVector<R>;
    type BasisValue = Complex<f64>;

    /// Runs the entire circuit on the gpu and then syncs state to cpu.
    /// Use function `run_without_sync` to avoid syncing state vector to cpu.
    ///
    /// Use this function if you need to access individual amplitudes in the
    /// state vector after running circuit.
    fn run(&mut self) {
        self.run_without_sync();
        self.sync();
    }

    /// Resets state vector, register and program counter to initial state.
    /// Syncs the state vector to cpu.
    /// Use function `reset_without_sync` to avoid syncing state vector to cpu.
    fn reset(&mut self) {
        self.reset_without_sync();
        self.sync();
    }

    fn state(&self) -> &Self::State {
        &self.gpu_state_vector
    }
}

impl<B, R> Sampleable<B> for GpuStateVectorSimulator<R>
where
    B: CircuitBehaviour,
    R: Runtime,
    Circuit<B>: Clone + Into<Circuit<HybridCircuit>>,
{
    fn sample_once<S: Sampler<Self>>(
        circuit: Circuit<B>,
        sampler: S,
    ) -> Result<S::Output, Self::E> {
        let mut sim = Self::try_from(circuit)?;
        sim.run_without_sync();
        Ok(sampler.sample(&sim))
    }

    fn sample<S: Sampler<Self>>(
        circuit: Circuit<B>,
        sampler: S,
        times: usize,
    ) -> Result<impl Iterator<Item = <S as Sampler<Self>>::Output>, Self::E> {
        let mut sim = Self::try_from(circuit)?;
        let iter = (0..times).map(move |_| {
            sim.run_without_sync();
            sampler.sample(&sim)
        });
        Ok(iter)
    }
}

impl<R: Runtime> StoredRegisters for GpuStateVectorSimulator<R> {
    fn registers(&self) -> &RegisterFile {
        &self.registers
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GPUSVError {}

#[cfg(test)]
mod tests {
    use cubecl::wgpu::WgpuRuntime;

    use crate::{
        circuit::{Circuit, PureCircuit},
        common_test,
        expr_dsl::expr_helpers::{r, rb},
        ext::equal_state_c,
        gpu_sv_simulator::GpuStateVectorSimulator,
        sampler::{CircuitSampler, QubitsSampler},
        simulator::{Buildable, Sampleable, Simulator},
        sv_simulator::StateVectorSimulator,
    };

    type WgpuSimulator = GpuStateVectorSimulator<WgpuRuntime>;

    #[test]
    fn test_qft() {
        let n_qubits = 4;
        let circuit = Circuit::<PureCircuit>::new_qft(n_qubits);

        let mut gpu = WgpuSimulator::build(circuit.clone()).unwrap();
        let mut cpu = StateVectorSimulator::build(circuit).unwrap();
        gpu.run();
        cpu.run();

        println!("{}", &gpu.state());
        println!("{}", &cpu.state());
        assert!(equal_state_c(gpu.state(), cpu.state(), n_qubits, 0.001));
    }

    #[test]
    fn hybrid_test() {
        common_test::hybrid_test::<WgpuSimulator>();
    }

    #[test]
    fn register_test() {
        common_test::register_test::<WgpuSimulator>();
    }

    #[test]
    fn test_measure_overwrites_with_zero() {
        common_test::test_measure_overwrites_with_zero::<WgpuSimulator>();
    }

    #[test]
    fn test_reset() {
        common_test::test_reset::<WgpuSimulator>();
    }

    #[test]
    fn test_reset_with_shared_scratch_register() {
        common_test::test_reset_with_shared_scratch_register::<WgpuSimulator>();
    }

    #[test]
    fn test_jump_into_batch() {
        let circuit = Circuit::new(3)
            .x(0)
            .jump("target")
            .x(1)
            .label("target")
            .x(2);

        let qubits = QubitsSampler::new([0, 2]);

        let gpu = WgpuSimulator::sample_once(circuit.clone(), &qubits).unwrap();
        let cpu = StateVectorSimulator::sample_once(circuit, &qubits).unwrap();

        assert_eq!(cpu, [1, 1]);
        assert_eq!(gpu, cpu);
    }

    #[test]
    fn test_subcircuit_and_control_flow() {
        let leaf = Circuit::new(1).x(0).h(0);
        let pair = Circuit::new(2)
            .new_sub_circuit("leaf", leaf.clone())
            .call("leaf", 0)
            .ccall("leaf", 1, &[0])
            .cx(&[1], 0);
        let nested =
            Circuit::new(3)
                .call_new("pair", pair, 0)
                .x(2)
                .ccall_new("leaf", leaf, 1, &[2]);

        let circuit = Circuit::new(4)
            .new_sub_circuit("nested", nested.clone())
            .new_sub_circuit("nested_inv", nested.inverse())
            .call("nested", 0)
            .call("nested_inv", 0)
            .new_reg("m", 3)
            .new_reg("p", 1)
            .new_reg("a", 1)
            .new_reg("ok", 1)
            .h(0)
            .h(1)
            .h(2)
            .measure_bits(&[0, 1, 2], "m")
            .assign("p", rb("m", 0) ^ rb("m", 1) ^ rb("m", 2))
            .jump_if(r("p").eq(0), "even")
            .x(3)
            .measure_bit(3, ("a", 0))
            .apply_if(rb("a", 0).eq(1))
            .x(3)
            .jump("after")
            .label("even")
            .h(3)
            .measure_bit(3, ("a", 0))
            .apply_if(rb("a", 0).eq(1))
            .x(3)
            .label("after")
            .apply_if(rb("m", 0).eq(1))
            .x(0)
            .apply_if(rb("m", 1).eq(1))
            .x(1)
            .apply_if(rb("m", 2).eq(1))
            .x(2)
            .assign("ok", r("p") ^ rb("m", 0) ^ rb("m", 1) ^ rb("m", 2))
            .jump_if(r("ok").eq(0), "done")
            .x(3)
            .label("done");

        let mut gpu = WgpuSimulator::build(circuit.clone()).unwrap();
        let mut cpu = StateVectorSimulator::build(circuit.clone()).unwrap();
        gpu.run();
        cpu.run();

        assert!(equal_state_c(gpu.state(), cpu.state(), 4, 0.001));

        assert!(
            WgpuSimulator::sample(circuit.clone(), CircuitSampler, 10)
                .unwrap()
                .all(|sample| sample == 0)
        );
        assert!(
            StateVectorSimulator::sample(circuit, CircuitSampler, 10)
                .unwrap()
                .all(|sample| sample == 0)
        );
    }
}
