use cubecl::Runtime;
use nalgebra::{Complex, DVector};

use crate::circuit::{CircuitBehaviour, HybridCircuit};
use crate::expr_dsl::{BitExpr, BoolExpr};
use crate::gate::QBits;
use crate::gpu_sv_simulator::batched_circuit::{BatchedCircuit, BatchedCircuitOp};
use crate::gpu_sv_simulator::gpu_state_vector::GpuStateVector;
use crate::simulator::RunnableSimulator;
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

    pub fn reset(&mut self) {
        self.gpu_state_vector.reset();
        self.registers.reset();
        self.pc = 0;
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
    Circuit<B>: Clone + Into<Circuit<HybridCircuit>>,
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

impl<R: Runtime> RunnableSimulator for GpuStateVectorSimulator<R> {
    type Storage = DVector<Complex<f64>>;
    type State = Complex<f64>;

    fn run(&self) -> usize {
        let mut sim = self.clone();
        sim.reset();
        sim.step_all().get_collapsed_state()
    }

    fn final_state(&self) -> Self::Storage {
        let mut sim = self.clone();
        sim.reset();
        sim.step_all();
        // This is extremely expensive, we should probably do something about this
        // Will be able to be fixed after #193 is merged
        sim.gpu_state_vector.sync_state_to_cpu();
        DVector::from_row_slice(sim.gpu_state_vector.as_slice())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GPUSVError {}

#[cfg(test)]
mod tests {
    use cubecl::wgpu::WgpuRuntime;

    use crate::{
        circuit::{Circuit, PureCircuit},
        expr_dsl::expr_helpers::{r, rb},
        ext::equal_state_c,
        gpu_sv_simulator::GpuStateVectorSimulator,
        simulator::{BuildSimulator, RunnableSimulator},
        sv_simulator::SVSimulator,
    };

    #[test]
    fn test_qft() {
        let n_qubits = 4;
        let circuit = Circuit::<PureCircuit>::new_qft(n_qubits);

        let gpu = GpuStateVectorSimulator::<WgpuRuntime>::build(circuit.clone()).unwrap();
        let cpu = SVSimulator::build(circuit).unwrap();

        println!("{}", &gpu.final_state());
        println!("{}", &cpu.final_state());
        assert!(equal_state_c(
            &gpu.final_state(),
            &cpu.final_state(),
            n_qubits,
            0.001
        ));
    }

    #[test]
    fn test_sampling() {
        let circuit = Circuit::<PureCircuit>::new(15).x(0).x(7).x(14);
        let gpu = GpuStateVectorSimulator::<WgpuRuntime>::build(circuit).unwrap();

        assert_eq!(gpu.run(), (1 << 14) | (1 << 7) | 1);
    }

    #[test]
    fn test_hybrid() {
        let circuit = Circuit::new(4)
            .new_reg("rbits", 4)
            // Init random state
            .h(0)
            .h(1)
            .h(2)
            .h(3)
            .measure_bit(0, ("rbits", 0))
            .measure_bit(1, ("rbits", 1))
            .measure_bit(2, ("rbits", 2))
            .measure_bit(3, ("rbits", 3))
            .apply_if(rb("rbits", 0).eq(1))
            .x(0)
            .apply_if(rb("rbits", 1).eq(1))
            .x(1)
            .apply_if(rb("rbits", 2).eq(1))
            .x(2)
            .apply_if(rb("rbits", 3).eq(1))
            .x(3);

        let sim = GpuStateVectorSimulator::<WgpuRuntime>::build(circuit).unwrap();

        for i in 0..20 {
            assert_eq!(sim.run(), 0, "in iter {i}");
        }
    }

    #[test]
    fn test_jump_into_batch() {
        let circuit = Circuit::new(3)
            .x(0)
            .jump("target")
            .x(1)
            .label("target")
            .x(2);

        let gpu = GpuStateVectorSimulator::<WgpuRuntime>::build(circuit.clone()).unwrap();
        let cpu = SVSimulator::build(circuit).unwrap();

        assert_eq!(cpu.run(), 0b101);
        assert_eq!(gpu.run(), cpu.run());
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

        let gpu = GpuStateVectorSimulator::<WgpuRuntime>::build(circuit.clone()).unwrap();
        let cpu = SVSimulator::build(circuit).unwrap();

        assert!(equal_state_c(
            &gpu.final_state(),
            &cpu.final_state(),
            4,
            0.001
        ));

        for i in 0..10 {
            assert_eq!(gpu.run(), 0, "gpu iter {i}");
            assert_eq!(cpu.run(), 0, "cpu iter {i}");
        }
    }
}
