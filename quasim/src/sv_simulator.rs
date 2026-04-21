use nalgebra::Complex;
use rand::distr::{Distribution, weighted::WeightedIndex};

use crate::circuit::{CircuitBehaviour, HybridCircuit};
use crate::expr_dsl::{BitExpr, BoolExpr};
use crate::register_file::RegisterError;
use crate::simulator::{Sampleable, StoredRegisters};
use crate::state_vector::StateVector;
use crate::{
    cart,
    circuit::{Circuit, pc::CircuitPc},
    instruction::Instruction,
    register_file::RegisterFile,
    simulator::{Debuggable, Simulator, StoredCircuit},
};

// SVSimulator
#[derive(Debug, Clone)]
pub struct StateVectorSimulator {
    state_vector: StateVector,
    circuit: Circuit<HybridCircuit>,
    pc: CircuitPc,
    registers: RegisterFile,
}

impl StateVectorSimulator {
    /// Step forward one instruction in the circuit
    pub fn step(&mut self) -> Option<&StateVector> {
        let Some(inst) = self.circuit.instruction(self.pc()) else {
            // End of (sub) circuit: Try to return
            if self.pc_mut().ret() {
                return Some(&self.state_vector);
            }

            // Could not return: End of circuit
            return None;
        };

        self.apply_instruction(&inst);

        Some(&self.state_vector)
    }

    /// Gets a collapsed result from the current state vector
    fn get_collapsed_state(&self) -> usize {
        let probs = self.state_vector.iter().map(|&c| c.norm_sqr());

        let dist = WeightedIndex::new(probs)
            .expect("Failed to create probability distribution. Invalid or empty state vector?");
        let mut rng = rand::rng();

        dist.sample(&mut rng)
    }

    fn measure_bit(&mut self, target: usize, reg: &str, bit_pos: usize) {
        let measured_bit = self.state_vector.measure_bit(target);

        self.registers[reg]
            .write_bit(bit_pos, measured_bit)
            .expect("invalid register write");

        self.pc_mut().increment();
    }

    fn measure_all(&mut self, reg: &str) {
        let measurement = self.state_vector.measure_all();

        self.registers[reg].write(measurement);

        self.pc_mut().increment();
    }

    fn jump(&mut self, label_pc: usize) {
        self.pc_mut().jump(label_pc);
    }

    fn jump_if(&mut self, expr: &BoolExpr, label_pc: usize) {
        if expr.eval(&self.registers) {
            self.jump(label_pc)
        } else {
            self.pc_mut().increment()
        }
    }

    fn assign(&mut self, expr: &BitExpr, reg: &str) {
        let value = expr.eval(&self.registers);
        self.registers[reg].write(value);
        self.pc_mut().increment();
    }

    fn apply_instruction(&mut self, inst: &Instruction) {
        match inst {
            Instruction::Gate(gate) => {
                self.state_vector.apply_gate(gate);
                self.pc_mut().increment();
            }
            Instruction::MeasureBit(qbit, (reg, bit_pos)) => self.measure_bit(*qbit, reg, *bit_pos),
            Instruction::MeasureAll(reg) => self.measure_all(reg),
            Instruction::Jump(pc) => self.jump(*pc),
            Instruction::JumpIf(expr, pc) => self.jump_if(expr, *pc),
            Instruction::Assign(expr, reg) => self.assign(expr, reg),
            Instruction::Call(name, lsq, ctrl) => {
                self.pc_mut().jump_and_link(name.clone(), *lsq, *ctrl)
            }
        }
    }

    fn pc(&self) -> &CircuitPc {
        &self.pc
    }

    fn pc_mut(&mut self) -> &mut CircuitPc {
        &mut self.pc
    }
}

impl<B> TryFrom<Circuit<B>> for StateVectorSimulator
where
    B: CircuitBehaviour,
    Circuit<B>: Into<Circuit<HybridCircuit>>,
{
    type Error = SVError;

    fn try_from(value: Circuit<B>) -> Result<Self, Self::Error> {
        let init_state_vector = StateVector::zeros(value.n_qubits());
        let registers = RegisterFile::from(value.registers());

        Ok(Self {
            state_vector: init_state_vector,
            circuit: value.into(),
            pc: Default::default(),
            registers: registers,
        })
    }
}

impl Simulator for StateVectorSimulator {
    type State = StateVector;
    type BasisValue = Complex<f32>;

    fn run(&mut self) {
        self.reset();
        while let Some(_) = self.step() {}
    }

    fn reset(&mut self) {
        self.state_vector.fill(cart!(0.0));
        self.state_vector[0] = cart!(1.0);
        self.pc = Default::default();
        self.registers.reset();
    }

    fn state(&self) -> &Self::State {
        &self.state_vector
    }
}

impl Debuggable for StateVectorSimulator {
    fn next(&mut self) -> bool {
        match self.step() {
            Some(_) => true,
            None => false,
        }
    }

    fn current_instruction(&self) -> (&CircuitPc, Option<Instruction>) {
        let pc = self.pc();
        (pc, self.circuit.instruction(pc))
    }

    fn prev(&mut self) -> bool {
        false
    }

    fn double_ended(&self) -> bool {
        false
    }
}

impl StoredCircuit for StateVectorSimulator {
    type B = HybridCircuit;

    fn circuit(&self) -> &Circuit<HybridCircuit> {
        &self.circuit
    }

    fn circuit_mut(&mut self) -> &mut Circuit<HybridCircuit> {
        &mut self.circuit
    }
}

impl StoredRegisters for StateVectorSimulator {
    fn registers(&self) -> &RegisterFile {
        &self.registers
    }
}

impl<B> Sampleable<B> for StateVectorSimulator
where
    B: CircuitBehaviour,
    Circuit<B>: Into<Circuit<HybridCircuit>>,
{
}

#[derive(Debug, thiserror::Error)]
pub enum SVError {
    #[error("{0}")]
    RegisterError(#[from] RegisterError),
}

#[cfg(test)]
mod tests {
    use std::f32::consts::FRAC_1_SQRT_2;

    use nalgebra::dvector;

    use crate::ext::equal_state_c;
    use crate::simulator::Debuggable;
    use crate::{cart, common_test};
    use crate::{
        circuit::Circuit,
        expr_dsl::expr_helpers::r,
        simulator::{Buildable, Simulator},
        state_vector::StateVector,
        sv_simulator::StateVectorSimulator,
    };

    #[test]
    fn test() {
        let circuit = Circuit::new(4)
            .new_reg("r0", 1)
            .new_reg("r1", 1)
            .new_reg("r2", 1)
            .new_reg("r3", 1)
            // Init random state
            .h(0)
            .h(1)
            .h(2)
            .h(3)
            .measure_bit(0, ("r0", 0))
            .measure_bit(1, ("r1", 0))
            .measure_bit(2, ("r2", 0))
            .measure_bit(3, ("r3", 0))
            .apply_if(r("r0").eq(1))
            .x(0)
            .apply_if(r("r1").eq(1))
            .x(1)
            .apply_if(r("r2").eq(1))
            .x(2)
            .apply_if(r("r3").eq(1))
            .x(3);

        let mut sim = StateVectorSimulator::build(circuit.clone()).unwrap();
        sim.run();

        assert!(equal_state_c(
            sim.state(),
            &StateVector::from(dvector![
                cart!(1), // |0000>
                cart!(0), // |0001>
                cart!(0), // |0010>
                cart!(0), // |0011>
                cart!(0), // |0100>
                cart!(0), // |0101>
                cart!(0), // |0110>
                cart!(0), // |0111>
                cart!(0), // |1000>
                cart!(0), // |1001>
                cart!(0), // |1010>
                cart!(0), // |1011>
                cart!(0), // |1100>
                cart!(0), // |1101>
                cart!(0), // |1110>
                cart!(0), // |1111>
            ]),
            4,
            0.0001
        ));
    }

    #[allow(unreachable_code)]
    #[test]
    fn test_sub() {
        let sub = Circuit::new(1).h(0).breakpoint();
        let circuit = Circuit::new(4)
            .new_reg("tmp", 1)
            .new_sub_circuit("U", sub)
            // Hybrid check
            .assign("tmp", 0)
            .call("U", 0)
            .measure_bit(0, ("tmp", 0))
            .apply_if(r("tmp").gt(0))
            .x(0)
            // Hybrid check
            .assign("tmp", 0)
            .call("U", 1)
            .measure_bit(1, ("tmp", 0))
            .apply_if(r("tmp").gt(0))
            .x(1)
            // Hybrid check
            .assign("tmp", 0)
            .call("U", 2)
            .measure_bit(2, ("tmp", 0))
            .apply_if(r("tmp").gt(0))
            .x(2)
            // Hybrid check
            .assign("tmp", 0)
            .call("U", 3)
            .measure_bit(3, ("tmp", 0))
            .apply_if(r("tmp").gt(0))
            .x(3);

        let mut sim = StateVectorSimulator::build(circuit).unwrap();
        sim.cont();

        assert!(equal_state_c(
            sim.state(),
            &StateVector::from(dvector![
                cart!(FRAC_1_SQRT_2), // |0000>
                cart!(FRAC_1_SQRT_2), // |0001>
                cart!(0),             // |0010>
                cart!(0),             // |0011>
                cart!(0),             // |0100>
                cart!(0),             // |0101>
                cart!(0),             // |0110>
                cart!(0),             // |0111>
                cart!(0),             // |1000>
                cart!(0),             // |1001>
                cart!(0),             // |1010>
                cart!(0),             // |1011>
                cart!(0),             // |1100>
                cart!(0),             // |1101>
                cart!(0),             // |1110>
                cart!(0),             // |1111>
            ]),
            4,
            0.0001
        ));
        sim.cont();
        assert!(equal_state_c(
            sim.state(),
            &StateVector::from(dvector![
                cart!(FRAC_1_SQRT_2), // |0000>
                cart!(0),             // |0001>
                cart!(FRAC_1_SQRT_2), // |0010>
                cart!(0),             // |0011>
                cart!(0),             // |0100>
                cart!(0),             // |0101>
                cart!(0),             // |0110>
                cart!(0),             // |0111>
                cart!(0),             // |1000>
                cart!(0),             // |1001>
                cart!(0),             // |1010>
                cart!(0),             // |1011>
                cart!(0),             // |1100>
                cart!(0),             // |1101>
                cart!(0),             // |1110>
                cart!(0),             // |1111>
            ]),
            4,
            0.0001
        ));
        sim.cont();
        assert!(equal_state_c(
            sim.state(),
            &StateVector::from(dvector![
                cart!(FRAC_1_SQRT_2), // |0000>
                cart!(0),             // |0001>
                cart!(0),             // |0010>
                cart!(0),             // |0011>
                cart!(FRAC_1_SQRT_2), // |0100>
                cart!(0),             // |0101>
                cart!(0),             // |0110>
                cart!(0),             // |0111>
                cart!(0),             // |1000>
                cart!(0),             // |1001>
                cart!(0),             // |1010>
                cart!(0),             // |1011>
                cart!(0),             // |1100>
                cart!(0),             // |1101>
                cart!(0),             // |1110>
                cart!(0),             // |1111>
            ]),
            4,
            0.0001
        ));
        sim.cont();
        assert!(equal_state_c(
            sim.state(),
            &StateVector::from(dvector![
                cart!(FRAC_1_SQRT_2), // |0000>
                cart!(0),             // |0001>
                cart!(0),             // |0010>
                cart!(0),             // |0011>
                cart!(0),             // |0100>
                cart!(0),             // |0101>
                cart!(0),             // |0110>
                cart!(0),             // |0111>
                cart!(FRAC_1_SQRT_2), // |1000>
                cart!(0),             // |1001>
                cart!(0),             // |1010>
                cart!(0),             // |1011>
                cart!(0),             // |1100>
                cart!(0),             // |1101>
                cart!(0),             // |1110>
                cart!(0),             // |1111>
            ]),
            4,
            0.0001
        ));
        sim.cont();
        assert!(equal_state_c(
            sim.state(),
            &StateVector::from(dvector![
                cart!(1), // |0000>
                cart!(0), // |0001>
                cart!(0), // |0010>
                cart!(0), // |0011>
                cart!(0), // |0100>
                cart!(0), // |0101>
                cart!(0), // |0110>
                cart!(0), // |0111>
                cart!(0), // |1000>
                cart!(0), // |1001>
                cart!(0), // |1010>
                cart!(0), // |1011>
                cart!(0), // |1100>
                cart!(0), // |1101>
                cart!(0), // |1110>
                cart!(0), // |1111>
            ]),
            4,
            0.0001
        ));
    }

    #[test]
    fn hybrid_test() {
        common_test::hybrid_test::<StateVectorSimulator>();
    }

    #[test]
    fn register_test() {
        common_test::register_test::<StateVectorSimulator>();
    }

    #[test]
    fn test_measure_overwrites_with_zero() {
        common_test::test_measure_overwrites_with_zero::<StateVectorSimulator>();
    }

    #[test]
    fn test_reset() {
        common_test::test_reset::<StateVectorSimulator>();
    }

    #[test]
    fn test_reset_with_shared_scratch_register() {
        common_test::test_reset_with_shared_scratch_register::<StateVectorSimulator>();
    }

    #[test]
    fn apply_gates() {
        common_test::apply_gates::<StateVectorSimulator>();
    }

    #[test]
    fn double_sub() {
        common_test::double_sub::<StateVectorSimulator>();
    }

    #[test]
    fn deep_sub() {
        common_test::deep_sub::<StateVectorSimulator>();
    }

    #[test]
    fn deep_ctrl_sub() {
        common_test::deep_ctrl_sub::<StateVectorSimulator>();
    }
}
