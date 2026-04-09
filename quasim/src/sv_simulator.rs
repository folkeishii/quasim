use nalgebra::{Complex, DVector, Matrix2};
use rand::distr::{Distribution, weighted::WeightedIndex};

use crate::circuit::{CircuitBehaviour, HybridCircuit};
use crate::expr_dsl::{BitExpr, BoolExpr};
use crate::ext::get_u_matrix2;
use crate::gate::GateType;
use crate::register_file::RegisterError;
use crate::simulator::{QuantumState, StoredRegisters};
use crate::{
    cart,
    circuit::{Circuit, pc::CircuitPc},
    gate::{Gate, QBits},
    instruction::Instruction,
    register_file::RegisterFile,
    simulator::{Debuggable, Simulator, StoredCircuit},
};

// SVSimulator

pub struct SVSimulator {
    state_vector: DVector<Complex<f64>>,
    circuit: Circuit<HybridCircuit>,
    pc: CircuitPc,
    registers: RegisterFile,
}

impl SVSimulator {
    /// Step forward one instruction in the circuit
    fn step(&mut self) -> Option<&DVector<Complex<f64>>> {
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

    /// Get current state of the quantum system
    pub fn state_vector(&self) -> &DVector<Complex<f64>> {
        &self.state_vector
    }

    /// Checks that all control bits are 1
    fn controls_active(i: usize, controls: QBits) -> bool {
        let control_mask = controls.get_bitstring();
        (i & control_mask) == control_mask
    }

    /// Checks that all target bits are 0
    fn is_block_base(i: usize, targets: QBits) -> bool {
        let target_mask = targets.get_bitstring();
        (i & target_mask) == 0
    }

    // 0 1
    // 1 0
    #[inline(always)]
    fn apply_x(&mut self, base_index: usize, target: QBits) {
        self.state_vector
            .as_mut_slice()
            .swap(base_index, base_index | target.get_bitstring());
    }

    // 0 -i
    // i  0
    #[inline(always)]
    fn apply_y(&mut self, base_index: usize, target: QBits) {
        let flipped_index = base_index | target.get_bitstring();
        let state = self.state_vector.as_mut_slice();
        let a = state[base_index];
        let b = state[flipped_index];

        state[base_index] = cart!(b.im, -b.re);
        state[flipped_index] = cart!(-a.im, a.re);
    }

    // 1  0
    // 0 -1
    #[inline(always)]
    fn apply_z(&mut self, base_index: usize, target: QBits) {
        let i = base_index | target.get_bitstring();
        let amp = &mut self.state_vector[i];

        amp.re = -amp.re;
        amp.im = -amp.im;
    }

    #[inline(always)]
    fn apply_h(&mut self, base_index: usize, target: QBits) {
        let flipped_index = base_index | target.get_bitstring();
        let state = self.state_vector.as_mut_slice();
        let a = state[base_index];
        let b = state[flipped_index];
        let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;

        state[base_index] = (a + b) * inv_sqrt2;
        state[flipped_index] = (a - b) * inv_sqrt2;
    }

    #[inline(always)]
    fn apply_s(&mut self, base_index: usize, target: QBits) {
        let i = base_index | target.get_bitstring();
        let amp = self.state_vector[i];

        self.state_vector[i].re = -amp.im;
        self.state_vector[i].im = amp.re;
    }

    #[inline(always)]
    fn apply_swap(&mut self, base_index: usize, targets: QBits) {
        let t0 = targets.get_indices()[0];
        let t1 = targets.get_indices()[1];

        let i01 = base_index | (1 << t0);
        let i10 = base_index | (1 << t1);

        self.state_vector.as_mut_slice().swap(i01, i10);
    }

    #[inline(always)]
    fn apply_unitary2(&mut self, base_index: usize, u: &Matrix2<Complex<f64>>, target: QBits) {
        let flipped_index = base_index | target.get_bitstring();
        let a = self.state_vector[base_index];
        let b = self.state_vector[flipped_index];

        self.state_vector[base_index] = u[(0, 0)] * a + u[(0, 1)] * b;
        self.state_vector[flipped_index] = u[(1, 0)] * a + u[(1, 1)] * b;
    }

    fn gate(&mut self, gate: &Gate) {
        let controls = gate.get_control_bits();
        let targets = gate.get_target_bits();
        let n = self.state_vector.len();

        // No parallelization
        // State vector is length 2^n , n=num qubits
        for i in 0..n {
            if !Self::is_block_base(i, targets) {
                continue;
            }

            if !Self::controls_active(i, controls) {
                continue;
            }

            match gate.get_type() {
                GateType::X => self.apply_x(i, targets),
                GateType::Y => self.apply_y(i, targets),
                GateType::Z => self.apply_z(i, targets),
                GateType::H => self.apply_h(i, targets),
                GateType::S => self.apply_s(i, targets),
                GateType::SWAP => self.apply_swap(i, targets),
                GateType::U(theta, phi, lambda) => {
                    self.apply_unitary2(i, &get_u_matrix2(theta, phi, lambda), targets)
                }
            }
        }

        self.pc_mut().increment();
    }

    fn measure_bit(&mut self, target: usize, reg: &str, bit_pos: usize) {
        let mask = 1 << target;
        let measurement = self.get_collapsed_state() & mask;
        let measured_bit = (measurement >> target) & 1;

        self.registers[reg]
            .write_bit(bit_pos, measured_bit)
            .expect("invalid register write");

        // Go through state vector and remove amplitude for all states that do not align with measurement
        for (i, amp) in self.state_vector.iter_mut().enumerate() {
            if (i & mask) != measurement {
                *amp = Complex::ZERO;
            }
        }

        // Renormalize state vector
        let norm = self
            .state_vector
            .iter()
            .map(|x| x.norm_sqr())
            .sum::<f64>()
            .sqrt();
        self.state_vector.iter_mut().for_each(|x| *x /= norm);

        self.pc_mut().increment();
    }

    fn measure_all(&mut self, reg: &str) {
        let measurement = self.get_collapsed_state();

        self.registers[reg].write(measurement);

        // Collapse whole state vector
        self.state_vector.fill(cart!(0.0));
        self.state_vector[measurement] = cart!(1.0);

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
            Instruction::Gate(gate) => self.gate(gate),
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

impl<B> TryFrom<Circuit<B>> for SVSimulator
where
    B: CircuitBehaviour,
    Circuit<B>: Into<Circuit<HybridCircuit>>,
{
    type Error = SVError;

    fn try_from(value: Circuit<B>) -> Result<Self, Self::Error> {
        let size = 1 << value.n_qubits();
        let mut init_state_vector: DVector<Complex<f64>> = DVector::from_element(size, cart![0.0]);
        init_state_vector[0] = cart![1.0];

        let registers = RegisterFile::from(value.registers());

        Ok(Self {
            state_vector: init_state_vector,
            circuit: value.into(),
            pc: Default::default(),
            registers: registers,
        })
    }
}

impl QuantumState for DVector<Complex<f64>> {
    type BasisValue = Complex<f64>;

    fn collapse(&self) -> usize {
        let probs = self.iter().map(|&c| c.norm_sqr());

        let dist = WeightedIndex::new(probs)
            .expect("Failed to create probability distribution. Invalid or empty state vector?");
        let mut rng = rand::rng();

        dist.sample(&mut rng)
    }

    fn basis_value(&self, basis: usize) -> Self::BasisValue {
        self[basis]
    }
}

impl Simulator for SVSimulator {
    type State = DVector<Complex<f64>>;
    type BasisValue = Complex<f64>;

    fn run(&mut self) -> &mut Self {
        self.reset();
        while let Some(_) = self.step() {}

        self
    }

    fn reset(&mut self) -> &mut Self {
        self.state_vector.fill(cart!(0.0));
        self.state_vector[0] = cart!(1.0);
        self.pc = Default::default();

        self
    }

    fn state(&self) -> &Self::State {
        &self.state_vector
    }
}

impl Debuggable for SVSimulator {
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

impl StoredCircuit for SVSimulator {
    type B = HybridCircuit;

    fn circuit(&self) -> &Circuit<HybridCircuit> {
        &self.circuit
    }

    fn circuit_mut(&mut self) -> &mut Circuit<HybridCircuit> {
        &mut self.circuit
    }
}

impl StoredRegisters for SVSimulator {
    fn registers(&self) -> &RegisterFile {
        &self.registers
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SVError {
    #[error("{0}")]
    RegisterError(#[from] RegisterError),
}

#[cfg(test)]
mod tests {
    use std::f64::consts::FRAC_1_SQRT_2;

    use nalgebra::dvector;

    use crate::ext::equal_state_c;
    use crate::simulator::{Debuggable, Sampleable};
    use crate::{cart, common_test};
    use crate::{
        circuit::Circuit,
        expr_dsl::expr_helpers::r,
        simulator::{Buildable, Simulator},
        sv_simulator::SVSimulator,
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

        let mut sim = SVSimulator::build(circuit.clone()).unwrap();

        assert!(equal_state_c(
            sim.run().state(),
            &dvector![
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
            ],
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

        let mut sim = SVSimulator::build(circuit).unwrap();
        sim.cont();

        assert!(equal_state_c(
            sim.state(),
            &dvector![
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
            ],
            4,
            0.0001
        ));
        sim.cont();
        assert!(equal_state_c(
            sim.state(),
            &dvector![
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
            ],
            4,
            0.0001
        ));
        sim.cont();
        assert!(equal_state_c(
            sim.state(),
            &dvector![
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
            ],
            4,
            0.0001
        ));
        sim.cont();
        assert!(equal_state_c(
            sim.state(),
            &dvector![
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
            ],
            4,
            0.0001
        ));
        sim.cont();
        assert!(equal_state_c(
            sim.state(),
            &dvector![
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
            ],
            4,
            0.0001
        ));
    }

    #[test]
    fn hybrid_test() {
        common_test::hybrid_test::<SVSimulator>();
    }

    #[test]
    fn register_test() {
        common_test::register_test::<SVSimulator>();
    }

    #[test]
    fn test_measure_overwrites_with_zero() {
        common_test::test_measure_overwrites_with_zero::<SVSimulator>();
    }

    #[test]
    fn test_reset_with_shared_scratch_register() {
        let circuit = Circuit::new(4)
            .h(0)
            .h(1)
            .h(2)
            .h(3)
            .reset(0)
            .reset(1)
            .reset(2)
            .reset(3);

        let sampler = circuit.sample();

        for _ in 0..100 {
            assert!(SVSimulator::sample_once(&sampler).unwrap() == 0);
        }
    }

    #[test]
    fn double_sub() {
        common_test::double_sub::<SVSimulator>();
    }

    #[test]
    fn deep_sub() {
        common_test::deep_sub::<SVSimulator>();
    }

    #[test]
    fn deep_ctrl_sub() {
        common_test::deep_ctrl_sub::<SVSimulator>();
    }
}
