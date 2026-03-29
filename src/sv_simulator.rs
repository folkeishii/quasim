use nalgebra::{Complex, DVector, Matrix2};
use rand::distr::{Distribution, weighted::WeightedIndex};

use crate::circuit::{CircuitBehaviour, HybridCircuit};
use crate::ext::get_u_matrix2;
use crate::gate::GateType;
use crate::simulator::HybridSimulator;
use crate::{
    cart,
    circuit::{Circuit, pc::CircuitPc},
    expr_dsl::{Expr, Value},
    gate::{Gate, QBits},
    instruction::Instruction,
    register_file::RegisterFile,
    simulator::{DebuggableSimulator, RunnableSimulator, StoredCircuitSimulator},
};

#[derive(Debug, Clone)]
pub struct SVExecutor {
    state_vector: DVector<Complex<f64>>,
    circuit: Circuit<HybridCircuit>,
    pc: CircuitPc,
    registers: RegisterFile<Value>,
}

impl SVExecutor {
    fn new(circuit: Circuit<HybridCircuit>) -> Self {
        let size = 1 << circuit.n_qubits();
        let mut init_state_vector: DVector<Complex<f64>> = DVector::from_element(size, cart![0.0]);
        init_state_vector[0] = cart![1.0];

        let registers = RegisterFile::from(circuit.registers());
        Self {
            state_vector: init_state_vector,
            circuit: circuit,
            pc: Default::default(),
            registers: registers,
        }
    }

    /// Step forward one instruction in the circuit
    pub fn step(&mut self) -> Option<&DVector<Complex<f64>>> {
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

    /// Run the entire circuit
    pub fn step_all(&mut self) -> &Self {
        while let Some(_) = self.step() {}
        self
    }

    /// Gets a collapsed result from the current state vector
    pub fn get_collapsed_state(&self) -> usize {
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
        for (i, amp) in self.state_vector.iter_mut().enumerate() {
            if (i & mask) != measured_bit {
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

        self.registers[reg] = Value::Int(measurement as i32);

        // Collapse whole state vector
        self.state_vector.fill(cart!(0.0));
        self.state_vector[measurement] = cart!(1.0);

        self.pc_mut().increment();
    }

    fn jump(&mut self, label_pc: usize) {
        self.pc_mut().jump(label_pc);
    }

    fn jump_if(&mut self, expr: &Expr, label_pc: usize) {
        match expr.eval(&self.registers) {
            Ok(Value::Bool(true)) => self.jump(label_pc),
            Ok(Value::Bool(false)) => self.pc_mut().increment(),
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

impl StoredCircuitSimulator for SVExecutor {
    type B = HybridCircuit;

    fn circuit(&self) -> &Circuit<HybridCircuit> {
        &self.circuit
    }

    fn circuit_mut(&mut self) -> &mut Circuit<HybridCircuit> {
        &mut self.circuit
    }
}

// SVSimulator

pub struct SVSimulator {
    circuit: Circuit<HybridCircuit>,
}

impl<T> TryFrom<Circuit<T>> for SVSimulator
where
    T: CircuitBehaviour,
    Circuit<T>: Into<Circuit<HybridCircuit>>,
{
    type Error = SVError;

    fn try_from(value: Circuit<T>) -> Result<Self, Self::Error> {
        Ok(Self {
            circuit: value.into(),
        })
    }
}

impl RunnableSimulator for SVSimulator {
    fn run(&self) -> usize {
        SVExecutor::new(self.circuit.clone())
            .step_all()
            .get_collapsed_state()
    }

    fn final_state(&self) -> DVector<Complex<f64>> {
        SVExecutor::new(self.circuit.clone())
            .step_all()
            .state_vector()
            .clone()
    }
}

// SVSimulatorDebugger

#[derive(Debug, Clone)]
pub struct SVSimulatorDebugger {
    executor: SVExecutor,
}

impl<T> TryFrom<Circuit<T>> for SVSimulatorDebugger
where
    T: CircuitBehaviour,
    Circuit<T>: Into<Circuit<HybridCircuit>>,
{
    type Error = SVError;

    fn try_from(value: Circuit<T>) -> Result<Self, Self::Error> {
        Ok(Self {
            executor: SVExecutor::new(value.into()),
        })
    }
}

impl DebuggableSimulator for SVSimulatorDebugger {
    fn next(&mut self) -> Option<&DVector<Complex<f64>>> {
        self.executor.step()
    }

    fn current_instruction(&self) -> (&CircuitPc, Option<Instruction>) {
        let pc = self.executor.pc();
        (pc, self.executor.circuit.instruction(pc))
    }

    fn current_state(&self) -> &DVector<Complex<f64>> {
        &self.executor.state_vector
    }

    fn prev(&mut self) -> Option<&DVector<Complex<f64>>> {
        None
    }

    fn double_ended(&self) -> bool {
        false
    }
}

impl StoredCircuitSimulator for SVSimulatorDebugger {
    type B = HybridCircuit;

    fn circuit(&self) -> &Circuit<HybridCircuit> {
        &self.executor.circuit
    }

    fn circuit_mut(&mut self) -> &mut Circuit<HybridCircuit> {
        &mut self.executor.circuit
    }
}

impl HybridSimulator<Value> for SVSimulatorDebugger {
    fn registers(&self) -> &RegisterFile<Value> {
        &self.executor.registers
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SVError {}

#[cfg(test)]
mod tests {
    use std::f64::consts::FRAC_1_SQRT_2;

    use nalgebra::dvector;

    use crate::expr_dsl::Value;
    use crate::ext::equal_to_matrix_c;
    use crate::simulator::{DebuggableSimulator, HybridSimulator};
    use crate::sv_simulator::SVSimulatorDebugger;
    use crate::{cart, common_test};
    use crate::{
        circuit::Circuit,
        expr_dsl::expr_helpers::r,
        simulator::{BuildSimulator, RunnableSimulator},
        sv_simulator::SVSimulator,
    };

    #[test]
    fn test() {
        let circuit = Circuit::new(4)
            .new_reg("r0")
            .new_reg("r1")
            .new_reg("r2")
            .new_reg("r3")
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

        let sim = SVSimulator::build(circuit.clone()).unwrap();

        assert!(equal_to_matrix_c(
            &sim.final_state(),
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
            0.0001
        ));
    }

    #[allow(unreachable_code)]
    #[test]
    fn test_sub() {
        let sub = Circuit::new(1).h(0).breakpoint();
        let circuit = Circuit::new(4)
            .new_reg("tmp")
            .new_sub_circuit("U", sub)
            // Hybrid check
            .assign("tmp", 0.into())
            .call("U", 0)
            .measure_bit(0, ("tmp", 0))
            .apply_if(r("tmp").gt(0))
            .x(0)
            // Hybrid check
            .assign("tmp", 0.into())
            .call("U", 1)
            .measure_bit(1, ("tmp", 0))
            .apply_if(r("tmp").gt(0))
            .x(1)
            // Hybrid check
            .assign("tmp", 0.into())
            .call("U", 2)
            .measure_bit(2, ("tmp", 0))
            .apply_if(r("tmp").gt(0))
            .x(2)
            // Hybrid check
            .assign("tmp", 0.into())
            .call("U", 3)
            .measure_bit(3, ("tmp", 0))
            .apply_if(r("tmp").gt(0))
            .x(3);

        let mut sim = SVSimulatorDebugger::build(circuit).unwrap();

        assert!(equal_to_matrix_c(
            sim.cont(),
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
            0.0001
        ));
        assert!(equal_to_matrix_c(
            sim.cont(),
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
            0.0001
        ));
        assert!(equal_to_matrix_c(
            sim.cont(),
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
            0.0001
        ));
        assert!(equal_to_matrix_c(
            sim.cont(),
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
            0.0001
        ));
        assert!(equal_to_matrix_c(
            sim.cont(),
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
            0.0001
        ));
    }

    #[test]
    fn test_register() {
        let circuit = Circuit::new(2).new_reg("r0").x(1).measure_bit(1, ("r0", 0));

        let mut sim = SVSimulatorDebugger::build(circuit).unwrap();
        sim.executor.step_all();

        assert_eq!(sim.register("r0"), Value::Int(1));
    }

    #[test]
    fn test_measure_bit_overwrites_existing_zero() {
        let circuit = Circuit::new(2)
            .new_reg("tmp")
            .x(0)
            .measure_bit(0, ("tmp", 0))
            .measure_bit(1, ("tmp", 0));

        let mut sim = SVSimulatorDebugger::build(circuit).unwrap();
        sim.executor.step_all();

        assert_eq!(sim.register("tmp"), Value::Int(0));
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

        let sim = SVSimulator::build(circuit).unwrap();

        for _ in 0..100 {
            assert!(sim.run() == 0);
        }
    }

    #[test]
    fn apply_gates() {
        common_test::apply_gates::<SVSimulatorDebugger>();
    }

    #[test]
    fn double_sub() {
        common_test::double_sub::<SVSimulatorDebugger>();
    }

    #[test]
    fn deep_sub() {
        common_test::deep_sub::<SVSimulatorDebugger>();
    }

    #[test]
    fn deep_ctrl_sub() {
        common_test::deep_ctrl_sub::<SVSimulatorDebugger>();
    }
}
