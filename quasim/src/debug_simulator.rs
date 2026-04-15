use crate::{
    cart,
    circuit::{Circuit, HybridCircuit, PureCircuit, pc::CircuitPc},
    expr_dsl::{BitExpr, BoolExpr},
    ext::{collapse, expand_matrix_from_gate, measure_and_observe_sv},
    instruction::Instruction,
    register_file::{RegisterError, RegisterFile},
    simulator::{Debuggable, Sampleable, Simulator, StoredCircuit, StoredRegisters},
};
use nalgebra::{Complex, DVector};

#[derive(Debug, Clone)]
pub struct DebugSimulator {
    current_state: DVector<Complex<f32>>,
    circuit: Circuit<HybridCircuit>,
    pc: CircuitPc,
    registers: RegisterFile,
}

impl TryFrom<Circuit<PureCircuit>> for DebugSimulator {
    type Error = DebugSimulatorError;

    fn try_from(value: Circuit<PureCircuit>) -> Result<Self, Self::Error> {
        Self::try_from(Circuit::<HybridCircuit>::from(value.into()))
    }
}

impl TryFrom<Circuit<HybridCircuit>> for DebugSimulator {
    type Error = DebugSimulatorError;

    fn try_from(value: Circuit<HybridCircuit>) -> Result<Self, Self::Error> {
        let circuit = value;
        let k = circuit.n_qubits();

        // Initial state assumed to be |000..>
        let mut init_state = vec![cart!(0.0); 1 << k];
        init_state[0] = cart!(1.0);

        let registers = RegisterFile::from(circuit.registers());

        let sim = DebugSimulator {
            current_state: DVector::from_vec(init_state),
            circuit: circuit,
            pc: Default::default(),
            registers: registers,
        };

        Ok(sim)
    }
}

impl Simulator for DebugSimulator {
    type State = DVector<Complex<f32>>;
    type BasisValue = Complex<f32>;

    fn run(&mut self) {
        while self.next() {}
    }

    fn reset(&mut self) {
        self.current_state.fill(cart!(0.0));
        self.current_state[0] = cart!(1.0);
        self.pc = Default::default();
        self.registers.reset();
    }

    fn state(&self) -> &Self::State {
        &self.current_state
    }
}

impl StoredRegisters for DebugSimulator {
    fn registers(&self) -> &RegisterFile {
        &self.registers
    }
}

impl Debuggable for DebugSimulator {
    fn next(&mut self) -> bool {
        let Some(inst) = self.circuit.instruction(self.pc()) else {
            // End of (sub) circuit: Try to return
            if self.pc_mut().ret() {
                return true;
            }

            // Could not return: End of circuit
            return false;
        };

        match inst {
            Instruction::Gate(gate) => {
                let mat = expand_matrix_from_gate(&gate, self.circuit.n_qubits());
                self.current_state = mat * self.current_state.clone();
                self.pc_mut().increment();
            }
            Instruction::MeasureBit(qbit, (reg, bit_pos)) => self.measure_bit(qbit, &reg, bit_pos),
            Instruction::MeasureAll(reg) => self.measure_all(&reg),
            Instruction::Jump(pc) => self.jump(pc),
            Instruction::JumpIf(expr, pc) => self.jump_if(&expr, pc),
            Instruction::Assign(expr, reg) => self.assign(&expr, &reg),
            Instruction::Call(name, lsq, ctrl) => self.pc_mut().jump_and_link(name, lsq, ctrl),
        }
        true
    }

    fn current_instruction(&self) -> (&CircuitPc, Option<Instruction>) {
        (self.pc(), self.circuit.instruction(self.pc()))
    }

    fn prev(&mut self) -> bool {
        if !self.pc_mut().decrement() {
            // Beginnning of (sub) circuit: Try to return
            if self.pc_mut().ret_backwards() {
                return true;
            }

            // Could not return: Beginning of circuit
            return false;
        }

        // Will happen if doing prev into a sub circuit
        // i.e. we are at the end of a sub circuit
        let Some(inst) = self.circuit.instruction(self.pc()) else {
            // Pc already decremented: do nothing
            return true;
        };

        match inst {
            Instruction::Gate(gate) => {
                let mat = expand_matrix_from_gate(&gate, self.circuit.n_qubits()).adjoint(); // All matricies are unitary --> inverse <=> adjoint
                self.current_state = mat * self.current_state.clone();
            }
            Instruction::MeasureBit(_, _) => todo!(),
            Instruction::MeasureAll(_) => todo!(),
            Instruction::Jump(_) => todo!(),
            Instruction::JumpIf(_, _) => todo!(),
            Instruction::Assign(_, _) => todo!(),
            Instruction::Call(name, lsq, ctrl) => {
                let inst_count = match self.circuit.current_sub_circuit(self.pc()) {
                    Some(sub_circuit) => sub_circuit.sub_circuit(&name).instructions().len(),
                    None => self.circuit.sub_circuit(&name).instructions().len(),
                };
                self.pc_mut().jump_and_link(name, lsq, ctrl);
                self.pc_mut().jump(inst_count); // Place pc at end of sub circuit
            }
        }
        true
    }

    fn double_ended(&self) -> bool {
        true
    }
}

impl DebugSimulator {
    fn measure_bit(&mut self, target: usize, reg: &str, bit_pos: usize) {
        let (measurement, new_state) =
            measure_and_observe_sv(target, &self.current_state, self.n_qubits());

        self.registers[reg]
            .write_bit(bit_pos, measurement)
            .expect("invalid register write");

        self.current_state = new_state;

        self.pc_mut().increment();
    }

    fn measure_all(&mut self, reg: &str) {
        let measurement = collapse(self.current_state.as_slice());

        self.registers[reg].write(measurement);

        // Collapse whole state vector
        self.current_state.fill(cart!(0.0));
        self.current_state[measurement] = cart!(1.0);

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

    fn pc(&self) -> &CircuitPc {
        &self.pc
    }

    fn pc_mut(&mut self) -> &mut CircuitPc {
        &mut self.pc
    }
}

impl StoredCircuit for DebugSimulator {
    type B = HybridCircuit;
    fn circuit(&self) -> &Circuit<HybridCircuit> {
        &self.circuit
    }

    fn circuit_mut(&mut self) -> &mut Circuit<HybridCircuit> {
        &mut self.circuit
    }
}

impl Sampleable<HybridCircuit> for DebugSimulator {}
impl Sampleable<PureCircuit> for DebugSimulator {}

#[derive(Debug, Clone, thiserror::Error)]
pub enum DebugSimulatorError {
    #[error("Measurement mid-circuit")]
    MidCircuitMeasurement,
    #[error("{0}")]
    RegisterError(#[from] RegisterError),
}

#[cfg(test)]
mod tests {
    use crate::common_test;
    use crate::ext::{
        collapse, equal_matrix_c, equal_state_c, expand_matrix, expand_matrix_from_gate,
        get_gate_matrix, measure_and_observe_sv,
    };
    use crate::simulator::Simulator;
    use crate::{
        cart,
        circuit::Circuit,
        debug_simulator::DebugSimulator,
        gate::{Gate, GateType},
        simulator::{Buildable, Debuggable},
    };
    use nalgebra::{Complex, DMatrix, DVector, dmatrix, dvector};
    use std::f32::consts::FRAC_1_SQRT_2;

    #[test]
    fn measure_hadamard_all() {
        let circ = Circuit::new(3).h(0).h(1).h(2);
        let mut sim = DebugSimulator::build(circ).expect("Circuit should be valid");
        sim.cont();
        let mut res = sim.state().clone();
        let plus_plus_plus: DVector<Complex<f32>> = dvector![
            cart!(0.5 * FRAC_1_SQRT_2), // |000>
            cart!(0.5 * FRAC_1_SQRT_2), // |001>
            cart!(0.5 * FRAC_1_SQRT_2), // |010>
            cart!(0.5 * FRAC_1_SQRT_2), // |011>
            cart!(0.5 * FRAC_1_SQRT_2), // |100>
            cart!(0.5 * FRAC_1_SQRT_2), // |101>
            cart!(0.5 * FRAC_1_SQRT_2), // |110>
            cart!(0.5 * FRAC_1_SQRT_2), // |111>
        ];
        assert!(equal_state_c(&res, &plus_plus_plus, 3, 0.001));
        let plus_plus_measure0: DVector<Complex<f32>> = dvector![
            cart!(0.5), // |000>
            cart!(0.0), // |001>
            cart!(0.5), // |010>
            cart!(0.0), // |011>
            cart!(0.5), // |100>
            cart!(0.0), // |101>
            cart!(0.5), // |110>
            cart!(0.0), // |111>
        ];
        let plus_plus_measure1: DVector<Complex<f32>> = dvector![
            cart!(0.0), // |000>
            cart!(0.5), // |001>
            cart!(0.0), // |010>
            cart!(0.5), // |011>
            cart!(0.0), // |100>
            cart!(0.5), // |101>
            cart!(0.0), // |110>
            cart!(0.5), // |111>
        ];
        (_, res) = measure_and_observe_sv(0, &res, 3);
        assert!(
            equal_state_c(&res, &plus_plus_measure0, 3, 0.001)
                || equal_state_c(&res, &plus_plus_measure1, 3, 0.001)
        );
        let plus_measure0_measure0: DVector<Complex<f32>> = dvector![
            cart!(FRAC_1_SQRT_2), // |000>
            cart!(0.0),           // |001>
            cart!(0.0),           // |010>
            cart!(0.0),           // |011>
            cart!(FRAC_1_SQRT_2), // |100>
            cart!(0.0),           // |101>
            cart!(0.0),           // |110>
            cart!(0.0),           // |111>
        ];
        let plus_measure0_measure1: DVector<Complex<f32>> = dvector![
            cart!(0.0),           // |000>
            cart!(FRAC_1_SQRT_2), // |001>
            cart!(0.0),           // |010>
            cart!(0.0),           // |011>
            cart!(0.0),           // |100>
            cart!(FRAC_1_SQRT_2), // |101>
            cart!(0.0),           // |110>
            cart!(0.0),           // |111>
        ];
        let plus_measure1_measure0: DVector<Complex<f32>> = dvector![
            cart!(0.0),           // |000>
            cart!(0.0),           // |001>
            cart!(FRAC_1_SQRT_2), // |010>
            cart!(0.0),           // |011>
            cart!(0.0),           // |100>
            cart!(0.0),           // |101>
            cart!(FRAC_1_SQRT_2), // |110>
            cart!(0.0),           // |111>
        ];
        let plus_measure1_measure1: DVector<Complex<f32>> = dvector![
            cart!(0.0),           // |000>
            cart!(0.0),           // |001>
            cart!(0.0),           // |010>
            cart!(FRAC_1_SQRT_2), // |011>
            cart!(0.0),           // |100>
            cart!(0.0),           // |101>
            cart!(0.0),           // |110>
            cart!(FRAC_1_SQRT_2), // |111>
        ];
        (_, res) = measure_and_observe_sv(1, &res, 3);
        assert!(
            equal_state_c(&res, &plus_measure0_measure0, 3, 0.001)
                || equal_state_c(&res, &plus_measure0_measure1, 3, 0.001)
                || equal_state_c(&res, &plus_measure1_measure0, 3, 0.001)
                || equal_state_c(&res, &plus_measure1_measure1, 3, 0.001)
        );
        (_, res) = measure_and_observe_sv(2, &res, 3);
        // Now collapsed to any 3-bit-string.
        assert!(state_is_collapsed(res));
    }

    fn state_is_collapsed(vector: DVector<Complex<f32>>) -> bool {
        let mut one_count = 0;

        for &value in vector.iter() {
            if nalgebra::ComplexField::abs(value - cart!(1.0)) < 0.001 {
                one_count += 1;
            } else if nalgebra::ComplexField::abs(value - cart!(0.0)) > 0.001 {
                return false; // Found a value not equal to 0.0
            }
        }

        one_count == 1
    }

    #[test]
    fn measure_entanglement() {
        let circ = Circuit::new(3).h(0).cx(&[0], 1);
        let mut sim = DebugSimulator::build(circ).expect("Circuit should be valid");
        sim.cont();
        let mut res = sim.state().clone();
        // Expected state vector before any measurments
        let bell: DVector<Complex<f32>> = dvector![
            cart!(FRAC_1_SQRT_2), // |000>
            cart!(0.0),
            cart!(0.0),
            cart!(FRAC_1_SQRT_2), // |011>
            cart!(0.0),
            cart!(0.0),
            cart!(0.0),
            cart!(0.0),
        ];
        assert!(equal_state_c(&bell, &res, 3, 0.001));
        // When any qubit is measured, state vector should collapse to either |00> or |11>.
        let colapse_00: DVector<Complex<f32>> = dvector![
            cart!(1.0), // |000>
            cart!(0.0),
            cart!(0.0),
            cart!(0.0),
            cart!(0.0),
            cart!(0.0),
            cart!(0.0),
            cart!(0.0),
        ];
        let colapse_11: DVector<Complex<f32>> = dvector![
            cart!(0.0),
            cart!(0.0),
            cart!(0.0),
            cart!(1.0), // |011>
            cart!(0.0),
            cart!(0.0),
            cart!(0.0),
            cart!(0.0),
        ];
        (_, res) = measure_and_observe_sv(0, &res, 3);

        assert!(
            equal_state_c(&res, &colapse_00, 3, 0.001)
                || equal_state_c(&res, &colapse_11, 3, 0.001)
        );

        (_, res) = measure_and_observe_sv(1, &res, 3);
        assert!(
            equal_state_c(&res, &colapse_00, 3, 0.001)
                || equal_state_c(&res, &colapse_11, 3, 0.001)
        );
    }

    #[test]
    fn bell_state_test() {
        let circ = Circuit::new(2).h(0).cx(&[0], 1);

        let mut sim = DebugSimulator::build(circ).expect("No mid-circuit measurements");
        sim.cont();
        let collapsed = collapse(sim.state().as_ref());

        println!("bell_state_test collapsed state: 0b{:02b}", collapsed);
        assert!(collapsed == 0b00 || collapsed == 0b11);
    }

    fn textbook_cnot() -> DMatrix<Complex<f32>> {
        #[rustfmt::skip]
        let textbook_cnot: DMatrix::<Complex<f32>> = dmatrix![
            cart!(1.0), cart!(0.0), cart!(0.0), cart!(0.0);
            cart!(0.0), cart!(0.0), cart!(0.0), cart!(1.0);
            cart!(0.0), cart!(0.0), cart!(1.0), cart!(0.0);
            cart!(0.0), cart!(1.0), cart!(0.0), cart!(0.0);
        ];
        textbook_cnot
    }

    #[test]
    fn test_textbook_cnot() {
        let cnot = Gate::new(GateType::X, &[0], &[1]).unwrap();
        let mat = expand_matrix_from_gate(&cnot, 2);
        assert!(equal_matrix_c(&mat, &textbook_cnot(), 4, 0.001));
    }

    fn textbook_toffoli() -> DMatrix<Complex<f32>> {
        #[rustfmt::skip]
        let textbook_toffoli: DMatrix::<Complex<f32>> = dmatrix![
            cart!(1.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0);
            cart!(0.0), cart!(1.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0);
            cart!(0.0), cart!(0.0), cart!(1.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0);
            cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(1.0);
            cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(1.0), cart!(0.0), cart!(0.0), cart!(0.0);
            cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(1.0), cart!(0.0), cart!(0.0);
            cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(1.0), cart!(0.0);
            cart!(0.0), cart!(0.0), cart!(0.0), cart!(1.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0);
        ];
        textbook_toffoli
    }
    #[test]
    fn test_textbook_toffoli() {
        let x = Gate::new(GateType::X, &[], &[0]).unwrap();
        let mat = expand_matrix(get_gate_matrix(&x), &[0, 1], &[2], 3);
        assert!(equal_matrix_c(&mat, &textbook_toffoli(), 6, 0.001));
    }

    /* Following tests are based on 'ControlledGates.tex' */

    fn cnot_01() -> DMatrix<Complex<f32>> {
        #[rustfmt::skip]
        let cnot_01: DMatrix::<Complex<f32>> = dmatrix![
            cart!(1.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0);
            cart!(0.0), cart!(0.0), cart!(0.0), cart!(1.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0);
            cart!(0.0), cart!(0.0), cart!(1.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0);
            cart!(0.0), cart!(1.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0);
            cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(1.0), cart!(0.0), cart!(0.0), cart!(0.0);
            cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(1.0);
            cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(1.0), cart!(0.0);
            cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(1.0), cart!(0.0), cart!(0.0);
        ];
        cnot_01
    }

    #[test]
    fn test_cnot_01() {
        let cnot = Gate::new(GateType::X, &[0], &[1]).unwrap();
        let mat = expand_matrix_from_gate(&cnot, 3);
        assert!(equal_matrix_c(&mat, &cnot_01(), 6, 0.001));
    }

    fn cnot_02() -> DMatrix<Complex<f32>> {
        #[rustfmt::skip]
        let cnot_02: DMatrix::<Complex<f32>> = dmatrix![
            cart!(1.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0);
            cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(1.0), cart!(0.0), cart!(0.0);
            cart!(0.0), cart!(0.0), cart!(1.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0);
            cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(1.0);
            cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(1.0), cart!(0.0), cart!(0.0), cart!(0.0);
            cart!(0.0), cart!(1.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0);
            cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(1.0), cart!(0.0);
            cart!(0.0), cart!(0.0), cart!(0.0), cart!(1.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0);
        ];
        cnot_02
    }

    #[test]
    fn test_cnot_02() {
        let cnot = Gate::new(GateType::X, &[0], &[2]).unwrap();
        let mat = expand_matrix_from_gate(&cnot, 3);
        assert!(equal_matrix_c(&mat, &cnot_02(), 6, 0.001));
    }

    fn cnot_12() -> DMatrix<Complex<f32>> {
        #[rustfmt::skip]
        let cnot_12: DMatrix::<Complex<f32>> = dmatrix![
            cart!(1.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0);
            cart!(0.0), cart!(1.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0);
            cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(1.0), cart!(0.0);
            cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(1.0);
            cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(1.0), cart!(0.0), cart!(0.0), cart!(0.0);
            cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(1.0), cart!(0.0), cart!(0.0);
            cart!(0.0), cart!(0.0), cart!(1.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0);
            cart!(0.0), cart!(0.0), cart!(0.0), cart!(1.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0);
        ];
        cnot_12
    }

    #[test]
    fn test_cnot_12() {
        let cnot = Gate::new(GateType::X, &[1], &[2]).unwrap();
        let mat = expand_matrix_from_gate(&cnot, 3);
        assert!(equal_matrix_c(&mat, &cnot_12(), 6, 0.001));
    }

    fn h_0() -> DMatrix<Complex<f32>> {
        #[rustfmt::skip]
        let h_0: DMatrix::<Complex<f32>> = dmatrix![
            cart!(FRAC_1_SQRT_2), cart!(FRAC_1_SQRT_2), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0);
            cart!(FRAC_1_SQRT_2), -cart!(FRAC_1_SQRT_2), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0);
            cart!(0.0), cart!(0.0), cart!(FRAC_1_SQRT_2), cart!(FRAC_1_SQRT_2), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0);
            cart!(0.0), cart!(0.0), cart!(FRAC_1_SQRT_2), -cart!(FRAC_1_SQRT_2), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0);
            cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(FRAC_1_SQRT_2), cart!(FRAC_1_SQRT_2), cart!(0.0), cart!(0.0);
            cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(FRAC_1_SQRT_2), -cart!(FRAC_1_SQRT_2), cart!(0.0), cart!(0.0);
            cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(FRAC_1_SQRT_2), cart!(FRAC_1_SQRT_2);
            cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(FRAC_1_SQRT_2), -cart!(FRAC_1_SQRT_2);
        ];
        h_0
    }

    #[test]
    fn test_h_0() {
        let h = Gate::new(GateType::H, &[], &[0]).unwrap();
        let mat = expand_matrix_from_gate(&h, 3);
        assert!(equal_matrix_c(&mat, &h_0(), 6, 0.001));
    }

    fn cnot_201() -> DMatrix<Complex<f32>> {
        let cnot_201: DMatrix<Complex<f32>> = dmatrix![
            cart!(1.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0);
            cart!(0.0), cart!(1.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0);
            cart!(0.0), cart!(0.0), cart!(1.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0);
            cart!(0.0), cart!(0.0), cart!(0.0), cart!(1.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0);
            cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(1.0);
            cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(1.0), cart!(0.0);
            cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(1.0), cart!(0.0), cart!(0.0);
            cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(1.0), cart!(0.0), cart!(0.0), cart!(0.0);
        ];
        cnot_201
    }

    #[test]
    fn test_cnot_201() {
        let x = Gate::new(GateType::X, &[], &[0]).unwrap();
        let mat = expand_matrix(get_gate_matrix(&x), &[2], &[0, 1], 3);
        assert!(equal_matrix_c(&mat, &cnot_201(), 6, 0.001));
    }

    #[test]
    fn test_hadamard_double_cnot_entanglement() {
        let circ = Circuit::new(3).h(0).cx(&[0], 1).cx(&[0], 2);

        let psi0: DVector<Complex<f32>> = dvector![
            cart!(1.0), // |000>
            cart!(0.0),
            cart!(0.0),
            cart!(0.0),
            cart!(0.0),
            cart!(0.0),
            cart!(0.0),
            cart!(0.0)
        ];
        let psi1: DVector<Complex<f32>> = dvector![
            cart!(FRAC_1_SQRT_2), //|000>
            cart!(FRAC_1_SQRT_2), //|001>
            cart!(0.0),
            cart!(0.0),
            cart!(0.0),
            cart!(0.0),
            cart!(0.0),
            cart!(0.0)
        ];
        let psi2: DVector<Complex<f32>> = dvector![
            cart!(FRAC_1_SQRT_2), // |000>
            cart!(0.0),
            cart!(0.0),
            cart!(FRAC_1_SQRT_2), // |011>
            cart!(0.0),
            cart!(0.0),
            cart!(0.0),
            cart!(0.0)
        ];
        let psi3: DVector<Complex<f32>> = dvector![
            cart!(FRAC_1_SQRT_2), // |000>
            cart!(0.0),
            cart!(0.0),
            cart!(0.0),
            cart!(0.0),
            cart!(0.0),
            cart!(0.0),
            cart!(FRAC_1_SQRT_2) // |111>
        ];
        let mut sim = DebugSimulator::build(circ).expect("Should be no measurements in circ.");
        assert!(equal_state_c(&psi0, sim.state(), 3, 0.001));
        sim.next();
        assert!(equal_state_c(&psi1, sim.state(), 3, 0.001));
        sim.next();
        assert!(equal_state_c(&psi2, sim.state(), 3, 0.001));
        sim.next();
        assert!(equal_state_c(&psi3, sim.state(), 3, 0.001));

        let res = sim.next();
        match res {
            true => panic!("Does not err correctly when stepping forwards."),
            false => println!("Errs correctly when stepping forwards"),
        }

        sim.prev();
        assert!(equal_state_c(&psi2, sim.state(), 3, 0.001));
        sim.prev();
        assert!(equal_state_c(&psi1, sim.state(), 3, 0.001));
        sim.prev();
        assert!(equal_state_c(&psi0, sim.state(), 3, 0.001));

        let res = sim.prev();
        match res {
            true => panic!("Does not err correctly when stepping forwards."),
            false => println!("Errs correctly when stepping backwards"),
        }
    }

    #[test]
    fn hybrid_test() {
        common_test::hybrid_test::<DebugSimulator>();
    }

    #[test]
    fn register_test() {
        common_test::register_test::<DebugSimulator>();
    }

    #[test]
    fn test_measure_overwrites_with_zero() {
        common_test::test_measure_overwrites_with_zero::<DebugSimulator>();
    }

    #[test]
    fn test_reset() {
        common_test::test_reset::<DebugSimulator>();
    }

    #[test]
    fn test_reset_with_shared_scratch_register() {
        common_test::test_reset_with_shared_scratch_register::<DebugSimulator>();
    }

    #[test]
    fn double_sub() {
        common_test::double_sub::<DebugSimulator>();
    }

    #[test]
    fn deep_sub() {
        common_test::deep_sub::<DebugSimulator>();
    }

    #[test]
    fn deep_ctrl_sub() {
        common_test::deep_ctrl_sub::<DebugSimulator>();
    }
}
