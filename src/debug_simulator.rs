use crate::{
    cart,
    circuit::{Circuit, pc::CircuitPc},
    ext::{Stack, expand_matrix_from_gate, measure},
    instruction::Instruction,
    simulator::{DebuggableSimulator, StoredCircuitSimulator},
};
use nalgebra::{Complex, DVector};

#[derive(Debug, Clone)]
pub struct DebugSimulator {
    current_state: DVector<Complex<f64>>,
    circuit: Circuit,
    pc_stack: Stack<CircuitPc>,
}

impl TryFrom<Circuit> for DebugSimulator {
    type Error = DebugSimulatorError;

    fn try_from(value: Circuit) -> Result<Self, Self::Error> {
        let circuit = value;
        let k = circuit.n_qubits();

        // Check for mid-cicuit measurement
        for (sc, insts) in circuit.all_instructions() {
            if sc.is_none() {
                let mut encountered = false;
                for inst in insts {
                    let is_measurement = matches!(inst, Instruction::Measurement(_, _));
                    if is_measurement {
                        encountered = true;
                    } else if encountered {
                        // There was a gate between measurements
                        return Err(DebugSimulatorError::MidCircuitMeasurement);
                    }
                }
            } else {
                if insts
                    .iter()
                    .any(|i| matches!(i, Instruction::Measurement(_, _)))
                {
                    // There was a gate between measurements
                    return Err(DebugSimulatorError::MidCircuitMeasurement);
                }
            }
        }

        // Initial state assumed to be |000..>
        let mut init_state = vec![cart!(0.0); 1 << k];
        init_state[0] = cart!(1.0);

        let sim = DebugSimulator {
            current_state: DVector::from_vec(init_state),
            circuit: circuit,
            pc_stack: Default::default(),
        };

        Ok(sim)
    }
}

impl DebuggableSimulator for DebugSimulator {
    fn next(&mut self) -> Option<&DVector<Complex<f64>>> {
        let Some(inst) = self.circuit.instruction(self.pc()) else {
            // End of (sub) circuit: step out
            return self.step_out();
        };

        match inst {
            Instruction::Gate(gate) => {
                let mat = expand_matrix_from_gate(&gate, self.circuit.n_qubits());
                self.current_state = mat * self.current_state.clone();
            }
            Instruction::Measurement(qbits, _) => {
                for qbit in qbits.get_indices() {
                    self.current_state =
                        measure(qbit, &self.current_state, self.circuit.n_qubits());
                }
            }
            Instruction::Jump(_) => todo!(),
            Instruction::JumpIf(_, _) => todo!(),
            Instruction::Assign(_, _) => todo!(),
            Instruction::Call(_, _) => todo!(),
        }
        self.pc_mut().increment();
        Some(&self.current_state)
    }

    fn current_instruction(&self) -> (&CircuitPc, Option<Instruction>) {
        (self.pc(), self.circuit.instruction(self.pc()))
    }

    fn current_state(&self) -> &DVector<Complex<f64>> {
        &self.current_state
    }

    fn prev(&mut self) -> Option<&DVector<Complex<f64>>> {
        if !self.pc_mut().decrement() {
            // Beginning of (sub) circuit: step out
            let state = self.step_out_prev();
            return state;
        }

        let Some(inst) = self.circuit.instruction(self.pc()) else {
            // Should not happen
            // Beginning of (sub) circuit: step out
            self.pc_stack[1].decrement(); // undo step in increment
            let state = self.step_out_prev();
            return state;
        };

        match inst {
            Instruction::Gate(gate) => {
                let mut mat = expand_matrix_from_gate(&gate, self.circuit.n_qubits());
                mat.try_inverse_mut();
                self.current_state = mat * self.current_state.clone();
            }
            Instruction::Measurement(_, _) => todo!(),
            Instruction::Jump(_) => todo!(),
            Instruction::JumpIf(_, _) => todo!(),
            Instruction::Assign(_, _) => todo!(),
            Instruction::Call(_, _) => todo!(),
        }
        Some(&self.current_state)
    }

    fn double_ended(&self) -> bool {
        true
    }
}

impl DebugSimulator {
    fn pc(&self) -> &CircuitPc {
        self.pc_stack.top()
    }

    fn pc_mut(&mut self) -> &mut CircuitPc {
        self.pc_stack.top_mut()
    }

    fn step_out(&mut self) -> Option<&DVector<Complex<f64>>> {
        if self.pc_stack.pop() {
            Some(&self.current_state)
        } else {
            None
        }
    }

    fn step_out_prev(&mut self) -> Option<&DVector<Complex<f64>>> {
        if self.pc_stack.pop() {
            self.pc_mut().decrement();
            Some(&self.current_state)
        } else {
            None
        }
    }
}

impl StoredCircuitSimulator for DebugSimulator {
    fn circuit(&self) -> &Circuit {
        &self.circuit
    }

    fn circuit_mut(&mut self) -> &mut Circuit {
        &mut self.circuit
    }
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum DebugSimulatorError {
    #[error("Measurement mid-circuit")]
    MidCircuitMeasurement,
}

#[cfg(test)]
mod tests {
    use crate::ext::{collapse, expand_matrix, expand_matrix_from_gate, get_gate_matrix, measure};
    use crate::{
        cart,
        circuit::Circuit,
        debug_simulator::DebugSimulator,
        gate::{Gate, GateType},
        simulator::{BuildSimulator, DebuggableSimulator},
    };
    use nalgebra::{Complex, DMatrix, DVector, dmatrix, dvector};
    use std::f64::consts::FRAC_1_SQRT_2;

    fn is_matrix_equal_to(m1: DMatrix<Complex<f64>>, m2: DMatrix<Complex<f64>>) -> bool {
        m1.iter()
            .zip(m2.iter())
            .all(|(a, b)| nalgebra::ComplexField::abs(a - b) < 0.001)
    }

    fn is_vector_equal_to(v1: DVector<Complex<f64>>, v2: DVector<Complex<f64>>) -> bool {
        let l = v1.len();
        let m1 = DMatrix::<Complex<f64>>::from_row_slice(l, 1, v1.as_slice());
        let m2 = DMatrix::<Complex<f64>>::from_row_slice(l, 1, v2.as_slice());
        l == v2.len() && is_matrix_equal_to(m1, m2)
    }

    macro_rules! assert_is_matrix_equal {
        ($m1: expr, $m2: expr) => {
            assert!(is_matrix_equal_to($m1, $m2))
        };
    }

    macro_rules! assert_is_vector_equal {
        ($m1: expr, $m2: expr) => {
            assert!(is_vector_equal_to($m1, $m2))
        };
    }

    #[test]
    fn measure_hadamard_all() {
        let circ = Circuit::new(3).hadamard(0).hadamard(1).hadamard(2);
        let mut sim = DebugSimulator::build(circ).expect("Circuit should be valid");
        let mut res = sim.cont().clone();
        let plus_plus_plus: DVector<Complex<f64>> = dvector![
            cart!(0.5 * FRAC_1_SQRT_2), // |000>
            cart!(0.5 * FRAC_1_SQRT_2), // |001>
            cart!(0.5 * FRAC_1_SQRT_2), // |010>
            cart!(0.5 * FRAC_1_SQRT_2), // |011>
            cart!(0.5 * FRAC_1_SQRT_2), // |100>
            cart!(0.5 * FRAC_1_SQRT_2), // |101>
            cart!(0.5 * FRAC_1_SQRT_2), // |110>
            cart!(0.5 * FRAC_1_SQRT_2), // |111>
        ];
        assert_is_vector_equal!(res.clone(), plus_plus_plus);
        let plus_plus_measure0: DVector<Complex<f64>> = dvector![
            cart!(0.5), // |000>
            cart!(0.0), // |001>
            cart!(0.5), // |010>
            cart!(0.0), // |011>
            cart!(0.5), // |100>
            cart!(0.0), // |101>
            cart!(0.5), // |110>
            cart!(0.0), // |111>
        ];
        let plus_plus_measure1: DVector<Complex<f64>> = dvector![
            cart!(0.0), // |000>
            cart!(0.5), // |001>
            cart!(0.0), // |010>
            cart!(0.5), // |011>
            cart!(0.0), // |100>
            cart!(0.5), // |101>
            cart!(0.0), // |110>
            cart!(0.5), // |111>
        ];
        res = measure(0, &res, 3);
        assert!(
            is_vector_equal_to(res.clone(), plus_plus_measure0)
                || is_vector_equal_to(res.clone(), plus_plus_measure1)
        );
        let plus_measure0_measure0: DVector<Complex<f64>> = dvector![
            cart!(FRAC_1_SQRT_2), // |000>
            cart!(0.0),           // |001>
            cart!(0.0),           // |010>
            cart!(0.0),           // |011>
            cart!(FRAC_1_SQRT_2), // |100>
            cart!(0.0),           // |101>
            cart!(0.0),           // |110>
            cart!(0.0),           // |111>
        ];
        let plus_measure0_measure1: DVector<Complex<f64>> = dvector![
            cart!(0.0),           // |000>
            cart!(FRAC_1_SQRT_2), // |001>
            cart!(0.0),           // |010>
            cart!(0.0),           // |011>
            cart!(0.0),           // |100>
            cart!(FRAC_1_SQRT_2), // |101>
            cart!(0.0),           // |110>
            cart!(0.0),           // |111>
        ];
        let plus_measure1_measure0: DVector<Complex<f64>> = dvector![
            cart!(0.0),           // |000>
            cart!(0.0),           // |001>
            cart!(FRAC_1_SQRT_2), // |010>
            cart!(0.0),           // |011>
            cart!(0.0),           // |100>
            cart!(0.0),           // |101>
            cart!(FRAC_1_SQRT_2), // |110>
            cart!(0.0),           // |111>
        ];
        let plus_measure1_measure1: DVector<Complex<f64>> = dvector![
            cart!(0.0),           // |000>
            cart!(0.0),           // |001>
            cart!(0.0),           // |010>
            cart!(FRAC_1_SQRT_2), // |011>
            cart!(0.0),           // |100>
            cart!(0.0),           // |101>
            cart!(0.0),           // |110>
            cart!(FRAC_1_SQRT_2), // |111>
        ];
        res = measure(1, &res, 3);
        assert!(
            is_vector_equal_to(res.clone(), plus_measure0_measure0)
                || is_vector_equal_to(res.clone(), plus_measure0_measure1)
                || is_vector_equal_to(res.clone(), plus_measure1_measure0)
                || is_vector_equal_to(res.clone(), plus_measure1_measure1)
        );
        res = measure(2, &res, 3);
        // Now collapsed to any 3-bit-string.
        assert!(state_is_collapsed(res));
    }

    fn state_is_collapsed(vector: DVector<Complex<f64>>) -> bool {
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
        let circ = Circuit::new(3).hadamard(0).cnot(0, 1);
        let mut sim = DebugSimulator::build(circ).expect("Circuit should be valid");
        let mut res = sim.cont().clone();
        // Expected state vector before any measurments
        let bell: DVector<Complex<f64>> = dvector![
            cart!(FRAC_1_SQRT_2), // |000>
            cart!(0.0),
            cart!(0.0),
            cart!(FRAC_1_SQRT_2), // |011>
            cart!(0.0),
            cart!(0.0),
            cart!(0.0),
            cart!(0.0),
        ];
        assert_is_vector_equal!(bell, res.clone());
        // When any qubit is measured, state vector should collapse to either |00> or |11>.
        let colapse_00: DVector<Complex<f64>> = dvector![
            cart!(1.0), // |000>
            cart!(0.0),
            cart!(0.0),
            cart!(0.0),
            cart!(0.0),
            cart!(0.0),
            cart!(0.0),
            cart!(0.0),
        ];
        let colapse_11: DVector<Complex<f64>> = dvector![
            cart!(0.0),
            cart!(0.0),
            cart!(0.0),
            cart!(1.0), // |011>
            cart!(0.0),
            cart!(0.0),
            cart!(0.0),
            cart!(0.0),
        ];
        res = measure(0, &res, 3);

        assert!(
            is_vector_equal_to(res.clone(), colapse_00.clone())
                || is_vector_equal_to(res.clone(), colapse_11.clone())
        );

        res = measure(1, &res, 3);
        assert!(
            is_vector_equal_to(res.clone(), colapse_00.clone())
                || is_vector_equal_to(res.clone(), colapse_11.clone())
        );
    }

    #[test]
    fn bell_state_test() {
        let circ = Circuit::new(2).hadamard(0).cnot(0, 1);

        let mut sim = DebugSimulator::build(circ).expect("No mid-circuit measurements");
        let collapsed = collapse(sim.cont().as_ref());

        println!("bell_state_test collapsed state: 0b{:02b}", collapsed);
        assert!(collapsed == 0b00 || collapsed == 0b11);
    }

    fn textbook_cnot() -> DMatrix<Complex<f64>> {
        #[rustfmt::skip]
        let textbook_cnot: DMatrix::<Complex<f64>> = dmatrix![
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
        assert_is_matrix_equal!(mat, textbook_cnot());
    }

    fn textbook_toffoli() -> DMatrix<Complex<f64>> {
        #[rustfmt::skip]
        let textbook_toffoli: DMatrix::<Complex<f64>> = dmatrix![
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
        assert_is_matrix_equal!(mat, textbook_toffoli());
    }

    /* Following tests are based on 'ControlledGates.tex' */

    fn cnot_01() -> DMatrix<Complex<f64>> {
        #[rustfmt::skip]
        let cnot_01: DMatrix::<Complex<f64>> = dmatrix![
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
        assert_is_matrix_equal!(mat, cnot_01());
    }

    fn cnot_02() -> DMatrix<Complex<f64>> {
        #[rustfmt::skip]
        let cnot_02: DMatrix::<Complex<f64>> = dmatrix![
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
        assert_is_matrix_equal!(mat, cnot_02());
    }

    fn cnot_12() -> DMatrix<Complex<f64>> {
        #[rustfmt::skip]
        let cnot_12: DMatrix::<Complex<f64>> = dmatrix![
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
        assert_is_matrix_equal!(mat, cnot_12());
    }

    fn h_0() -> DMatrix<Complex<f64>> {
        #[rustfmt::skip]
        let h_0: DMatrix::<Complex<f64>> = dmatrix![
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
        assert_is_matrix_equal!(mat, h_0());
    }

    fn cnot_201() -> DMatrix<Complex<f64>> {
        let cnot_201: DMatrix<Complex<f64>> = dmatrix![
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
        assert_is_matrix_equal!(mat, cnot_201());
    }

    #[test]
    fn test_hadamard_double_cnot_entanglement() {
        let circ = Circuit::new(3).hadamard(0).cnot(0, 1).cnot(0, 2);

        let psi0: DVector<Complex<f64>> = dvector![
            cart!(1.0), // |000>
            cart!(0.0),
            cart!(0.0),
            cart!(0.0),
            cart!(0.0),
            cart!(0.0),
            cart!(0.0),
            cart!(0.0)
        ];
        let psi1: DVector<Complex<f64>> = dvector![
            cart!(FRAC_1_SQRT_2), //|000>
            cart!(FRAC_1_SQRT_2), //|001>
            cart!(0.0),
            cart!(0.0),
            cart!(0.0),
            cart!(0.0),
            cart!(0.0),
            cart!(0.0)
        ];
        let psi2: DVector<Complex<f64>> = dvector![
            cart!(FRAC_1_SQRT_2), // |000>
            cart!(0.0),
            cart!(0.0),
            cart!(FRAC_1_SQRT_2), // |011>
            cart!(0.0),
            cart!(0.0),
            cart!(0.0),
            cart!(0.0)
        ];
        let psi3: DVector<Complex<f64>> = dvector![
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
        assert_is_vector_equal!(psi0.clone(), sim.current_state().clone());
        sim.next().expect("Apply Hadamard.");
        assert_is_vector_equal!(psi1.clone(), sim.current_state().clone());
        sim.next().expect("Apply first CNOT.");
        assert_is_vector_equal!(psi2.clone(), sim.current_state().clone());
        sim.next().expect("Apply second CNOT.");
        assert_is_vector_equal!(psi3.clone(), sim.current_state().clone());

        let res = sim.next();
        match res {
            Some(_) => panic!("Does not err correctly when stepping forwards."),
            None => println!("Errs correctly when stepping forwards"),
        }

        sim.prev().expect("Revert second CNOT");
        assert_is_vector_equal!(psi2.clone(), sim.current_state().clone());
        sim.prev().expect("Revert first CNOT");
        assert_is_vector_equal!(psi1.clone(), sim.current_state().clone());
        sim.prev().expect("Revert Hadamard");
        assert_is_vector_equal!(psi0.clone(), sim.current_state().clone());

        let res = sim.prev();
        match res {
            Some(_) => panic!("Does not err correctly when stepping forwards."),
            None => println!("Errs correctly when stepping backwards"),
        }
    }
}
