use crate::{
    cart,
    circuit::Circuit,
    ext::get_gate_matrix,
    gate::Gate,
    instruction::Instruction,
    simulator::{DebuggableSimulator, DoubleEndedSimulator},
};
use nalgebra::{Complex, DMatrix, DVector, dmatrix};
use rand::prelude::*;

#[derive(Debug, Clone)]
pub struct DebugSimulator {
    current_state: DVector<Complex<f32>>,
    circuit: Circuit,
    current_step: usize,
}

impl TryFrom<Circuit> for DebugSimulator {
    type Error = DebugSimulatorError;

    fn try_from(value: Circuit) -> Result<Self, Self::Error> {
        let circuit = value;
        let k = circuit.n_qubits();

        // Check for mid-cicuit measurement
        let mut encountered = false;
        for inst in circuit.instructions() {
            let _ = inst; // avoid warning for now
            let is_measurement = false; // matches!(inst, Instruction::Measurement(_));
            if is_measurement {
                encountered = true;
            } else if encountered {
                // There was a gate between measurements
                return Err(DebugSimulatorError::MidCircuitMeasurement);
            }
        }

        // Initial state assumed to be |000..>
        let mut init_state = vec![cart!(0.0); 1 << k];
        init_state[0] = cart!(1.0);

        let sim = DebugSimulator {
            current_state: DVector::from_vec(init_state),
            circuit: circuit,
            current_step: 0,
        };

        Ok(sim)
    }
}

impl DebuggableSimulator for DebugSimulator {
    fn next(&mut self) -> Option<&DVector<Complex<f32>>> {
        if self.current_step >= self.n_instructions() {
            return None;
        }
        match &self.circuit.instructions()[self.current_step] {
            Instruction::Gate(gate) => {
                let mat = Self::expand_matrix_from_gate(&gate, self.circuit.n_qubits());
                self.current_state = mat * self.current_state.clone();
            }
            Instruction::Measurement(qbits) => {
                for qbit in qbits.get_indices() {
                    self.current_state =
                        Self::measure(qbit, &self.current_state, self.circuit.n_qubits());
                }
            }
        }
        self.current_step += 1;
        Some(&self.current_state)
    }

    fn current_instruction(&self) -> Option<(usize, &Instruction)> {
        self.circuit
            .instructions()
            .get(self.current_step)
            .map(|inst| (self.current_step, inst))
    }

    fn current_state(&self) -> &DVector<Complex<f32>> {
        &self.current_state
    }
}

impl DoubleEndedSimulator for DebugSimulator {
    fn prev(&mut self) -> Option<&DVector<Complex<f32>>> {
        if self.current_step <= 0 {
            return None;
        }
        self.current_step -= 1;
        match &self.circuit.instructions()[self.current_step] {
            Instruction::Gate(gate) => {
                let mut mat = Self::expand_matrix_from_gate(&gate, self.circuit.n_qubits());
                mat.try_inverse_mut();
                self.current_state = mat * self.current_state.clone();
            }
            Instruction::Measurement(_) => todo!(), //Not possible without saving states before
                                                    //measurements.
        }
        Some(&self.current_state)
    }
}

impl DebugSimulator {
    fn n_instructions(&self) -> usize {
        self.circuit.instructions().len()
    }

    /// # measure
    /// Returns a probable state vector after measurement.
    fn measure(
        target: usize,
        state: &DVector<Complex<f32>>,
        n_qubits: usize,
    ) -> DVector<Complex<f32>> {
        // Choose a collapsed state
        let prob_target_eq_zero = state
            .iter()
            .enumerate()
            .filter(|&(idx, _)| (1 << target) & idx == 0) // Using |..q_1q_0> convetion
            .map(|(_, c)| c.norm_sqr())
            .sum::<f32>();

        let mut rng = rand::rng();
        let random_value = rng.random_range(0.0..1.0);
        let mut result = dmatrix![cart!(0.0), cart!(0.0); cart!(0.0), cart!(1.0)]; // |1><1|
        if random_value < prob_target_eq_zero {
            // 0 was chosen as collapsed state.
            result = dmatrix![cart!(1.0), cart!(0.0); cart!(0.0), cart!(0.0)]; // |0><0|
        }

        // Calculate projection operator, M
        let mut projection_operator_prod = DebugSimulator::identity_tensor_factors(n_qubits);
        projection_operator_prod[target] = result;
        let projection_operator = DebugSimulator::eval_tensor_product(projection_operator_prod);

        /*
         * Use formula for next state,
         *
         *                 M|s>
         *  |s'>  ==   ___________
         *              _________
         *             √ <s|M|s>
         * */
        let bra_state = state.adjoint(); // <s|
        let proj_times_ket_state = projection_operator * state; // M|s>
        let normalization = (bra_state * proj_times_ket_state.clone())[(0, 0)].sqrt(); // √ <s|M|s>

        proj_times_ket_state / normalization
    }

    /// # expand_matrix_from_gate
    /// Returns the 2^n by 2^n matrix describing a gate in a n-qubit system.
    fn expand_matrix_from_gate(gate: &Gate, n_qubits: usize) -> DMatrix<Complex<f32>> {
        DebugSimulator::expand_matrix(
            get_gate_matrix(gate),
            &gate.get_controls(),
            &gate.get_targets(),
            n_qubits,
        )
    }

    /// # identity_tensor_factors
    /// A `Vec` of `n_factor` number of 2 by 2 identity matricies.
    fn identity_tensor_factors(n_factors: usize) -> Vec<DMatrix<Complex<f32>>> {
        vec![DMatrix::<Complex<f32>>::identity(2, 2); n_factors]
    }

    /// # eval_tensor_product
    /// Evaluates the tensor product of a `Vec` of matricies.
    fn eval_tensor_product(tensor_factors: Vec<DMatrix<Complex<f32>>>) -> DMatrix<Complex<f32>> {
        tensor_factors.iter().rev().fold(
            DMatrix::<Complex<f32>>::identity(1, 1),
            |product, factor| product.kronecker(factor),
        )
    }

    /// # expand_matrix
    /// Returns the 2^n by 2^n matrix describing a gate in a n-qubit system.
    fn expand_matrix(
        matrix_2x2: DMatrix<Complex<f32>>,
        controls: &[usize], //TODO: Allow for neg_controls.
        targets: &[usize],
        n_qubits: usize,
    ) -> DMatrix<Complex<f32>> {
        let ketbra = [
            dmatrix![cart!(1.0), cart!(0.0); cart!(0.0), cart!(0.0)], // |0><0|
            dmatrix![cart!(0.0), cart!(0.0); cart!(0.0), cart!(1.0)], // |1><1|
        ];
        let n_controls = controls.len();

        // Create one term for each entry as in a 'classical truth-table'.
        // ex: |0><0| * |0><0| * I +
        //   + |0><0| * |1><1| * I +
        //   + |1><1| * |0><0| * I +
        //   + |1><1| * |1><1| * U
        let n_terms = 1 << n_controls;
        let mut terms = vec![DebugSimulator::identity_tensor_factors(n_qubits); n_terms];
        for i in 0..n_terms {
            let mut j: usize = 0;
            for &control in controls {
                terms[i][control] = ketbra[(i >> j) & 1].clone();
                j += 1;
            }
        }

        // For several controls, all controls must equal 1 for gates to be applied.
        for &target in targets {
            terms[n_terms - 1][target] = matrix_2x2.clone();
        }

        // Evaluate sum of tensor products.
        let dim = 1 << n_qubits;
        terms
            .iter()
            .fold(DMatrix::<Complex<f32>>::zeros(dim, dim), |sum, term| {
                sum + DebugSimulator::eval_tensor_product(term.clone())
            })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DebugSimulatorError {
    #[error("Measurement mid-circuit")]
    MidCircuitMeasurement,
}

#[cfg(test)]
mod tests {
    use crate::{
        cart,
        circuit::Circuit,
        debug_simulator::DebugSimulator,
        ext::get_gate_matrix,
        gate::{Gate, GateType},
        simulator::{BuildSimulator, DebuggableSimulator, DoubleEndedSimulator},
    };
    use nalgebra::{Complex, DMatrix, DVector, dmatrix, dvector};
    use rand::distr::{Distribution, weighted::WeightedIndex};
    use std::f32::consts::FRAC_1_SQRT_2;

    fn is_matrix_equal_to(m1: DMatrix<Complex<f32>>, m2: DMatrix<Complex<f32>>) -> bool {
        m1.iter()
            .zip(m2.iter())
            .all(|(a, b)| nalgebra::ComplexField::abs(a - b) < 0.001)
    }

    fn is_vector_equal_to(v1: DVector<Complex<f32>>, v2: DVector<Complex<f32>>) -> bool {
        let l = v1.len();
        let m1 = DMatrix::<Complex<f32>>::from_row_slice(l, 1, v1.as_slice());
        let m2 = DMatrix::<Complex<f32>>::from_row_slice(l, 1, v2.as_slice());
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
        let mut res = sim.continue_until(None).clone();
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
        assert_is_vector_equal!(res.clone(), plus_plus_plus);
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
        res = DebugSimulator::measure(0, &res, 3);
        assert!(
            is_vector_equal_to(res.clone(), plus_plus_measure0)
                || is_vector_equal_to(res.clone(), plus_plus_measure1)
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
        res = DebugSimulator::measure(1, &res, 3);
        assert!(
            is_vector_equal_to(res.clone(), plus_measure0_measure0)
                || is_vector_equal_to(res.clone(), plus_measure0_measure1)
                || is_vector_equal_to(res.clone(), plus_measure1_measure0)
                || is_vector_equal_to(res.clone(), plus_measure1_measure1)
        );
        res = DebugSimulator::measure(2, &res, 3);
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
        let circ = Circuit::new(3).hadamard(0).cnot(0, 1);
        let mut sim = DebugSimulator::build(circ).expect("Circuit should be valid");
        let mut res = sim.continue_until(None).clone();
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
        assert_is_vector_equal!(bell, res.clone());
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
        res = DebugSimulator::measure(0, &res, 3);

        assert!(
            is_vector_equal_to(res.clone(), colapse_00.clone())
                || is_vector_equal_to(res.clone(), colapse_11.clone())
        );

        res = DebugSimulator::measure(1, &res, 3);
        assert!(
            is_vector_equal_to(res.clone(), colapse_00.clone())
                || is_vector_equal_to(res.clone(), colapse_11.clone())
        );
    }

    #[test]
    fn bell_state_test() {
        let circ = Circuit::new(2).hadamard(0).cnot(0, 1);

        let mut sim = DebugSimulator::build(circ).expect("No mid-circuit measurements");
        let probs = sim.continue_until(None).iter().map(|&c| c.norm_sqr());

        let dist = WeightedIndex::new(probs)
            .expect("Failed to create probability distribution. Invalid or empty state vector?");
        let mut rng = rand::rng();

        let collapsed = dist.sample(&mut rng);
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
        let mat = DebugSimulator::expand_matrix_from_gate(&cnot, 2);
        assert_is_matrix_equal!(mat, textbook_cnot());
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
        let mat = DebugSimulator::expand_matrix(get_gate_matrix(&x), &[0, 1], &[2], 3);
        assert_is_matrix_equal!(mat, textbook_toffoli());
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
        let mat = DebugSimulator::expand_matrix_from_gate(&cnot, 3);
        assert_is_matrix_equal!(mat, cnot_01());
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
        let mat = DebugSimulator::expand_matrix_from_gate(&cnot, 3);
        assert_is_matrix_equal!(mat, cnot_02());
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
        let mat = DebugSimulator::expand_matrix_from_gate(&cnot, 3);
        assert_is_matrix_equal!(mat, cnot_12());
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
        let mat = DebugSimulator::expand_matrix_from_gate(&h, 3);
        assert_is_matrix_equal!(mat, h_0());
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
        let mat = DebugSimulator::expand_matrix(get_gate_matrix(&x), &[2], &[0, 1], 3);
        assert_is_matrix_equal!(mat, cnot_201());
    }

    #[test]
    fn test_hadamard_double_cnot_entanglement() {
        let circ = Circuit::new(3).hadamard(0).cnot(0, 1).cnot(0, 2);

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
