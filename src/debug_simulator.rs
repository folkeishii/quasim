use crate::{
    cart, circuit::Circuit, ext::get_gate_matrix, gate::Gate, instruction::Instruction, simulator::{DebuggableSimulator, DoubleEndedSimulator}
};
use nalgebra::{Complex, DMatrix, DVector, dmatrix};

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
        match &self.circuit.instructions()[self.current_step] {
            Instruction::Gate(gate) => {
                if self.current_step >= self.n_instructions() {
                    return None;
                }
                let mat = Self::expand_matrix_from_gate(
                    &gate,
                    self.circuit.n_qubits(),
                );
                self.current_state = mat * self.current_state.clone();
                self.current_step += 1;
                Some(&self.current_state)
            },
            Instruction::Measurement(qbits) => todo!(),
        }
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
        match &self.circuit.instructions()[self.current_step] {
            Instruction::Gate(gate) => {
                if self.current_step <= 0 {
                    return None;
                }
        
                self.current_step -= 1;
                let mut mat = Self::expand_matrix_from_gate(
                    &gate,
                    self.circuit.n_qubits(),
                );
                mat = mat
                    .try_inverse()
                    .expect("Unitary matricies should be invertible.");
                self.current_state = mat * self.current_state.clone();
            },
            Instruction::Measurement(qbits) => todo!(),
        }
        Some(&self.current_state)
    }
}

impl DebugSimulator {
    fn n_instructions(&self) -> usize {
        self.circuit.instructions().len()
    }

    fn expand_matrix_from_gate(gate: &Gate, n_qubits: usize) -> DMatrix<Complex<f32>> {
        DebugSimulator::expand_matrix(
            get_gate_matrix(gate),
            &gate.get_controls(),
            &gate.get_targets(),
            n_qubits,
        )
    }

    fn identity_tensor_factors(n_factors: usize) -> Vec<DMatrix<Complex<f32>>> {
        vec![DMatrix::<Complex<f32>>::identity(2, 2); n_factors]
    }

    fn eval_tensor_product(tensor_factors: Vec<DMatrix<Complex<f32>>>) -> DMatrix<Complex<f32>> {
        tensor_factors.iter().rev().fold(
            DMatrix::<Complex<f32>>::identity(1, 1),
            |product, factor| product.kronecker(factor),
        )
    }

    //TODO: Allow for neg_controls.
    fn expand_matrix(
        matrix_2x2: DMatrix<Complex<f32>>,
        controls: &[usize],
        targets: &[usize],
        n_qubits: usize,
    ) -> DMatrix<Complex<f32>> {
        let ketbra = [
            dmatrix![cart!(1.0), cart!(0.0); cart!(0.0), cart!(0.0)], // |0><0|
            dmatrix![cart!(0.0), cart!(0.0); cart!(0.0), cart!(1.0)], // |1><1|
        ];
        let n_controls = controls.len();

        // one term for each entry in a 'classical truth-table'
        let n_terms = 1 << n_controls;
        let mut terms = vec![DebugSimulator::identity_tensor_factors(n_qubits); n_terms];
        for i in 0..n_terms {
            let mut j: usize = 0;
            for &control in controls {
                terms[i][control] = ketbra[(i >> j) & 1].clone();
                j += 1;
            }
        }

        // Several controls -> all controls == 1 for gates to be applied.
        for &target in targets {
            terms[n_terms - 1][target] = matrix_2x2.clone();
        }

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
        cart, circuit::Circuit, debug_simulator::DebugSimulator, ext::get_gate_matrix, gate::{Gate, GateType}, instruction::Instruction, simulator::{BuildSimulator, DebuggableSimulator, DoubleEndedSimulator}
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
