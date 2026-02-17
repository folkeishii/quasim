use crate::{cart, Circuit, Instruction, SimpleSimulator};
use nalgebra::{Complex, DMatrix, DVector, dmatrix};
use rand::{distr::weighted::WeightedIndex, prelude::*};

pub struct DebugSimulator {
    pub states: Vec<DVector<Complex<f32>>>,
}

impl SimpleSimulator for DebugSimulator {
    type E = SimpleError;

    fn build(circuit: Circuit) -> Result<Self, Self::E> {
        let k = circuit.n_qubits;

        // Initial state assumed to be |000..>
        let mut init_state = vec![cart!(0.0); 1 << k];
        init_state[0] = cart!(1.0);
        let mut sim = DebugSimulator {
            states: vec![DVector::from_vec(init_state)],
        };

        for inst in circuit.instructions {
            let mat = DebugSimulator::expand_matrix_from_instruction(inst, k);
            let v = sim.final_state();
            let w = mat * v;
            sim.states.push(w);
        }

        Ok(sim)
    }

    fn final_state(&self) -> DVector<nalgebra::Complex<f32>> {
        return self
            .states
            .last()
            .expect("states should, at least, contain the initial state")
            .clone();
    }

    fn run(&self) -> usize {
        let probs: Vec<f32> = self.final_state().iter().map(|&c| c.norm_sqr()).collect();

        let dist = WeightedIndex::new(probs).unwrap();
        let mut rng = rand::rng();

        dist.sample(&mut rng)
    }
}

impl DebugSimulator {
    fn expand_matrix_from_instruction(inst: Instruction, n_qubits: usize) -> DMatrix<Complex<f32>> {
        DebugSimulator::expand_matrix(
            inst.get_matrix(),
            &inst.get_controls(),
            &inst.get_targets(),
            n_qubits,
        )
    }

    fn identity_tensor_factors(n_factors: usize) -> Vec<DMatrix<Complex<f32>>> {
        vec![DMatrix::<Complex<f32>>::identity(2, 2); n_factors]
    }

    fn eval_tensor_product(tensor_factors: Vec<DMatrix<Complex<f32>>>) -> DMatrix<Complex<f32>> {
        tensor_factors.iter().fold(
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
pub enum SimpleError {
    #[error("Measurement mid-circuit")]
    MidCircuitMeasurement
}

#[cfg(test)]
mod tests {
    use crate::{cart, Circuit, DebugSimulator, Instruction, SimpleSimulator};
    use nalgebra::{Complex, DMatrix, DVector, dmatrix, dvector};
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
        let circ = Circuit {
            instructions: vec![Instruction::H(0), Instruction::CNOT(0, 1)],
            n_qubits: 2,
        };

        let sim = DebugSimulator::build(circ).unwrap();
        let collapsed = sim.run();
        println!("bell_state_test collapsed state: 0b{:02b}", collapsed);
        assert!(collapsed == 0b00 || collapsed == 0b11);
    }

    fn textbook_cnot() -> DMatrix<Complex<f32>> {
        #[rustfmt::skip]
        let textbook_cnot: DMatrix::<Complex<f32>> = dmatrix![
            cart!(1.0),cart!(0.0),cart!(0.0),cart!(0.0);
            cart!(0.0),cart!(1.0),cart!(0.0),cart!(0.0);
            cart!(0.0),cart!(0.0),cart!(0.0),cart!(1.0);
            cart!(0.0),cart!(0.0),cart!(1.0),cart!(0.0);
        ];
        textbook_cnot
    }

    #[test]
    fn test_textbook_cnot() {
        let mat = DebugSimulator::expand_matrix_from_instruction(Instruction::CNOT(0, 1), 2);
        assert_is_matrix_equal!(mat, textbook_cnot());
    }

    fn textbook_toffoli() -> DMatrix<Complex<f32>> {
        #[rustfmt::skip]
        let textbook_toffoli: DMatrix::<Complex<f32>> = dmatrix![
            cart!(1.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0);
            cart!(0.0),cart!(1.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0);
            cart!(0.0),cart!(0.0),cart!(1.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0);
            cart!(0.0),cart!(0.0),cart!(0.0),cart!(1.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0);
            cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(1.0),cart!(0.0),cart!(0.0),cart!(0.0);
            cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(1.0),cart!(0.0),cart!(0.0);
            cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(1.0);
            cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(1.0),cart!(0.0);
        ];
        textbook_toffoli
    }
    #[test]
    fn test_textbook_toffoli() {
        let mat = DebugSimulator::expand_matrix(Instruction::X(0).get_matrix(), &[0, 1], &[2], 3);
        assert_is_matrix_equal!(mat, textbook_toffoli());
    }

    /* Following tests are based on 'ControlledGates.tex' */

    fn cnot_01() -> DMatrix<Complex<f32>> {
        #[rustfmt::skip]
        let cnot_01: DMatrix::<Complex<f32>> = dmatrix![
            cart!(1.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0);
            cart!(0.0),cart!(1.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0);
            cart!(0.0),cart!(0.0),cart!(1.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0);
            cart!(0.0),cart!(0.0),cart!(0.0),cart!(1.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0);
            cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(1.0),cart!(0.0);
            cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(1.0);
            cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(1.0),cart!(0.0),cart!(0.0),cart!(0.0);
            cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(1.0),cart!(0.0),cart!(0.0);
        ];
        cnot_01
    }

    #[test]
    fn test_cnot_01() {
        let mat = DebugSimulator::expand_matrix_from_instruction(Instruction::CNOT(0, 1), 3);
        assert_is_matrix_equal!(mat, cnot_01());
    }

    fn cnot_02() -> DMatrix<Complex<f32>> {
        #[rustfmt::skip]
        let cnot_02: DMatrix::<Complex<f32>> = dmatrix![
            cart!(1.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0);
            cart!(0.0),cart!(1.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0);
            cart!(0.0),cart!(0.0),cart!(1.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0);
            cart!(0.0),cart!(0.0),cart!(0.0),cart!(1.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0);
            cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(1.0),cart!(0.0),cart!(0.0);
            cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(1.0),cart!(0.0),cart!(0.0),cart!(0.0);
            cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(1.0);
            cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(1.0),cart!(0.0);
        ];
        cnot_02
    }

    #[test]
    fn test_cnot_02() {
        let mat = DebugSimulator::expand_matrix_from_instruction(Instruction::CNOT(0, 2), 3);
        assert_is_matrix_equal!(mat, cnot_02());
    }

    fn cnot_12() -> DMatrix<Complex<f32>> {
        #[rustfmt::skip]
        let cnot_12: DMatrix::<Complex<f32>> = dmatrix![
            cart!(1.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0);
            cart!(0.0),cart!(1.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0);
            cart!(0.0),cart!(0.0),cart!(0.0),cart!(1.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0);
            cart!(0.0),cart!(0.0),cart!(1.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0);
            cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(1.0),cart!(0.0),cart!(0.0),cart!(0.0);
            cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(1.0),cart!(0.0),cart!(0.0);
            cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(1.0);
            cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(1.0),cart!(0.0);
        ];
        cnot_12
    }

    #[test]
    fn test_cnot_12() {
        let mat = DebugSimulator::expand_matrix_from_instruction(Instruction::CNOT(1, 2), 3);
        assert_is_matrix_equal!(mat, cnot_12());
    }

    fn h_0() -> DMatrix<Complex<f32>> {
        #[rustfmt::skip]
        let h_0: DMatrix::<Complex<f32>> = dmatrix![
            cart!(FRAC_1_SQRT_2),cart!(0.0),cart!(0.0),cart!(0.0), cart!(FRAC_1_SQRT_2), cart!(0.0), cart!(0.0), cart!(0.0);
            cart!(0.0),cart!(FRAC_1_SQRT_2),cart!(0.0),cart!(0.0), cart!(0.0), cart!(FRAC_1_SQRT_2), cart!(0.0), cart!(0.0);
            cart!(0.0),cart!(0.0),cart!(FRAC_1_SQRT_2),cart!(0.0), cart!(0.0), cart!(0.0), cart!(FRAC_1_SQRT_2), cart!(0.0);
            cart!(0.0),cart!(0.0),cart!(0.0),cart!(FRAC_1_SQRT_2), cart!(0.0), cart!(0.0), cart!(0.0), cart!(FRAC_1_SQRT_2);
            cart!(FRAC_1_SQRT_2),cart!(0.0),cart!(0.0),cart!(0.0),-cart!(FRAC_1_SQRT_2), cart!(0.0), cart!(0.0), cart!(0.0);
            cart!(0.0),cart!(FRAC_1_SQRT_2),cart!(0.0),cart!(0.0), cart!(0.0),-cart!(FRAC_1_SQRT_2), cart!(0.0), cart!(0.0);
            cart!(0.0),cart!(0.0),cart!(FRAC_1_SQRT_2),cart!(0.0), cart!(0.0), cart!(0.0),-cart!(FRAC_1_SQRT_2), cart!(0.0);
            cart!(0.0),cart!(0.0),cart!(0.0),cart!(FRAC_1_SQRT_2), cart!(0.0), cart!(0.0), cart!(0.0),-cart!(FRAC_1_SQRT_2);
        ];
        h_0
    }

    #[test]
    fn test_h_0() {
        let mat = DebugSimulator::expand_matrix_from_instruction(Instruction::H(0), 3);
        assert_is_matrix_equal!(mat, h_0());
    }

    fn cnot_201() -> DMatrix<Complex<f32>> {
        let cnot_201: DMatrix<Complex<f32>> = dmatrix![
            cart!(1.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0);
            cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(1.0);
            cart!(0.0),cart!(0.0),cart!(1.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0);
            cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(1.0),cart!(0.0),cart!(0.0);
            cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(1.0),cart!(0.0),cart!(0.0),cart!(0.0);
            cart!(0.0),cart!(0.0),cart!(0.0),cart!(1.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0);
            cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(1.0),cart!(0.0);
            cart!(0.0),cart!(1.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0),cart!(0.0);
        ];
        cnot_201
    }

    #[test]
    fn test_cnot_201() {
        let mat = DebugSimulator::expand_matrix(Instruction::X(0).get_matrix(), &[2], &[0, 1], 3);
        assert_is_matrix_equal!(mat, cnot_201());
    }

    #[test]
    fn test_hadamard_double_cnot_entanglement() {
        let instructions = vec![
            Instruction::H(0),
            Instruction::CNOT(0, 1),
            Instruction::CNOT(0, 2),
        ];
        let circ = Circuit {
            instructions: instructions,
            n_qubits: 3,
        };
        let sim = DebugSimulator::build(circ).unwrap();
        assert!(sim.states.len() == 4);
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
        assert_is_vector_equal!(psi0, sim.states[0].clone());
        let psi1: DVector<Complex<f32>> = dvector![
            cart!(FRAC_1_SQRT_2), //|000>
            cart!(0.0),
            cart!(0.0),
            cart!(0.0),
            cart!(FRAC_1_SQRT_2), //|100>
            cart!(0.0),
            cart!(0.0),
            cart!(0.0)
        ];
        assert_is_vector_equal!(psi1, sim.states[1].clone());
        let psi2: DVector<Complex<f32>> = dvector![
            cart!(FRAC_1_SQRT_2), // |000>
            cart!(0.0),
            cart!(0.0),
            cart!(0.0),
            cart!(0.0),
            cart!(0.0),
            cart!(FRAC_1_SQRT_2), // |110>
            cart!(0.0)
        ];
        assert_is_vector_equal!(psi2, sim.states[2].clone());
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
        assert_is_vector_equal!(psi3.clone(), sim.states[3].clone());
        assert_is_vector_equal!(psi3, sim.final_state());
    }
}
