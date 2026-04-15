use std::marker::PhantomData;
use std::mem::replace;
use std::ops::{Deref, Index};
use std::{iter::Map, ops::Range};

use nalgebra::{Complex, DMatrix, DVector, Matrix2, dmatrix};
use rand::distr::weighted::WeightedIndex;
use rand::prelude::Distribution;

use crate::gate::{Gate, GateType};
use crate::simulator::QuantumState;

#[macro_export]
macro_rules! cart {
    ($re:expr) => {
        nalgebra::Complex {
            re: $re as f32,
            im: 0.0,
        }
    };
    ($re:expr, $im:expr) => {
        nalgebra::Complex {
            re: $re as f32,
            im: $im as f32,
        }
    };
}

#[macro_export]
macro_rules! polar {
    ($r: expr, $theta: expr) => {
        Complex::from_polar($r as f32, $theta as f32)
    };
}

#[macro_export]
macro_rules! cexp {
    ($exp: expr) => {
        Complex::exp($exp as f32)
    };
}

/// Compares two complex numbers
///
/// Two complex numbers are determined to be equal if the distance
/// between them is at most `margin`
pub fn equal_to_c(lhs: Complex<f32>, rhs: Complex<f32>, margin: f32) -> bool {
    (lhs - rhs).norm().le(&margin)
}

pub fn equal_matrix_c<'a>(
    lhs: &'a impl Index<usize, Output = Complex<f32>>,
    rhs: &'a impl Index<usize, Output = Complex<f32>>,
    n_qubits: usize,
    margin: f32,
) -> bool {
    for state in 0..(1 << n_qubits) {
        if !equal_to_c(lhs[state], rhs[state], margin) {
            return false;
        }
    }

    true
}

pub fn equal_state_c<'a>(
    lhs: &'a impl QuantumState<BasisValue = Complex<f32>>,
    rhs: &'a impl QuantumState<BasisValue = Complex<f32>>,
    n_qubits: usize,
    margin: f32,
) -> bool {
    for basis in 0..(1 << n_qubits) {
        if !equal_to_c(lhs.basis_value(basis), rhs.basis_value(basis), margin) {
            return false;
        }
    }

    true
}

pub fn reverse_indices(
    range: Range<usize>,
    len: usize,
) -> Map<Range<usize>, impl FnMut(usize) -> usize> {
    range.map(move |i| len - i - 1)
}

// Helper function for get_gate_matrix
fn u(theta: f32, phi: f32, lambda: f32) -> [Complex<f32>; 4] {
    // https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.UGate for definition
    let theta_half = theta / 2.0;

    let cos = theta_half.cos();
    let sin = theta_half.sin();
    [
        cart!(cos),
        -polar!(sin, lambda),
        polar!(sin, phi),
        polar!(cos, lambda + phi),
    ]
}

pub fn get_gate2_data(gate: &Gate) -> Option<[Complex<f32>; 4]> {
    match gate.get_type() {
        GateType::X => Some(Gate::PAULI_X_DATA),
        GateType::Y => Some(Gate::PAULI_Y_DATA),
        GateType::Z => Some(Gate::PAULI_Z_DATA),
        GateType::H => Some(Gate::HADAMARD_DATA),
        GateType::U(theta, phi, lambda) => Some(u(theta, phi, lambda)),
        GateType::S => Some(Gate::PHASE_S_DATA),
        _ => None,
    }
}

pub fn get_gate2_matrix(gate: &Gate) -> Option<Matrix2<Complex<f32>>> {
    get_gate2_data(gate).map(|d| Matrix2::from_row_slice(&d))
}

pub fn get_u_matrix2(theta: f32, phi: f32, lambda: f32) -> Matrix2<Complex<f32>> {
    Matrix2::from_row_slice(&u(theta, phi, lambda))
}

pub fn get_gate_matrix(gate: &Gate) -> DMatrix<Complex<f32>> {
    let data: &[Complex<f32>] = match gate.get_type() {
        GateType::X => &Gate::PAULI_X_DATA,
        GateType::Y => &Gate::PAULI_Y_DATA,
        GateType::Z => &Gate::PAULI_Z_DATA,
        GateType::H => &Gate::HADAMARD_DATA,
        GateType::SWAP => &Gate::SWAP_DATA,
        GateType::U(theta, phi, lambda) => &u(theta, phi, lambda),
        GateType::S => &Gate::PHASE_S_DATA,
    };

    let dim = 1 << gate.get_type().arity();

    return DMatrix::from_row_slice(dim, dim, data);
}

/// Collapse a state vector into a value
///
/// The sum of the squares of each item should equal to one
pub fn collapse(state: &[Complex<f32>]) -> usize {
    let probs = state.iter().map(|&c| c.norm_sqr());

    let dist = WeightedIndex::new(probs)
        .expect("Failed to create probability distribution. Invalid or empty state vector?");
    let mut rng = rand::rng();

    dist.sample(&mut rng)
}

/// # expand_matrix_from_gate
/// Returns the 2^n by 2^n matrix describing a gate in a n-qubit system.
pub fn expand_matrix_from_gate(gate: &Gate, n_qubits: usize) -> DMatrix<Complex<f32>> {
    match gate.get_type() {
        GateType::SWAP => swap_matrix(
            &gate.get_controls(),
            gate.get_targets()[0],
            gate.get_targets()[1],
            n_qubits,
        ),
        _ => expand_matrix(
            get_gate_matrix(gate),
            &gate.get_controls(),
            &gate.get_targets(),
            n_qubits,
        ),
    }
}

/// # identity_tensor_factors
/// A `Vec` of `n_factor` number of 2 by 2 identity matricies.
pub fn identity_tensor_factors(n_factors: usize) -> Vec<DMatrix<Complex<f32>>> {
    vec![DMatrix::<Complex<f32>>::identity(2, 2); n_factors]
}

/// # eval_tensor_product
/// Evaluates the tensor product of a `Vec` of matricies.
pub fn eval_tensor_product(tensor_factors: Vec<DMatrix<Complex<f32>>>) -> DMatrix<Complex<f32>> {
    tensor_factors.iter().rev().fold(
        DMatrix::<Complex<f32>>::identity(1, 1),
        |product, factor| product.kronecker(factor),
    )
}

/// # swap_matrix
/// Returns the 2^n by 2^n matrix describing a swap gate in a n-qubit system.
pub fn swap_matrix(
    controls: &[usize],
    target1: usize,
    target2: usize,
    n_qubits: usize,
) -> DMatrix<Complex<f32>> {
    /* This swap gate is implemented by a series of CNOT gates,
     *
     * 1 --*--X--*--
     * 2 --X--*--X--
     * */
    let mut controls_12: Vec<usize> = controls.to_vec();
    controls_12.push(target1);
    let mut controls_21: Vec<usize> = controls.to_vec();
    controls_21.push(target2);
    let cnot_12 = expand_matrix_from_gate(
        &Gate::new(GateType::X, &controls_12, &[target2]).unwrap(),
        n_qubits,
    );
    let cnot_21 = expand_matrix_from_gate(
        &Gate::new(GateType::X, &controls_21, &[target1]).unwrap(),
        n_qubits,
    );
    cnot_12.clone() * cnot_21 * cnot_12
}

/// # reverse_matrix
/// Returns the 2^n by 2^n matrix describing reversing the order of qubits.
pub fn reverse_matrix(
    controls: &[usize],
    targets: &[usize],
    n_qubits: usize,
) -> DMatrix<Complex<f32>> {
    let dim = 1 << n_qubits;
    let mut mat = DMatrix::<Complex<f32>>::identity(dim, dim);
    for i in 0..(n_qubits >> 1) {
        mat *= swap_matrix(controls, targets[i], targets[n_qubits - 1 - i], n_qubits);
    }
    mat
}

/// # convention_convertion_matrix
/// Returns the 2^n by 2^n matrix that converts a state vector/density matrix between,
/// little-endian and big-endian convention. |q_0 q_1 q_2> <-> |q_2 q_1 q_0>.
pub fn convention_convertion_matrix(n_qubits: usize) -> DMatrix<Complex<f32>> {
    reverse_matrix(&[], &(0..n_qubits).collect::<Vec<usize>>(), n_qubits)
}

/// # convert_vector
/// converts a state vector between little-endian and big-endian convention. |q_0 q_1 q_2> <-> |q_2 q_1 q_0>.
pub fn convert_vector(vector: &DVector<Complex<f32>>) -> DVector<Complex<f32>> {
    let n_qubits = (vector.nrows() as f32).log2() as usize;
    convention_convertion_matrix(n_qubits) * vector
}

/// # convert_matrix
/// converts a matrix between little-endian and big-endian convention. |q_0 q_1 q_2> <-> |q_2 q_1 q_0>.
pub fn convert_matrix(matrix: &DMatrix<Complex<f32>>) -> DMatrix<Complex<f32>> {
    let n_qubits = (matrix.nrows() as f32).log2() as usize;
    let mat = convention_convertion_matrix(n_qubits);
    let adj = mat.adjoint();
    mat * matrix * adj
}

/// # expand_matrix
/// Returns the 2^n by 2^n matrix describing a gate in a n-qubit system.
pub fn expand_matrix(
    matrix_2x2: DMatrix<Complex<f32>>,
    controls: &[usize], //TODO: Allow for neg_controls.
    targets: &[usize],
    n_qubits: usize,
) -> DMatrix<Complex<f32>> {
    let ketbra = [
        dmatrix![cart!(1.0), cart!(0.0); cart!(0.0), cart!(0.0)], // |0><0|
        dmatrix![cart!(0.0), cart!(0.0); cart!(0.0), cart!(1.0)], // |1><1|
    ];

    // Create one term for each entry as in a 'classical truth-table'.
    // ex: |0><0| * |0><0| * I +
    //   + |0><0| * |1><1| * I +
    //   + |1><1| * |0><0| * I +
    //   + |1><1| * |1><1| * U
    let n_terms = 1 << controls.len();
    let dim = 1 << n_qubits;
    let mut sum = DMatrix::<Complex<f32>>::zeros(dim, dim);

    for i in 0..n_terms {
        let mut term = identity_tensor_factors(n_qubits);
        let mut j: usize = 0;
        for &control in controls {
            term[control] = ketbra[(i >> j) & 1].clone();
            j += 1;
        }
        if i == n_terms - 1 {
            // Last term, all controls == 1.
            for &target in targets {
                term[target] = matrix_2x2.clone();
            }
        }
        sum += eval_tensor_product(term);
    }
    sum
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SortedVec<T, K = T>(Vec<T>, PhantomData<K>);
impl<T: OrdByKey<K>, K: Ord> SortedVec<T, K> {
    pub fn new() -> Self {
        Self(Default::default(), PhantomData)
    }

    pub fn insert(&mut self, value: T) -> Option<T> {
        match self.index_of(&value) {
            Ok(index) => Some(replace(&mut self.0[index], value)),
            Err(index) => {
                self.0.insert(index, value);
                None
            }
        }
    }

    pub fn remove<Q: OrdByKey<K>>(&mut self, key: &Q) -> Option<T> {
        match self.index_of(key) {
            Ok(index) => Some(self.0.remove(index)),
            Err(_) => None,
        }
    }

    pub fn get<Q: OrdByKey<K>>(&self, key: &Q) -> Option<&T> {
        match self.index_of(key) {
            Ok(index) => Some(&self.0[index]),
            Err(_) => None,
        }
    }

    pub fn get_or_next<Q: OrdByKey<K>>(&self, key: &Q) -> Option<&T> {
        match self.index_of(key) {
            Ok(index) => Some(&self.0[index]),
            Err(index) => self.0.get(index),
        }
    }

    pub fn map<Q: OrdByKey<K>>(&mut self, key: &Q, f: &mut impl FnMut(&mut T)) -> bool
    where
        K: Clone,
    {
        match self.index_of(key) {
            Ok(index) => {
                let value = &mut self.0[index];
                let old_key = value.key().clone();
                f(value);
                if value.key() != &old_key {
                    let value = self.0.remove(index);
                    self.insert(value);
                }
                true
            }
            Err(_) => false,
        }
    }

    fn index_of<Q: OrdByKey<K>>(&self, key: &Q) -> Result<usize, usize> {
        let key = key.key();
        self.0.binary_search_by(|t| t.key().cmp(key))
    }
}
impl<T, K> Default for SortedVec<T, K> {
    fn default() -> Self {
        Self(Default::default(), Default::default())
    }
}
impl<T> Deref for SortedVec<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub trait OrdByKey<K: PartialOrd> {
    fn key(&self) -> &K;
}
impl<T: Ord> OrdByKey<T> for T {
    fn key(&self) -> &T {
        self
    }
}

#[cfg(test)]
mod tests {
    use crate::ext::{
        convert_matrix, convert_vector, equal_matrix_c, equal_state_c, expand_matrix_from_gate,
        get_gate_matrix, swap_matrix,
    };
    use crate::gate::{Gate, GateType};
    use crate::state_vector::StateVector;
    use nalgebra::{dmatrix, dvector};
    use std::f32::consts::FRAC_1_SQRT_2;

    #[test]
    fn swap_test() {
        assert!(equal_matrix_c(
            &swap_matrix(&[], 0, 1, 2),
            &get_gate_matrix(&Gate::new(GateType::SWAP, &[], &[0, 1]).unwrap()),
            4,
            0.001
        ));
    }
    #[test]
    fn fredkin_test() {
        assert!(equal_matrix_c(
            &swap_matrix(&[2], 1, 0, 3),
            &dmatrix![
                cart!(1.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0);
                cart!(0.0), cart!(1.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0);
                cart!(0.0), cart!(0.0), cart!(1.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0);
                cart!(0.0), cart!(0.0), cart!(0.0), cart!(1.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0);
                cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(1.0), cart!(0.0), cart!(0.0), cart!(0.0);
                cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(1.0), cart!(0.0);
                cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(1.0), cart!(0.0), cart!(0.0);
                cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0), cart!(1.0);
            ],
            6,
            0.001
        ));
    }

    #[test]
    fn convention_test() {
        let vec_msb = dvector![
            cart!(0.0), //|000>
            cart!(1.0), //|001>
            cart!(2.0), //|010>
            cart!(3.0), //|011>
            cart!(4.0), //|100>
            cart!(5.0), //|101>
            cart!(6.0), //|110>
            cart!(7.0), //|111>
        ];
        let vec_lsb = dvector![
            cart!(0.0), //|000>
            cart!(4.0), //|001>
            cart!(2.0), //|010>
            cart!(6.0), //|011>
            cart!(1.0), //|100>
            cart!(5.0), //|101>
            cart!(3.0), //|110>
            cart!(7.0), //|111>
        ];
        assert!(equal_state_c(
            &StateVector::from(vec_lsb.clone()),
            &StateVector::from(convert_vector(&vec_msb)),
            3,
            0.001
        ));
        assert!(equal_state_c(
            &StateVector::from(vec_msb.clone()),
            &StateVector::from(convert_vector(&vec_lsb)),
            3,
            0.001
        ));
        let textbook_ch = dmatrix![
            cart!(1.0), cart!(0.0), cart!(0.0), cart!(0.0);
            cart!(0.0), cart!(1.0), cart!(0.0), cart!(0.0);
            cart!(0.0), cart!(0.0), cart!(FRAC_1_SQRT_2), cart!(FRAC_1_SQRT_2);
            cart!(0.0), cart!(0.0), cart!(FRAC_1_SQRT_2), cart!(-FRAC_1_SQRT_2);
        ];
        let sim_ch = expand_matrix_from_gate(&Gate::new(GateType::H, &[0], &[1]).unwrap(), 2);

        assert!(equal_matrix_c(
            &convert_matrix(&sim_ch),
            &textbook_ch,
            4,
            0.001
        ));
        assert!(equal_matrix_c(
            &convert_matrix(&textbook_ch),
            &sim_ch,
            4,
            0.001
        ));
    }
}
