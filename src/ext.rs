use std::marker::PhantomData;
use std::mem::replace;
use std::ops::Deref;
use std::{iter::Map, ops::Range};

use nalgebra::{dmatrix, Complex, DMatrix, DVector, Dim, Matrix, RawStorage};
use rand::distr::weighted::WeightedIndex;
use rand::{prelude::Distribution, Rng};

use crate::gate::{Gate, GateType};

#[macro_export]
macro_rules! cart {
    ($re:expr) => {
        nalgebra::Complex {
            re: $re as f64,
            im: 0.0,
        }
    };
    ($re:expr, $im:expr) => {
        nalgebra::Complex {
            re: $re as f64,
            im: $im as f64,
        }
    };
}

#[macro_export]
macro_rules! polar {
    ($r: expr, $theta: expr) => {
        Complex::from_polar($r as f64, $theta as f64)
    };
}

#[macro_export]
macro_rules! cexp {
    ($exp: expr) => {
        Complex::exp($exp as f64)
    };
}

/// Compares two complex numbers
///
/// Two complex numbers are determined to be equal if the distance
/// between them is at most `margin`
pub fn equal_to_c(lhs: Complex<f64>, rhs: Complex<f64>, margin: f64) -> bool {
    (lhs - rhs).norm().le(&margin)
}

/// Compares complex elements using ´cmp_c´
///
/// Equality is determined by every element being equal
///
/// Returns `Ordering::Greater` if all elements preceeding an element
/// are equal and the same element is greater
///
/// Return `Ordering::Less` if all elements preceeding an element
/// are equal and the same element is less
pub fn equal_to_matrix_c<R, C, S>(
    lhs: &Matrix<Complex<f64>, R, C, S>,
    rhs: &Matrix<Complex<f64>, R, C, S>,
    margin: f64,
) -> bool
where
    R: Dim,
    C: Dim,
    S: RawStorage<Complex<f64>, R, C>,
{
    let ((l1, l2), (r1, r2)) = (lhs.shape(), rhs.shape());
    if l1 != r1 || l2 != r2 {
        return false;
    }

    for (lel, rel) in lhs.iter().zip(rhs.iter()) {
        if !equal_to_c(*lel, *rel, margin) {
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
fn u(theta: f64, phi: f64, lambda: f64) -> [Complex<f64>; 4] {
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

pub fn get_gate_matrix(gate: &Gate) -> DMatrix<Complex<f64>> {
    let data: &[Complex<f64>] = match gate.get_type() {
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
pub fn collapse(state: &[Complex<f64>]) -> usize {
    let probs = state.iter().map(|&c| c.norm_sqr());

    let dist = WeightedIndex::new(probs)
        .expect("Failed to create probability distribution. Invalid or empty state vector?");
    let mut rng = rand::rng();

    dist.sample(&mut rng)
}

/// # measure
/// Returns a probable state vector after measurement.
pub fn measure(
    target: usize,
    state: &DVector<Complex<f64>>,
    n_qubits: usize,
) -> DVector<Complex<f64>> {
    // Choose a collapsed state
    let prob_target_eq_zero = state
        .iter()
        .enumerate()
        .filter(|&(idx, _)| (1 << target) & idx == 0) // Using |..q_1q_0> convetion
        .map(|(_, c)| c.norm_sqr())
        .sum::<f64>();

    let mut rng = rand::rng();
    let random_value = rng.random_range(0.0..1.0);
    let mut result = dmatrix![cart!(0.0), cart!(0.0); cart!(0.0), cart!(1.0)]; // |1><1|
    if random_value < prob_target_eq_zero {
        // 0 was chosen as collapsed state.
        result = dmatrix![cart!(1.0), cart!(0.0); cart!(0.0), cart!(0.0)]; // |0><0|
    }

    // Calculate projection operator, M
    let mut projection_operator_prod = identity_tensor_factors(n_qubits);
    projection_operator_prod[target] = result;
    let projection_operator = eval_tensor_product(projection_operator_prod);

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
pub fn expand_matrix_from_gate(gate: &Gate, n_qubits: usize) -> DMatrix<Complex<f64>> {
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
fn identity_tensor_factors(n_factors: usize) -> Vec<DMatrix<Complex<f64>>> {
    vec![DMatrix::<Complex<f64>>::identity(2, 2); n_factors]
}

/// # eval_tensor_product
/// Evaluates the tensor product of a `Vec` of matricies.
fn eval_tensor_product(tensor_factors: Vec<DMatrix<Complex<f64>>>) -> DMatrix<Complex<f64>> {
    tensor_factors.iter().rev().fold(
        DMatrix::<Complex<f64>>::identity(1, 1),
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
) -> DMatrix<Complex<f64>> {
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
) -> DMatrix<Complex<f64>> {
    let dim = 1 << n_qubits;
    let mut mat = DMatrix::<Complex<f64>>::identity(dim, dim);
    for i in 0..(n_qubits >> 1) {
        mat *= swap_matrix(controls, targets[i], targets[n_qubits - 1 - i], n_qubits);
    }
    mat
}

/// # convention_convertion_matrix
/// Returns the 2^n by 2^n matrix that converts a state vector/density matrix between,
/// little-endian and big-endian convention. |q_0 q_1 q_2> <-> |q_2 q_1 q_0>.
pub fn convention_convertion_matrix(n_qubits: usize) -> DMatrix<Complex<f64>> {
    reverse_matrix(&[], &(0..n_qubits).collect::<Vec<usize>>(), n_qubits)
}

/// # convert_vector
/// converts a state vector between little-endian and big-endian convention. |q_0 q_1 q_2> <-> |q_2 q_1 q_0>.
pub fn convert_vector(vector: &DVector<Complex<f64>>) -> DVector<Complex<f64>> {
    let n_qubits = (vector.nrows() as f32).log2() as usize;
    convention_convertion_matrix(n_qubits) * vector
}

/// # convert_matrix
/// converts a matrix between little-endian and big-endian convention. |q_0 q_1 q_2> <-> |q_2 q_1 q_0>.
pub fn convert_matrix(matrix: &DMatrix<Complex<f64>>) -> DMatrix<Complex<f64>> {
    let n_qubits = (matrix.nrows() as f32).log2() as usize;
    let mat = convention_convertion_matrix(n_qubits);
    let adj = mat.adjoint();
    mat * matrix * adj
}

/// # expand_matrix
/// Returns the 2^n by 2^n matrix describing a gate in a n-qubit system.
pub fn expand_matrix(
    matrix_2x2: DMatrix<Complex<f64>>,
    controls: &[usize], //TODO: Allow for neg_controls.
    targets: &[usize],
    n_qubits: usize,
) -> DMatrix<Complex<f64>> {
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
    let mut sum = DMatrix::<Complex<f64>>::zeros(dim, dim);

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
        convert_matrix, convert_vector, equal_to_matrix_c, expand_matrix_from_gate,
        get_gate_matrix, swap_matrix,
    };
    use crate::gate::{Gate, GateType};
    use nalgebra::{dmatrix, dvector};
    use std::f64::consts::FRAC_1_SQRT_2;

    #[test]
    fn swap_test() {
        assert!(equal_to_matrix_c(
            &swap_matrix(&[], 0, 1, 2),
            &get_gate_matrix(&Gate::new(GateType::SWAP, &[], &[0, 1]).unwrap()),
            0.001
        ));
    }
    #[test]
    fn fredkin_test() {
        assert!(equal_to_matrix_c(
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
        assert!(equal_to_matrix_c(
            &vec_lsb,
            &convert_vector(&vec_msb),
            0.001
        ));
        assert!(equal_to_matrix_c(
            &vec_msb,
            &convert_vector(&vec_lsb),
            0.001
        ));
        let textbook_ch = dmatrix![
            cart!(1.0), cart!(0.0), cart!(0.0), cart!(0.0);
            cart!(0.0), cart!(1.0), cart!(0.0), cart!(0.0);
            cart!(0.0), cart!(0.0), cart!(FRAC_1_SQRT_2), cart!(FRAC_1_SQRT_2);
            cart!(0.0), cart!(0.0), cart!(FRAC_1_SQRT_2), cart!(-FRAC_1_SQRT_2);
        ];
        let sim_ch = expand_matrix_from_gate(&Gate::new(GateType::H, &[0], &[1]).unwrap(), 2);

        assert!(equal_to_matrix_c(
            &convert_matrix(&sim_ch),
            &textbook_ch,
            0.001
        ));
        assert!(equal_to_matrix_c(
            &convert_matrix(&textbook_ch),
            &sim_ch,
            0.001
        ));
    }
}
