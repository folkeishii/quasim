use std::{iter::Map, ops::Range};

use nalgebra::{Complex, DMatrix, Dim, Matrix, RawStorage};
use rand::distr::weighted::WeightedIndex;
use rand::prelude::Distribution;

use crate::gate::{Gate, GateType};

/// Compares two complex numbers
///
/// Two complex numbers are determined to be equal if the distance
/// between them is at most `margin`
pub fn equal_to_c(lhs: Complex<f32>, rhs: Complex<f32>, margin: f32) -> bool {
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
    lhs: &Matrix<Complex<f32>, R, C, S>,
    rhs: &Matrix<Complex<f32>, R, C, S>,
    margin: f32,
) -> bool
where
    R: Dim,
    C: Dim,
    S: RawStorage<Complex<f32>, R, C>,
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

pub fn get_gate_matrix(gate: &Gate) -> DMatrix<Complex<f32>> {
    let data: &[Complex<f32>] = match gate.get_type() {
        GateType::X => &Gate::PAULI_X_DATA,
        GateType::Y => &Gate::PAULI_Y_DATA,
        GateType::Z => &Gate::PAULI_Z_DATA,
        GateType::H => &Gate::HADAMARD_DATA,
        GateType::SWAP => &Gate::SWAP_DATA,
    };

    let dim = 1 << gate.get_type().arity();

    return DMatrix::from_row_slice(dim, dim, data);
}

/// Collapse a state vector into a value
pub fn collapse(state: &Vec<Complex<f32>>) -> usize {
    let probs = state.iter().map(|&c| c.norm_sqr());

    let dist = WeightedIndex::new(probs)
        .expect("Failed to create probability distribution. Invalid or empty state vector?");
    let mut rng = rand::rng();

    dist.sample(&mut rng)
}

#[macro_export]
macro_rules! cart {
    ($re:expr) => {
        Complex { re: $re, im: 0.0 }
    };
    ($re:expr, $im:expr) => {
        Complex { re: $re, im: $im }
    };
}

#[macro_export]
macro_rules! polar {
    ($r: expr, $theta: expr) => {
        Complex::from_polar($r, $theta)
    };
}

#[macro_export]
macro_rules! cexp {
    ($exp: expr) => {
        Complex::exp($exp)
    };
}
