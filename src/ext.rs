use std::{iter::Map, ops::Range};

use nalgebra::{Complex, DMatrix, Dim, Matrix, RawStorage};

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

pub fn call_unary<F, T, U>(f: &F, arg: T) -> U
where
    F: Fn(T) -> U,
{
    f(arg)
}

pub trait DSLEq<Rhs = Self> {
    type Output;
    // Required method
    fn eq(&self, other: &Rhs) -> Self::Output;

    // Provided method
    fn ne(&self, other: &Rhs) -> Self::Output;
}
pub trait DSLOrd<Rhs = Self>: DSLEq<Rhs> {
    fn lt(&self, other: &Rhs) -> Self::Output;
    fn le(&self, other: &Rhs) -> Self::Output;
    fn gt(&self, other: &Rhs) -> Self::Output;
    fn ge(&self, other: &Rhs) -> Self::Output;
    fn max(self, other: Self) -> Self::Output;
    fn min(self, other: Self) -> Self::Output;
    fn clamp(self, min: Self, max: Self) -> Self::Output;
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

#[macro_export]
macro_rules! impl_deref {
    ($cont:ident<$generic:ident$(,$gg:ident)*>) => {
        impl<$generic$(,$gg)*> std::ops::Deref for $cont<$generic$(,$gg)*> {
            type Target = $generic;

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }
    };
    ($cont:ident($target:ty)) => {
        impl std::ops::Deref for $cont {
            type Target = $target;

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }
    };
}
#[macro_export]
macro_rules! impl_deref_mut {
    ($cont:ident<$generic:ident$(,$gg:ident)*>) => {
        impl_deref!($cont<$generic$(,$gg)*>);
        impl<$generic$(,$gg)*> std::ops::DerefMut for $cont<$generic$(,$gg)*> {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.0
            }
        }
    };
    ($cont:ident($target:ty)) => {
        impl_deref!($cont($target));
        impl std::ops::DerefMut for $cont {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.0
            }
        }
    };
}
#[macro_export]
macro_rules! impl_from {
    ($cont:ident<$generic:ident$(,$gg:ident)*> $(,$other:expr)*) => {
        impl<$generic$(,$gg)*> From<$generic> for $cont<$generic$(,$gg)*> {
            fn from(value: $generic) -> Self {
                Self(value$(,$other)*)
            }
        }
    };
    ($cont:ident($target:ty) $(,$other:expr)*) => {
        impl From<$target> for $cont {
            fn from(value: $target) -> Self {
                Self(value $(,$other)*)
            }
        }
    };
}
