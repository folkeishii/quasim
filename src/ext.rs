use std::cmp::Ordering;

use nalgebra::{Complex, Dim, Matrix, RawStorage};

/// Compares two complex numbers
///
/// Equality is determined by norm or arg being within an
/// offset of margin from each other
///
/// Returns `Ordering::Greater` if norm is greater or if the
/// norm is equal then a greater phase on the range [-π, π)
///
/// Returns `Ordering::Greater` if norm is less or if the
/// norm is equal then a lesser phase on the range [-π, π)
///
/// If any component is `NAN` then `None` will be returned
pub fn cmp_c(lhs: Complex<f32>, rhs: Complex<f32>, margin: f32) -> Option<Ordering> {
    let lnorm = &lhs.norm();
    let rnorm = &rhs.norm();
    let larg = &lhs.arg();
    let rarg = &rhs.arg();
    match (
        (lnorm - rnorm).abs().partial_cmp(&margin.abs()),
        lnorm.partial_cmp(rnorm),
        (larg - rarg).abs().partial_cmp(&margin.abs()),
        larg.partial_cmp(rarg),
    ) {
        (None, _, _, _) => None,
        (_, None, _, _) => None,
        (_, _, None, _) => None,
        (_, _, _, None) => None,
        (Some(Ordering::Equal | Ordering::Less), _, Some(Ordering::Equal | Ordering::Less), _) => {
            Some(Ordering::Equal)
        }
        (Some(Ordering::Equal | Ordering::Less), _, _, ord) => ord,
        (_, ord, _, _) => ord,
    }
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
pub fn cmp_elements<R, C, S>(
    lhs: &Matrix<Complex<f32>, R, C, S>,
    rhs: &Matrix<Complex<f32>, R, C, S>,
    margin: f32,
) -> Option<Ordering>
where
    R: Dim,
    C: Dim,
    S: RawStorage<Complex<f32>, R, C>,
{
    for (lel, rel) in lhs.iter().zip(rhs.iter()) {
        let cc = cmp_c(*lel, *rel, margin);
        match cc {
            Some(Ordering::Equal) => continue,
            ord => return ord,
        }
    }

    return Some(Ordering::Equal);
}
