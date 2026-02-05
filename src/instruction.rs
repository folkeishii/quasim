use nalgebra::{Complex, DMatrix, dmatrix};
use std::f32::consts::{FRAC_1_SQRT_2, PI};

pub enum Instruction {
    CNOT(usize, usize),
    X(usize),
    Z(usize),
    Y(usize),
    T(usize),
}
