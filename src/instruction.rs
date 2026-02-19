use crate::{Gate, QBits, cart};
use nalgebra::{Complex, DMatrix, dmatrix};
use std::{
    f32::consts::{FRAC_1_SQRT_2, PI},
    vec,
};

#[derive(Debug, Clone)]
pub enum Instruction {
    Gate(Gate),
    Measurement(QBits),
}
