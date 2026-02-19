use crate::{cart, gate::{Gate, QBits}};
use nalgebra::{Complex, DMatrix};
use std::{f32::consts::FRAC_1_SQRT_2, vec};

#[derive(Debug, Clone)]
pub enum Instruction {
    Gate(Gate),
    Measurement(QBits),
}
