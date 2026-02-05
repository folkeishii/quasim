use nalgebra::{Complex, DMatrix, DVector};
use crate::{Circuit, Instruction};
use rand::prelude::*;

// pub struct Qubit {
//     alfa: Complex<f32>,
//     beta: Complex<f32>,
// }

// impl Default for Qubit {
//     // Default qubit is a guaranteed 0 with no phase AKA |0>
//     fn default() -> Qubit {
//         Self {
//             alfa: Complex::new(1.0, 0.0),
//             beta: Complex::new(0.0, 0.0)
//         } 
//     } 
// }

pub trait SimpleSimulator: Sized {
    type E: std::error::Error;

    fn build(circuit: Circuit) -> Result<Self, Self::E>;
    fn run(&self) -> usize;
    fn final_state(&self) -> DVector<Complex<f32>>;
}
