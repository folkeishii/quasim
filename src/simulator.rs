use crate::{circuit::Circuit, instruction::Instruction};
use nalgebra::{Complex, DVector};

pub trait SimpleSimulator: Sized {
    type E: std::error::Error;

    fn build(circuit: Circuit) -> Result<Self, Self::E>;
    fn run(&self) -> usize;
    fn final_state(&self) -> DVector<Complex<f32>>;
}

pub trait Debugger {
    fn next(&mut self) -> Option<&DVector<Complex<f32>>>;
    fn prev(&mut self) -> Option<&DVector<Complex<f32>>>;
    fn current_instruction(&self) -> Option<&Instruction>;
    fn current_state(&self) -> &DVector<Complex<f32>>;
}
