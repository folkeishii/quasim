use nalgebra::{Complex, DVector};

use crate::Circuit;

pub trait SimpleSimulator: Sized {
    type E: std::error::Error;

    fn build(circuit: Circuit) -> Result<Self, Self::E>;
    fn run(&self) -> usize;
    fn final_state(&self) -> DVector<Complex<f32>>;
}

pub trait DebugSimulator: SimpleSimulator {
    fn state_at(&self, time_step: usize) -> DVector<Complex<f32>>;
}
