use nalgebra::{Complex, DMatrix, DVector};

pub struct Instruction {
    pub matrix: DMatrix<Complex<f32>>,
    pub target: Vec<usize>,
}
