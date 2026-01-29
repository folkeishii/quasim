use nalgebra::{Complex, DMatrix, DVector};

#[derive(Debug, Clone)]
pub struct Instruction {
    pub matrix: DMatrix<Complex<f32>>,
    pub target: Vec<usize>,
}
