use nalgebra::{Complex, DMatrix, DVector};

#[derive(Debug, Clone)]
pub struct Gate {
    pub matrix: DMatrix<Complex<f32>>,
    pub target: Vec<usize>,
}
