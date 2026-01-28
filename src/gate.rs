use nalgebra::{Complex, DMatrix, DVector};

pub struct Gate {
    pub matrix: DMatrix<Complex<f32>>,
    pub target: Vec<usize>,
}
