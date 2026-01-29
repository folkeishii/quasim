use std::f32;

use nalgebra::{Complex, DMatrix, DVector};

pub struct Instruction {
    pub matrix: DMatrix<Complex<f32>>,
    pub target: Vec<usize>,
}

impl Instruction {
    const HADAMARD_DATA: [Complex<f32>; 4] = [
        Complex::new(f32::consts::FRAC_1_SQRT_2, 0.0), Complex::new(f32::consts::FRAC_1_SQRT_2, 0.0),
        Complex::new(f32::consts::FRAC_1_SQRT_2, 0.0), Complex::new(-f32::consts::FRAC_1_SQRT_2, 0.0),
    ];

    pub fn hadamard(target: usize) -> Instruction{
        Self {
            matrix: DMatrix::from_row_slice(2, 2, &Self::HADAMARD_DATA),
            target: vec![target],
        }
    }
}
