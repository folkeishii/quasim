use nalgebra::{Complex, DMatrix, dmatrix};
use std::f32::consts::{FRAC_1_SQRT_2, PI};

pub enum Instruction {
    CNOT(usize, usize),
    X(usize),
    Z(usize),
    Y(usize),
    H(usize),
}

impl Instruction {
    #[rustfmt::skip]
    const PAULI_X_DATA: [Complex<f32>; 4] = [
        Complex::new(0.0, 0.0), Complex::new(1.0, 0.0),
        Complex::new(1.0, 0.0), Complex::new(0.0, 0.0),
    ];

    #[rustfmt::skip]
    const PAULI_Y_DATA: [Complex<f32>; 4] = [
        Complex::new(0.0, 0.0), Complex::new(0.0, -1.0),
        Complex::new(0.0, 1.0), Complex::new(0.0, 0.0),
    ];

    #[rustfmt::skip]
    const PAULI_Z_DATA: [Complex<f32>; 4] = [
        Complex::new(1.0, 0.0), Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0), Complex::new(-1.0, 0.0),
    ];

    #[rustfmt::skip]
    const HADAMARD_DATA: [Complex<f32>; 4] = [
        Complex::new(FRAC_1_SQRT_2, 0.0), Complex::new(FRAC_1_SQRT_2, 0.0),
        Complex::new(FRAC_1_SQRT_2, 0.0), Complex::new(-FRAC_1_SQRT_2, 0.0),
    ];

    // Can add an argument specifying if we should include full matrix including for the control bits of the gate
    // Right now just return the basic gate so for a CNOT gate we just return X data
    pub fn get_matrix(self) -> DMatrix<Complex<f32>> {
        let data = match self {
            Instruction::X(_) => &Self::PAULI_X_DATA,
            Instruction::Y(_) => &Self::PAULI_Y_DATA,
            Instruction::Z(_) => &Self::PAULI_Z_DATA,
            Instruction::H(_) => &Self::HADAMARD_DATA,
            Instruction::CNOT(_, _) => &Self::PAULI_X_DATA,
        };

        return DMatrix::from_row_slice(2, 2, data);
    }
}
