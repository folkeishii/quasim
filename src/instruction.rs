use crate::cart;
use nalgebra::{Complex, DMatrix};
use std::{f32::consts::FRAC_1_SQRT_2, vec};

#[derive(Debug, Clone)]
pub enum Instruction {
    CNOT(usize, usize),
    X(usize),
    Z(usize),
    Y(usize),
    H(usize),
    SWAP(usize, usize),
    CSWAP(usize, usize, usize), // Fredkin gate
}

impl Instruction {
    pub const PAULI_X_DATA: [Complex<f32>; 4] = [cart!(0.0), cart!(1.0), cart!(1.0), cart!(0.0)];

    #[rustfmt::skip]
    pub const PAULI_Y_DATA: [Complex<f32>; 4] = [
        cart!(0.0), cart!(0.0, -1.0),
        cart!(0.0, 1.0), cart!(0.0),
    ];

    #[rustfmt::skip]
    pub const PAULI_Z_DATA: [Complex<f32>; 4] = [
        cart!(1.0), cart!(0.0),
        cart!(0.0), cart!(-1.0, 0.0),
    ];

    #[rustfmt::skip]
    pub const HADAMARD_DATA: [Complex<f32>; 4] = [
        cart!(FRAC_1_SQRT_2, 0.0), cart!(FRAC_1_SQRT_2, 0.0),
        cart!(FRAC_1_SQRT_2, 0.0), cart!(-FRAC_1_SQRT_2, 0.0),
    ];

    #[rustfmt::skip]
    pub const SWAP_DATA: [Complex<f32>; 16] = [
        cart!(1.0), cart!(0.0), cart!(0.0), cart!(0.0),
        cart!(0.0), cart!(0.0), cart!(1.0), cart!(0.0),
        cart!(0.0), cart!(1.0), cart!(0.0), cart!(0.0),
        cart!(0.0), cart!(0.0), cart!(0.0), cart!(1.0)
    ];

    // Can add an argument specifying if we should include full matrix including for the control bits of the gate
    // Right now just return the basic gate so for a CNOT gate we just return X data
    pub fn get_matrix(&self) -> DMatrix<Complex<f32>> {
        let (dim, data): (usize, &[Complex<f32>]) = match self {
            Instruction::X(_) => (2, &Self::PAULI_X_DATA),
            Instruction::Y(_) => (2, &Self::PAULI_Y_DATA),
            Instruction::Z(_) => (2, &Self::PAULI_Z_DATA),
            Instruction::H(_) => (2, &Self::HADAMARD_DATA),
            Instruction::CNOT(_, _) => (2, &Self::PAULI_X_DATA),
            Instruction::SWAP(_, _) => (4, &Self::SWAP_DATA),
            Instruction::CSWAP(_, _, _) => (4, &Self::SWAP_DATA),
        };

        return DMatrix::from_row_slice(dim, dim, data);
    }

    pub fn get_controls(&self) -> Vec<usize> {
        match self {
            Instruction::CNOT(c, _) | Instruction::CSWAP(c, _, _) => vec![*c],
            _ => vec![],
        }
    }

    pub fn get_targets(&self) -> Vec<usize> {
        match self {
            Instruction::CNOT(_, t)
            | Instruction::X(t)
            | Instruction::Y(t)
            | Instruction::Z(t)
            | Instruction::H(t) => vec![*t],
            Instruction::SWAP(t1, t2) | Instruction::CSWAP(_, t1, t2) => vec![*t1, *t2],
        }
    }
}
