use crate::macros::{imag, real};
use nalgebra::{Complex, DMatrix, dmatrix};
use std::f32::consts::{FRAC_1_SQRT_2, PI};

pub struct Instruction {
    pub matrix: DMatrix<Complex<f32>>,
    pub target: Vec<usize>,
}

impl Instruction {
    #[rustfmt::skip]
    const PAULI_X_DATA: [Complex<f32>; 4] = [
        real!(0.0), real!(1.0),
        real!(1.0), real!(0.0),
    ];

    #[rustfmt::skip]
    const PAULI_Y_DATA: [Complex<f32>; 4] = [
        imag!(0.0), imag!(-1.0),
        imag!(1.0), imag!(0.0),
    ];

    #[rustfmt::skip]
    const PAULI_Z_DATA: [Complex<f32>; 4] = [
        real!(1.0), real!(0.0),
        real!(0.0), real!(-1.0),
    ];

    #[rustfmt::skip]
    const HADAMARD_DATA: [Complex<f32>; 4] = [
        real!(FRAC_1_SQRT_2), real!(FRAC_1_SQRT_2),
        real!(FRAC_1_SQRT_2), real!(-FRAC_1_SQRT_2),
    ];

    #[rustfmt::skip]
    const CNOT_DATA: [Complex<f32>; 16] = [
        real!(1.0), real!(0.0), real!(0.0), real!(0.0),
        real!(0.0), real!(1.0), real!(0.0), real!(0.0),
        real!(0.0), real!(0.0), real!(0.0), real!(1.0),
        real!(0.0), real!(0.0), real!(1.0), real!(0.0)
    ];

    pub fn x(target: usize) -> Instruction {
        Self {
            matrix: DMatrix::from_row_slice(2, 2, &Self::PAULI_X_DATA),
            target: vec![target],
        }
    }

    pub fn y(target: usize) -> Instruction {
        Self {
            matrix: DMatrix::from_row_slice(2, 2, &Self::PAULI_Y_DATA),
            target: vec![target],
        }
    }

    pub fn z(target: usize) -> Instruction {
        Self {
            matrix: DMatrix::from_row_slice(2, 2, &Self::PAULI_Z_DATA),
            target: vec![target],
        }
    }

    pub fn hadamard(target: usize) -> Instruction {
        Self {
            matrix: DMatrix::from_row_slice(2, 2, &Self::HADAMARD_DATA),
            target: vec![target],
        }
    }

    pub fn t(target: usize) -> Instruction {
        Self {
            matrix: dmatrix![
                real!(1.0), real!(0.0);
                real!(0.0), Complex::exp(imag!(PI / 8.0));
            ],
            target: vec![target],
        }
    }

    pub fn cnot(control: usize, target: usize) -> Instruction {
        Self {
            matrix: DMatrix::from_row_slice(4, 4, &Self::CNOT_DATA),
            target: vec![control, target],
        }
    }
}

#[cfg(test)]
mod tests {
    use core::f32;

    use super::*;
    use crate::macros::complex;
    use nalgebra::{ComplexField, dmatrix};

    fn is_unitary(instruction: Instruction, size: usize) -> bool {
        let adjoint = instruction.matrix.adjoint();
        let identity = DMatrix::<Complex<f32>>::identity(size, size);
        is_equal_to(instruction.matrix * adjoint, identity)
    }

    fn is_hermitian(instruction: Instruction) -> bool {
        let conjugate = instruction.matrix.conjugate();
        is_equal_to(instruction.matrix, conjugate)
    }

    fn is_equal_to(m1: DMatrix<Complex<f32>>, m2: DMatrix<Complex<f32>>) -> bool {
        m1.iter()
            .zip(m2.iter())
            .all(|(a, b)| Complex::abs(a - b) < 0.001)
    }

    macro_rules! assert_is_matrix_equal {
        ($m1: expr, $m2: expr) => {
            assert!(is_equal_to($m1, $m2))
        };
    }

    #[test]
    fn test_is_x_unitary() {
        assert!(is_unitary(Instruction::x(0), 2));
    }

    #[test]
    fn test_x_value() {
        let a = complex!(0.7, 2.3);
        let b = complex!(9.2, 0.0);

        let qubit = dmatrix![a; b];
        let target_qubit = dmatrix![b; a];
        let transform = Instruction::x(0).matrix * qubit;

        assert_is_matrix_equal!(transform, target_qubit);
    }

    #[test]
    fn test_is_y_unitary() {
        assert!(is_unitary(Instruction::y(0), 2));
    }

    #[test]
    fn test_y_value() {
        let qubit = dmatrix![real!(0.0); real!(1.0)];
        let qubit_target = dmatrix![imag!(-1.0); imag!(0.0)];
        let transform = Instruction::y(0).matrix * qubit;

        assert_is_matrix_equal!(transform, qubit_target);
    }

    #[test]
    fn test_is_z_unitary() {
        assert!(is_unitary(Instruction::z(0), 2));
    }

    #[test]
    fn test_z_value() {
        let qubit = dmatrix![real!(0.0); real!(1.0)];
        let qubit_target = dmatrix![real!(0.0); real!(-1.0)];
        let transform = Instruction::z(0).matrix * qubit;

        assert_is_matrix_equal!(transform, qubit_target);
    }

    #[test]
    fn test_is_hadamard_unitary() {
        assert!(is_unitary(Instruction::hadamard(0), 2));
    }

    #[test]
    fn test_hadamard_twice() {
        let qubit = dmatrix![real!(0.0); real!(1.0)];
        let transform =
            Instruction::hadamard(0).matrix * Instruction::hadamard(0).matrix * qubit.clone();

        assert_is_matrix_equal!(transform, qubit);
    }

    #[test]
    fn test_hzh() {
        let transform = Instruction::hadamard(0).matrix
            * Instruction::z(0).matrix
            * Instruction::hadamard(0).matrix;

        assert_is_matrix_equal!(transform, Instruction::x(0).matrix);
    }

    #[test]
    fn test_hxh() {
        let transform = Instruction::hadamard(0).matrix
            * Instruction::x(0).matrix
            * Instruction::hadamard(0).matrix;

        assert_is_matrix_equal!(transform, Instruction::z(0).matrix);
    }

    #[test]
    fn test_hyh() {
        let transform = Instruction::hadamard(0).matrix
            * Instruction::y(0).matrix
            * Instruction::hadamard(0).matrix;

        assert_is_matrix_equal!(transform, -Instruction::y(0).matrix);
    }

    #[test]
    fn test_hadamard_is_hermitian() {
        assert!(is_hermitian(Instruction::hadamard(0)));
    }

    #[test]
    fn test_hadamard_value0() {
        let plus_state = dmatrix![real!(FRAC_1_SQRT_2); real!(FRAC_1_SQRT_2)];
        let zero_state = dmatrix![real!(1.0); real!(0.0)];

        let transform = Instruction::hadamard(0).matrix * zero_state;
        assert_is_matrix_equal!(transform, plus_state);
    }

    #[test]
    fn test_hadamard_value1() {
        let minus_state = dmatrix![real!(FRAC_1_SQRT_2); real!(-FRAC_1_SQRT_2)];
        let one_state = dmatrix![real!(0.0); real!(1.0)];

        let transform = Instruction::hadamard(0).matrix * one_state;
        assert_is_matrix_equal!(transform, minus_state);
    }

    #[test]
    fn test_is_t_unitary() {
        assert!(is_unitary(Instruction::t(0), 2))
    }

    #[test]
    fn test_t_value() {
        let qubit = dmatrix![real!(FRAC_1_SQRT_2); real!(FRAC_1_SQRT_2)];
        let qubit_target = dmatrix![
            real!(FRAC_1_SQRT_2);
            real!(FRAC_1_SQRT_2) * Complex::exp(imag!(PI / 8.0))
        ];
        let transform = Instruction::t(0).matrix * qubit;

        assert_is_matrix_equal!(transform, qubit_target);
    }

    #[test]
    fn test_is_cnot_unitary() {
        assert!(is_unitary(Instruction::cnot(0, 0), 4));
    }

    #[test]
    fn test_cnot_value() {
        let a = complex!(0.7, 2.3);
        let b = complex!(9.2, 0.0);
        let c = complex!(2.2, 0.8);
        let d = complex!(5.1, 1.9);

        let qubit = dmatrix![a; b; c; d];
        let qubit_target = dmatrix![a; b; d; c];
        let transform = Instruction::cnot(0, 0).matrix * qubit;

        assert_is_matrix_equal!(transform, qubit_target);
    }
}
