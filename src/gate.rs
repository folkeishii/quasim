use nalgebra::{Complex, DMatrix};

pub struct Gate {
    pub matrix: DMatrix<Complex<f32>>,
    pub target: Vec<usize>,
}

impl Gate {
    const PAULI_X_DATA: [Complex<f32>; 4] = [
        Complex::new(0.0, 0.0), Complex::new(1.0, 0.0),
        Complex::new(1.0, 0.0), Complex::new(0.0, 0.0),
    ];

    const PAULI_Y_DATA: [Complex<f32>; 4] = [
        Complex::new(0.0, 0.0), Complex::new(0.0, -1.0),
        Complex::new(0.0, 1.0), Complex::new(0.0, 0.0),
    ];

    const PAULI_Z_DATA: [Complex<f32>; 4] = [
        Complex::new(1.0, 0.0), Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0), Complex::new(-1.0, 0.0),
    ];

    pub fn x(target: usize) -> Gate {
        Self {
            matrix: DMatrix::from_row_slice(2, 2, &Self::PAULI_X_DATA),
            target: vec![target],
        }
    }

    pub fn y(target: usize) -> Gate {
        Self {
            matrix: DMatrix::from_row_slice(2, 2, &Self::PAULI_Y_DATA),
            target: vec![target],
        }
    }

    pub fn z(target: usize) -> Gate {
        Self {
            matrix: DMatrix::from_row_slice(2, 2, &Self::PAULI_Z_DATA),
            target: vec![target],
        }
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{dmatrix, dvector, ComplexField};
    use super::*;

    fn is_unitary(gate: Gate) -> bool {
        let adjoint = gate.matrix.adjoint();
        let identity = DMatrix::<Complex<f32>>::identity(2, 2);
        is_equal_to(gate.matrix * adjoint, identity)
    }

    fn is_equal_to(m1: DMatrix<Complex<f32>>, m2: DMatrix<Complex<f32>>) -> bool {
        m1.iter()
            .zip(m2.iter())
            .all(|(a, b)|
                Complex::abs(a - b) < 0.001
            )
    }

    macro_rules! assert_is_matrix_equal {
        ($m1: expr, $m2: expr) => { assert!(is_equal_to($m1, $m2)) };
    }

    #[test]
    fn test_is_x_unitary() {
        assert!(is_unitary(Gate::x(0)));
    }

    #[test]
    fn test_x_value() {
        let a = Complex::new(0.7, 2.3);
        let b = Complex::new(9.2, 0.0);

        let qubit = dmatrix![a; b];
        let target_qubit = dmatrix![b; a];
        let transform = Gate::x(0).matrix * qubit;

        assert_is_matrix_equal!(transform, target_qubit);
    }

    #[test]
    fn test_is_y_unitary() {
        assert!(is_unitary(Gate::y(0)));
    }

    #[test]
    fn test_y_value() {
        let qubit = dmatrix![Complex::new(0.0, 0.0); Complex::new(1.0, 0.0)];
        let qubit_target = dmatrix![Complex::new(0.0, -1.0); Complex::new(0.0, 0.0)];
        let transform = Gate::y(0).matrix * qubit;

        assert_is_matrix_equal!(transform, qubit_target);
    }

    #[test]
    fn test_is_z_unitary() {
        assert!(is_unitary(Gate::z(0)));
    }

    #[test]
    fn test_z_value() {
        let qubit = dmatrix![Complex::new(0.0, 0.0); Complex::new(1.0, 0.0)];
        let qubit_target = dmatrix![Complex::new(0.0, 0.0); Complex::new(-1.0, 0.0)];
        let transform = Gate::z(0).matrix * qubit;

        assert_is_matrix_equal!(transform, qubit_target);
    }
}
