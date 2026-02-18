use nalgebra::{Complex, DMatrix, dmatrix};
use std::{f32::consts::{FRAC_1_SQRT_2, PI}, vec};
use crate::cart;

#[derive(Clone)]
pub enum Gate {
    CNOT(usize, usize),
    X(usize),
    Z(usize),
    Y(usize),
    H(usize),
    SWAP(usize, usize),
    CSWAP(usize, usize, usize), // Fredkin gate
}

pub enum Instruction {
    Gate(Gate),
    Measurement(usize),
    If {condition: Expr, gate: Gate},
    IfElse {condition: Expr, ifgate: Gate, elsegate: Gate},
}

pub enum Expr {
    True,
    False,
    Bit(usize),
    Not(Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    Xor(Box<Expr>, Box<Expr>),
}

impl Expr {
    pub fn eval(&self, bits: usize) -> bool {
        match self {
            Self::True => true,
            Self::False => false,
            Self::Bit(b) => ((bits >> b) & 1) == 1,
            Self::Not(e) => !Self::eval(e, bits),
            Self::And(e1, e2) => Self::eval(e1, bits) && Self::eval(e2, bits),
            Self::Or(e1, e2) => Self::eval(e1, bits) || Self::eval(e2, bits),
            Self::Xor(e1, e2) => Self::eval(e1, bits) ^ Self::eval(e2, bits),
        }
    }
}

pub mod expr_helpers {
    use crate::Expr;
    
    pub fn bit(i: usize) -> Expr {
        Expr::Bit(i)
    }

    pub fn not(e: Expr) -> Expr {
        Expr::Not(Box::new(e))
    }

    pub fn and(e1: Expr, e2: Expr) -> Expr {
        Expr::And(Box::new(e1), Box::new(e2))
    }

    pub fn or(e1: Expr, e2: Expr) -> Expr {
        Expr::Or(Box::new(e1), Box::new(e2))
    }

    pub fn xor(e1: Expr, e2: Expr) -> Expr {
        Expr::Xor(Box::new(e1), Box::new(e2))
    }
}

mod tests {
    use crate::{Expr, expr_helpers::*};

    #[test]
    fn test() {
        
        let my_expr1 = and(xor(bit(0), bit(1)), bit(2));
        println!("{}", my_expr1.eval(110));
    }
}

impl Gate {
    pub const PAULI_X_DATA: [Complex<f32>; 4] = [
        cart!(0.0), cart!(1.0),
        cart!(1.0), cart!(0.0),
    ];

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
            Self::X(_) => (2, &Self::PAULI_X_DATA),
            Self::Y(_) => (2, &Self::PAULI_Y_DATA),
            Self::Z(_) => (2, &Self::PAULI_Z_DATA),
            Self::H(_) => (2, &Self::HADAMARD_DATA),
            Self::CNOT(_, _) => (2, &Self::PAULI_X_DATA),
            Self::SWAP(_, _) => (4, &Self::SWAP_DATA),
            Self::CSWAP(_, _, _) => (4, &Self::SWAP_DATA),
        };

        return DMatrix::from_row_slice(dim, dim, data);
    }

    pub fn get_controls(&self) -> Vec<usize> {
        match self {
            Self::CNOT(c, _)
            | Self::CSWAP(c, _, _) => vec![*c],
            _ => vec![],
        }
    }

    pub fn get_targets(&self) -> Vec<usize> {
        match self {
            Self::CNOT(_, t)
            | Self::X(t)
            | Self::Y(t)
            | Self::Z(t)
            | Self::H(t) => vec![*t],
            Self::SWAP(t1, t2)
            | Self::CSWAP(_, t1, t2) => vec![*t1, *t2],
        }
    }
}
