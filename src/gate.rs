use std::f32::consts::FRAC_1_SQRT_2;

use nalgebra::{Complex};

use crate::cart;

#[derive(Debug, Copy, Clone)]
pub struct QBits(usize);

impl QBits {
    /// Specify qubits from bitstring
    pub fn from_bitstring(bits: usize) -> Self {
        Self(bits)
    }

    /// Specify qubits from a list of indices
    pub fn from_indices<'a, I>(indices: I) -> Self
    where
        I: IntoIterator<Item = &'a usize>
    {
        let mut bits = 0;
        for i in indices {
            bits |= 1 << i;
        }
        Self(bits)
    }

    pub fn get_bitstring(&self) -> usize {
        self.0
    }

    pub fn get_indices(&self) -> Vec<usize> {
        let mut bits = self.0;
        let mut vec = Vec::new();
        let mut index = 0;

        while bits != 0 {
            if bits & 1 == 1 {
                vec.push(index);
            }
            bits >>= 1;
            index += 1;
        }

        vec
    }
}

#[derive(Debug, Copy, Clone)]
pub enum GateType {
    X,
    Y,
    Z,
    H,
    SWAP,
}

impl GateType {
    pub fn arity(&self) -> usize {
        match self {
            Self::X | Self::Y | Self::Z | Self::H => 1,
            Self::SWAP => 2,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GateError {
    #[error("Invalid targete")]
    InvalidTargets,
}

#[derive(Debug, Clone)]
pub struct Gate {
    ty: GateType,
    controls: QBits,
    targets: QBits,
}

impl Gate {
    #[rustfmt::skip]
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

    pub fn new(ty: GateType, controls: &[usize], targets: &[usize]) -> Result<Self, GateError> {
        if targets.len() != ty.arity() {
            return Err(GateError::InvalidTargets);
        }

        Ok(Self {
            ty: ty,
            controls: QBits::from_indices(controls),
            targets: QBits::from_indices(targets),
        })
    }

    pub fn get_type(&self) -> GateType {
        self.ty
    }

    pub fn get_control_bits(&self) -> QBits {
        self.controls
    }

    pub fn get_target_bits(&self) -> QBits {
        self.targets
    }

    pub fn get_controls(&self) -> Vec<usize> {
        self.get_control_bits().get_indices()
    }

    pub fn get_targets(&self) -> Vec<usize> {
        self.get_target_bits().get_indices()
    }
}
