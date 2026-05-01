use core::f32;
use std::{
    fmt::{Debug, Display},
    ops::{Add, Mul, Neg},
};

use nalgebra::Complex;

use crate::syntax_simulator::{ScaledState, state::Sum};

#[derive(Clone, PartialEq, Copy)]
pub struct Scalar {
    frac_1_sqrt_2_power: usize,
    number: Complex<f32>,
}

impl Scalar {
    pub const ZERO: Scalar = Scalar {
        frac_1_sqrt_2_power: 0,
        number: Complex::new(0.0, 0.0),
    };

    pub const ONE: Scalar = Scalar {
        frac_1_sqrt_2_power: 0,
        number: Complex::new(1.0, 0.0),
    };

    pub const I: Scalar = Scalar {
        frac_1_sqrt_2_power: 0,
        number: Complex::new(0.0, 1.0),
    };

    pub const FRAC_1_SQRT_2: Scalar = Scalar {
        frac_1_sqrt_2_power: 1,
        number: Complex::new(1.0, 0.0),
    };

    pub fn mul_frac_1_sqrt_2(&mut self) {
        self.frac_1_sqrt_2_power += 1;
    }

    pub fn is_zero(&self) -> bool {
        self.number == Complex::new(0.0, 0.0)
    }

    pub fn probability(&self) -> f32 {
        let c: Complex<f32> = (*self).into();
        c.norm_sqr()
    }
}

impl Add for Scalar {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        if self.frac_1_sqrt_2_power == rhs.frac_1_sqrt_2_power {
            Self {
                frac_1_sqrt_2_power: self.frac_1_sqrt_2_power,
                number: self.number + rhs.number,
            }
        } else {
            Self {
                frac_1_sqrt_2_power: 0,
                number: Into::<Complex<f32>>::into(self) + Into::<Complex<f32>>::into(rhs),
            }
        }
    }
}

impl Neg for Scalar {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            frac_1_sqrt_2_power: self.frac_1_sqrt_2_power,
            number: -self.number,
        }
    }
}

impl Mul for Scalar {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            frac_1_sqrt_2_power: self.frac_1_sqrt_2_power + rhs.frac_1_sqrt_2_power,
            number: self.number * rhs.number,
        }
    }
}

impl Mul<ScaledState> for Scalar {
    type Output = ScaledState;

    fn mul(self, rhs: ScaledState) -> Self::Output {
        let ScaledState(state, scalar) = rhs;
        ScaledState(state, self * scalar)
    }
}

impl Mul<Sum> for Scalar {
    type Output = Sum;

    fn mul(self, rhs: Sum) -> Sum {
        rhs.into_iter().map(|ss| self * ss).collect()
    }
}

impl Debug for Scalar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

impl Display for Scalar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.frac_1_sqrt_2_power == 0 {
            write!(f, "{}", self.number)
        } else {
            write!(f, "({} / √2^{}))", self.number, self.frac_1_sqrt_2_power)
        }
    }
}

impl From<Complex<f32>> for Scalar {
    fn from(value: Complex<f32>) -> Self {
        Scalar {
            frac_1_sqrt_2_power: 0,
            number: value,
        }
    }
}

impl Into<Complex<f32>> for Scalar {
    fn into(self) -> Complex<f32> {
        f32::consts::FRAC_1_SQRT_2.powi(self.frac_1_sqrt_2_power as i32) * self.number
    }
}
