use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Not, Rem, Sub};

use serde::{Deserialize, Serialize};

use crate::register_file::RegisterFile;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BitExpr {
    Val(u64),
    Reg(String),
    RegBit(String, usize),

    Not(Box<Self>),
    And(Box<Self>, Box<Self>),
    Or(Box<Self>, Box<Self>),
    Xor(Box<Self>, Box<Self>),

    Add(Box<Self>, Box<Self>),
    Sub(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    Div(Box<Self>, Box<Self>),
    Rem(Box<Self>, Box<Self>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BoolExpr {
    Eq(Box<BitExpr>, Box<BitExpr>),
    Lt(Box<BitExpr>, Box<BitExpr>),

    Not(Box<Self>),
    And(Box<Self>, Box<Self>),
    Or(Box<Self>, Box<Self>),
    Xor(Box<Self>, Box<Self>),
}

impl BitExpr {
    pub fn eq<T: Into<BitExpr>>(self, rhs: T) -> BoolExpr {
        BoolExpr::Eq(Box::new(self), Box::new(rhs.into()))
    }

    pub fn lt<T: Into<BitExpr>>(self, rhs: T) -> BoolExpr {
        BoolExpr::Lt(Box::new(self), Box::new(rhs.into()))
    }

    pub fn lte<T: Into<BitExpr>>(self, rhs: T) -> BoolExpr {
        BoolExpr::Not(Box::new(BoolExpr::Lt(Box::new(rhs.into()), Box::new(self))))
    }

    pub fn gt<T: Into<BitExpr>>(self, rhs: T) -> BoolExpr {
        BoolExpr::Lt(Box::new(rhs.into()), Box::new(self))
    }

    pub fn gte<T: Into<BitExpr>>(self, rhs: T) -> BoolExpr {
        BoolExpr::Not(Box::new(BoolExpr::Lt(Box::new(self), Box::new(rhs.into()))))
    }

    pub fn eval(&self, regs: &RegisterFile) -> u64 {
        match self {
            Self::Val(v) => *v,
            Self::Reg(name) => regs[name].read(),
            Self::RegBit(name, usize) => regs[name].read_bit(*usize),

            Self::Not(e) => e.eval(regs).not(),
            Self::And(a, b) => a.eval(regs) & b.eval(regs),
            Self::Or(a, b) => a.eval(regs) | b.eval(regs),
            Self::Xor(a, b) => a.eval(regs) ^ b.eval(regs),

            Self::Add(a, b) => a.eval(regs) + b.eval(regs),
            Self::Sub(a, b) => a.eval(regs) - b.eval(regs),
            Self::Mul(a, b) => a.eval(regs) * b.eval(regs),
            Self::Div(a, b) => a.eval(regs) / b.eval(regs),
            Self::Rem(a, b) => a.eval(regs) % b.eval(regs),
        }
    }
}

impl BoolExpr {
    pub fn eval(&self, regs: &RegisterFile) -> bool {
        match self {
            Self::Not(e) => !e.eval(regs),
            Self::And(a, b) => a.eval(regs) && b.eval(regs),
            Self::Or(a, b) => a.eval(regs) || b.eval(regs),
            Self::Xor(a, b) => a.eval(regs) ^ b.eval(regs),

            Self::Eq(a, b) => a.eval(regs) == b.eval(regs),
            Self::Lt(a, b) => a.eval(regs) < b.eval(regs),
        }
    }
}

// Into types

impl From<u64> for BitExpr {
    fn from(v: u64) -> Self {
        Self::Val(v)
    }
}

// Arithmetic operators for BitExpr

impl<V: Into<Self>> Add<V> for BitExpr {
    type Output = Self;

    fn add(self, rhs: V) -> Self::Output {
        Self::Add(Box::new(self), Box::new(rhs.into()))
    }
}

impl<V: Into<Self>> Sub<V> for BitExpr {
    type Output = Self;

    fn sub(self, rhs: V) -> Self::Output {
        Self::Sub(Box::new(self), Box::new(rhs.into()))
    }
}

impl<V: Into<Self>> Mul<V> for BitExpr {
    type Output = Self;

    fn mul(self, rhs: V) -> Self::Output {
        Self::Mul(Box::new(self), Box::new(rhs.into()))
    }
}

impl<V: Into<Self>> Div<V> for BitExpr {
    type Output = Self;

    fn div(self, rhs: V) -> Self::Output {
        Self::Div(Box::new(self), Box::new(rhs.into()))
    }
}

impl<V: Into<Self>> Rem<V> for BitExpr {
    type Output = Self;

    fn rem(self, rhs: V) -> Self::Output {
        Self::Rem(Box::new(self), Box::new(rhs.into()))
    }
}

impl<V: Into<Self>> BitXor<V> for BitExpr {
    type Output = Self;

    fn bitxor(self, rhs: V) -> Self::Output {
        Self::Xor(Box::new(self), Box::new(rhs.into()))
    }
}

impl<V: Into<Self>> BitAnd<V> for BitExpr {
    type Output = Self;

    fn bitand(self, rhs: V) -> Self::Output {
        Self::And(Box::new(self), Box::new(rhs.into()))
    }
}

impl<V: Into<Self>> BitOr<V> for BitExpr {
    type Output = Self;

    fn bitor(self, rhs: V) -> Self::Output {
        Self::Or(Box::new(self), Box::new(rhs.into()))
    }
}

impl Not for BitExpr {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self::Not(Box::new(self))
    }
}

// Boolean operators for BoolExpr

impl<V: Into<Self>> BitXor<V> for BoolExpr {
    type Output = Self;

    fn bitxor(self, rhs: V) -> Self::Output {
        Self::Xor(Box::new(self), Box::new(rhs.into()))
    }
}

impl<V: Into<Self>> BitAnd<V> for BoolExpr {
    type Output = Self;

    fn bitand(self, rhs: V) -> Self::Output {
        Self::And(Box::new(self), Box::new(rhs.into()))
    }
}

impl<V: Into<Self>> BitOr<V> for BoolExpr {
    type Output = Self;

    fn bitor(self, rhs: V) -> Self::Output {
        Self::Or(Box::new(self), Box::new(rhs.into()))
    }
}

impl Not for BoolExpr {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self::Not(Box::new(self))
    }
}

pub mod expr_helpers {
    use crate::expr_dsl::BitExpr;

    /// Read whole register
    pub fn r<S: Into<String>>(reg: S) -> BitExpr {
        BitExpr::Reg(reg.into())
    }

    /// Read bit in register
    pub fn rb<S: Into<String>>(reg: S, bit: usize) -> BitExpr {
        BitExpr::RegBit(reg.into(), bit)
    }
}
