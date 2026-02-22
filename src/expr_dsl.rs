use std::{ops::{Add, BitAnd, BitOr, BitXor, Mul, Not, Sub}};


#[derive(Debug, Clone)]
pub enum ValueExpr {
    Val(i32),
    Reg(usize),

    Not(Box<ValueExpr>),
    And(Box<ValueExpr>, Box<ValueExpr>),
    Or(Box<ValueExpr>, Box<ValueExpr>),
    Xor(Box<ValueExpr>, Box<ValueExpr>),

    Add(Box<ValueExpr>, Box<ValueExpr>),
    Sub(Box<ValueExpr>, Box<ValueExpr>),
    Mul(Box<ValueExpr>, Box<ValueExpr>),
}

#[derive(Debug, Clone)]
pub enum BoolExpr {
    True,
    False,

    NonZero(ValueExpr),

    Eq(ValueExpr, ValueExpr),
    Lt(ValueExpr, ValueExpr),

    Not(Box<BoolExpr>),
    And(Box<BoolExpr>, Box<BoolExpr>),
    Or(Box<BoolExpr>, Box<BoolExpr>),
}

impl ValueExpr {
    pub fn eq<V: Into<ValueExpr>>(self, rhs: V) -> BoolExpr {
        BoolExpr::Eq(self, rhs.into())
    }

    pub fn lt<V: Into<ValueExpr>>(self, rhs: V) -> BoolExpr {
        BoolExpr::Lt(self, rhs.into())
    }

    pub fn lte<V: Into<ValueExpr>>(self, rhs: V) -> BoolExpr {
        BoolExpr::Not(Box::new(BoolExpr::Lt(rhs.into(), self)))
    }

    pub fn gt<V: Into<ValueExpr>>(self, rhs: V) -> BoolExpr {
        BoolExpr::Lt(rhs.into(), self)
    }

    pub fn gte<V: Into<ValueExpr>>(self, rhs: V) -> BoolExpr {
        BoolExpr::Not(Box::new(BoolExpr::Lt(self, rhs.into())))
    }
}

impl<T: Into<i32>> From<T> for ValueExpr {
    fn from(v: T) -> Self {
        ValueExpr::Val(v.into())
    }
}

// Addition

impl<V: Into<ValueExpr>> Add<V> for ValueExpr {
    type Output = ValueExpr;

    fn add(self, rhs: V) -> Self::Output {
        ValueExpr::Add(Box::new(self), Box::new(rhs.into()))
    }
}

impl Add<ValueExpr> for i32 {
    type Output = ValueExpr;

    fn add(self, rhs: ValueExpr) -> Self::Output {
        ValueExpr::Val(self) + rhs
    }
}

// Subtraction

impl<V: Into<ValueExpr>> Sub<V> for ValueExpr {
    type Output = ValueExpr;

    fn sub(self, rhs: V) -> Self::Output {
        ValueExpr::Sub(Box::new(self), Box::new(rhs.into()))
    }
}

impl Sub<ValueExpr> for i32 {
    type Output = ValueExpr;

    fn sub(self, rhs: ValueExpr) -> Self::Output {
        ValueExpr::Val(self) - rhs
    }
}

// Multiplication

impl<V: Into<ValueExpr>> Mul<V> for ValueExpr {
    type Output = ValueExpr;

    fn mul(self, rhs: V) -> Self::Output {
        ValueExpr::Mul(Box::new(self), Box::new(rhs.into()))
    }
}

impl Mul<ValueExpr> for i32 {
    type Output = ValueExpr;

    fn mul(self, rhs: ValueExpr) -> Self::Output {
        ValueExpr::Val(self) * rhs
    }
}

// Bitwise Xor

impl<V: Into<ValueExpr>> BitXor<V> for ValueExpr {
    type Output = ValueExpr;

    fn bitxor(self, rhs: V) -> Self::Output {
        ValueExpr::Xor(Box::new(self), Box::new(rhs.into()))
    }
}

impl BitXor<ValueExpr> for i32 {
    type Output = ValueExpr;

    fn bitxor(self, rhs: ValueExpr) -> Self::Output {
        ValueExpr::Val(self) ^ rhs
    }
}

// Bitwise And

impl<V: Into<ValueExpr>> BitAnd<V> for ValueExpr {
    type Output = ValueExpr;

    fn bitand(self, rhs: V) -> Self::Output {
        ValueExpr::And(Box::new(self), Box::new(rhs.into()))
    }
}

impl BitAnd<ValueExpr> for i32 {
    type Output = ValueExpr;

    fn bitand(self, rhs: ValueExpr) -> Self::Output {
        ValueExpr::Val(self) & rhs
    }
}

// Bitwise Or

impl<V: Into<ValueExpr>> BitOr<V> for ValueExpr {
    type Output = ValueExpr;

    fn bitor(self, rhs: V) -> Self::Output {
        ValueExpr::Or(Box::new(self), Box::new(rhs.into()))
    }
}

impl BitOr<ValueExpr> for i32 {
    type Output = ValueExpr;

    fn bitor(self, rhs: ValueExpr) -> Self::Output {
        ValueExpr::Val(self) | rhs
    }
}

// Bitwise Not

impl Not for ValueExpr {
    type Output = ValueExpr;

    fn not(self) -> Self::Output {
        ValueExpr::Not(Box::new(self))
    }
}

// --- Boolean expressions ---

impl Not for BoolExpr {
    type Output = BoolExpr;

    fn not(self) -> Self::Output {
        BoolExpr::Not(Box::new(self))
    }
}

impl BitAnd for BoolExpr {
    type Output = BoolExpr;

    fn bitand(self, rhs: BoolExpr) -> Self::Output {
        BoolExpr::And(Box::new(self), Box::new(rhs))
    }
}

impl BitOr for BoolExpr {
    type Output = BoolExpr;

    fn bitor(self, rhs: BoolExpr) -> Self::Output {
        BoolExpr::Or(Box::new(self), Box::new(rhs))
    }
}

pub mod expr_helpers {
    use crate::expr_dsl::{BoolExpr, ValueExpr};

    /// Read register as value
    pub fn rv(reg: usize) -> ValueExpr {
        ValueExpr::Reg(reg)
    }

    /// Read register as boolean
    pub fn rb(reg: usize) -> BoolExpr {
        BoolExpr::NonZero(ValueExpr::Reg(reg))
    }
}

struct DummySimWithRegisters {
    registers: Vec<i32>,
}

impl crate::simulator::HybridSimulator for DummySimWithRegisters {
    fn allocate(&mut self, classical_regs: usize) {
        self.registers.resize(classical_regs, 0);
    }

    fn get(&self, reg: usize) -> i32 {
        self.registers[reg]
    }
}

#[cfg(test)]
mod tests {
    use crate::{expr_dsl::{DummySimWithRegisters, expr_helpers::{rb, rv}}, simulator::HybridSimulator};

    #[test]
	fn test() {
	    let mut sim = DummySimWithRegisters { registers: Vec::new() };
	    sim.allocate(16);
	
	    sim.registers[0] = 2;

        let expr = (2 + rv(0) + 5).eq(rv(2) | 2) & rb(1);

        let expr2 = rv(0) + 2i16;

        println!("{:#?}", expr);

	}
}
