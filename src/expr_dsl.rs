use std::ops::{Add, BitAnd, BitOr, BitXor, Mul, Not, Sub};

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Value {
    Int(i32),
    Float(f32),
    Bool(bool),

    Err,
}

#[derive(Debug, Clone)]
pub enum Expr {
    Val(Value),
    Reg(usize),

    Not(Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    Xor(Box<Expr>, Box<Expr>),

    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),

    Eq(Box<Expr>, Box<Expr>),
    Lt(Box<Expr>, Box<Expr>),
}

impl Expr {
    pub fn eq<V: Into<Expr>>(self, rhs: V) -> Expr {
        Expr::Eq(Box::new(self), Box::new(rhs.into()))
    }

    pub fn lt<V: Into<Expr>>(self, rhs: V) -> Expr {
        Expr::Lt(Box::new(self), Box::new(rhs.into()))
    }

    pub fn lte<V: Into<Expr>>(self, rhs: V) -> Expr {
        Expr::Not(Box::new(Expr::Lt(Box::new(rhs.into()), Box::new(self))))
    }

    pub fn gt<V: Into<Expr>>(self, rhs: V) -> Expr {
        Expr::Lt(Box::new(rhs.into()), Box::new(self))
    }

    pub fn gte<V: Into<Expr>>(self, rhs: V) -> Expr {
        Expr::Not(Box::new(Expr::Lt(Box::new(self), Box::new(rhs.into()))))
    }
}

// Into types

impl From<i32> for Expr {
    fn from(v: i32) -> Self {
        Expr::Val(Value::Int(v))
    }
}

impl From<f32> for Expr {
    fn from(v: f32) -> Self {
        Expr::Val(Value::Float(v))
    }
}

impl From<bool> for Expr {
    fn from(v: bool) -> Self {
        Expr::Val(Value::Bool(v))
    }
}

// Addition

impl<V: Into<Expr>> Add<V> for Expr {
    type Output = Expr;

    fn add(self, rhs: V) -> Self::Output {
        Expr::Add(Box::new(self), Box::new(rhs.into()))
    }
}

// Subtraction

impl<V: Into<Expr>> Sub<V> for Expr {
    type Output = Expr;

    fn sub(self, rhs: V) -> Self::Output {
        Expr::Sub(Box::new(self), Box::new(rhs.into()))
    }
}

// Multiplication

impl<V: Into<Expr>> Mul<V> for Expr {
    type Output = Expr;

    fn mul(self, rhs: V) -> Self::Output {
        Expr::Mul(Box::new(self), Box::new(rhs.into()))
    }
}

// Bitwise Xor

impl<V: Into<Expr>> BitXor<V> for Expr {
    type Output = Expr;

    fn bitxor(self, rhs: V) -> Self::Output {
        Expr::Xor(Box::new(self), Box::new(rhs.into()))
    }
}

// Bitwise And

impl<V: Into<Expr>> BitAnd<V> for Expr {
    type Output = Expr;

    fn bitand(self, rhs: V) -> Self::Output {
        Expr::And(Box::new(self), Box::new(rhs.into()))
    }
}

// Bitwise Or

impl<V: Into<Expr>> BitOr<V> for Expr {
    type Output = Expr;

    fn bitor(self, rhs: V) -> Self::Output {
        Expr::Or(Box::new(self), Box::new(rhs.into()))
    }
}

// Bitwise Not

impl Not for Expr {
    type Output = Expr;

    fn not(self) -> Self::Output {
        Expr::Not(Box::new(self))
    }
}

pub mod expr_helpers {
    use crate::expr_dsl::Expr;

    /// Read register
    pub fn r(reg: usize) -> Expr {
        Expr::Reg(reg)
    }

    // pub fn rb(reg: usize) -> Expr {
    //     !(Expr::Reg(reg).eq(0))
    // }
}

struct DummySimWithRegisters {
    registers: Vec<Value>,
}

impl crate::simulator::HybridSimulator for DummySimWithRegisters {
    // fn allocate(&mut self, classical_regs: usize) {
    //     self.registers.resize(classical_regs, Value::Int(0));
    // }

    fn get(&self, reg: usize) -> Value {
        self.registers[reg]
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        expr_dsl::{
            DummySimWithRegisters, Value,
            expr_helpers::{r},
        },
        simulator::HybridSimulator,
    };

    #[test]
    fn test() {
        let mut sim = DummySimWithRegisters {
            registers: Vec::new(),
        };
        // sim.allocate(16);

        sim.registers[0] = Value::Int(10);

        let expr = (r(0) + 5.0).gt(14.9) & 4.5;

        println!("{:?}", expr);
        println!("{:?}", sim.eval(&expr));
    }
}
