use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Not, Rem, Sub};

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Value {
    Int(i32),
    Float(f32),
    Bool(bool),

    Err,
}

impl Value {
    pub fn not(self) -> Value {
        match self {
            Value::Int(x) => Value::Int(!x),
            Value::Bool(x) => Value::Bool(!x),
            _ => Value::Err,
        }
    }

    pub fn and(self, rhs: Value) -> Value {
        match (self, rhs) {
            (Value::Int(x), Value::Int(y)) => Value::Int(x & y),
            (Value::Bool(x), Value::Bool(y)) => Value::Bool(x && y),
            _ => Value::Err,
        }
    }

    pub fn or(self, rhs: Value) -> Value {
        match (self, rhs) {
            (Value::Int(x), Value::Int(y)) => Value::Int(x | y),
            (Value::Bool(x), Value::Bool(y)) => Value::Bool(x || y),
            _ => Value::Err,
        }
    }

    pub fn xor(self, rhs: Value) -> Value {
        match (self, rhs) {
            (Value::Int(x), Value::Int(y)) => Value::Int(x ^ y),
            (Value::Bool(x), Value::Bool(y)) => Value::Bool(x ^ y),
            _ => Value::Err,
        }
    }

    pub fn add(self, rhs: Value) -> Value {
        self.numeric_binop(rhs, |a, b| Value::Int(a + b), |x, y| Value::Float(x + y))
    }

    pub fn sub(self, rhs: Value) -> Value {
        self.numeric_binop(rhs, |a, b| Value::Int(a - b), |x, y| Value::Float(x - y))
    }

    pub fn mul(self, rhs: Value) -> Value {
        self.numeric_binop(rhs, |a, b| Value::Int(a * b), |x, y| Value::Float(x * y))
    }

    pub fn div(self, rhs: Value) -> Value {
        self.numeric_binop(rhs, |a, b| Value::Int(a / b), |x, y| Value::Float(x / y))
    }

    pub fn rem(self, rhs: Value) -> Value {
        self.numeric_binop(rhs, |a, b| Value::Int(a % b), |x, y| Value::Float(x % y))
    }

    pub fn lt(self, rhs: Value) -> Value {
        self.numeric_binop(rhs, |a, b| Value::Bool(a < b), |x, y| Value::Bool(x < y))
    }

    pub fn eq(self, rhs: Value) -> Value {
        Value::Bool(self == rhs)
    }

    fn numeric_binop<FInt, FFloat>(self, rhs: Value, int_fn: FInt, float_fn: FFloat) -> Value
    where
        FInt: Fn(i32, i32) -> Value,
        FFloat: Fn(f32, f32) -> Value,
    {
        match (self, rhs) {
            (Value::Int(x), Value::Int(y)) => int_fn(x, y),
            (Value::Float(x), Value::Float(y)) => float_fn(x, y),
            (Value::Int(x), Value::Float(y)) => float_fn(x as f32, y),
            (Value::Float(x), Value::Int(y)) => float_fn(x, y as f32),
            _ => Value::Err,
        }
    }
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
    Div(Box<Expr>, Box<Expr>),
    Rem(Box<Expr>, Box<Expr>),

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

// Arithmetic operators for expressions

impl<V: Into<Expr>> Add<V> for Expr {
    type Output = Expr;

    fn add(self, rhs: V) -> Self::Output {
        Expr::Add(Box::new(self), Box::new(rhs.into()))
    }
}

impl<V: Into<Expr>> Sub<V> for Expr {
    type Output = Expr;

    fn sub(self, rhs: V) -> Self::Output {
        Expr::Sub(Box::new(self), Box::new(rhs.into()))
    }
}

impl<V: Into<Expr>> Mul<V> for Expr {
    type Output = Expr;

    fn mul(self, rhs: V) -> Self::Output {
        Expr::Mul(Box::new(self), Box::new(rhs.into()))
    }
}

impl<V: Into<Expr>> Div<V> for Expr {
    type Output = Expr;

    fn div(self, rhs: V) -> Self::Output {
        Expr::Div(Box::new(self), Box::new(rhs.into()))
    }
}

impl<V: Into<Expr>> Rem<V> for Expr {
    type Output = Expr;

    fn rem(self, rhs: V) -> Self::Output {
        Expr::Rem(Box::new(self), Box::new(rhs.into()))
    }
}

impl<V: Into<Expr>> BitXor<V> for Expr {
    type Output = Expr;

    fn bitxor(self, rhs: V) -> Self::Output {
        Expr::Xor(Box::new(self), Box::new(rhs.into()))
    }
}

impl<V: Into<Expr>> BitAnd<V> for Expr {
    type Output = Expr;

    fn bitand(self, rhs: V) -> Self::Output {
        Expr::And(Box::new(self), Box::new(rhs.into()))
    }
}

impl<V: Into<Expr>> BitOr<V> for Expr {
    type Output = Expr;

    fn bitor(self, rhs: V) -> Self::Output {
        Expr::Or(Box::new(self), Box::new(rhs.into()))
    }
}

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
        expr_dsl::{expr_helpers::r, DummySimWithRegisters, Value},
        simulator::HybridSimulator,
    };

    #[test]
    fn test() {
        let mut sim = DummySimWithRegisters {
            registers: Vec::new(),
        };
        // sim.allocate(16);

        // sim.registers[0] = Value::Int(10);

        let expr = (r(0) % 5.0).gt(14.9) & 4.5;

        println!("{:?}", expr);
        // println!("{:?}", sim.eval(&expr));
    }
}
