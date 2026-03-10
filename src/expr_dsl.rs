use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Not, Rem, Sub};

use serde::{Deserialize, Serialize};

use crate::register_file::RegisterFile;

#[derive(Debug, thiserror::Error)]
pub enum ValueError {
    #[error("Operation type mismatch")]
    TypeMismatch,
}

#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    Int(i32),
    Float(f32),
    Bool(bool),
}

impl Default for Value {
    fn default() -> Self {
        Value::Int(0)
    }
}

impl Value {
    pub fn not(self) -> Result<Value, ValueError> {
        match self {
            Value::Int(x) => Ok(Value::Int(!x)),
            Value::Bool(x) => Ok(Value::Bool(!x)),
            _ => Err(ValueError::TypeMismatch),
        }
    }

    pub fn and(self, rhs: Value) -> Result<Value, ValueError> {
        match (self, rhs) {
            (Value::Int(x), Value::Int(y)) => Ok(Value::Int(x & y)),
            (Value::Bool(x), Value::Bool(y)) => Ok(Value::Bool(x && y)),
            _ => Err(ValueError::TypeMismatch),
        }
    }

    pub fn or(self, rhs: Value) -> Result<Value, ValueError> {
        match (self, rhs) {
            (Value::Int(x), Value::Int(y)) => Ok(Value::Int(x | y)),
            (Value::Bool(x), Value::Bool(y)) => Ok(Value::Bool(x || y)),
            _ => Err(ValueError::TypeMismatch),
        }
    }

    pub fn xor(self, rhs: Value) -> Result<Value, ValueError> {
        match (self, rhs) {
            (Value::Int(x), Value::Int(y)) => Ok(Value::Int(x ^ y)),
            (Value::Bool(x), Value::Bool(y)) => Ok(Value::Bool(x ^ y)),
            _ => Err(ValueError::TypeMismatch),
        }
    }

    pub fn add(self, rhs: Value) -> Result<Value, ValueError> {
        self.numeric_binop(rhs, |a, b| Value::Int(a + b), |x, y| Value::Float(x + y))
    }

    pub fn sub(self, rhs: Value) -> Result<Value, ValueError> {
        self.numeric_binop(rhs, |a, b| Value::Int(a - b), |x, y| Value::Float(x - y))
    }

    pub fn mul(self, rhs: Value) -> Result<Value, ValueError> {
        self.numeric_binop(rhs, |a, b| Value::Int(a * b), |x, y| Value::Float(x * y))
    }

    pub fn div(self, rhs: Value) -> Result<Value, ValueError> {
        self.numeric_binop(rhs, |a, b| Value::Int(a / b), |x, y| Value::Float(x / y))
    }

    pub fn rem(self, rhs: Value) -> Result<Value, ValueError> {
        self.numeric_binop(rhs, |a, b| Value::Int(a % b), |x, y| Value::Float(x % y))
    }

    pub fn lt(self, rhs: Value) -> Result<Value, ValueError> {
        self.numeric_binop(rhs, |a, b| Value::Bool(a < b), |x, y| Value::Bool(x < y))
    }

    pub fn eq(self, rhs: Value) -> Value {
        Value::Bool(self == rhs)
    }

    fn numeric_binop<FInt, FFloat>(
        self,
        rhs: Value,
        int_fn: FInt,
        float_fn: FFloat,
    ) -> Result<Value, ValueError>
    where
        FInt: Fn(i32, i32) -> Value,
        FFloat: Fn(f32, f32) -> Value,
    {
        match (self, rhs) {
            (Value::Int(x), Value::Int(y)) => Ok(int_fn(x, y)),
            (Value::Float(x), Value::Float(y)) => Ok(float_fn(x, y)),
            (Value::Int(x), Value::Float(y)) => Ok(float_fn(x as f32, y)),
            (Value::Float(x), Value::Int(y)) => Ok(float_fn(x, y as f32)),
            _ => Err(ValueError::TypeMismatch),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Expr {
    Val(Value),
    Reg(String),

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

    pub fn eval(&self, regs: &RegisterFile<Value>) -> Result<Value, ValueError> {
        match self {
            Expr::Val(v) => Ok(*v),
            Expr::Reg(name) => Ok(regs[name]),

            Expr::Not(e) => e.eval(regs)?.not(),
            Expr::And(a, b) => a.eval(regs)?.and(b.eval(regs)?),
            Expr::Or(a, b) => a.eval(regs)?.or(b.eval(regs)?),
            Expr::Xor(a, b) => a.eval(regs)?.xor(b.eval(regs)?),

            Expr::Add(a, b) => a.eval(regs)?.add(b.eval(regs)?),
            Expr::Sub(a, b) => a.eval(regs)?.sub(b.eval(regs)?),
            Expr::Mul(a, b) => a.eval(regs)?.mul(b.eval(regs)?),
            Expr::Div(a, b) => a.eval(regs)?.div(b.eval(regs)?),
            Expr::Rem(a, b) => a.eval(regs)?.rem(b.eval(regs)?),

            Expr::Eq(a, b) => Ok(a.eval(regs)?.eq(b.eval(regs)?)),
            Expr::Lt(a, b) => a.eval(regs)?.lt(b.eval(regs)?),
        }
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
    pub fn r(reg: &str) -> Expr {
        Expr::Reg(reg.to_owned())
    }
}
