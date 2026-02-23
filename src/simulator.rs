use crate::{
    circuit::Circuit,
    expr_dsl::{Expr, Value},
    instruction::Instruction,
};
use nalgebra::{Complex, DVector};

/// # BuildSimulator
/// Any simulator that is able to be built from a
/// circuit should implement this trait.
///
/// To support `TryFrom<Cicuit>` there is an auto
/// implementation of `BuildSimulator` for any type
/// that implements `TryFrom<Circuit>`
pub trait BuildSimulator: Sized {
    type E: std::error::Error;

    fn build(circuit: Circuit) -> Result<Self, Self::E>;
}
impl<T, E> BuildSimulator for T
where
    T: TryFrom<Circuit, Error = E>,
    E: std::error::Error,
{
    type E = E;

    fn build(circuit: Circuit) -> Result<Self, Self::E> {
        Self::try_from(circuit)
    }
}

/// # RunnableSimulator
/// Any simulator that can calculate the circuits
/// final state without changing internal state
/// should implement this trait
pub trait RunnableSimulator {
    fn run(&self) -> usize;
    fn final_state(&self) -> DVector<Complex<f32>>;
}

/// # DebuggableSimulator
/// Any simulator that can step through a circuit
/// one gate at a time should implement this trait
pub trait DebuggableSimulator {
    fn next(&mut self) -> Option<&DVector<Complex<f32>>>;
    fn current_instruction(&self) -> Option<(usize, &Instruction)>;
    fn current_state(&self) -> &DVector<Complex<f32>>;

    fn continue_until(&mut self, breakpoint: Option<usize>) -> &DVector<Complex<f32>> {
        while let Some(index) = self.current_instruction().map(|(i, _)| i) {
            if Some(index) == breakpoint {
                break;
            }
            self.next();
        }
        self.current_state()
    }
}

/// # DebuggableSimulator
/// Any simulator that can step back through a circuit
/// one gate at a time should implement this trait
pub trait DoubleEndedSimulator: DebuggableSimulator {
    fn prev(&mut self) -> Option<&DVector<Complex<f32>>>;
}

#[cfg(test)]
mod tests {

    use crate::{
        circuit::Circuit,
        debug_simulator::DebugSimulator,
        ext::equal_to_matrix_c,
        simulator::{BuildSimulator, DebuggableSimulator},
    };

    #[test]
    fn test_continue_until() {
        let circ = Circuit::new(3).hadamard(0).hadamard(1).hadamard(2);
        let mut sim1 = DebugSimulator::build(circ).unwrap();
        let mut sim2 = sim1.clone();

        sim1.next().unwrap();
        sim1.next().unwrap();
        assert!(equal_to_matrix_c(
            sim1.next().unwrap(),
            sim2.continue_until(None),
            0.001
        ))
    }
}

/// # HybridSimulator
/// Any simulator that contains classical registers and can
/// evaluate expressions from the contents of its registers
pub trait HybridSimulator {
    // fn allocate(&mut self, reg_count: usize);
    fn get(&self, idx: usize) -> Value;

    fn eval(&self, expr: &Expr) -> Value {
        match expr {
            Expr::Val(v) => v.clone(),

            Expr::Reg(idx) => self.get(*idx),

            Expr::Not(e) => match self.eval(e) {
                Value::Int(x) => Value::Int(!x),
                Value::Bool(x) => Value::Bool(!x),
                _ => Value::Err,
            },

            Expr::And(a, b) => match (self.eval(a), self.eval(b)) {
                (Value::Int(x), Value::Int(y)) => Value::Int(x & y),
                (Value::Bool(x), Value::Bool(y)) => Value::Bool(x && y),
                (Value::Int(x), Value::Bool(y)) => Value::Bool((x != 0) && y),
                (Value::Bool(x), Value::Int(y)) => Value::Bool(x && (y != 0)),
                _ => Value::Err,
            },

            Expr::Or(a, b) => match (self.eval(a), self.eval(b)) {
                (Value::Int(x), Value::Int(y)) => Value::Int(x | y),
                (Value::Bool(x), Value::Bool(y)) => Value::Bool(x || y),
                (Value::Int(x), Value::Bool(y)) => Value::Bool((x != 0) || y),
                (Value::Bool(x), Value::Int(y)) => Value::Bool(x || (y != 0)),
                _ => Value::Err,
            },

            Expr::Xor(a, b) => match (self.eval(a), self.eval(b)) {
                (Value::Int(x), Value::Int(y)) => Value::Int(x ^ y),
                (Value::Bool(x), Value::Bool(y)) => Value::Bool(x ^ y),
                (Value::Int(x), Value::Bool(y)) => Value::Bool((x != 0) ^ y),
                (Value::Bool(x), Value::Int(y)) => Value::Bool(x ^ (y != 0)),
                _ => Value::Err,
            },

            Expr::Add(a, b) => match (self.eval(a), self.eval(b)) {
                (Value::Int(x), Value::Int(y)) => Value::Int(x + y),
                (Value::Float(x), Value::Float(y)) => Value::Float(x + y),
                (Value::Int(x), Value::Float(y)) => Value::Float(x as f32 + y),
                (Value::Float(x), Value::Int(y)) => Value::Float(x + y as f32),
                _ => Value::Err,
            },

            Expr::Sub(a, b) => match (self.eval(a), self.eval(b)) {
                (Value::Int(x), Value::Int(y)) => Value::Int(x - y),
                (Value::Float(x), Value::Float(y)) => Value::Float(x - y),
                (Value::Int(x), Value::Float(y)) => Value::Float(x as f32 - y),
                (Value::Float(x), Value::Int(y)) => Value::Float(x - y as f32),
                _ => Value::Err,
            },

            Expr::Mul(a, b) => match (self.eval(a), self.eval(b)) {
                (Value::Int(x), Value::Int(y)) => Value::Int(x * y),
                (Value::Float(x), Value::Float(y)) => Value::Float(x * y),
                (Value::Int(x), Value::Float(y)) => Value::Float(x as f32 * y),
                (Value::Float(x), Value::Int(y)) => Value::Float(x * y as f32),
                _ => Value::Err,
            },

            Expr::Eq(a, b) => Value::Bool(self.eval(a) == self.eval(b)),

            Expr::Lt(a, b) => match (self.eval(a), self.eval(b)) {
                (Value::Int(x), Value::Int(y)) => Value::Bool(x < y),
                (Value::Float(x), Value::Float(y)) => Value::Bool(x < y),
                (Value::Int(x), Value::Float(y)) => Value::Bool((x as f32) < y),
                (Value::Float(x), Value::Int(y)) => Value::Bool(x < (y as f32)),
                _ => Value::Err,
            },
        }
    }
}
