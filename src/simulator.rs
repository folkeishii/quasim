use crate::{circuit::Circuit, expr_dsl::{BoolExpr, ValueExpr}, instruction::Instruction};
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
    fn allocate(&mut self, reg_count: usize);
    fn get(&self, idx: usize) -> i32;

    fn eval(&self, expr: &ValueExpr) -> i32 {
        match expr {
            ValueExpr::Val(i) => *i,
            ValueExpr::Reg(r) => self.get(*r),

            ValueExpr::Not(e) => !self.eval(e),
            ValueExpr::And(e1, e2) => self.eval(e1) & self.eval(e2),
            ValueExpr::Or(e1, e2) => self.eval(e1) | self.eval(e2),
            ValueExpr::Xor(e1, e2) => self.eval(e1) ^ self.eval(e2),

            ValueExpr::Add(e1, e2) => self.eval(e1) + self.eval(e2),
            ValueExpr::Sub(e1, e2) => self.eval(e1) - self.eval(e2),
            ValueExpr::Mul(e1, e2) => self.eval(e1) * self.eval(e2),
        }
    }

    fn eval_bool(&self, expr: &BoolExpr) -> bool {
        match expr {
            BoolExpr::True => true,
            BoolExpr::False => false,

            BoolExpr::NonZero(e) => self.eval(e) != 0,
            BoolExpr::Eq(e1, e2) => self.eval(e1) == self.eval(e2),
            BoolExpr::Lt(e1, e2) => self.eval(e1) < self.eval(e2),

            BoolExpr::Not(e) => !self.eval_bool(e),
            BoolExpr::And(e1, e2) => self.eval_bool(e1) && self.eval_bool(e2),
            BoolExpr::Or(e1, e2) => self.eval_bool(e1) || self.eval_bool(e2),
        }
    }
}