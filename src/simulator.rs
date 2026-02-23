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
    fn get(&self, reg: usize) -> Value;

    fn eval(&self, expr: &Expr) -> Value {
        match expr {
            Expr::Val(v) => *v,
            Expr::Reg(i) => self.get(*i),

            Expr::Not(e) => self.eval(e).not(),
            Expr::And(a, b) => self.eval(a).and(self.eval(b)),
            Expr::Or(a, b) => self.eval(a).or(self.eval(b)),
            Expr::Xor(a, b) => self.eval(a).xor(self.eval(b)),

            Expr::Add(a, b) => self.eval(a).add(self.eval(b)),
            Expr::Sub(a, b) => self.eval(a).sub(self.eval(b)),
            Expr::Mul(a, b) => self.eval(a).mul(self.eval(b)),
            Expr::Div(a, b) => self.eval(a).div(self.eval(b)),
            Expr::Rem(a, b) => self.eval(a).rem(self.eval(b)),

            Expr::Eq(a, b) => self.eval(a).eq(self.eval(b)),
            Expr::Lt(a, b) => self.eval(a).lt(self.eval(b)),
        }
    }
}
