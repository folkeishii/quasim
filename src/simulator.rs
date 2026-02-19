use crate::{circuit::Circuit, instruction::Instruction};
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
