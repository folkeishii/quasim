use crate::{circuit::Circuit, instruction::Instruction};
use nalgebra::{Complex, DVector};

pub trait BuildSimulator: Sized {
    type E: std::error::Error;

    fn build(circuit: Circuit) -> Result<Self, Self::E>;
}
impl<T: TryFrom<Circuit, Error = E>, E: std::error::Error> BuildSimulator for T {
    type E = E;

    fn build(circuit: Circuit) -> Result<Self, Self::E> {
        Self::try_from(circuit)
    }
}

pub trait RunnableSimulator {
    fn run(&self) -> usize;
    fn final_state(&self) -> DVector<Complex<f32>>;
}

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

pub trait DoubleEndedSimulator: DebuggableSimulator {
    fn prev(&mut self) -> Option<&DVector<Complex<f32>>>;
}
