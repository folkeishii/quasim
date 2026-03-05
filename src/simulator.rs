use crate::circuit::pc::CircuitPc;
use crate::register_file::RegisterFile;
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
    fn final_state(&self) -> DVector<Complex<f64>>;
}

/// # DebuggableSimulator
/// Any simulator that can step through a circuit
/// one gate at a time should implement this trait
pub trait DebuggableSimulator {
    fn next(&mut self) -> Option<&DVector<Complex<f64>>>;
    /// Not guaranteed to be implemented for every simulator
    ///
    /// `prev` should be implemented if `fn double_ended(&self)`
    /// returns true
    fn prev(&mut self) -> Option<&DVector<Complex<f64>>> {
        todo!()
    }
    fn double_ended(&self) -> bool;
    /// Returns current pc and instruction
    ///
    /// If returned value is (pc, None)
    /// then we have reached the end of (sub) circuit
    fn current_instruction(&self) -> (&CircuitPc, Option<&Instruction>);
    fn current_state(&self) -> &DVector<Complex<f64>>;

    fn continue_until(&mut self, breakpoint: Option<CircuitPc>) -> &DVector<Complex<f64>> {
        let breakpoint = breakpoint.map(Into::into);
        while let (pc, Some(_)) = self.current_instruction() {
            if Some(pc) == breakpoint.as_ref() {
                break;
            }
            self.next();
        }
        self.current_state()
    }
}

/// # StoredCircuitSimulator
/// Any simulator that stores the underlying circuit
/// internally should implment this trait
pub trait StoredCircuitSimulator {
    fn circuit(&self) -> &Circuit;
    fn circuit_mut(&mut self) -> &mut Circuit;
    fn instructions(&self, sub_circuit: Option<&str>) -> &[Instruction] {
        self.circuit().instructions(sub_circuit)
    }
    fn instruction_count(&self, sub_circuit: Option<&str>) -> usize {
        self.circuit().instructions(sub_circuit).len()
    }
}

/// # HybridSimulator
/// Any simulator that implements classical operations
/// and stores registers should implement this trait
pub trait HybridSimulator<T: Copy> {
    fn registers(&self) -> &RegisterFile<T>;

    fn register(&self, register: &str) -> T {
        self.registers()[register]
    }
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
