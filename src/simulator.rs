use std::ops::Index;

use crate::circuit::CircuitBehaviour;
use crate::circuit::pc::CircuitPc;
use crate::register_file::RegisterFile;
use crate::{circuit::Circuit, instruction::Instruction};

/// # BuildSimulator
/// Any simulator that is able to be built from a
/// circuit should implement this trait.
///
/// To support `TryFrom<Cicuit>` there is an auto
/// implementation of `BuildSimulator` for any type
/// that implements `TryFrom<Circuit>`
pub trait BuildSimulator<B: CircuitBehaviour>: Sized {
    type E: std::error::Error;

    fn build(circuit: Circuit<B>) -> Result<Self, Self::E>;
}
impl<T, B: CircuitBehaviour, E> BuildSimulator<B> for T
where
    T: TryFrom<Circuit<B>, Error = E>,
    E: std::error::Error,
{
    type E = E;

    fn build(circuit: Circuit<B>) -> Result<Self, Self::E> {
        Self::try_from(circuit)
    }
}

/// # RunnableSimulator
/// Any simulator that can calculate the circuits
/// final state without changing internal state
/// should implement this trait
pub trait RunnableSimulator {
    type Storage: Index<usize, Output = Self::State>;
    type State;

    fn run(&self) -> usize;
    fn final_state(&self) -> Self::Storage;
}

/// # DebuggableSimulator
/// Any simulator that can step through a circuit
/// one gate at a time should implement this trait
pub trait DebuggableSimulator {
    type Storage;
    type State;

    fn next(&mut self) -> bool;
    /// Unlike `next`, `next_over` will execute all instructions
    /// inside a sub circuit
    fn next_over(&mut self) -> bool {
        let (before, _) = self.current_instruction();
        let before_depth = before.depth();

        // Allways do at least one next
        let mut ret = self.next();

        let (after, _) = self.current_instruction();
        let mut after_depth = after.depth();

        while before_depth < after_depth {
            // Inside sub circuit
            ret = self.next();

            let (after, _) = self.current_instruction();
            after_depth = after.depth();
        }

        ret
    }
    /// Not guaranteed to be implemented for every simulator
    ///
    /// `prev` should be implemented if `fn double_ended(&self)`
    /// returns true
    fn prev(&mut self) -> bool {
        todo!()
    }
    fn double_ended(&self) -> bool;
    /// Returns current pc and instruction
    ///
    /// If returned value is (pc, None)
    /// then we have reached the end of (sub) circuit
    fn current_instruction(&self) -> (&CircuitPc, Option<Instruction>);
    fn current_state(&self) -> &Self::Storage;
    fn collapse_peek(&self) -> usize;

    fn cont(&mut self) -> bool
    where
        Self: StoredCircuitSimulator,
    {
        while self.next() {
            let (pc, _) = self.current_instruction();
            if self.circuit().enabled_breakpoint_at(pc) {
                return true;
            }
        }
        false
    }
}

/// # StoredCircuitSimulator
/// Any simulator that stores the underlying circuit
/// internally should implment this trait
pub trait StoredCircuitSimulator {
    type B: CircuitBehaviour;

    fn circuit(&self) -> &Circuit<Self::B>;
    fn circuit_mut(&mut self) -> &mut Circuit<Self::B>;
    fn instructions(&self) -> &[<Self::B as CircuitBehaviour>::InstructionTy] {
        self.circuit().instructions()
    }
    fn instruction_count(&self) -> usize {
        self.circuit().instructions().len()
    }
    fn n_qubits(&self) -> usize {
        self.circuit().n_qubits()
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
        ext::equal_state_c,
        simulator::{BuildSimulator, DebuggableSimulator},
    };

    #[test]
    fn test_continue_until() {
        let circ = Circuit::new(3).h(0).h(1).h(2);
        let mut sim1 = DebugSimulator::build(circ).unwrap();
        let mut sim2 = sim1.clone();

        sim1.next();
        sim1.next();
        sim1.next();
        sim2.cont();
        assert!(equal_state_c(sim1.current_state(), sim2.current_state(), 3, 0.001))
    }
}
