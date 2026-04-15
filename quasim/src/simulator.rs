use crate::circuit::CircuitBehaviour;
use crate::circuit::pc::CircuitPc;
use crate::register_file::{Register, RegisterFile};
use crate::sampler::Sampler;
use crate::{circuit::Circuit, instruction::Instruction};

/// # Buildable
/// Any simulator that is able to be built from a
/// circuit should implement this trait.
///
/// To support `TryFrom<Cicuit>` there is an auto
/// implementation of `BuildSimulator` for any type
/// that implements `TryFrom<Circuit>`
pub trait Buildable<B: CircuitBehaviour>: Simulator + Sized {
    type E: std::error::Error;

    fn build(circuit: Circuit<B>) -> Result<Self, Self::E>;
}
impl<T, B: CircuitBehaviour, E> Buildable<B> for T
where
    T: Simulator + TryFrom<Circuit<B>, Error = E>,
    E: std::error::Error,
{
    type E = E;

    fn build(circuit: Circuit<B>) -> Result<Self, Self::E> {
        Self::try_from(circuit)
    }
}

pub trait QuantumState {
    type BasisValue;

    fn collapse(&self) -> usize;
    fn basis_value(&self, basis: usize) -> Self::BasisValue;
}

/// # Simulator
/// The base simulator trait
///
/// Mutates internal state using `run` and `reset` functions
pub trait Simulator {
    type State: QuantumState<BasisValue = Self::BasisValue>;
    type BasisValue;

    fn run(&mut self);
    fn reset(&mut self);
    fn state(&self) -> &Self::State;
}

/// # Debuggable
/// Any simulator that can step through a circuit
/// one gate at a time should implement this trait
pub trait Debuggable: Simulator {
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
        false
    }
    fn double_ended(&self) -> bool;
    /// Returns current pc and instruction
    ///
    /// If returned value is (pc, None)
    /// then we have reached the end of (sub) circuit
    fn current_instruction(&self) -> (&CircuitPc, Option<Instruction>);

    fn cont(&mut self) -> bool
    where
        Self: StoredCircuit,
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

/// # StoredCircuit
/// Any simulator that stores the underlying circuit
/// internally should implement this trait
pub trait StoredCircuit: Simulator {
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

/// # StoredRegisters
/// Any simulator that implements classical operations
/// and stores registers should implement this trait
pub trait StoredRegisters: Simulator {
    fn registers(&self) -> &RegisterFile;

    fn register(&self, register: &str) -> Register {
        self.registers()[register]
    }
}

/// # Sampleable
/// Simulators that implement this trait can be sampled using a Sampler.
/// Is autoimplemented by all simulators, but requires the simulator to be buildable
/// in order to use the sampling functions.
pub trait Sampleable<B>: Simulator + Buildable<B>
where
    B: CircuitBehaviour,
{
    fn sample_once<S: Sampler<Self>>(
        circuit: Circuit<B>,
        sampler: S,
    ) -> Result<S::Output, Self::E> {
        let mut sim = Self::build(circuit)?;
        sim.run();
        Ok(sampler.sample(&sim))
    }

    fn sample<S: Sampler<Self>>(
        circuit: Circuit<B>,
        sampler: S,
        times: usize,
    ) -> Result<impl Iterator<Item = <S as Sampler<Self>>::Output>, Self::E> {
        let mut sim = Self::build(circuit)?;
        let iter = (0..times).map(move |_| {
            sim.run();
            sampler.sample(&sim)
        });
        Ok(iter)
    }
}

#[cfg(test)]
mod tests {

    use crate::{
        circuit::Circuit,
        debug_simulator::DebugSimulator,
        ext::equal_state_c,
        simulator::{Buildable, Debuggable, Simulator},
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
        assert!(equal_state_c(sim1.state(), sim2.state(), 3, 0.001))
    }
}
