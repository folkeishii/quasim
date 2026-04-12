mod basis;
mod scalar;
mod state;
#[cfg(test)]
mod tests;

use rand::random;

use crate::{
    circuit::{Circuit, PureCircuit, pc::CircuitPc},
    instruction::PureInstruction,
    syntax_simulator::state::{ScaledState, SumOfScaledStates},
};

pub struct SyntaxSimulator {
    circuit: Circuit<PureCircuit>,
    pc: CircuitPc,
    state: SumOfScaledStates,
}

#[derive(Debug, thiserror::Error)]
pub enum SyntaxSimError {
    #[error("Tried to build syntax simulator with non-pure circuit")]
    NonPureCircuit,
}

impl SyntaxSimulator {
    fn step(&mut self) -> Option<&SumOfScaledStates> {
        let Some(instruction) = self.circuit.instruction(&self.pc) else {
            return None;
        };
        match instruction {
            PureInstruction::Gate(gate) => {
                self.state.apply_gate(&gate);
                self.pc.increment();
            }
            PureInstruction::Call(_, _, _) => {
                todo!();
            }
        }
        Some(&self.state)
    }

    fn step_all(&mut self) -> &SumOfScaledStates {
        while self.step().is_some() {}
        &self.state
    }

    /// Returns the probability distribution over binary states,
    /// all states containing an extended basis (eg, |+⟩, |−⟩, |i⟩, |−i⟩),
    /// will be expanded into the binary states they represent.
    ///
    /// **CAUTION**: There is no deduplication or simplification of the resulting states,
    /// so the same binary state may appear multiple times with different scalars, and these scalars should be summed to get the actual probability of that binary state.
    fn unsugared_probability_distribution(&self) -> impl Iterator<Item = ScaledState> {
        self.state
            .probability_distribution()
            .flat_map(|s| s.all_inherent_states())
    }

    pub fn run(&mut self) -> usize {
        self.step_all();

        let mut probability_so_far = 0f32;
        let guess = random::<f32>();
        for ScaledState(state, scalar) in self.unsugared_probability_distribution() {
            probability_so_far += scalar.probability();
            if probability_so_far >= guess {
                return state.into_binary();
            }
        }

        panic!(
            "Probabilities did not sum to 1! Total probability: {}",
            probability_so_far
        );
    }
}

impl TryFrom<Circuit<PureCircuit>> for SyntaxSimulator {
    type Error = SyntaxSimError;

    fn try_from(circuit: Circuit<PureCircuit>) -> Result<Self, Self::Error> {
        let n_qubits = circuit.n_qubits();

        // Turn SWAP gates into 3 CX gates
        let circuit = circuit.swaps_to_cxs();

        Ok(Self {
            circuit,
            pc: CircuitPc::default(),
            state: SumOfScaledStates::guaranteed_full_zero(n_qubits),
        })
    }
}
