mod basis;
mod scalar;
mod state;
#[cfg(test)]
mod tests;

use std::collections::HashMap;

use log::{debug, trace};
use rand::random;

use crate::{
    circuit::{Circuit, PureCircuit, pc::CircuitPc},
    instruction::PureInstruction,
    simulator::{BuildSimulator, RunnableSimulator},
    syntax_simulator::{
        scalar::Scalar,
        state::{ScaledState, SumOfScaledStates},
    },
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
                #[cfg(debug_assertions)]
                {
                    // In debug mode, we can afford to compute state between every step, so we do it
                    // at every step to check for consistency and correctness.
                    debug!(
                        "All scaled states: {:?}",
                        self.state.all_scaled_states().collect::<Vec<_>>()
                    );
                }
            }
            PureInstruction::Call(name, lsq, ctrl) => {
                self.pc.jump_and_link(name.clone(), lsq, ctrl);
            }
        }
        Some(&self.state)
    }

    fn step_all(&mut self) -> &SumOfScaledStates {
        while self.step().is_some() {}
        &self.state
    }

    /// Returns the all expanded states of the current state,
    /// where any state that contains an extended basis is expanded into the binary states it represents.
    /// All states containing an extended basis (eg, |+⟩, |−⟩, |i⟩, |−i⟩),
    /// will be expanded into the binary states they represent.
    ///
    /// **CAUTION**: There is no deduplication or simplification of the resulting states,
    /// so the same binary state may appear multiple times with different scalars,
    /// and these scalars should be summed to get the actual probability of that binary state.
    fn all_states_unsugared(&self) -> impl Iterator<Item = ScaledState> {
        debug!(
            "All states unsugared: {:?}",
            self.state
                .all_scaled_states()
                .flat_map(|s| s.all_inherent_states())
                .collect::<Vec<_>>()
        );

        self.state
            .all_scaled_states()
            .flat_map(|s| s.all_inherent_states())
    }

    fn probability_distribution(&self) -> HashMap<usize, Scalar> {
        let mut summed_scalars: HashMap<usize, Scalar> = HashMap::new();
        let non_zero_states = self.all_states_unsugared().filter(|s| !s.has_zero_coef());
        for ScaledState(basis, scalar) in non_zero_states {
            let basis = basis.into_binary(); // Unsugaring guarantees binary state here

            debug_assert!(
                basis < (1 << self.circuit.n_qubits()),
                "Basis {} is too large for {} qubits",
                basis,
                self.circuit.n_qubits()
            );

            if let Some(old_scalar) = summed_scalars.get(&basis) {
                // There already was a scalar for this basis
                let new_scalar = *old_scalar + scalar;
                if new_scalar.probability() == 0f32 {
                    summed_scalars.remove(&basis);
                } else {
                    summed_scalars.insert(basis, new_scalar);
                }
            } else {
                // This basis has no previous terms
                summed_scalars.insert(basis, scalar);
            }
        }

        debug!("Probability distribution: {:?}", summed_scalars);

        summed_scalars
    }

    pub fn run_and_measure_all(&mut self) -> usize {
        self.step_all();

        let mut probability_so_far = 0f32;
        let guess = random::<f32>();
        for (state, scalar) in self.probability_distribution() {
            probability_so_far += scalar.probability();
            if probability_so_far >= guess {
                trace!("Ran to completion! Measured state: {}", state);
                return state;
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

        Ok(Self {
            circuit,
            pc: CircuitPc::default(),
            state: SumOfScaledStates::guaranteed_full_zero(n_qubits),
        })
    }
}

impl RunnableSimulator for SyntaxSimulator {
    type Storage = SumOfScaledStates;
    type State = SumOfScaledStates;

    fn run(&self) -> usize {
        SyntaxSimulator::build(self.circuit.clone())
            .unwrap()
            .run_and_measure_all()
    }

    fn final_state(&self) -> Self::Storage {
        SyntaxSimulator::build(self.circuit.clone())
            .unwrap()
            .step_all()
            .clone()
    }
}
