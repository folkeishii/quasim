mod basis;
mod scalar;
mod state;
#[cfg(test)]
mod tests;

use std::collections::HashMap;

use log::debug;

use crate::{
    circuit::{Circuit, PureCircuit, pc::CircuitPc},
    instruction::PureInstruction,
    simulator::{Sampleable, Simulator},
    syntax_simulator::{
        scalar::Scalar,
        state::{ScaledState, SumOfScaledStates},
    },
};

pub struct SyntaxSimulator {
    circuit: Circuit<PureCircuit>,
    pc: CircuitPc,
    sum: SumOfScaledStates,
}

impl Sampleable<PureCircuit> for SyntaxSimulator {}

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
                self.sum.apply_gate(&gate);
                self.pc.increment();
                #[cfg(debug_assertions)]
                {
                    // In debug mode, we can afford to compute state between every step, so we do it
                    // at every step to check for consistency and correctness.
                    debug!(
                        "All scaled states: {:?}",
                        self.sum.all_scaled_states().collect::<Vec<_>>()
                    );
                }
            }
            PureInstruction::Call(name, lsq, ctrl) => {
                self.pc.jump_and_link(name.clone(), lsq, ctrl);
            }
        }
        Some(&self.sum)
    }

    fn step_all(&mut self) -> &SumOfScaledStates {
        while self.step().is_some() {}
        &self.sum
    }
}

impl TryFrom<Circuit<PureCircuit>> for SyntaxSimulator {
    type Error = SyntaxSimError;

    fn try_from(circuit: Circuit<PureCircuit>) -> Result<Self, Self::Error> {
        let n_qubits = circuit.n_qubits();

        Ok(Self {
            circuit,
            pc: CircuitPc::default(),
            sum: SumOfScaledStates::guaranteed_full_zero(n_qubits),
        })
    }
}

impl Simulator for SyntaxSimulator {
    type BasisValue = Scalar;
    type State = HashMap<usize, Scalar>;

    fn run(&mut self) {
        self.reset();
        self.step_all();
    }

    fn reset(&mut self) {
        self.pc = CircuitPc::default();
        // Old probability cache gets destroyed when old sum gets destroyed
        self.sum = SumOfScaledStates::guaranteed_full_zero(self.circuit.n_qubits());
    }

    fn state(&self) -> &Self::State {
        &self.sum.calculate_probability_distribution()
    }
}
