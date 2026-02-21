use nalgebra::{Complex, DVector};
use rand::distr::{Distribution, weighted::WeightedIndex};

use crate::{
    circuit::{self, Circuit},
    instruction::Instruction,
    simulator::{BuildSimulator, RunnableSimulator},
    sv_simulator::{SVExecutor, SVSimulator},
};

struct SimpleSimulator {
    state_vector: DVector<Complex<f32>>,
    dist: WeightedIndex<f32>,
}

impl TryFrom<Circuit> for SimpleSimulator {
    type Error = SimpleError;

    fn try_from(value: Circuit) -> Result<Self, Self::Error> {
        // Lets check so that circuit only contains quantum gates, otherwise we cant precompute state vector
        let only_gates = value
            .instructions()
            .iter()
            .all(|inst| matches!(inst, Instruction::Gate(_)));
        if !only_gates {
            return Err(SimpleError::UnsupportedInstruction);
        }

        match SVSimulator::build(value) {
            Ok(sv_sim) => {
                let state_vector = sv_sim.final_state();

                let probs = state_vector.iter().map(|&c| c.norm_sqr());
                let dist = WeightedIndex::new(probs).expect(
                    "Failed to create probability distribution. Invalid or empty state vector?",
                );

                Ok(Self {
                    state_vector: state_vector,
                    dist: dist,
                })
            }
            Err(_) => Err(SimpleError::StateVectorSimCreationFailed),
        }
    }
}

impl RunnableSimulator for SimpleSimulator {
    fn run(&self) -> usize {
        let mut rng = rand::rng();
        self.dist.sample(&mut rng)
    }

    fn final_state(&self) -> DVector<Complex<f32>> {
        self.state_vector.clone()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SimpleError {
    #[error("Failed to create a state vector simulator from circuit")]
    StateVectorSimCreationFailed,
    #[error("Unsupported instruction in circuit")]
    UnsupportedInstruction,
}
