use nalgebra::Complex;
use rand::distr::{Distribution, weighted::WeightedIndex};

use crate::{
    circuit::Circuit,
    simulator::{BuildSimulator, RunnableSimulator},
    state_vector::StateVector,
    sv_simulator::{SVError, SVSimulator},
};

struct SimpleSimulator {
    state_vector: StateVector,
    dist: WeightedIndex<f64>,
}

impl TryFrom<Circuit> for SimpleSimulator {
    type Error = SimpleError;

    fn try_from(value: Circuit) -> Result<Self, Self::Error> {
        // Lets check so that circuit only contains quantum gates, otherwise we cant precompute state vector

        let sv_sim = SVSimulator::build(value)?;
        let state_vector = sv_sim.final_state();
        let probs = state_vector.iter().map(|&c| c.norm_sqr());
        let dist = WeightedIndex::new(probs)
            .expect("Failed to create probability distribution. Invalid or empty state vector?");

        Ok(Self {
            state_vector: state_vector,
            dist: dist,
        })
    }
}

impl RunnableSimulator for SimpleSimulator {
    type Storage = StateVector;
    type State = Complex<f64>;

    fn run(&self) -> usize {
        let mut rng = rand::rng();
        self.dist.sample(&mut rng)
    }

    fn final_state(&self) -> StateVector {
        self.state_vector.clone()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SimpleError {
    #[error("{0}")]
    SVError(#[from] SVError),
    #[error("Unsupported instruction in circuit")]
    UnsupportedInstruction,
}
