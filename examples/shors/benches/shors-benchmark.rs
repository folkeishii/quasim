use std::time::Duration;

use shors::circuit;
use quasim::{circuit::HybridCircuit, sampler::CircuitSampler, simulator::{Buildable, Sampleable}, sv_simulator::StateVectorSimulator};

extern crate quasim;

fn main() {
    divan::Divan::from_args().main();
}

#[divan::bench(
    types = [StateVectorSimulator],
    args = [0b11, 0b1111, 0b1111_1111],
    sample_count = 10,
    max_time = Duration::from_secs(1)
)]
fn shors<S>(n: usize)
where
    S: Buildable<HybridCircuit> + Sampleable<HybridCircuit>
{
    let a = n-1;
    S::sample_once(circuit(n, a), CircuitSampler).unwrap();
}
