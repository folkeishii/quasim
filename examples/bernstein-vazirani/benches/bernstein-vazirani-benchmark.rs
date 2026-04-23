use std::time::Duration;

use bernstein_vazirani::circuit;
use quasim::{
    circuit::HybridCircuit, fmm_simulator::FullMatMulSimulator, sampler::CircuitSampler, simulator::{Buildable, Sampleable}, sv_simulator::StateVectorSimulator
};

extern crate quasim;

fn main() {
    divan::Divan::from_args().main();
}

#[divan::bench(
    types = [StateVectorSimulator, FullMatMulSimulator],
    args = [2,3,4,5,6,7,8,9,10,11],
    sample_count = 10,
    max_time = Duration::from_secs(1)
)]
fn bernstein_vazirani_worst_case<S>(n_qubits: usize)
where
    S: Buildable<HybridCircuit> + Sampleable<HybridCircuit>,
{
    let secret = !(usize::MAX << n_qubits);
    S::sample_once(circuit(n_qubits, secret), CircuitSampler).unwrap();
}

#[divan::bench(
    types = [StateVectorSimulator, FullMatMulSimulator],
    args = [2,3,4,5,6,7,8,9,10,11],
    sample_count = 10,
    max_time = Duration::from_secs(1)
)]
fn bernstein_vazirani_rand_secret<S>(n_qubits: usize)
where
    S: Buildable<HybridCircuit> + Sampleable<HybridCircuit>,
{
    let secret = !(usize::MAX << (n_qubits / 2));
    S::sample_once(circuit(n_qubits, secret), CircuitSampler).unwrap();
}
