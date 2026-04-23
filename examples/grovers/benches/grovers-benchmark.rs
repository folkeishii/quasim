use std::time::Duration;

use grovers::circuit;
use quasim::{
    circuit::HybridCircuit,
    fmm_simulator::FullMatMulSimulator,
    sampler::CircuitSampler,
    simulator::{Buildable, Sampleable},
    sv_simulator::StateVectorSimulator,
};

extern crate quasim;

fn main() {
    divan::Divan::from_args().main();
}

#[divan::bench(
    types = [StateVectorSimulator, FullMatMulSimulator],
    args = [2,3,4,5,6],//,7,8,9,10,11],
    sample_count = 10,
    max_time = Duration::from_secs(1)
)]
fn grovers<S>(n_qubits: usize)
where
    S: Buildable<HybridCircuit> + Sampleable<HybridCircuit>,
{
    let mut func = vec![0; n_qubits];
    func[0] = 1;
    S::sample_once(circuit(&func), CircuitSampler).unwrap();
}
