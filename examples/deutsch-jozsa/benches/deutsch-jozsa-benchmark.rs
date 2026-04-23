use std::time::Duration;

use deutsch_jozsa::{FunctionType, circuit};
use quasim::{circuit::HybridCircuit, fmm_simulator::FullMatMulSimulator, sampler::CircuitSampler, simulator::{Buildable, Sampleable, StoredRegisters}, sv_simulator::StateVectorSimulator};

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
fn deutsch_jozsa_constant0<S>(n_qubits: usize)
where
    S: Buildable<HybridCircuit> + Sampleable<HybridCircuit>
{
    S::sample_once(circuit(n_qubits, FunctionType::Constant0), CircuitSampler).unwrap();
}

#[divan::bench(
    types = [StateVectorSimulator, FullMatMulSimulator],
    args = [2,3,4,5,6,7,8,9,10,11],
    sample_count = 10,
    max_time = Duration::from_secs(1)
)]
fn deutsch_jozsa_constant1<S>(n_qubits: usize)
where
    S: Buildable<HybridCircuit> + Sampleable<HybridCircuit> + StoredRegisters
{
    S::sample_once(circuit(n_qubits, FunctionType::Constant1), CircuitSampler).unwrap();
}

#[divan::bench(
    types = [StateVectorSimulator, FullMatMulSimulator],
    args = [2,3,4,5,6,7,8,9,10,11],
    sample_count = 10,
    max_time = Duration::from_secs(1)
)]
fn deutsch_jozsa_balanced<S>(n_qubits: usize)
where
    S: Buildable<HybridCircuit> + Sampleable<HybridCircuit>
{
    S::sample_once(circuit(n_qubits, FunctionType::Balanced), CircuitSampler).unwrap();
}
