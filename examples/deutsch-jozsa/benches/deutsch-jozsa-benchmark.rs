extern crate quasim;

use deutsch_jozsa::{FunctionType, find_function_type_quantum};
use divan::AllocProfiler;
use quasim::circuit::HybridCircuit;
use quasim::debug_simulator::DebugSimulator;
use quasim::simulator::{
    BuildSimulator, DebuggableSimulator, HybridSimulator, StoredCircuitSimulator,
};
use quasim::sv_simulator::SVSimulatorDebugger;
use std::time::Duration;

#[global_allocator]
static ALLOCATOR: AllocProfiler = AllocProfiler::system();

fn main() {
    divan::Divan::from_args().main();
}

#[divan::bench(
    types = [SVSimulatorDebugger, DebugSimulator],
    args = [2,3,4,5,6,7,8,9,10,11],
    sample_count = 10,
    max_time = Duration::from_secs(1)
)]
fn deutsch_jozsa_constant0<S>(n_qubits: usize)
where
    S: DebuggableSimulator
        + BuildSimulator<HybridCircuit>
        + StoredCircuitSimulator
        + HybridSimulator,
{
    find_function_type_quantum::<S>(n_qubits, FunctionType::Constant0);
}

#[divan::bench(
    types = [SVSimulatorDebugger, DebugSimulator],
    args = [2,3,4,5,6,7,8,9,10,11],
    sample_count = 10,
    max_time = Duration::from_secs(1)
)]
fn deutsch_jozsa_constant1<S>(n_qubits: usize)
where
    S: DebuggableSimulator
        + BuildSimulator<HybridCircuit>
        + StoredCircuitSimulator
        + HybridSimulator,
{
    find_function_type_quantum::<S>(n_qubits, FunctionType::Constant1);
}

#[divan::bench(
    types = [SVSimulatorDebugger, DebugSimulator],
    args = [2,3,4,5,6,7,8,9,10,11],
    sample_count = 10,
    max_time = Duration::from_secs(1)
)]
fn deutsch_jozsa_balanced<S>(n_qubits: usize)
where
    S: DebuggableSimulator
        + BuildSimulator<HybridCircuit>
        + StoredCircuitSimulator
        + HybridSimulator,
{
    find_function_type_quantum::<S>(n_qubits, FunctionType::Balanced);
}
