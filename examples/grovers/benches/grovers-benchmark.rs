use std::time::Duration;

use divan::AllocProfiler;
use grovers::check_quantum;
use quasim::{
    circuit::HybridCircuit,
    debug_simulator::DebugSimulator,
    simulator::{BuildSimulator, DebuggableSimulator, HybridSimulator, StoredCircuitSimulator},
    sv_simulator::SVSimulatorDebugger,
};

extern crate quasim;

#[global_allocator]
static ALLOCATOR: AllocProfiler = AllocProfiler::system();

fn main() {
    divan::Divan::from_args().main();
}

#[divan::bench(
    types = [SVSimulatorDebugger, DebugSimulator],
    args = [2,3,4,5,6],//,7,8,9,10,11],
    sample_count = 10,
    max_time = Duration::from_secs(1)
)]
fn grovers<S>(n_qubits: usize)
where
    S: DebuggableSimulator
        + BuildSimulator<HybridCircuit>
        + StoredCircuitSimulator
        + HybridSimulator,
{
    let mut func = vec![0; n_qubits];
    func[0] = 1;
    check_quantum::<S>(&func);
}
