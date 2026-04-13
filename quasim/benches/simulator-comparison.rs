use divan::AllocProfiler;
use nalgebra::{Complex, DVector};
use quasim::{
    circuit::{Circuit, HybridCircuit},
    debug_simulator::DebugSimulator,
    pot_sim::{GenericSim, StateMaybe},
    simulator::{BuildSimulator, DebuggableSimulator, StoredCircuitSimulator},
    sv_simulator::SVSimulatorDebugger,
};
use std::collections::BTreeMap;

extern crate quasim;
#[global_allocator]
static ALLOCATOR: AllocProfiler = AllocProfiler::system();

fn main() {
    divan::main();
}

#[divan::bench(
    types = [SVSimulatorDebugger, DebugSimulator, GenericSim<DVector<Complex<f64>>>, GenericSim<StateMaybe<BTreeMap<usize, Complex<f64>>>>],
    args = [2,3,4,5,6,7,8,9,10,11],
    sample_count = 10,
)]
fn circuit_size<S>(n_qubits: usize)
where
    S: DebuggableSimulator + BuildSimulator<HybridCircuit> + StoredCircuitSimulator,
{
    let mut circuit = Circuit::new(n_qubits);

    for i in 0..n_qubits {
        circuit = circuit.h(i);
    }

    let mut sim = S::build(circuit.into()).expect("Couldnt build circuit...");
    sim.cont();
}

#[divan::bench(
    types = [SVSimulatorDebugger, DebugSimulator, GenericSim<DVector<Complex<f64>>>, GenericSim<StateMaybe<BTreeMap<usize, Complex<f64>>>>],
    args = [1000,2000,4000,8000,16000,32000],
    sample_count = 10,
)]
fn num_gates<S>(n_gates: usize)
where
    S: DebuggableSimulator + BuildSimulator<HybridCircuit> + StoredCircuitSimulator,
{
    let mut circuit = Circuit::new(6);

    for i in 0..n_gates {
        circuit = circuit.h(i % 6);
    }

    let mut sim = S::build(circuit.into()).expect("Couldnt build circuit...");
    sim.cont();
}
