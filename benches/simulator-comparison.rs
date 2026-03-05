use quasim::{circuit::Circuit, debug_simulator::DebugSimulator, simulator::{BuildSimulator, DebuggableSimulator}, sv_simulator::SVSimulatorDebugger};

extern crate quasim;

fn main() {
    divan::main();
}

#[divan::bench(
    types = [SVSimulatorDebugger, DebugSimulator],
    args = [2,3,4,5,6,7,8,9,10,11],
    sample_count = 10,
)]
fn circuit_size<S>(n_qubits: usize)
where 
    S: DebuggableSimulator + BuildSimulator
{
    let mut circuit = Circuit::new(n_qubits);

    for i in 0..n_qubits {
        circuit = circuit.hadamard(i);
    }

    let mut sim = S::build(circuit).expect("Couldnt build circuit...");
    sim.continue_until(None);
}


#[divan::bench(
    types = [SVSimulatorDebugger, DebugSimulator],
    args = [1000,2000,4000,8000,16000,32000],
    sample_count = 10,
)]
fn num_gates<S>(n_gates: usize)
where 
    S: DebuggableSimulator + BuildSimulator
{
    let mut circuit = Circuit::new(6);

    for i in 0..n_gates {
        circuit = circuit.hadamard(i % 6);
    }

    let mut sim = S::build(circuit).expect("Couldnt build circuit...");
    sim.continue_until(None);
}
