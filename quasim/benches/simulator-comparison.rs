use quasim::{
    circuit::{Circuit, PureCircuit},
    debug_simulator::DebugSimulator,
    simulator::{BuildSimulator, RunnableOnceSimulator},
    sv_simulator::SVSimulator,
    syntax_simulator::SyntaxSimulator,
};

extern crate quasim;

fn main() {
    divan::main();
}

#[divan::bench(
    types = [SVSimulator, DebugSimulator, SyntaxSimulator],
    args = [2,10,20],
    sample_count = 10,
)]
fn uniform_distribution_n_qubits<S>(n_qubits: usize)
where
    S: BuildSimulator<PureCircuit> + RunnableOnceSimulator,
{
    let mut circuit = Circuit::new(n_qubits);

    for i in 0..n_qubits {
        circuit = circuit.h(i);
    }

    let mut sim = S::build(circuit.into()).expect("Couldnt build circuit...");
    sim.run_once();
}

#[divan::bench(
    types = [SVSimulator, DebugSimulator, SyntaxSimulator],
    args = [1000,2000,4000,8000,16000,32000],
    sample_count = 10,
)]
fn n_gates_6_qubits<S>(n_gates: usize)
where
    S: BuildSimulator<PureCircuit> + RunnableOnceSimulator,
{
    let mut circuit = Circuit::new(6);

    for i in 0..n_gates {
        circuit = circuit.h(i % 6);
    }

    let mut sim = S::build(circuit.into()).expect("Couldnt build circuit...");
    sim.run_once();
}

#[divan::bench(
    types = [SVSimulator, DebugSimulator, SyntaxSimulator],
    args = [2,10,20],
    sample_count = 10,
)]
fn hadamard_cnot_n_qubits<S>(n_qubits: usize)
where
    S: BuildSimulator<PureCircuit> + RunnableOnceSimulator,
{
    let mut circuit = Circuit::new(n_qubits);

    circuit = circuit.h(0);

    for i in 0..(n_qubits - 1) {
        circuit = circuit.cx(&[i], i + 1);
    }

    let mut sim = S::build(circuit.into()).expect("Couldnt build circuit...");
    let result = sim.run_once();
    if result != 0 {
        assert_eq!(result, usize::MAX >> (usize::BITS as usize - n_qubits));
    }
}
