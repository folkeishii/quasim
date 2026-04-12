use cubecl::wgpu::WgpuRuntime;
use quasim::{
    circuit::{Circuit, HybridCircuit, PureCircuit},
    debug_simulator::DebugSimulator,
    gpu_sv_simulator::GpuStateVectorSimulator,
    prod_simulator::ProdSimulator,
    simulator::{BuildSimulator, DebuggableSimulator, RunnableSimulator, StoredCircuitSimulator},
    sv_simulator::{SVSimulator, SVSimulatorDebugger},
};

extern crate quasim;

fn main() {
    divan::main();
}

#[divan::bench(
    types = [SVSimulatorDebugger, ProdSimulator, DebugSimulator],
    args = [2,3,4,5,6,7,8,9,10,11,12],
    sample_count = 10,
)]
fn circuit_size_debug<S>(n_qubits: usize)
where
    S: DebuggableSimulator + BuildSimulator<HybridCircuit> + StoredCircuitSimulator,
{
    let mut circuit = Circuit::new(n_qubits);

    for i in 0..n_qubits {
        circuit = circuit.h(i);
    }

    let mut sim = S::build(circuit.into()).expect("Couldnt build circuit...");
    while sim.next() {}
}
#[divan::bench(
    types = [SVSimulatorDebugger, ProdSimulator, DebugSimulator],
    args = [1000,2000,4000,8000,16000,32000],
    sample_count = 10,
)]
fn num_gates_debug<S>(n_gates: usize)
where
    S: DebuggableSimulator + BuildSimulator<HybridCircuit> + StoredCircuitSimulator,
{
    let mut circuit = Circuit::new(6);

    for i in 0..n_gates {
        circuit = circuit.h(i % 6);
    }

    let mut sim = S::build(circuit.into()).expect("Couldnt build circuit...");
    while sim.next() {}
}
#[divan::bench(
    types = [SVSimulatorDebugger, ProdSimulator, DebugSimulator],
    args = [2,3,4,5,6,7,8],
    sample_count = 10,
)]
fn circuit_size_full_entagnlement<S>(n_qubits: usize)
where
    S: DebuggableSimulator + BuildSimulator<HybridCircuit> + StoredCircuitSimulator,
{
    let mut circuit = Circuit::new(n_qubits);

    circuit = circuit.h(0);

    for i in 1..n_qubits {
        circuit = circuit.cx(&[0], i);
    }

    let mut sim = S::build(circuit.into()).expect("Couldnt build circuit...");
    while sim.next() {}
}
#[divan::bench(
    types = [SVSimulatorDebugger, ProdSimulator, DebugSimulator],
    args = [
        (6,2), (6,3), (6,4), (6,5),
        (7,2), (7,3), (7,4), (7,5), (7,6),
        (8,2), (8,3), (8,4), (8,5), (8,6), (8,7),
    ],
    sample_count = 10,
)]
fn entanglement_size<S>(inp: (usize, usize))
where
    S: DebuggableSimulator + BuildSimulator<HybridCircuit> + StoredCircuitSimulator,
{
    let (n_qubits, entangle_size) = inp;

    let n_systems = n_qubits / entangle_size;

    let mut circuit = Circuit::new(n_qubits);

    for s in 0..n_systems {
        circuit = circuit.ch(
            &Vec::from_iter((s * entangle_size..(s + 1) * entangle_size).skip(1)),
            s,
        );
    }

    let mut sim = S::build(circuit.into()).expect("Couldnt build circuit...");
    while sim.next() {}
}

#[divan::bench(
    types = [SVSimulator, GpuStateVectorSimulator<WgpuRuntime>],
    args = [15,16,17,18,19,20,21,22],
    sample_count = 10,
)]
fn circuit_size_sample<S>(n_qubits: usize)
where
    S: RunnableSimulator + BuildSimulator<HybridCircuit>,
{
    let circuit = Circuit::<PureCircuit>::new_qft(n_qubits);

    let sim = S::build(circuit.into()).expect("Couldnt build circuit...");
    sim.run();
}

#[divan::bench(
    types = [SVSimulator, GpuStateVectorSimulator<WgpuRuntime>],
    args = [250, 500, 1000, 2000],
    sample_count = 10,
)]
fn num_gates_sample<S>(n_gates: usize)
where
    S: RunnableSimulator + BuildSimulator<HybridCircuit>,
{
    let mut circuit = Circuit::new(18);

    for i in 0..n_gates {
        circuit = circuit.h(i % 18);
    }

    let sim = S::build(circuit.into()).expect("Couldnt build circuit...");
    sim.run();
}

// Measurement benchmark

#[divan::bench(
    types = [SVSimulator, GpuStateVectorSimulator<WgpuRuntime>],
    args = [15,16,17,18,19,20,21,22],
    sample_count = 10,
)]
fn circuit_size_measurement<S>(n_qubits: usize)
where
    S: RunnableSimulator + BuildSimulator<HybridCircuit>,
{
    let circuit = Circuit::new(n_qubits)
        .new_reg("r0", 1)
        .measure_bit(0, ("r0", 0));

    let sim = S::build(circuit).expect("Couldnt build circuit...");
    sim.run();
}

#[divan::bench(
    types = [SVSimulator, GpuStateVectorSimulator<WgpuRuntime>],
    args = [40, 80, 160, 320],
    sample_count = 20,
)]
fn circuit_num_measurements<S>(n_measure: usize)
where
    S: RunnableSimulator + BuildSimulator<HybridCircuit>,
{
    let mut circuit = Circuit::new(18).new_reg("r0", 1);

    for _ in 0..n_measure {
        circuit = circuit.measure_bit(0, ("r0", 0));
    }

    let sim = S::build(circuit).expect("Couldnt build circuit...");
    sim.run();
}
