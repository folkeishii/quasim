use cubecl::{cpu::CpuRuntime, wgpu::WgpuRuntime};
use quasim::{
    circuit::{Circuit, HybridCircuit},
    debug_simulator::DebugSimulator,
    gpu_sv_simulator::GpuStateVectorSimulator,
    simulator::{BuildSimulator, DebuggableSimulator, RunnableSimulator, StoredCircuitSimulator},
    sv_simulator::{SVSimulator, SVSimulatorDebugger},
};

extern crate quasim;

fn main() {
    divan::main();
}

#[divan::bench(
    types = [SVSimulatorDebugger, DebugSimulator],
    args = [2,3,4,5,6,7,8,9,10,11],
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
    sim.cont();
}

#[divan::bench(
    types = [SVSimulatorDebugger, DebugSimulator],
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
    sim.cont();
}

#[divan::bench(
    types = [SVSimulator, GpuStateVectorSimulator<WgpuRuntime>],
    args = [19,20,21,22,23,24],
    sample_count = 10,
)]
fn circuit_size_sample<S>(n_qubits: usize)
where
    S: RunnableSimulator + BuildSimulator<HybridCircuit>,
{
    let mut circuit = Circuit::new(n_qubits);

    for i in 0..n_qubits {
        circuit = circuit.h(i);
    }

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
