use cubecl::wgpu::WgpuRuntime;
use quasim::{
    circuit::{Circuit, HybridCircuit, PureCircuit},
    debug_simulator::DebugSimulator,
    gpu_sv_simulator::GpuStateVectorSimulator,
    sampler::CircuitSampler,
    simulator::Sampleable,
    sv_simulator::StateVectorSimulator,
    product_state_simulator::ProductStateSimulator,
};

extern crate quasim;

const QUBITS_UPPER_BOUND: usize = 22;

fn main() {
    divan::main();
}

#[divan::bench(
    types = [DebugSimulator, StateVectorSimulator, GpuStateVectorSimulator<WgpuRuntime>, ProductStateSimulator],
    args = [15,16,17,18,19,20,21,22],
    sample_count = 10,
)]
fn circuit_size<Sim>(n_qubits: usize)
where
    Sim: Sampleable<PureCircuit>,
{
    let circuit = Circuit::<PureCircuit>::new_qft(n_qubits);

    Sim::sample_once(circuit, CircuitSampler).expect("couldn't build circuit");
}

#[divan::bench(
    types = [DebugSimulator, StateVectorSimulator, GpuStateVectorSimulator<WgpuRuntime>, ProductStateSimulator],
    args = [250, 500, 1000, 2000],
    sample_count = 10,
)]
fn num_gates<Sim>(n_gates: usize)
where
    Sim: Sampleable<PureCircuit>,
{
    let mut circuit = Circuit::new(18);

    for i in 0..n_gates {
        circuit = circuit.h(i % 18);
    }

    Sim::sample_once(circuit, CircuitSampler).expect("couldn't build circuit");
}

// Measurement benchmark

#[divan::bench(
    types = [DebugSimulator, StateVectorSimulator, GpuStateVectorSimulator<WgpuRuntime>, ProductStateSimulator],
    args = [15,16,17,18,19,20,21,22],
    sample_count = 10,
)]
fn circuit_size_measurement<Sim>(n_qubits: usize)
where
    Sim: Sampleable<HybridCircuit>,
{
    let circuit = Circuit::new(n_qubits)
        .new_reg("r0", 1)
        .measure_bit(0, ("r0", 0));

    Sim::sample_once(circuit, CircuitSampler).expect("couldn't build circuit");
}

#[divan::bench(
    types = [DebugSimulator, StateVectorSimulator, GpuStateVectorSimulator<WgpuRuntime>, ProductStateSimulator],
    args = [40, 80, 160, 320],
    sample_count = 20,
)]
fn num_measurements<Sim>(n_measure: usize)
where
    Sim: Sampleable<HybridCircuit>,
{
    let mut circuit = Circuit::new(18).new_reg("r0", 1);

    for _ in 0..n_measure {
        circuit = circuit.measure_bit(0, ("r0", 0));
    }

    Sim::sample_once(circuit, CircuitSampler).expect("couldn't build circuit");
}
#[divan::bench(
    types = [DebugSimulator, StateVectorSimulator, GpuStateVectorSimulator<WgpuRuntime>, ProductStateSimulator],
    args = [15,16,17,18,19,20,21,22],
    sample_count = 10,
)]
fn circuit_size_full_entagnlement<Sim>(n_qubits: usize)
where
    Sim: Sampleable<PureCircuit>,
{
    let mut circuit = Circuit::new(n_qubits);

    circuit = circuit.h(0);

    for i in 1..n_qubits {
        circuit = circuit.cx(&[0], i);
    }

    Sim::sample_once(circuit, CircuitSampler).expect("couldn't build circuit");
}
#[divan::bench(
    types = [DebugSimulator, StateVectorSimulator, GpuStateVectorSimulator<WgpuRuntime>, ProductStateSimulator],
    args = [1, 2, 4, 8, 16, 20, 22],
    sample_count = 10,
)]
fn entanglement_size<Sim>(entangle_size: usize)
where
    Sim: Sampleable<PureCircuit>,
{
    let n_systems = QUBITS_UPPER_BOUND / entangle_size;

    let mut circuit = Circuit::new(QUBITS_UPPER_BOUND);

    for q in 0..QUBITS_UPPER_BOUND {
        circuit = circuit.h(q);
    }

    for s in 0..n_systems {
        circuit = circuit.cx(
            &Vec::from_iter((s * entangle_size..(s + 1) * entangle_size).skip(1)),
            s,
        );
    }

    Sim::sample_once(circuit, CircuitSampler).expect("couldn't build circuit");
}

#[divan::bench(
    types = [DebugSimulator, StateVectorSimulator, GpuStateVectorSimulator<WgpuRuntime>, ProductStateSimulator],
    args = [1, 2, 4, 8, 16, 20, 22],
    sample_count = 10,
)]
fn mid_measure_bits<Sim>(n_measurements: usize)
where
    Sim: Sampleable<HybridCircuit>,
{
    let mut circuit = Circuit::new(QUBITS_UPPER_BOUND).new_reg("dummy", n_measurements);

    for q in 0..QUBITS_UPPER_BOUND {
        circuit = circuit.h(q).cx(&[q], (q + 1) % QUBITS_UPPER_BOUND);
    }

    for m in 0..n_measurements {
        circuit = circuit.measure_bit(m, ("dummy", m));
    }

    for q in 0..QUBITS_UPPER_BOUND {
        circuit = circuit.h(q).cx(&[q], (q + 1) % QUBITS_UPPER_BOUND);
    }

    Sim::sample_once(circuit.into(), CircuitSampler).expect("couldn't build circuit");
}

#[divan::bench(
    types = [DebugSimulator, StateVectorSimulator, GpuStateVectorSimulator<WgpuRuntime>, ProductStateSimulator],
    args = [15,16,17,18,19,20,21,22],
    sample_count = 10,
)]
fn mid_measure_all<Sim>(n_qubits: usize)
where
    Sim: Sampleable<HybridCircuit>,
{
    let mut circuit = Circuit::new(n_qubits).new_reg("dummy", n_qubits);

    for q in 0..n_qubits {
        circuit = circuit.h(q).cx(&[q], (q + 1) % n_qubits);
    }

    circuit = circuit.measure("dummy");

    for q in 0..n_qubits {
        circuit = circuit.h(q).cx(&[q], (q + 1) % n_qubits);
    }

    Sim::sample_once(circuit.into(), CircuitSampler).expect("couldn't build circuit");
}
