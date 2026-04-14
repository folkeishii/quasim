use cubecl::wgpu::WgpuRuntime;
use divan::AllocProfiler;
use nalgebra::{Complex, DVector};
use quasim::{
    circuit::{Circuit, HybridCircuit, PureCircuit},
    debug_simulator::DebugSimulator,
    gpu_sv_simulator::GpuStateVectorSimulator,
    pot_sim::{GenericSim, StateMaybe},
    sampler::CircuitSampler,
    simulator::Sampleable,
    sv_simulator::StateVectorSimulator,
};
use std::collections::BTreeMap;

extern crate quasim;
#[global_allocator]
static ALLOCATOR: AllocProfiler = AllocProfiler::system();

fn main() {
    divan::main();
}

#[divan::bench(
    types = [DebugSimulator, StateVectorSimulator, GpuStateVectorSimulator<WgpuRuntime>],
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
    types = [DebugSimulator, StateVectorSimulator, GpuStateVectorSimulator<WgpuRuntime>],
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
    types = [DebugSimulator, StateVectorSimulator, GpuStateVectorSimulator<WgpuRuntime>],
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
    types = [DebugSimulator, StateVectorSimulator, GpuStateVectorSimulator<WgpuRuntime>],
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
