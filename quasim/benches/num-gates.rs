use divan::Bencher;
extern crate quasim;
#[cfg(feature = "cpu")]
use quasim::cubecl::cpu::CpuRuntime;
#[cfg(feature = "cuda")]
use quasim::cubecl::cuda::CudaRuntime;
#[cfg(feature = "hip")]
use quasim::cubecl::hip::HipRuntime;
#[cfg(feature = "wgpu")]
use quasim::cubecl::wgpu::WgpuRuntime;
#[cfg(feature = "gpu")]
use quasim::gpu_sv_simulator::GpuStateVectorSimulator;
use quasim::{
    circuit::{Circuit, PureCircuit},
    cube_simulator::CubeSimulator,
    fmm_simulator::FullMatMulSimulator,
    product_state_simulator::ProductStateSimulator,
    sampler::{CircuitSampler, Sampler},
    simulator::{Buildable, Sampleable},
    sv_simulator::StateVectorSimulator,
};

macro_rules! bench {
    ($($feat:literal =>)? $name:ident, $sim:ty, $qubits:expr, $arg0:expr $(;)?) => {
        $(#[cfg(feature = $feat)])?
        #[divan::bench(
            args = $arg0,
            consts = $qubits
        )]
        fn $name<const N: usize>(bencher: Bencher, num_gates: usize) {
            let mut sim = build::<$sim, N>(num_gates);
            bench(bencher, &mut sim);
        }
    };

    (
        $($ffeat:literal =>)? $first:ident, $fsim:ty, $fqubits:expr, $farg0:expr  $(;
            $($feat:literal =>)? $name:ident, $sim:ty, $qubits:expr, $arg0:expr
        )+ $(;)?
    ) => {
        bench!($($ffeat =>)? $first, $fsim, $fqubits, $farg0);
        bench!($($($feat =>)? $name, $sim, $qubits, $arg0);+);
    };
}

const QUBITS: &[usize] = &[12, 18];
const FMM_QUBITS: &[usize] = &[12];
const NUM: &[usize] = &[250, 500, 1000, 2000];
bench!(
    state_vector_simulator, StateVectorSimulator, QUBITS, NUM;
    "wgpu" => gpu_accelerated_wgpu, GpuStateVectorSimulator<WgpuRuntime>, QUBITS, NUM;
    "cuda" => gpu_accelerated_cuda, GpuStateVectorSimulator<CudaRuntime>, QUBITS, NUM;
    "cpu" => gpu_accelerated_cpu, GpuStateVectorSimulator<CpuRuntime>, QUBITS, NUM;
    "hip" => gpu_accelerated_hip, GpuStateVectorSimulator<HipRuntime>, QUBITS, NUM;
    full_mat_mul_simulator, FullMatMulSimulator, FMM_QUBITS, NUM;
    product_state_simulator, ProductStateSimulator, QUBITS, NUM;
    cube_simulator, CubeSimulator, QUBITS, NUM;
);

fn main() {
    divan::Divan::from_args().main();
}

fn build<S, const N: usize>(num_gates: usize) -> S
where
    S: Buildable<PureCircuit>,
{
    S::build({
        let mut circuit = Circuit::new(N);

        for i in 0..num_gates {
            circuit = circuit.h(i % N);
        }

        circuit
    })
    .unwrap()
}

fn bench<S>(bencher: Bencher, sim: &mut S)
where
    S: Sampleable<PureCircuit>,
{
    bencher.bench_local(|| {
        divan::black_box(CircuitSampler.sample({
            sim.run();
            sim
        }));
    });
}
