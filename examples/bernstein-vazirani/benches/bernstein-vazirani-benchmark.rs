use bernstein_vazirani::circuit;
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
#[cfg(any(feature = "cpu", feature = "cuda", feature = "hip", feature = "wgpu"))]
use quasim::gpu_sv_simulator::GpuStateVectorSimulator;
use quasim::{
    circuit::PureCircuit,
    cube_simulator::CubeSimulator,
    fmm_simulator::FullMatMulSimulator,
    product_state_simulator::ProductStateSimulator,
    sampler::{CircuitSampler, Sampler},
    simulator::{Buildable, Sampleable},
    sv_simulator::StateVectorSimulator,
    syntax_simulator::SyntaxSimulator,
};

macro_rules! bench {
    ($($feat:literal =>)? $name:ident, $sim:ty, $qubits:expr $(;)?) => {
        $(#[cfg(feature = $feat)])?
        #[divan::bench(
            consts = $qubits
        )]
        fn $name<const N: usize>(bencher: Bencher) {
            let mut sim = build::<$sim, N>();
            bench(bencher, &mut sim);
        }
    };

    (
        $($ffeat:literal =>)? $first:ident, $fsim:ty, $fqubits:expr $(;
            $($feat:literal =>)? $name:ident, $sim:ty, $qubits:expr
        )+ $(;)?
    ) => {
        bench!($($ffeat =>)? $first, $fsim, $fqubits);
        bench!($($($feat =>)? $name, $sim, $qubits);+);
    };
}

const QUBITS: &[usize] = &[2, 4, 8, 10, 12, 16];
const FMM_QUBITS: &[usize] = &[2, 4, 8, 10];
bench!(
    state_vector_simulator, StateVectorSimulator, QUBITS;
    "wgpu" => gpu_accelerated_wgpu, GpuStateVectorSimulator<WgpuRuntime>, QUBITS;
    "cuda" => gpu_accelerated_cuda, GpuStateVectorSimulator<CudaRuntime>, QUBITS;
    "cpu" => gpu_accelerated_cpu, GpuStateVectorSimulator<CpuRuntime>, QUBITS;
    "hip" => gpu_accelerated_hip, GpuStateVectorSimulator<HipRuntime>, QUBITS;
    full_mat_mul_simulator, FullMatMulSimulator, FMM_QUBITS;
    product_state_simulator, ProductStateSimulator, QUBITS;
    cube_simulator, CubeSimulator, QUBITS;
    syntax_simulator, SyntaxSimulator, QUBITS;
);

fn main() {
    divan::Divan::from_args().main();
}

fn build<S, const N: usize>() -> S
where
    S: Buildable<PureCircuit>,
{
    let secret = !(usize::MAX << N);
    S::build(circuit(N, secret)).unwrap()
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
