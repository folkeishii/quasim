use divan::Bencher;
use shors::circuit;
extern crate quasim;
use quasim::{
    circuit::HybridCircuit,
    cube_simulator::CubeSimulator,
    fmm_simulator::FullMatMulSimulator,
    product_state_simulator::ProductStateSimulator,
    sampler::{CircuitSampler, Sampler},
    simulator::{Buildable, Sampleable},
    sv_simulator::StateVectorSimulator,
};
#[cfg(feature = "gpu")]
use quasim::{gpu_sv_simulator::GpuStateVectorSimulator};
#[cfg(feature = "wgpu")]
use quasim::{cubecl::wgpu::WgpuRuntime};
#[cfg(feature = "cuda")]
use quasim::{cubecl::cuda::CudaRuntime};
#[cfg(feature = "cpu")]
use quasim::{cubecl::cpu::CpuRuntime};
#[cfg(feature = "hip")]
use quasim::{cubecl::hip::HipRuntime};

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

const NS: &[usize] = &[0x3, 0x7, 0xF, 0x3F, 0x7F, 0xFF];
const FMM_NS: &[usize] = &[0x3, 0x7];
bench!(
    state_vector_simulator, StateVectorSimulator, NS;
    "wgpu" => gpu_accelerated_wgpu, GpuStateVectorSimulator<WgpuRuntime>, NS;
    "cuda" => gpu_accelerated_cuda, GpuStateVectorSimulator<CudaRuntime>, NS;
    "cpu" => gpu_accelerated_cpu, GpuStateVectorSimulator<CpuRuntime>, NS;
    "hip" => gpu_accelerated_hip, GpuStateVectorSimulator<HipRuntime>, NS;
    full_mat_mul_simulator, FullMatMulSimulator, FMM_NS;
    product_state_simulator, ProductStateSimulator, NS;
    cube_simulator, CubeSimulator, NS;
);

fn main() {
    divan::Divan::from_args().main();
}

fn build<S, const N: usize>() -> S
where
    S: Buildable<HybridCircuit>,
{
    let a = N-1;
    S::build(circuit(N, a)).unwrap()
}

fn bench<S>(bencher: Bencher, sim: &mut S)
where
    S: Sampleable<HybridCircuit>,
{
    bencher.bench_local(|| {
        divan::black_box(CircuitSampler.sample({
            sim.run();
            sim
        }));
    });
}
