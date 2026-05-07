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
    circuit::HybridCircuit,
    cube_simulator::CubeSimulator,
    fmm_simulator::FullMatMulSimulator,
    product_state_simulator::ProductStateSimulator,
    sampler::{CircuitSampler, Sampler},
    simulator::{Buildable, Sampleable},
    sv_simulator::StateVectorSimulator, syntax_simulator::SyntaxSimulator,
};

#[rustfmt::skip]
bench_utils::bench!(state_vector_simulator, StateVectorSimulator, "bernstein-vazirani-benchmark", "state_vector_simulator");
#[rustfmt::skip]
bench_utils::bench!("wgpu" => gpu_accelerated_wgpu, GpuStateVectorSimulator<WgpuRuntime>, "bernstein-vazirani-benchmark", "gpu_accelerated_wgpu");
#[rustfmt::skip]
bench_utils::bench!("cuda" => gpu_accelerated_cuda, GpuStateVectorSimulator<CudaRuntime>, "bernstein-vazirani-benchmark", "gpu_accelerated_cuda");
#[rustfmt::skip]
bench_utils::bench!("cpu" => gpu_accelerated_cpu, GpuStateVectorSimulator<CpuRuntime>, "bernstein-vazirani-benchmark", "gpu_accelerated_cpu");
#[rustfmt::skip]
bench_utils::bench!("hip" => gpu_accelerated_hip, GpuStateVectorSimulator<HipRuntime>, "bernstein-vazirani-benchmark", "gpu_accelerated_hip");
#[rustfmt::skip]
bench_utils::bench!(full_mat_mul_simulator, FullMatMulSimulator, "bernstein-vazirani-benchmark", "full_mat_mul_simulator");
#[rustfmt::skip]
bench_utils::bench!(product_state_simulator, ProductStateSimulator, "bernstein-vazirani-benchmark", "product_state_simulator");
#[rustfmt::skip]
bench_utils::bench!(cube_simulator, CubeSimulator, "bernstein-vazirani-benchmark", "cube_simulator");
#[rustfmt::skip]
bench_utils::bench!(syntax_simulator, SyntaxSimulator, "bernstein-vazirani-benchmark", "syntax_simulator");

fn main() {
    divan::Divan::from_args().main();
}

fn build<S, const N: usize>() -> Option<S>
where
    S: Buildable<HybridCircuit>,
{
    let secret = !(usize::MAX << N);
    S::build(circuit(N, secret)).ok()
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
