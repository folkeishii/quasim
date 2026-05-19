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
    circuit::{Circuit, HybridCircuit},
    cube_simulator::CubeSimulator,
    fmm_simulator::FullMatMulSimulator,
    product_state_simulator::ProductStateSimulator,
    sampler::{CircuitSampler, Sampler},
    simulator::{Buildable, Sampleable},
    sv_simulator::StateVectorSimulator,
    syntax_simulator::SyntaxSimulator,
};

#[rustfmt::skip]
bench_utils::bench!(state_vector_simulator, StateVectorSimulator, "mid-measure-all", "state_vector_simulator");
#[rustfmt::skip]
bench_utils::bench!("wgpu" => gpu_accelerated_wgpu, GpuStateVectorSimulator<WgpuRuntime>, "mid-measure-all", "gpu_accelerated_wgpu");
#[rustfmt::skip]
bench_utils::bench!("cuda" => gpu_accelerated_cuda, GpuStateVectorSimulator<CudaRuntime>, "mid-measure-all", "gpu_accelerated_cuda");
#[rustfmt::skip]
bench_utils::bench!("cpu" => gpu_accelerated_cpu, GpuStateVectorSimulator<CpuRuntime>, "mid-measure-all", "gpu_accelerated_cpu");
#[rustfmt::skip]
bench_utils::bench!("hip" => gpu_accelerated_hip, GpuStateVectorSimulator<HipRuntime>, "mid-measure-all", "gpu_accelerated_hip");
#[rustfmt::skip]
bench_utils::bench!(full_mat_mul_simulator, FullMatMulSimulator, "mid-measure-all", "full_mat_mul_simulator");
#[rustfmt::skip]
bench_utils::bench!(product_state_simulator, ProductStateSimulator, "mid-measure-all", "product_state_simulator");
#[rustfmt::skip]
bench_utils::bench!(cube_simulator, CubeSimulator, "mid-measure-all", "cube_simulator");
#[rustfmt::skip]
bench_utils::bench!(syntax_simulator, SyntaxSimulator, "mid-measure-all", "syntax_simulator");

fn main() {
    divan::Divan::from_args().main();
}

fn build<S, const N: usize>() -> Option<S>
where
    S: Buildable<HybridCircuit>,
{
    S::build(
        Circuit::new(N)
            .new_reg("res", N)
            .call_new("qft", Circuit::new_qft(N), 0)
            .measure("res")
            .call("qft", 0),
    )
    .ok()
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
