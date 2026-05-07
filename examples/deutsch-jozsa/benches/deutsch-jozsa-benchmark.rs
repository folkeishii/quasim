use deutsch_jozsa::FunctionType;
use deutsch_jozsa::circuit;
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
use quasim::syntax_simulator::SyntaxSimulator;
use quasim::{
    circuit::HybridCircuit,
    cube_simulator::CubeSimulator,
    fmm_simulator::FullMatMulSimulator,
    product_state_simulator::ProductStateSimulator,
    sampler::{RegisterSampler, Sampler},
    simulator::{Buildable, Sampleable, StoredRegisters},
    sv_simulator::StateVectorSimulator,
};

#[rustfmt::skip]
bench_utils::bench!(state_vector_simulator, StateVectorSimulator, "deutsch-jozsa-benchmark", "state_vector_simulator", use_args);
#[rustfmt::skip]
bench_utils::bench!("wgpu" => gpu_accelerated_wgpu, GpuStateVectorSimulator<WgpuRuntime>, "deutsch-jozsa-benchmark", "gpu_accelerated_wgpu", use_args);
#[rustfmt::skip]
bench_utils::bench!("cuda" => gpu_accelerated_cuda, GpuStateVectorSimulator<CudaRuntime>, "deutsch-jozsa-benchmark", "gpu_accelerated_cuda", use_args);
#[rustfmt::skip]
bench_utils::bench!("cpu" => gpu_accelerated_cpu, GpuStateVectorSimulator<CpuRuntime>, "deutsch-jozsa-benchmark", "gpu_accelerated_cpu", use_args);
#[rustfmt::skip]
bench_utils::bench!("hip" => gpu_accelerated_hip, GpuStateVectorSimulator<HipRuntime>, "deutsch-jozsa-benchmark", "gpu_accelerated_hip", use_args);
#[rustfmt::skip]
bench_utils::bench!(full_mat_mul_simulator, FullMatMulSimulator, "deutsch-jozsa-benchmark", "full_mat_mul_simulator", use_args);
#[rustfmt::skip]
bench_utils::bench!(product_state_simulator, ProductStateSimulator, "deutsch-jozsa-benchmark", "product_state_simulator", use_args);
#[rustfmt::skip]
bench_utils::bench!(cube_simulator, CubeSimulator, "deutsch-jozsa-benchmark", "cube_simulator", use_args);
#[rustfmt::skip]
bench_utils::bench!(syntax_simulator, SyntaxSimulator, "deutsch-jozsa-benchmark", "syntax_simulator", use_args);

fn main() {
    divan::Divan::from_args().main();
}

fn build<S, const N: usize>(ft: FunctionType) -> Option<S>
where
    S: Buildable<HybridCircuit>,
{
    S::build(circuit(N, ft)).ok()
}

fn bench<S>(bencher: Bencher, sim: &mut S) -> bool
where
    S: Sampleable<HybridCircuit> + StoredRegisters,
{
    let mut ret = false;
    bencher.bench_local(|| {
        ret = divan::black_box(RegisterSampler::new("res").sample({
            sim.run();
            sim
        })) == 0;
    });
    ret
}
