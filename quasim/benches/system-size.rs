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
    circuit::{Circuit, PureCircuit},
    cube_simulator::CubeSimulator,
    fmm_simulator::FullMatMulSimulator,
    product_state_simulator::ProductStateSimulator,
    sampler::{CircuitSampler, Sampler},
    simulator::{Buildable, Sampleable},
    sv_simulator::StateVectorSimulator,
    syntax_simulator::SyntaxSimulator,
};

#[rustfmt::skip]
bench_utils::bench!(state_vector_simulator, StateVectorSimulator, "system-size", "state_vector_simulator");
#[rustfmt::skip]
bench_utils::bench!("wgpu" => gpu_accelerated_wgpu, GpuStateVectorSimulator<WgpuRuntime>, "system-size", "gpu_accelerated_wgpu");
#[rustfmt::skip]
bench_utils::bench!("cuda" => gpu_accelerated_cuda, GpuStateVectorSimulator<CudaRuntime>, "system-size", "gpu_accelerated_cuda");
#[rustfmt::skip]
bench_utils::bench!("cpu" => gpu_accelerated_cpu, GpuStateVectorSimulator<CpuRuntime>, "system-size", "gpu_accelerated_cpu");
#[rustfmt::skip]
bench_utils::bench!("hip" => gpu_accelerated_hip, GpuStateVectorSimulator<HipRuntime>, "system-size", "gpu_accelerated_hip");
#[rustfmt::skip]
bench_utils::bench!(full_mat_mul_simulator, FullMatMulSimulator, "system-size", "full_mat_mul_simulator");
#[rustfmt::skip]
bench_utils::bench!(product_state_simulator, ProductStateSimulator, "system-size", "product_state_simulator");
#[rustfmt::skip]
bench_utils::bench!(cube_simulator, CubeSimulator, "system-size", "cube_simulator");
#[rustfmt::skip]
bench_utils::bench!(syntax_simulator, SyntaxSimulator, "system-size", "syntax_simulator");

fn main() {
    divan::Divan::from_args().main();
}

fn build<S, const N: usize>() -> Option<S>
where
    S: Buildable<PureCircuit>,
{
    S::build({
        let mut circuit = Circuit::new(N);

        circuit = circuit.h(0);

        for i in 1..N {
            circuit = circuit.cx(&[0], i);
        }

        circuit
    }).ok()
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
