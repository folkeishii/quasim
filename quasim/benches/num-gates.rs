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
bench_utils::bench!(state_vector_simulator, StateVectorSimulator, "num-gates", "state_vector_simulator", use_args);
#[rustfmt::skip]
bench_utils::bench!("wgpu" => gpu_accelerated_wgpu, GpuStateVectorSimulator<WgpuRuntime>, "num-gates", "gpu_accelerated_wgpu", use_args);
#[rustfmt::skip]
bench_utils::bench!("cuda" => gpu_accelerated_cuda, GpuStateVectorSimulator<CudaRuntime>, "num-gates", "gpu_accelerated_cuda", use_args);
#[rustfmt::skip]
bench_utils::bench!("cpu" => gpu_accelerated_cpu, GpuStateVectorSimulator<CpuRuntime>, "num-gates", "gpu_accelerated_cpu", use_args);
#[rustfmt::skip]
bench_utils::bench!("hip" => gpu_accelerated_hip, GpuStateVectorSimulator<HipRuntime>, "num-gates", "gpu_accelerated_hip", use_args);
#[rustfmt::skip]
bench_utils::bench!(full_mat_mul_simulator, FullMatMulSimulator, "num-gates", "full_mat_mul_simulator", use_args);
#[rustfmt::skip]
bench_utils::bench!(product_state_simulator, ProductStateSimulator, "num-gates", "product_state_simulator", use_args);
#[rustfmt::skip]
bench_utils::bench!(cube_simulator, CubeSimulator, "num-gates", "cube_simulator", use_args);
#[rustfmt::skip]
bench_utils::bench!(syntax_simulator, SyntaxSimulator, "num-gates", "syntax_simulator", use_args);

fn main() {
    bench_utils::ttt();
    divan::Divan::from_args().main();
}

fn build<S, const N: usize>(num_gates: usize) -> Option<S>
where
    S: Buildable<PureCircuit>,
{
    S::build({
        let mut circuit = Circuit::new(N);

        for i in 0..num_gates {
            circuit = circuit.h(i % N);
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
