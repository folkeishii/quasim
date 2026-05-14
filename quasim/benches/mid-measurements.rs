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
bench_utils::bench!(state_vector_simulator, StateVectorSimulator, "mid-measurements", "state_vector_simulator", use_args);
#[rustfmt::skip]
bench_utils::bench!("wgpu" => gpu_accelerated_wgpu, GpuStateVectorSimulator<WgpuRuntime>, "mid-measurements", "gpu_accelerated_wgpu", use_args);
#[rustfmt::skip]
bench_utils::bench!("cuda" => gpu_accelerated_cuda, GpuStateVectorSimulator<CudaRuntime>, "mid-measurements", "gpu_accelerated_cuda", use_args);
#[rustfmt::skip]
bench_utils::bench!("cpu" => gpu_accelerated_cpu, GpuStateVectorSimulator<CpuRuntime>, "mid-measurements", "gpu_accelerated_cpu", use_args);
#[rustfmt::skip]
bench_utils::bench!("hip" => gpu_accelerated_hip, GpuStateVectorSimulator<HipRuntime>, "mid-measurements", "gpu_accelerated_hip", use_args);
#[rustfmt::skip]
bench_utils::bench!(full_mat_mul_simulator, FullMatMulSimulator, "mid-measurements", "full_mat_mul_simulator", use_args);
#[rustfmt::skip]
bench_utils::bench!(product_state_simulator, ProductStateSimulator, "mid-measurements", "product_state_simulator", use_args);
#[rustfmt::skip]
bench_utils::bench!(cube_simulator, CubeSimulator, "mid-measurements", "cube_simulator", use_args);
#[rustfmt::skip]
bench_utils::bench!(syntax_simulator, SyntaxSimulator, "mid-measurements", "syntax_simulator", use_args);

fn main() {
    divan::Divan::from_args().main();
}

fn build<S, const N: usize>(measurements: usize) -> Option<S>
where
    S: Buildable<HybridCircuit>,
{
    if N < measurements {
        return None;
    }
    S::build({
        let mut circuit =
            Circuit::new(N)
                .new_reg("dummy", measurements)
                .call_new("qft", Circuit::new_qft(N), 0);

        for m in 0..measurements {
            circuit = circuit.measure_bit(m, ("dummy", m));
        }

        circuit.call("qft", 0)
    })
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
