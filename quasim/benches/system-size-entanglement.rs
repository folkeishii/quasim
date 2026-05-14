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
bench_utils::bench!(state_vector_simulator, StateVectorSimulator, "system-size-entanglement", "state_vector_simulator", use_args);
#[rustfmt::skip]
bench_utils::bench!("wgpu" => gpu_accelerated_wgpu, GpuStateVectorSimulator<WgpuRuntime>, "system-size-entanglement", "gpu_accelerated_wgpu", use_args);
#[rustfmt::skip]
bench_utils::bench!("cuda" => gpu_accelerated_cuda, GpuStateVectorSimulator<CudaRuntime>, "system-size-entanglement", "gpu_accelerated_cuda", use_args);
#[rustfmt::skip]
bench_utils::bench!("cpu" => gpu_accelerated_cpu, GpuStateVectorSimulator<CpuRuntime>, "system-size-entanglement", "gpu_accelerated_cpu", use_args);
#[rustfmt::skip]
bench_utils::bench!("hip" => gpu_accelerated_hip, GpuStateVectorSimulator<HipRuntime>, "system-size-entanglement", "gpu_accelerated_hip", use_args);
#[rustfmt::skip]
bench_utils::bench!(full_mat_mul_simulator, FullMatMulSimulator, "system-size-entanglement", "full_mat_mul_simulator", use_args);
#[rustfmt::skip]
bench_utils::bench!(product_state_simulator, ProductStateSimulator, "system-size-entanglement", "product_state_simulator", use_args);
#[rustfmt::skip]
bench_utils::bench!(cube_simulator, CubeSimulator, "system-size-entanglement", "cube_simulator", use_args);
#[rustfmt::skip]
bench_utils::bench!(syntax_simulator, SyntaxSimulator, "system-size-entanglement", "syntax_simulator", use_args);

fn main() {
    divan::Divan::from_args().main();
}

fn build<S, const N: usize>(entangle_size: usize) -> Option<S>
where
    S: Buildable<PureCircuit>,
{
    if entangle_size > N {
        return None;
    }
    S::build({
        let n_systems = N / entangle_size;

        let mut circuit = Circuit::new(N).new_sub_circuit("QFT", Circuit::new_qft(entangle_size));

        for s in 0..n_systems {
            circuit = circuit.call("QFT", s * entangle_size);
        }

        let remainder = N % n_systems;

        circuit.call_new(
            "QFT remainder",
            Circuit::new_qft(remainder),
            n_systems * entangle_size,
        )
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
