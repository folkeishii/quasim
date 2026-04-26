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
        fn $name<const N: usize>(bencher: Bencher, entangle_size: usize) {
            if entangle_size > N {
                // Skip bench
                return;
            }
            let mut sim = build::<$sim, N>(entangle_size);
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

const QUBITS: &[usize] = &[10, 16, 22];
const ENTANGLE_SIZE: &[usize] = &[2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22];
const FMM_QUBITS: &[usize] = &[10, 16];
bench!(
    state_vector_simulator, StateVectorSimulator, QUBITS, ENTANGLE_SIZE;
    "wgpu" => gpu_accelerated_wgpu, GpuStateVectorSimulator<WgpuRuntime>, QUBITS, ENTANGLE_SIZE;
    "cuda" => gpu_accelerated_cuda, GpuStateVectorSimulator<CudaRuntime>, QUBITS, ENTANGLE_SIZE;
    "cpu" => gpu_accelerated_cpu, GpuStateVectorSimulator<CpuRuntime>, QUBITS, ENTANGLE_SIZE;
    "hip" => gpu_accelerated_hip, GpuStateVectorSimulator<HipRuntime>, QUBITS, ENTANGLE_SIZE;
    full_mat_mul_simulator, FullMatMulSimulator, FMM_QUBITS, ENTANGLE_SIZE;
    product_state_simulator, ProductStateSimulator, QUBITS, ENTANGLE_SIZE;
    cube_simulator, CubeSimulator, QUBITS, ENTANGLE_SIZE;
);

fn main() {
    divan::Divan::from_args().main();
}

fn build<S, const N: usize>(entangle_size: usize) -> S
where
    S: Buildable<PureCircuit>,
{
    S::build({
        let n_systems = N / entangle_size;

        let mut circuit = Circuit::new(N);

        for q in 0..N {
            circuit = circuit.h(q);
        }

        for s in 0..n_systems {
            circuit = circuit.cx(
                &Vec::from_iter((s * entangle_size..(s + 1) * entangle_size).skip(1)),
                s,
            );
        }

        circuit
    }).unwrap()
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
