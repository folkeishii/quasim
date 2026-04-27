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
    circuit::{Circuit, HybridCircuit},
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
        fn $name<const N: usize>(bencher: Bencher, measurements: usize) {
            if measurements > N {
                // Skip bench
                return;
            }
            let mut sim = build::<$sim, N>(measurements);
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

const QUBITS: &[usize] = &[12, 16, 22];
const MEASUREMENTS: &[usize] = &[2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22];
const FMM_QUBITS: &[usize] = &[12];
bench!(
    state_vector_simulator, StateVectorSimulator, QUBITS, MEASUREMENTS;
    "wgpu" => gpu_accelerated_wgpu, GpuStateVectorSimulator<WgpuRuntime>, QUBITS, MEASUREMENTS;
    "cuda" => gpu_accelerated_cuda, GpuStateVectorSimulator<CudaRuntime>, QUBITS, MEASUREMENTS;
    "cpu" => gpu_accelerated_cpu, GpuStateVectorSimulator<CpuRuntime>, QUBITS, MEASUREMENTS;
    "hip" => gpu_accelerated_hip, GpuStateVectorSimulator<HipRuntime>, QUBITS, MEASUREMENTS;
    full_mat_mul_simulator, FullMatMulSimulator, FMM_QUBITS, MEASUREMENTS;
    product_state_simulator, ProductStateSimulator, QUBITS, MEASUREMENTS;
    cube_simulator, CubeSimulator, QUBITS, MEASUREMENTS;
);

fn main() {
    divan::Divan::from_args().main();
}

fn build<S, const N: usize>(measurements: usize) -> S
where
    S: Buildable<HybridCircuit>,
{
    S::build({
        let mut circuit = Circuit::new(N).new_reg("dummy", measurements);

        for q in 0..N {
            circuit = circuit.h(q).cx(&[q], (q + 1) % N);
        }

        for m in 0..measurements {
            circuit = circuit.measure_bit(m, ("dummy", m));
        }

        for q in 0..N {
            circuit = circuit.h(q).cx(&[q], (q + 1) % N);
        }

        circuit
    }).unwrap()
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
