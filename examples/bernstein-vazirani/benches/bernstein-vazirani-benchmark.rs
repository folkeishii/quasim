use bernstein_vazirani::circuit;
use divan::Bencher;
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
use quasim::{cubecl::wgpu::WgpuRuntime, gpu_sv_simulator::GpuStateVectorSimulator};

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

const QUBITS: &[usize] = &[2, 4, 8, 12, 16];
const FMM_QUBITS: &[usize] = &[2, 4, 8, 12];
bench!(
    state_vector_simulator, StateVectorSimulator, QUBITS;
    "gpu" => gpu_accelerated, GpuStateVectorSimulator<WgpuRuntime>, QUBITS;
    full_mat_mul_simulator, FullMatMulSimulator, FMM_QUBITS;
    product_state_simulator, ProductStateSimulator, QUBITS;
    cube_simulator, CubeSimulator, QUBITS;
);

fn main() {
    divan::Divan::from_args().main();
}

fn build<S, const N: usize>() -> S
where
    S: Buildable<HybridCircuit>,
{
    let secret = !(usize::MAX << N);
    S::build(circuit(N, secret)).unwrap()
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
