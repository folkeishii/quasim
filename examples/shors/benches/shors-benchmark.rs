use divan::Bencher;
use shors::circuit;
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

const NS: &[usize] = &[0x3, 0x7, 0xF, 0x3F, 0x7F, 0xFF];
const FMM_NS: &[usize] = &[0x3, 0x7];
bench!(
    state_vector_simulator, StateVectorSimulator, NS;
    "gpu" => gpu_accelerated, GpuStateVectorSimulator<WgpuRuntime>, NS;
    full_mat_mul_simulator, FullMatMulSimulator, FMM_NS;
    product_state_simulator, ProductStateSimulator, NS;
    cube_simulator, CubeSimulator, NS;
);

fn main() {
    divan::Divan::from_args().main();
}

fn build<S, const N: usize>() -> S
where
    S: Buildable<HybridCircuit>,
{
    let a = N-1;
    S::build(circuit(N, a)).unwrap()
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
