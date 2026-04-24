use deutsch_jozsa::FunctionType;
use divan::Bencher;
use deutsch_jozsa::circuit;
extern crate quasim;
use quasim::{
    circuit::HybridCircuit,
    cube_simulator::CubeSimulator,
    fmm_simulator::FullMatMulSimulator,
    product_state_simulator::ProductStateSimulator,
    sampler::{RegisterSampler, Sampler},
    simulator::{Buildable, Sampleable, StoredRegisters},
    sv_simulator::StateVectorSimulator,
};
#[cfg(feature = "gpu")]
use quasim::{cubecl::wgpu::WgpuRuntime, gpu_sv_simulator::GpuStateVectorSimulator};

macro_rules! bench {
    ($($feat:literal =>)? $name:ident, $sim:ty, $qubits:expr $(;)?) => {
        $(#[cfg(feature = $feat)])?
        #[divan::bench(
            args = [FunctionType::Constant0, FunctionType::Constant1, FunctionType::Balanced],
            consts = $qubits
        )]
        fn $name<const N: usize>(bencher: Bencher, ft: FunctionType) {
            let mut sim = build::<$sim, N>(ft);
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

const QUBITS: &[usize] = &[2, 4, 6, 8, 10, 12, 14, 16];
const FMM_QUBITS: &[usize] = &[2, 4, 6, 8, 10];
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

fn build<S, const N: usize>(ft: FunctionType) -> S
where
    S: Buildable<HybridCircuit>,
{
    S::build(circuit(N, ft)).unwrap()
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
