use divan::Bencher;
use grovers::circuit;
extern crate quasim;
use quasim::{
    circuit::HybridCircuit, cube_simulator::CubeSimulator, fmm_simulator::FullMatMulSimulator, product_state_simulator::ProductStateSimulator, sampler::{RegisterSampler, Sampler}, simulator::{Buildable, Sampleable, StoredRegisters}, sv_simulator::StateVectorSimulator
};
#[cfg(feature="gpu")]
use quasim::{gpu_sv_simulator::GpuStateVectorSimulator, cubecl::wgpu::WgpuRuntime,};

macro_rules! bench {
    ($($feat:literal =>)? $name:ident, $sim:ty, $qubits:expr $(;)?) => {
        $(#[cfg(feature = $feat)])?
        #[divan::bench(
            consts = $qubits
        )]
        fn $name<const N: usize>(bencher: Bencher) {
            let (mut sim, correct) = build::<$sim, N>();
            bench(bencher, &mut sim, correct);
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

const QUBITS: &[usize] = &[2,4,8,12,16];
bench!(
    state_vector_simulator, StateVectorSimulator, QUBITS;
    "gpu" => gpu_accelerated, GpuStateVectorSimulator<WgpuRuntime>, QUBITS;
    full_mat_mul_simulator, FullMatMulSimulator, QUBITS;
    product_state_simulator, ProductStateSimulator, QUBITS;
    cube_simulator, CubeSimulator, QUBITS;
);

fn main() {
    divan::Divan::from_args().main();
}

fn build<S, const N: usize>() -> (S, usize)
where
    S: Buildable<HybridCircuit>,
{
    let mut func = [0; N];
    func[0] = 1;
    let fun_res: usize = func.iter().rev().enumerate().map(|(i, &b)| b << i).sum();
    (S::build(circuit(&func)).unwrap(), fun_res)
}

fn bench<S>(bencher: Bencher, sim: &mut S, correct: usize) -> bool
where
    S: Sampleable<HybridCircuit> + StoredRegisters,
{
    let mut ret = false;
    bencher.bench_local(|| {
        ret = divan::black_box(RegisterSampler::new("res").sample({sim.run(); sim})) == correct;
    });
    ret
}
