include!(concat!(env!("OUT_DIR"), "/bench_consts.rs"));

#[macro_export]
macro_rules! bench {
    ($($feat:literal =>)? $name:ident, $sim:ty, $bench:literal, $sim_name:literal) => {
        $(#[cfg(feature = $feat)])?
        #[divan::bench(
            consts = $crate::sim_qubits($bench, $sim_name)
        )]
        fn $name<const N: usize>(bencher: Bencher) {
            let Some(mut sim) = build::<$sim, N>() else {return;};
            bench(bencher, &mut sim);
        }
    };
    ($($feat:literal =>)? $name:ident, $sim:ty, $bench:literal, $sim_name:literal, use_args) => {
        $(#[cfg(feature = $feat)])?
        #[divan::bench(
            args = $crate::sim_args($bench, $sim_name),
            consts = $crate::sim_qubits($bench, $sim_name)
        )]
        fn $name<const N: usize>(bencher: Bencher, arg: usize) {
            let Some(mut sim) = build::<$sim, N>(arg.into()) else {return;};
            bench(bencher, &mut sim);
        }
    };
}

pub const fn sim_qubits(bench: &str, sim: &str) -> &'static [usize] {
    let Some(bench_i) = bench_id(bench) else {
        return env_qubits();
    };
    let Some(sim_i) = sim_id(sim) else {
        return bench_qubits(bench);
    };

    if !SIMULATOR_RUN[bench_i][sim_i] {
        return &[1];
    }

    if SIMULATOR_QUBITS[bench_i][sim_i].is_empty() {
        bench_qubits(bench)
    } else {
        SIMULATOR_QUBITS[bench_i][sim_i]
    }
}

pub const fn bench_qubits(bench: &str) -> &'static [usize] {
    let Some(bench_i) = bench_id(bench) else {
        return env_qubits();
    };

    if BENCHMARK_QUBITS[bench_i].is_empty() {
        env_qubits()
    } else {
        BENCHMARK_QUBITS[bench_i]
    }
}

pub const fn env_qubits() -> &'static [usize] {
    &QUBITS
}

pub fn sim_args(bench: &str, sim: &str) -> &'static [usize] {
    let Some(bench_i) = bench_id(bench) else {
        return env_args();
    };
    let Some(sim_i) = sim_id(sim) else {
        return bench_args(bench);
    };

    if !SIMULATOR_RUN[bench_i][sim_i] {
        return &[1];
    }

    if SIMULATOR_ARGS[bench_i][sim_i].is_empty() {
        bench_args(bench)
    } else {
        SIMULATOR_ARGS[bench_i][sim_i]
    }
}

pub fn bench_args(bench: &str) -> &'static [usize] {
    let Some(bench_i) = bench_id(bench) else {
        return env_args();
    };

    if BENCHMARK_ARGS[bench_i].is_empty() {
        env_args()
    } else {
        BENCHMARK_ARGS[bench_i]
    }
}

pub fn env_args() -> &'static [usize] {
    &ARGS
}

pub const fn sim_id(sim: &str) -> Option<usize> {
    indexed_id(sim, &SIMULATORS)
}

pub const fn bench_id(bench: &str) -> Option<usize> {
    indexed_id(bench, &BENCHMARKS)
}

pub const fn indexed_id(name: &str, indexed: &[&str]) -> Option<usize> {
    let mut i = 0;
    while i < indexed.len() {
        if name.len() != indexed[i].len() {
            i += 1;
            continue;
        }
        let mut e = true;
        let bs1 = name.as_bytes();
        let bs2 = indexed[i].as_bytes();
        let mut j = 0;
        while j < bs1.len() {
            if bs1[j] != bs2[j] {
                e = false;
                break;
            }
            j += 1;
        }
        if e {
            return Some(i);
        }

        i += 1;
    }
    None
}
