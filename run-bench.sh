#!/bin/sh

# USAGE:
# > run-bench.sh <out>

# BENCH PARAMETERS
# Change `bench-parameters.json` to alter the qubits and arguments used for each benchmark and simulator
# OR set the BENCHMARK_PARAMETERS environment variable to a path pointing at a JSON file.

export DIVAN_SAMPLE_COUNT=10
export DIVAN_SAMPLE_SIZE=10
#export BENCHMARk_PARAMETERS="/path/to/json"
#export BENCH_PARAMETERS_TAKE=<count> # only take the first <count> bench_parameters for each bench.
cargo bench --bench mid-measure-all > $1
cargo bench --bench mid-measurements >> $1
cargo bench --bench num-gates >> $1
cargo bench --bench qft >> $1
cargo bench --bench system-size >> $1
cargo bench --bench system-size-entanglement >> $1
cargo bench --bench bernstein-vazirani-benchmark >> $1
cargo bench --bench deutsch-jozsa-benchmark >> $1
cargo bench --bench grovers-benchmark >> $1
cargo bench --bench shors-benchmark >> $1
