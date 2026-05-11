#!/bin/sh

export DIVAN_SAMPLE_COUNT=10
export DIVAN_SAMPLE_SIZE=10
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
