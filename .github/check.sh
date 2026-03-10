#!/usr/bin/env bash

set -euo pipefail

shopt -s globstar
for filename in **/Cargo.toml; do
    [ -e "$filename" ] || continue
    cargo fmt --manifest-path "$filename" --all -- --check
    cargo check --manifest-path "$filename" --benches
    cargo test --manifest-path "$filename"
done
