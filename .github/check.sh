#!/usr/bin/env bash

set -euo pipefail

for filename in examples/*/Cargo.toml; do
    [ -e "$filename" ] || continue
    cargo fmt --manifest-path "$filename" --all -- --check
    cargo check --manifest-path "$filename"
    cargo test --manifest-path "$filename"
done
