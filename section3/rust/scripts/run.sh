#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CRATE=$(cd "$SCRIPT_DIR/.." && pwd)
REPO=$(cd "$CRATE/.." && pwd)
export CARGO_HOME="${CARGO_HOME:-$REPO/.scratch/cargo-home}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$REPO/.scratch/cargo-target}"
cargo test --release --workspace --manifest-path "$CRATE/Cargo.toml" --all-targets
cargo build --release -p rust-unified-prototype --manifest-path "$CRATE/Cargo.toml"
for example in laplace gaussian screened; do
  for n in 256 512 1024 2048; do
    cargo run --quiet --release --manifest-path "$CRATE/Cargo.toml" \
      --example "$example" -- "$n" > "$CRATE/results/${example}-${n}.json"
  done
done
for n in 512 1024 2048 4096 8192; do
  cargo run --quiet --release --manifest-path "$CRATE/Cargo.toml" \
    --example fft -- "$n" > "$CRATE/results/fft-${n}.json"
done
cargo run --quiet --release --manifest-path "$CRATE/Cargo.toml" \
  --example dft_as_butterfly > "$CRATE/results/dft-as-butterfly.txt"
python3 "$SCRIPT_DIR/verify_results.py"
python3 "$SCRIPT_DIR/verify_reference_core.py"
