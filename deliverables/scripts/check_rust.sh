#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
WORKSPACE=$(cd "$ROOT/.." && pwd)
export CARGO_HOME="${CARGO_HOME:-$WORKSPACE/.scratch/deliverable-cargo-home}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$WORKSPACE/.scratch/deliverable-cargo-target}"

cargo build --release --locked --workspace --manifest-path "$ROOT/rust/Cargo.toml"
cargo test --release --locked --workspace --manifest-path "$ROOT/rust/Cargo.toml" \
  --all-targets > "$ROOT/results/verification-rust.txt"
python3 "$ROOT/rust/scripts/verify_results.py" > "$ROOT/results/verify-kernel-saved.txt"
python3 "$ROOT/rust/scripts/verify_reference_core.py" > "$ROOT/results/verify-reference-core.txt"
printf 'Rust checks completed. Saved outputs checked in %s and %s.\n' \
  "$ROOT/results/verify-kernel-saved.txt" "$ROOT/results/verify-reference-core.txt"
