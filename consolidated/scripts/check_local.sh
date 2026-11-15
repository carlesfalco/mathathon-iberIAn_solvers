#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
mkdir -p results/local/target-only .build/cargo-home .build/cargo-target

python3 examples/fft_periodic_solver.py > results/local/fft-console.txt
PYTHONPATH=python python3 examples/framework_laplace.py > results/local/framework-laplace-example.txt
(
  cd examples/butterfly
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 test_butterfly.py \
    > ../../results/local/butterfly-tests.txt 2>&1
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 compare.py \
    > ../../results/local/butterfly-comparison-console.txt 2>&1
)
(
  cd examples/target-only
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python3 compare_butterfly.py > ../../results/local/target-only/svd-comparison.txt 2>&1
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python3 compare_idbf.py > ../../results/local/target-only/idbf-comparison.txt 2>&1
)
python/scripts/test-core.sh > results/local/python-core.txt
python/scripts/test-showcase.sh > results/local/python-showcase.txt

export CARGO_HOME="$ROOT/.build/cargo-home"
export CARGO_TARGET_DIR="$ROOT/.build/cargo-target"
cargo build --release --locked --workspace --manifest-path rust/Cargo.toml \
  > results/local/rust-build.txt 2>&1
cargo test --release --locked --workspace --manifest-path rust/Cargo.toml --all-targets \
  > results/local/rust-tests.txt
python3 rust/scripts/verify_results.py > results/local/rust-saved-results-check.txt
python3 rust/scripts/verify_reference_core.py > results/local/rust-reference-core-check.txt

cp examples/butterfly/results/comparison.png figures/butterfly-dft.png
cp examples/butterfly/results/non_dft_comparison.png figures/butterfly-non-dft.png
printf 'Local report examples and core/showcase/Python/Rust checks passed.\n'
printf 'The longer twelve research suites were not run by this wrapper.\n'
