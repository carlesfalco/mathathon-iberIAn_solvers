# `rust-unified`: a small Rust compression/apply prototype

## Status in one paragraph

This workspace contains two explicitly different prototypes:

1. **`unified-rs` (the root crate):** five short example programs on one Rust library: log/Laplace, Gaussian covariance, exponential screening, a direct FFT with a dense reference, and the exact rank-one DFT butterfly. It demonstrates that kernel-facing examples can be very small and that compressed apply/storage can be efficient.
2. **[`core-prototype/`](core-prototype/README.md):** a separate dependency-free feasibility port of the Python tree, weak/strong admissibility, exact/proxy compression, complex FIO butterfly, and recursive LU solve. It has 72 exact golden keys and 17 documented tolerance-boundary count differences; parity is error-based, not bit-for-bit rank portability.

Neither is a complete production port of `updated/unified`. The root crate does not claim its own Python-compatible tree/admissibility or recursive-solver API, and the companion prototype uses hand-written numerical routines rather than a production LAPACK backend. There is no special-function physics package or E6–E12 parity yet. See [PORT_STATUS.md](PORT_STATUS.md) for the staged discussion and [BENCHMARKING.md](BENCHMARKING.md) for the comparison protocol.

## Lean one-file examples

| Example | Lines | Run |
|---|---:|---|
| `examples/laplace.rs` | 12 | `cargo run --release --example laplace -- 2048` |
| `examples/gaussian.rs` | 11 | `cargo run --release --example gaussian -- 2048` |
| `examples/screened.rs` | 11 | `cargo run --release --example screened -- 2048` |
| `examples/fft.rs` | 38 | `cargo run --release --example fft -- 8192` |
| `examples/dft_as_butterfly.rs` | one-file claim + checks | `cargo run --release --example dft_as_butterfly` |

For example, the complete Laplace program is only:

```rust
use unified_rs::problem::{grid, print_records};
use unified_rs::KernelProblem;

// One file per problem: the library owns trees, compression and timing.
fn main() {
    let n = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(512);
    let problem = KernelProblem::new("log/Laplace, I + K", grid(n), |x, y| {
        let r = (x - y).abs();
        if r == 0.0 { 1.0 } else { -r.ln() / (2.0 * std::f64::consts::PI) }
    });
    print_records(&[problem.benchmark(32, 1e-6)]);
}
```

The other two kernel examples differ essentially by the closure:

```rust
// Gaussian
(-0.5 * ((x - y) / 0.08_f64).powi(2)).exp()
// Screened exponential
(-8.0_f64 * (x - y).abs()).exp()
```

## What the library actually does

- Builds a real `f64` dense matrix from a kernel closure.
- Recursively partitions diagonal/near-field blocks and compresses off-diagonal blocks with **column-pivoted QR**.
- Uses the relative diagonal criterion $|R_{ii}|>\varepsilon|R_{00}|$ and stores $Q_rR_rP^{-1}$ when this is smaller than a dense block.
- Keeps small or unprofitable blocks dense and zero blocks at rank zero.
- Measures storage, a dense-reference apply error, dense construction, compression, and repeated apply times.

**Important:** `HMatrix::from_dense` is a dense-to-compressed **oracle** for testing the representation. It is not a hierarchical fast builder, and its SVD/QR-style construction must not be described as such. The low-rank factor stores a $Q_rR_r$ compression, not Python's column-skeleton interpolation matrix $T$.

## Verified results in this VM

Environment: Rust 1.93.1, release build, `nalgebra` 0.34.2, `rustfft` 6.4.1, warm-up, raw samples, min/median timings, and peak RSS. The VM is shared; raw values and protocol are in [`results/summary.json`](results/summary.json). These are not Rust-versus-Python language benchmarks.

At $N=2048$, leaf size 32, $\varepsilon=10^{-6}$:

| Problem | Storage / dense | Relative apply error | Dense apply | Compressed apply | Apply speedup | Compression time | Approx. apply break-even* |
|---|---:|---:|---:|---:|---:|---:|---:|
| log/Laplace | 0.0767 | $2.77\times10^{-7}$ | 0.676 ms | 0.057 ms | 11.8× | 0.965 s | 1,560 |
| Gaussian | 0.0576 | $2.82\times10^{-7}$ | 0.647 ms | 0.051 ms | 12.6× | 0.964 s | 1,619 |
| Screened exponential | 0.0361 | $1.04\times10^{-15}$ | 0.661 ms | 0.045 ms | 14.7× | 0.961 s | 1,559 |

\* Break-even is compression time divided by the apply-time saving per call, ignoring dense construction and all other application work. It is a useful rule of thumb for these measurements, not a general theorem.

The screened kernel has rank-one separated interactions in this simple 1-D geometry, so its exceptionally small error is expected, not a universal error guarantee. All 12 compressed benchmark records pass the saved check `relative_apply_error < 1e-6`; the largest is $8.17\times10^{-7}$.

### FFT

The direct FFT uses a reusable RustFFT plan and is checked against the crate's naive negative-sign DFT at $N=512,\ldots,8192$. Errors range from $7.2\times10^{-14}$ to $1.12\times10^{-12}$. Unit tests also check Parseval's identity.

The independent `dft_butterfly` modules additionally construct the exact complementary factorization. At $N=1024$, every stage has rank one, storage is exactly $2N\log_2N+2N=22{,}528$, and the one-file example agrees with a separate radix-two FFT to $1.83\times10^{-16}$. Fourteen invariants check full operator reconstruction, unimodular factors, Parseval, round trips, mixed radix, natural/mixed-order falsifiers, and this builder's exact $3N^2-2N$ construction-entry count. Row-sampled IDBF would be a different, cheaper construction path; the generic FIO butterfly is not implemented in this root crate, though a separate reference implementation exists in `core-prototype`.

The naive reference evaluates sine/cosine for an $\mathrm O(N^2)$ DFT. **Do not quote its timing ratio as a comparison against NumPy's optimized FFT or any production FFT implementation.** Its purpose is correctness and asymptotic scaling.

## Rebuild and test

From this directory, with Cargo installed:

```bash
cargo test --release --all-targets
```

From the project root, run the full deterministic sweep and validation:

```bash
rust-unified/scripts/run.sh
```

`run.sh` uses `.scratch/cargo-home` and `.scratch/cargo-target` in the project for downloaded dependencies and build artifacts. It writes 12 compressed-kernel JSON files, five FFT JSON files, `dft-as-butterfly.txt`, and `results/summary.json`. `scripts/verify_results.py` rechecks every accuracy/storage assertion and reconstructs the summary. Timings naturally differ when repeated on a different or loaded machine; correctness checks do not depend on speedups. The saved records contain every sample, min/median values, and high-water RSS (about 61 MiB at $N=2048$).

## Deliberate limits

1. Only the companion `core-prototype` has tree/admissibility, exact/proxy ID, complex FIO, and recursive LU solve; these are not unified into the root crate's production API.
2. The hand-written companion routines are not a production LAPACK backend, and bit-identical rank decisions are not guaranteed. Seventeen documented tolerance-boundary differences remain.
3. Dense working construction still exists; there is no claim of an $O(N\log N)$ end-to-end constructor.
4. No verified Bessel/Hankel/Legendre special-function package or E6–E12 physics parity.
5. No generic ID butterfly in the root crate (`core-prototype` has the complex FIO reference; exact DFT exists here too).
6. Benchmarks use process-local dense work arrays, not a tuned matched-backend Rust-versus-Python benchmark.
7. There is no complete E6–E12 reproduction or production direct-solver cost claim.

## Numerical regressions included

The combined 21 checks cover zero matrices, diagonal preservation, exact rank-one reconstruction, a smooth log-kernel tolerance check, FFT-versus-dense correctness, Parseval, and the 14 exact DFT-butterfly invariants. The exact rank-one test is important: it catches factorization/convention mistakes that storage counts and plausible-looking timings cannot detect.
