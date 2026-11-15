# Consolidated numerical package

Python and Rust implementations, teaching examples, figures, data, and verification results assembled for the numerical-PDE project.

LaTeX sources are intentionally not tracked. The compiled [`report.pdf`](report.pdf) is retained only as a reference document; report editing happens separately in the **Final Report** chat.

## Code and evidence

| Path | Contents |
|---|---|
| `examples/fft_periodic_solver.py` | Periodic FFT-solver, sparse-stage, and discretization checks |
| `examples/butterfly/` | Minimal phase-based butterfly, four tests, accuracy/timing data, and plots |
| `python/` | Larger Python experimental package and longer research suites |
| `rust/` | Apply benchmarks, DFT invariants, and reference-core prototype |
| `figures/` | Figures generated from the included numerical data |
| `results/local/` | Locally generated numerical and test receipts |
| `results/package/` | Historical review notes, results, and verification receipts |
| `scripts/check_local.sh` | Reproduction wrapper for the core numerical checks |

## Reproduce the small examples

```bash
python3 examples/fft_periodic_solver.py
cd examples/butterfly
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 test_butterfly.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 compare.py
```

## Reproduce the core package checks

Run from this directory:

```bash
scripts/check_local.sh
```

The wrapper runs the FFT example, four butterfly tests and figure regeneration, 126 Python core tests, 24 showcase tests, all 21 Rust unit/integration checks, saved Rust-data validation, and reference-core golden controls. It **does not** rerun the full twelve research suites; `python/scripts/test-all.sh` is the separate longer job.

## Recorded verification scope

- FFT algebra and PDE-error checks ran under Python 3.12.14 / NumPy 2.5.3.
- Four butterfly tests passed, with comparison figures and CSV data generated.
- Python: 126 core + 24 showcase tests passed; 264 total tests collected, not all run.
- Rust: 7 unit + 14 DFT-invariant checks passed.
- Saved Rust data: 12 compressed, five FFT, and one exact DFT-butterfly record validated.
- Reference core: 72 exact golden values and 17 documented tolerance-boundary count differences; all 11 achieved errors below $10^{-8}$ and maximum LU unit residual $1.72\times10^{-15}$.

Rust wall-time values in the reference report are preserved historical measurements, not timings regenerated in this pass. No Lean verification is claimed. See the result files for exact commands, parameters, and limitations.
