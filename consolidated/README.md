# Consolidated numerical package

Python and Rust implementations, teaching examples, figures, data, and verification results assembled for the numerical-PDE project.

The report LaTeX source is included alongside the compiled [`report.pdf`](report.pdf). The **Final Report** chat remains the authoritative report-editing workflow.

## Code and evidence

| Path | Contents |
|---|---|
| `examples/fft_periodic_solver.py` | Periodic FFT-solver, sparse-stage, and discretization checks |
| `examples/framework_laplace.py` | Section 3 framework-to-code example: tree, weak interactions, proxy compression, recursive solve |
| `examples/butterfly/` | Minimal phase-based butterfly, four tests, accuracy/timing data, and plots |
| `examples/target-only/` | Original two-invariant, target-only QR, and IDBF comparison drivers with raw results |
| `python/` | Larger Python experimental package and longer research suites |
| `rust/` | Apply benchmarks, DFT invariants, and reference-core prototype |
| `figures/` | Figures generated from the included numerical data |
| `results/local/` | Locally generated numerical and test receipts |
| `results/package/` | Historical review notes, results, and verification receipts |
| `scripts/check_local.sh` | Reproduction wrapper for the core numerical checks |

## Reproduce the small examples

```bash
python3 examples/fft_periodic_solver.py
PYTHONPATH=python python3 examples/framework_laplace.py
cd examples/butterfly
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 test_butterfly.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 compare.py

# Section 5: target-only SVD/QR and IDBF comparisons
cd ../target-only
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 compare_butterfly.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 compare_idbf.py
```

## Reproduce the core package checks

Run from this directory:

```bash
scripts/check_local.sh
```

The wrapper runs the FFT and framework-to-code Laplace examples, four butterfly tests and figure regeneration, both target-only SVD/IDBF comparison drivers, 126 Python core tests, 24 showcase tests, all 21 Rust unit/integration checks, saved Rust-data validation, and reference-core golden controls. It **does not** rerun the full twelve research suites; `python/scripts/test-all.sh` is the separate longer job.

## Recorded verification scope

- FFT algebra and PDE-error checks ran under Python 3.12.14 / NumPy 2.5.3.
- Four butterfly tests passed, with comparison figures and CSV data generated.
- The Section 3 Laplace example returned an independent kernel residual of $9.412014\times10^{-10}$ at $N=512$.
- Target-only SVD and IDBF drivers passed on exact-rank matrices at $N=16,64,256$; application ratios, entry counts, and raw errors are reported in Section 5 and `results/local/target-only/`.
- Python: 126 core + 24 showcase tests passed; 264 total tests collected, not all run.
- The updated reference PDF has 40 A4 pages, including the explicit FFT sparse-stage definitions and $N=8$ diagram, Section 3 levers, five intuitive Martin figures, and the Appendix B code-reuse/line-count table. The separate Final Report workflow owns its LaTeX source.
- Rust: 7 unit + 14 DFT-invariant checks passed.
- Saved Rust data: 12 compressed, five FFT, and one exact DFT-butterfly record validated.
- Reference core: 72 exact golden values and 17 documented tolerance-boundary count differences; all 11 achieved errors below $10^{-8}$ and maximum LU unit residual $1.72\times10^{-15}$.

Rust wall-time values in the package benchmark table are preserved historical measurements. The target-only timings are fresh local medians on this VM and apply only to exact-rank oracle matrices, not nontrivial approximation problems or optimized sampled construction. No Lean verification is claimed. See the result files for exact commands, parameters, and limitations.
