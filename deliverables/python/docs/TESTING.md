# Testing, dependencies and reproduction

## Commands run from `python/`

The shortest reproducible workflow is:

```bash
PYTHONPATH=. python3 -m pytest -q unified/tests unified/showcase
PYTHONPATH=. python3 -m unified.showcase.reproduce --quick --no-write
PYTHONPATH=. python3 -m pytest --collect-only -q unified
```

The wrapper scripts set single-thread numerical-backend variables before invoking these tests. Change them explicitly if you are benchmarking threads, but do not compare the resulting times with the reference pinned runs.

## What is a quick test versus a research experiment?

- `unified/tests`: core tree/ID/compression/driver tests, including the black-box spec tests. `RUN_SLOW=1` adds the largest stated cases.
- `unified/showcase`: ten model smoke checks at reduced sizes plus catalogue/line-count tests.
- `unified/explore/e1_...` through `e12_...`: research experiments. These have independent correctness tests, saved parameter tables, figures, and (for many) receipts of the production-sized run. The entire research set is a substantially longer job than the core and showcase tests.
- `unified/bench`: driver-scaling drivers and reference data, not a portable speed certificate. `run_butterfly` uses the packaged reference SVD helper at `unified.reference.butterfly`.
- `unified/showcase.reproduce --quick --no-write`: a full ten-model no-write reproduction, rather than only catalogue assertions.

The saved JSON and PNG files are **evidence snapshots**, not automatically regenerated expectations. Keep reproducibility and document prose separate by checking directly against JSON or an independent reference.

## Dependency inventory

The core and most tests use NumPy/SciPy arrays and linear algebra, including pivoted QR, LU and sparse factors. Plotting uses Matplotlib. Experiment-specific modules additionally use SciPy's Hankel functions and derivatives, $K_\nu$, scaled $K_\nu$, $J_\nu$, $Y_0$, $J_0$, Gamma/log-Gamma, $E_1$, factorial, Riemann zeta, associated Legendre functions, spherical $j_n$, Gauss nodes, Bessel zeros and ODE integration. Python 3.12 is required by the source's `type` alias syntax.

The manifest records NumPy, SciPy, Matplotlib, and the optional pytest extra. There is no bundled binary `research/` directory, `.venv` dependency, missing Git submodule, or hidden package installation step.

## Environment and accuracy caveats

- BLAS and LAPACK versions alter pivot tie-breaking on near-tolerance blocks. Gate on achieved error rather than portable bit-identical rank counts.
- Test kernels can be ill-conditioned. A residual/backward error is not the same as error against a continuous PDE solution.
- Subnormal tiles may yield zeros on one platform and NaNs on another. E10 tests the protective clipping/absolute-tolerance path, not that one library version reproduces a NaN.
- Native-thread settings can substantially change runtime. Pin threads for fair comparisons, and report hardware, package versions and whether assembly was included.
- The complete 264-test upstream collection exceeded one earlier audit's 2400 s time limit. This cleanup documents each suite separately instead of claiming an unmeasured full-suite pass.

## Benchmark validity

The package keeps dense working matrices in its reference solver. These are useful for correctness and exploring finite-$N$ behavior, but are not a production sparse implementation. Always state:

1. problem, $N$, kernel, depth, rule, compressor and $\varepsilon$;
2. achieved error and the norm/reference used;
3. assembly, compression/factorization and apply/solve times separately;
4. stored entries and process memory separately;
5. cold/warm runs, raw samples and min/median values where relevant.

Do not compare a Python call backed by optimized BLAS with a naive Rust loop and interpret the ratio as a language property. See [../../rust/BENCHMARKING.md](../../rust/BENCHMARKING.md) for the measured comparison methodology.
