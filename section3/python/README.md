# Unified fast algorithms — Python reference implementation

Small, readable Python code for hierarchical matrix compression, operator application, recursive skeletonization and butterfly transforms. The package shows **shared vocabulary and reusable components** across several fast methods. It is not a universal solver, and it does not reproduce every published algorithm by changing one menu entry.

Use this directory directly, or install it as the distribution `unified-fast-algorithms`. The imported package is named **`unified`**.

## Quick start

Python **3.12 or newer** is required (the source uses Python 3.12 type syntax). Scientific dependencies are NumPy, SciPy and Matplotlib; tests use pytest.

```bash
# Optional, but recommended if you want an isolated environment.
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q unified/tests
```

If those dependencies already exist, no installation or virtual environment is needed:

```bash
# Run from this directory (python-unified/).
PYTHONPATH=. python3 -c "from unified.core import Tree, Weak, ProxyCompressor; print('import OK')"
```

`[test]` is an optional dependency set in `pyproject.toml`, not a shell command. On Windows PowerShell, use `.venv\Scripts\python` instead of `.venv/bin/python`.

### A small solve

This is the complete construction for the Laplace starfish test problem. It is a weak-admissibility recursive solve using proxy sampling; no new solver is written in the example.

```python
import numpy as np
from unified.core import ProxyCompressor, Tree, Weak
from unified.bench import problems as P
from unified.drivers.rskel import rskel_factor

prob = P.p1_curve(512)
tree = Tree(prob.X, max_leaf=32)
compressor = ProxyCompressor(prob.kernel, prob.X, 1e-8, A=prob.A)
factor = rskel_factor(prob.A, tree, Weak(), compressor)
b = np.ones(512)
x = factor.solve(b)
print(P.relres_kernel(prob, x, b))  # 9.412e-10 in the documented run
```

The maximum permitted residual for this family is $100\varepsilon=10^{-6}$. `rskel_factor` can also use `Strong()` or `ExactCompressor`; pairs can be compared at the same input and tolerance.

To apply a Fourier integral operator with the butterfly driver:

```python
from unified.drivers.butterfly_id import b1_problem, butterfly_id, exact_apply
import numpy as np
kernel, x, xi = b1_problem(512, t=1.0)
bf = butterfly_id(kernel, x, xi, eps=1e-8)
g = np.random.default_rng(0).standard_normal(512)
f = exact_apply(kernel, x, xi, g)
print(np.linalg.norm(bf.apply(g) - f) / np.linalg.norm(f))  # 2.4e-9 <= 10*eps
```

`exact_apply` is a dense-entry reference, not a fast constructor. The butterfly construction here also reads all tile rows; row-sampled IDBF is a different optimization.

## What is implemented

| Layer | Files | Purpose |
|---|---|---|
| Geometry and rules | `unified/core/` | Uniform binary/quadtree, weak/strong/complementary interaction rules, kernel matrices |
| Compression | `unified/core/compress.py` | Column interpolative decomposition (ID), exact far blocks, proxy-circle sampling |
| Solve | `unified/drivers/rskel.py` | Bottom-up sparsification, small LU factors, Schur complements, forward/backward solve |
| Transform | `unified/drivers/butterfly_id.py` | Two trees, staged ID compression, sparse factor application, DFT/FIO examples |
| Test problems | `unified/bench/` | Starfish curve, area points, Helmholtz curve, Fourier integral operator |
| Showcase | `unified/showcase/` | Ten physics/model families with analytic solutions or invariant checks |
| Exploratory studies | `unified/explore/e1_*` … `e12_*` | Projection/sampling/stability/cost experiments and saved results |

The design idea can be organized into five parts: **partition**, **interaction pattern**, **representation**, **sampling**, and **algebraic assembly**. Each needs three separate checks: approximation error, numerical stability, and total cost. Shared interfaces do not eliminate the need for kernel-specific analysis.

### Important terminology

- **Apply:** calculate $y=Ax$.
- **Solve:** recover $x$ such that $Ax=b$; this adds elimination and pivot-stability questions.
- **Low rank:** represent a large block using a few shared patterns.
- **Skeleton:** selected columns retained by an ID.
- **Proxy sampling:** use auxiliary observations instead of reading every far entry. Its validity depends on the kernel and geometry.
- **Dense working matrix:** an $N\times N$ array retained for clarity. The recursive solver still has $O(N^2)$ memory and bookkeeping at this stage.

## Ten-model showcase

```bash
PYTHONPATH=. python3 -m unified.showcase.reproduce --quick --no-write
PYTHONPATH=. python3 -m pytest -q unified/showcase
```

The quick catalogue covers Laplace, Helmholtz, DFT, FIO, Stokes, Yukawa, heat, fractional diffusion, Lippmann–Schwinger scattering and spherical harmonics. **Ten quick agreement checks passed in the independent review environment.** These checks used exact solutions, algebraic invariants, or documented behaviour; they were not all physics-validation experiments or complete dense-free timings.

Read [showcase/RESULTS.md](unified/showcase/RESULTS.md) for the recorded full-mode checks, parameters, code-line counts and caveats. Its `results.json` was produced on an arm64 laptop and is not interchangeable with this VM's timings. At those recorded sizes, four fast paths were faster, five slower, and one approximately tied. A fast path can lose while remaining accurate.

After `pip install -e .`, the same reproduction is available as:

```bash
unified-showcase --quick --no-write
```

Omit `--no-write` to regenerate `unified/showcase/results.json` and `unified/showcase/RESULTS.md` in place. Do this only if you intend to update the published snapshot.

## Tests and builds

```bash
scripts/test-core.sh              # core + shared-driver tests
scripts/test-showcase.sh         # showcase smoke tests
scripts/test-all.sh               # core, showcase and E1–E12: a longer run
scripts/build-wheel.sh           # creates a wheel in dist/
scripts/collect-tests.sh         # collection count; no heavy runs
```

`RUN_SLOW=1` adds the $N=4096$ black-box spec cases. For deliberate numerical-backend comparisons, pin `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`; these are set by the scripts. See [docs/TESTING.md](docs/TESTING.md) for exact scope, timings and known platform limitations.

## Submission contents

- The actual package and all twelve experiment directories.
- The ten-model showcase and its saved checks.
- Saved figures and JSON for the exploratory results; no recomputation is needed to inspect them.
- [docs/STATUS.md](docs/STATUS.md): verified status and limitations.
- [docs/PROVENANCE.md](docs/PROVENANCE.md): relationship to the supplied archive and deliberate cleanup changes.
- [docs/RESULTS_GUIDE.md](docs/RESULTS_GUIDE.md): what the result tables mean and which claims not to make.
- [docs/LEGACY_README.md](docs/LEGACY_README.md): historical project description, **not** the run guide. Some of its commands and references are to directories that were not supplied.

The `md/` material in the supplied archive is intentionally not elevated into a second authoritative README. Its older numerical/textual claims are reviewed in `../results/update-review.md`; this README describes the executable code and current reproducibility evidence.

## Companion Rust implementation

[`../rust`](../rust/README.md) contains a QR-compression/apply library with short one-file examples, an exact DFT butterfly, separate tree/ID/complex-FIO/recursive-solve prototypes, raw timings and golden fixtures. It is **not feature-equivalent to this Python tree**. Python remains the more complete scientific experiment package.

## Scope and honest limits

- This is a readable research harness, not a complete adaptive 3-D PDE solver.
- `rskel_factor` retains a dense working matrix. Its wall-clock slopes are not proof of sparse direct-solver asymptotics.
- Compression error, discretization error, solver stability, and construction cost are different measurements.
- Some special-function and proxy results are platform/regime dependent. Never quote a rank without its tolerance, geometry and norm convention.
- The "one framework" vocabulary has prior art. This project's useful contribution is its map, small implementations and controlled comparisons, not a claim to have invented hierarchical low-rank methods.

For the independent E6–E12 review and corrections, see [../results/update-review.md](../results/update-review.md). For the beginner presentation, see [../slides.pdf](../slides.pdf).
