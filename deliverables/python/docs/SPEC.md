**Historical design/specification, retained for reference.** This is not the current run guide or submission status page. Use `TESTING.md`, `STATUS.md` and `../README.md`; some commands and paths below refer to the original, unbundled project environment.

# Unified fast-algorithm code: spec

Goal: one small, readable code base in which **HBS-style recursive skeletonization (weak admissibility)**,
**FMM-LU-style strong recursive skeletonization (strong admissibility)** and a **1D butterfly** are the same
program with swappable parts:

| Part | Choices | Where |
|---|---|---|
| tree | uniform 2^d-ary tree over points (1D or 2D) | `unified/core/tree.py` (`Tree`, `Node`) |
| admissibility | `Weak`, `Strong`, `Complementary` | `unified/core/admissibility.py` |
| compression | `interp_decomp` (one ID) fed by `ExactCompressor` or `ProxyCompressor` | `unified/core/compress.py` |
| algebra | factor (eliminate box by box) or apply (butterfly) | `unified/drivers/` |

Clarity beats speed. Every function gets a docstring that names the step in the papers' language
(skeleton, redundant, interpolation matrix T, Schur complement, tile, phase-space cell). Target size:
each driver ≤ ~300 lines. No new dependencies beyond numpy, scipy, matplotlib, pytest.

## Core API (frozen; owned by the coordinator)

Do not edit `unified/core/*`. If you need something, put a helper in your own file and mention it in
your handoff. Tests: `.venv/bin/python -m pytest -q unified/tests/test_core.py` (6 passing).

- `Tree(points, depth=None, max_leaf=32)`: points (n,) or (n,d). `tree.levels[l]` = list of non-empty
  `Node`s at level l; `tree.depth`; `tree.nodes_postorder()` (leaves first);
  `tree.neighbors(node)`; `tree.interaction_list(node)`; `node.idx` (ALL point indices in the box),
  `node.children`, `node.parent`, `node.center`, `node.half_width`, `node.level`, `node.coord`.
- `Weak().near_nodes(tree, node) -> []`, `Strong().near_nodes(tree, node) -> tree.neighbors(node)`,
  `Complementary().pairs(row_tree, col_tree, l)` → list of (row node at depth l, col node at depth L-l).
- `interp_decomp(M, eps) -> (skel, red, T)` with `M[:, red] ≈ M[:, skel] @ T` (indices into M's columns).
- Compressors: `comp(box_idx, far_idx, node) -> (skel, red, T)` (indices into `box_idx`).
  `ExactCompressor(A, eps)` uses true entries `A[far, box]`, `A[box, far]`.
  `ProxyCompressor(kernel, points, eps, A=None, n_proxy=64, radius_factor=2.5)` uses proxy points on a
  circle plus the far unknowns inside the circle (explicit rows from `A`).
- `kernels.laplace2d`, `kernels.helmholtz2d(k)`, `kernels.fio1d(t, c)` (diagonal set to 0).

## Standard test problems (use these names in tests and benchmarks)

- **P1 curve**: starfish `r = 1 + 0.3 cos 5θ`, N points equispaced in θ, weights `w_j = |γ'(θ_j)| 2π/N`,
  `A = I + K * w[None, :]` with `K = laplace2d(X, X)`. Second-kind-like, well conditioned.
- **P2 area**: N uniform random points in the unit square (seed 0), `A = I + K / N`, `K = laplace2d`.
  Weak admissibility ranks should grow with N here; strong should stay bounded.
- **P3 Helmholtz curve**: as P1 with `helmholtz2d(k)`, k in {5, 20, 50}: ranks grow with k.
- **B1 butterfly**: `kernels.fio1d(t)` with x = arange(N)/N, xi = arange(N) - N/2, N = 2^L;
  t = 0 is a shifted inverse DFT: `K @ g == (-1)**j * N * ifft(g)`.

## Deliverables

### A1: `unified/drivers/rskel.py` (+ `unified/tests/test_rskel.py`)
`rskel_factor(A, tree, admissibility, compressor) -> Factorization` with `.solve(b)` and `.stats`
(skeleton sizes per level, root size, number of eliminated unknowns per level, wall time).
ONE implementation for both admissibilities: weak = strong with an empty near list.
Algorithm (Ho–Greengard 2012 for weak; Minden–Ho–Damle–Ying 2017 / Sushnikova et al. 2023 for strong):
leaves to root, for each box: ID against the far field allowed by the admissibility rule → sparsify with
T (rows and columns) → eliminate the redundant unknowns by block Gaussian elimination (Schur complement);
skeletons of the children become the parent's active unknowns; dense LU at the root.
**Key subtlety (strong case):** elimination modifies blocks among a box's skeletons and its near
neighbours. Those modified entries are no longer kernel values, so when later compressing a box with the
proxy trick, every far unknown whose interaction with that box has been modified must be included
explicitly (from the current, updated matrix), like FMM-LU's "Q" set. Keep the current matrix as a dense
array or a dict of blocks; dense is acceptable for clarity at N ≤ 16k if memory allows, but say so.
Acceptance: relative residual ‖A x − b‖/‖b‖ ≤ 100·eps on P1 and P2 for N up to 4096 with both rules and
both compressors; nonsymmetric matrices work; stats show weak root size growing on P2 and strong bounded.

### A2: `unified/drivers/butterfly_id.py` (+ `unified/tests/test_butterfly_id.py`)
1D butterfly built from the SAME `Tree` (one over x, one over xi), `Complementary` pairs and
`interp_decomp`. Produce explicit sparse factors (scipy.sparse) and `apply(g)`. Use the ID (skeleton
columns / interpolative butterfly in the style of O'Neil–Woolfe–Rokhlin 2010, Pang–Ho–Yang IDBF) rather
than SVDs, so the whole repo uses one compression primitive. A reference SVD version exists in
`code/butterfly.py` (do not edit it; use it to compare storage and accuracy).
Acceptance: B1 error ≤ 10·eps for N = 256…4096, t ∈ {0, 1, 3}; t = 0 compared with numpy FFT;
storage (nnz) reported and compared with `code/butterfly.py`.

### A3: `unified/tests/test_spec_*.py`, `unified/bench/`, `unified/README.md`
Independent black-box tester and benchmarker: write tests from THIS spec only (not from A1/A2 code),
including edge cases (clustered points, empty boxes, N not a power of 2 for rskel, complex matrices,
eps sweep). Benchmarks with matplotlib PNG + JSON in `unified/bench/results/`: factor time, solve time
and root size vs N for weak vs strong on P1 and P2; rank growth vs k on P3; butterfly error and nnz vs N.
README: the "one algorithm, swappable parts" table above, how to run everything, and a claim table
labelling each result as proved / computed / cited. If A1/A2 files don't exist yet, write tests first
and re-run them when they appear.

## Rules for all workers
- Edit only the files you own (listed above). Do not commit. Do not edit `unified/core/*` or `code/*`.
- Run `.venv/bin/python -m pytest -q unified/tests` before finishing; report exact results.
- Label claims honestly in docstrings and README (computed vs cited vs conjecture).
