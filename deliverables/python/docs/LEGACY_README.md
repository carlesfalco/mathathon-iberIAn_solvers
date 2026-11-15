# Historical project README — retained for provenance

**Use [../README.md](../README.md) for the current submission and run commands.**

The text below is the original project README. It mentions `.venv/` and `research/`, which were not supplied, and `code/butterfly.py`, which was absent from the original archive. Its stronger "one program" framing should be read as an organizing abstraction: some choices are coupled, and accuracy, solver stability and total cost must be certified separately.

- `docs/TESTING.md` replaces the `.venv` commands.
- `docs/STATUS.md` gives the verified status and known limitations.
- `docs/RESULTS_GUIDE.md` separates saved measurements from proof and from prior-art claims.

---

# Unified fast algorithms: one program, swappable parts

This folder holds small, tested reference code. It shows that three fast algorithms are **one program
with swappable parts**:

- **HBS-style recursive skeletonization**, which uses *weak* admissibility.
- **FMM-LU-style strong recursive skeletonization**, which uses *strong* admissibility.
- **The 1D butterfly**, which uses *complementary* admissibility.

The design document is [`SPEC.md`](SPEC.md). The math background is in `research/02_fmm_lu.md` and
`research/03_butterfly.md`.

## One algorithm, swappable parts

| Part | Choices | Where |
|---|---|---|
| tree | uniform 2^d-ary tree over points (1D or 2D) | `core/tree.py` (`Tree`, `Node`) |
| admissibility | `Weak`, `Strong`, `Complementary` | `core/admissibility.py` |
| compression | `interp_decomp` (one ID) fed by `ExactCompressor` or `ProxyCompressor` | `core/compress.py` |
| algebra | factor (eliminate box by box) or apply (butterfly) | `drivers/rskel.py`, `drivers/butterfly_id.py` |

All three algorithms repeat the same step at every box (or tile):

1. **Ask the admissibility rule** which interactions count as "far".
2. **Run one interpolative decomposition (ID)** against that far field. The ID splits the box's
   unknowns into *skeleton* and *redundant* unknowns. It also returns an *interpolation matrix T* with
   `far-field(redundant) ≈ far-field(skeleton) @ T`.
3. **Use T.**
   - The **factor** drivers sparsify with T. They then eliminate the redundant unknowns with a small
     **Schur complement**, so that only skeletons pass up to the parent box.
   - The **butterfly** stores T as one block of a sparse factor. Its "boxes" are **tiles**: a row box at
     depth l paired with a column box at depth L − l. Each tile is one phase-space cell.

The three variants differ only in the admissibility rule:

| | near field kept exactly | compressed against |
|---|---|---|
| `Weak` (HBS) | nothing | everything outside the box, including the touching neighbours |
| `Strong` (FMM-LU) | the touching neighbours | only well-separated boxes |
| `Complementary` (butterfly) | — | the paired box of the *other* tree |

## How to run everything

Run all commands from the repository root. They use the project venv.

```bash
# all tests: core, the driver authors' tests, and the black-box spec tests (≈ 1 min)
.venv/bin/python -m pytest -q unified/tests

# include N = 4096 in the black-box spec tests (≈ 1 min more)
RUN_SLOW=1 .venv/bin/python -m pytest -q unified/tests/test_spec_rskel.py unified/tests/test_spec_butterfly.py

# benchmarks: PNG + JSON written to unified/bench/results/ (≈ 1 min each; run_rskel peaks at ≈ 4 GB RAM)
.venv/bin/python -m unified.bench.run_rskel        # P1, P2 scaling to N = 16384; P3 rank vs k  (--max-n 4096 for less memory)
.venv/bin/python -m unified.bench.run_butterfly    # B1 error and storage vs N, ID vs SVD
#   add --quick to either script for N <= 1024
```

### Files

| File | What it is |
|---|---|
| `tests/test_core.py` | Tests of the frozen core (tree, ID, compressors, admissibility). |
| `tests/test_rskel.py`, `tests/test_butterfly_id.py` | The driver authors' own tests. |
| `tests/test_spec_rskel.py`, `tests/test_spec_butterfly.py` | **Black-box** tests. They were written from `SPEC.md` before the drivers existed. They cover the SPEC acceptance criteria and the following edge cases: clustered points and empty boxes, deep trees on a curve, N not a power of 2, a depth-0 tree, a tiny N, nonsymmetric row and column scaling, complex Helmholtz matrices, eps sweeps, reuse and linearity of `solve`/`apply`, and point-source columns. |
| `tests/test_spec_problems.py` | Checks that the test matrices themselves match the SPEC definitions. |
| `bench/problems.py` | The SPEC problems P1, P2, P3 and B1, plus a clustered-points problem. The tests and the benchmarks share this file. |
| `bench/adapters.py` | The one place that maps loosely specified driver details (stats keys, butterfly constructor) to the tests. |
| `bench/run_rskel.py`, `bench/run_butterfly.py`, `bench/common.py` | The benchmarks and their plot style. |

### Test problems (from SPEC)

| Name | Definition |
|---|---|
| **P1 curve** | Starfish `r = 1 + 0.3 cos 5θ`. Matrix `A = I + K w`, with K the 2D Laplace kernel and w the trapezoid weights. |
| **P2 area** | N uniform random points in the unit square. Matrix `A = I + K / N`. |
| **P3** | P1 with the Helmholtz kernel `(i/4) H0(k r)`. This makes A complex. |
| **B1** | The 1D Fourier integral operator `exp(2πi(x ξ + t c(x)|ξ|))`. At t = 0 it is a shifted inverse DFT. |

For the proxy compressor, the quadrature weights of P1 and P3 are folded into the kernel function. The
starfish is star-shaped, so θ can be recovered from a point as θ = atan2(y, x).

## Results and claims

Every row below is labelled with how it is known:

- **proved**: follows from a short argument given here.
- **computed**: measured by this repo's tests or benchmarks. The numbers come from
  `bench/results/*.json`, measured on one laptop. They show *trends* and are not portable timings.
- **cited**: a statement from the literature that this repo does not re-prove. Sources are in `research/`.

| # | Claim | Label | Evidence |
|---|---|---|---|
| 1 | The B1 kernel at t = 0 equals `(-1)^j N · ifft`. | **proved** | Substitute: `x_j ξ = j(m − N/2)/N`, so `e^{2πi x_j ξ} = (−1)^j e^{2πi jm/N}`. Checked in `test_b1_t0_is_shifted_inverse_dft`. |
| 2 | Weak and strong are the same code. Weak is strong with an empty near list. | **proved** (by construction) | `Weak.near_nodes` returns `[]`. `rskel_factor` takes the rule as an argument. |
| 3 | rskel meets residual ≤ 100·eps on P1 and P2, for N = 512…4096 (tests) and up to N = 16384 (benchmark), with both rules and both compressors (eps = 1e-8). | **computed** | `test_spec_rskel.py` with `RUN_SLOW=1`. The benchmark residuals (proxy compressor) are 2e-13 to 1.5e-10 for N = 1024…16384. |
| 4 | rskel also meets residual ≤ 100·eps for: nonsymmetric `D1 A D2`, complex P3 (k = 5, 20), clustered points, N ∈ {777, 1001, 1500}, depth-0 trees, and eps from 1e-4 to 1e-10. | **computed** | `test_spec_rskel.py` |
| 5 | **The headline difference.** On P2 (area), the max skeleton per box at level 2 grows with N under weak but stays nearly flat under strong. For N = 1024 / 4096 / 16384, weak is 55 / 111 / 235 (≈ N^0.52, like the box perimeter in points) and strong is 36 / 40 / 44. | **computed** | `rskel_scaling.json` (`skel_max_per_level`) |
| 6 | At the finest two levels on P2, the strong max skeleton stays at 25–38 for every N. The weak max skeleton at the next-to-finest level reaches 55–86. The finest-level curves zigzag because the depth only increases every 4× in N, so leaf occupancy alternates. | **computed** | `rskel_scaling.png`, bottom right |
| 7 | Root size is a secondary number and misleads when comparing rules. Under strong in 2D the four level-1 boxes have no far field, so they are not compressed, and the root holds all 16 level-2 skeletons by construction. The P2 root goes 314 → 1055 (weak) and 517 → 684 (strong) for N = 1024 → 16384. Strong is larger up to N = 4096 and smaller from N = 8192 on. | **computed** + explanation | `rskel_scaling.json` |
| 8 | On P1 (curve), strong ranks are flat or decrease with N: level 2 stays at 22–23 and the finest level goes 16 → 13. Weak level-2 ranks grow slowly (28 → 37). The roots are nearly flat (weak 171 → 199, strong 227 → 233). | **computed** | `rskel_scaling.json` |
| 9 | On P3, skeleton ranks grow with k. At N = 2048, the max skeleton per box at level 2 for k = 5, 20, 50 is 31 → 38 → 47 (weak) and 24 → 31 → 43 (strong). | **computed** | `rskel_helmholtz.json` |
| 10 | The weak (HBS) rule gives an O(N^{1/2}) root for area problems in 2D, hence an O(N^{3/2}) factorization. The strong rule gives bounded or polylog ranks and roughly O(N) cost for non-oscillatory kernels. | **cited** | Ho–Greengard 2012; Ho–Ying 2016; Minden–Ho–Damle–Ying 2017; Sushnikova et al. 2023 (`research/02_fmm_lu.md`) |
| 11 | The ID butterfly meets error ≤ 10·eps on B1 for N = 256…4096 and t ∈ {0, 1, 3}. At eps = 1e-8 the worst error is 4.4e-9. At t = 0 it matches `numpy.fft`. | **computed** | `test_spec_butterfly.py`, `butterfly.json` |
| 12 | ID butterfly storage is about 0.55–0.6× that of the SVD reference `code/butterfly.py` at the same eps. For example, at N = 4096, t = 3: 2.39M vs 4.04M nonzeros, against 16.8M for dense. | **computed** | `butterfly.json` |
| 13 | Storage grows far more slowly than dense, but somewhat faster than N log N at these sizes. At t = 3, nnz grows 7.3× from N = 256 to 1024 and 5.9× from 1024 to 4096. N log N predicts 5× and 4.8×; dense grows 16×. The asymptotic N log N is cited, not demonstrated here. | **computed** (trend); **cited** (asymptotics) | `butterfly.json`, `test_storage_is_n_log_n` (asserts < 8× per 4N); O'Neil–Woolfe–Rokhlin 2010; Pang–Ho–Yang IDBF (`research/03_butterfly.md`) |
| 14 | Butterfly tiles (complementary pairs) have rank bounded independently of N for Fourier integral operators. | **cited** | Candès–Demanet–Ying 2009; Michielssen–Boag 1996 (`research/03_butterfly.md`) |

### Caveats

- **Times are not a performance claim.** `rskel` keeps a dense N × N working matrix, which the SPEC
  allows for clarity. The benchmark passes `overwrite_a=True` and recomputes residuals from the kernel.
  The fitted log-log slopes of factor time against N, for N = 1024…16384, are about 1.03 on P1 (both
  rules) and 1.49 (weak) / 1.53 (strong) on P2. Over the largest three N they are 1.2 and 1.55–1.6.
  The P2 slopes are the same for both rules, even though the strong ranks are bounded. So the time is most likely
  dominated by dense-matrix bookkeeping, not by the rank-dependent work (not profiled). Do not read these slopes as
  O(N) vs O(N^{3/2}) evidence; the ranks in claims 5–6 are the evidence.
- **The benchmarks use one compressor.** They use `ProxyCompressor`. The tests use both compressors.

## Figures

- `bench/results/rskel_scaling.png` has two rows, P1 (top) and P2 (bottom). Each row has four columns,
  all plotted against N for N = 1024…16384, weak vs strong: factor time with its fitted slope, solve
  time, max skeleton per box at level 2, and max skeleton per box at the finest two levels (solid =
  finest, dashed = finest − 1). Root sizes appear only as a footnote.
- `bench/results/rskel_helmholtz.png` shows the max skeleton per box at each tree level for k = 5, 20, 50
  (weak and strong), and the level-2 rank against k.
- `bench/results/butterfly.png` has two panels. The left panel shows the ID butterfly's error against
  N for t = 0, 1, 3, with the SPEC bound 10·eps marked. The right panel compares the storage (nnz) of
  the ID butterfly, the SVD reference and the dense matrix.

## Showcase: ten physics models, one harness (`showcase/`)

`showcase/models.py` is a catalogue with one typed entry per physics problem. Each entry holds: the physics
in one sentence, references, the kernel, the dial settings, `build` and `check`. The models are Laplace,
Helmholtz with the E5 congruent fix, the DFT as an FFT butterfly, a Fourier integral operator, Stokes
particles (E6), Yukawa and the heat equation (E7), the fractional Laplacian with the lifted proxy (E8),
Lippmann–Schwinger (E9) and the spherical harmonic transform (E10). Every model calls the unchanged
`rskel_factor` or `butterfly_id`.

```bash
.venv/bin/python -m unified.showcase.reproduce --quick   # all ten models, small sizes (seconds)
.venv/bin/python -m unified.showcase.reproduce           # full sizes (≈ 2 min, ≈ 5 GB peak RAM)
.venv/bin/python -m pytest -q unified/showcase           # quick-mode smoke test per model
```

The script prints one table and writes `showcase/results.json` and `showcase/RESULTS.md`. The table has
these columns:
- the matched result and its reference;
- lines of solver-dial code (kernel, sampler or proxy override, wrappers, forked driver);
- lines of physics set-up (formulation, quadrature, geometry, analytic reference);
- the error against the analytic answer or invariant, with the dense solve's own error alongside;
- fast vs dense;
- time;
- PASS/FAIL.

Line counts are computed with `inspect.getsource`. They exclude blank lines, comments and docstrings, and
are never typed in by hand.

**What "matched" means.** Each model says which of three kinds of match it uses:
- **exact solution**: a closed-form solution of the continuous problem. Agreement is limited by the
  discretization, not by ε.
- **invariant**: an identity that must hold, such as Parseval or the FFT entry count.
- **published behaviour**: for example, bounded butterfly tile rank.

No row reproduces a table from a paper number for number. Failed checks are reported as FAIL or ERROR,
never dropped.

Two tolerances deserve a note:
- **Helmholtz.** The analytic tolerance was set after the first run, because no prior error fit exists for
  this kernel. The binding test is the observed O(h³) convergence order, which was not tuned.
- **Yukawa.** Quick mode first failed the order test at n = 1024, because the error there is still
  pre-asymptotic. Quick mode now uses n = 2048. Both cases are recorded in the model's `note`.
