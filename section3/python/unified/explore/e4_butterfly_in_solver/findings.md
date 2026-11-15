# E4: the butterfly inside a solver (two swaps)

Reproduce: `.venv/bin/python -m unified.explore.e4_butterfly_in_solver.run` (≈ 5 min; `--quick` caps N at 1024).
Tests: `.venv/bin/python -m pytest -q unified/explore/e4_butterfly_in_solver` (7 checks).
Files: `arcs.py` (swap a), `fio_solve.py` (swap b), `results/*.json|png`, `results/run_log.txt`.

## (a) Compression swap: one far block of 2D Helmholtz stored as a low-rank ID or as a butterfly

**Setup.** Take two well-separated pieces of curve: target and source, each with N points at 10 points per
wavelength, so k = 2πN/(10·length). The far block is K_ij = (i/4)H0(k|x_i − y_j|). We compare two storage formats:
- **low rank**: one interpolative decomposition (ID) of the whole block, `K ≈ K[:, skel] [I | T]`, storing
  m·r + r(n−r) numbers. This is what strong-admissibility skeletonization keeps for a far block.
- **butterfly**: the repo's ID butterfly (`drivers/butterfly_id.py`), unchanged, applied to the block viewed as
  a kernel of the two 1D **arc parameters**. Its complementary tiles pair a short target sub-arc with a long
  source sub-arc. We count its nnz *minus* the identity entries of each [I | T], so both formats are counted alike.

Fairness: for each format we pick the largest ID tolerance whose **achieved** relative Frobenius error
(Gaussian probe with 8 vectors) is ≤ 1e-6. Butterfly errors add up over log2 N stages; a single ID's do not.
Geometries: quarter arcs in opposite quadrants of the unit circle (`opposite_arcs`); parallel unit segments
at distance 1 (`parallel_segments`); a control case of collinear segments, where k|x−y| = k(x−y) separates,
so the rank stays O(1) (`collinear_segments`).

**Hypothesis (before running).** The low-rank rank grows like k·(aperture), so the ID stores ~cN², while the
butterfly stores O(r N log N) with tile rank r = O(1). The butterfly should therefore win above some k of
order 10². For the collinear control it should lose at every k.

| geometry | N (k) | LR rank (SVD rank) | LR stored | BF max tile rank | BF stored | BF / LR |
|---|---|---|---|---|---|---|
| opposite arcs | 128 (51) | 21 | 4.9e3 | 4 | 8.2e3 | 1.66 |
| | 512 (205) | 52 | 5.1e4 | 4 | 4.9e4 | **0.97** |
| | 1024 (410) | 92 | 1.8e5 | 4 | 1.1e5 | 0.64 |
| | 4096 (1638) | 325 | 2.56e6 | 4 | 5.9e5 | **0.23** |
| parallel segs | 512 (322) | 57 | 5.5e4 | 4 | 4.9e4 | 0.89 |
| | 4096 (2574) | 360 | 2.82e6 | 4 | 5.9e5 | 0.21 |
| collinear segs | 4096 (2574) | 4 | 3.3e4 | 2 | 1.8e5 | 5.50 (LR wins) |

All achieved errors are in [1e-10, 5.6e-7]. The full table is in `results/compression.json`.
**Crossover** (log-log interpolation): **k ≈ 190** (N ≈ 480) for opposite arcs, **k ≈ 250** (N ≈ 400) for
parallel segments, and none for the collinear control. The low-rank rank grows like k^0.84 over k ≥ 100;
the local slope tends to 1 (171 → 325 when k doubles from 819 to 1638). The butterfly tile rank stays at 4
for every k, and its storage grows like N log N (×2.25 per doubling).
**Conclusion.** In a high-frequency H-matrix, far blocks above k·diam ≈ 200–300 should be stored as
butterflies. This matches Michielssen–Boag 1996 (MLMDA) and Liu et al. 2021 (butterfly-compressed
admissible blocks, ButterflyPACK). The swap fails when the block's phase is (nearly) separable:
collinear or grazing geometry, or low k.

## (b) Algebra swap: the butterfly as an approximate inverse for GMRES on the 1D FIO

**Setup.** K[x, ξ] = a(ξ)·exp(2πi(xξ + t c(x)|ξ|)), the B1 kernel with an optional damping
a(ξ) = exp(−σ|ξ|/(N/2)). We solve K x = f with f = K x_true, using full GMRES with a right preconditioner M
(so the GMRES residual is the true residual) and tolerance 1e-8. Every product with K and with K* goes
through the butterfly factors (eps 1e-10); K* = W* G0* … U* costs one more O(N log N) pass. Preconditioners:
- `adjoint`: M = K*/N.
- `jacobian`: M = (1/N)[Π₊K*D₊ + Π₋K*D₋], with D± = diag(1 ± t c′(x)) and Π± restricting to ξ > 0 or ξ < 0.
  This is the adjoint with the Jacobian of the canonical map x ↦ x ± t c(x) folded in. It uses the identity
  Σ_x Φ′ e^{2πi(η−ξ)Φ(x)} ≈ N δ_ξη, valid while Φ is monotone.
- `parametrix`: `jacobian` followed by multiplication by 1/a(ξ)², which assumes the damping is known.

The caustic: 1 − t c′(x) = 0 has a solution iff t ≥ 16/(2π) ≈ 2.55.

**Hypothesis.** K/√N is close to unitary, so K*/N is a good approximate inverse until (i) the canonical map
folds (a caustic), or (ii) damping makes K non-normal and ill-conditioned.

| case (N = 1024 unless stated) | cond(K) | none | adjoint | jacobian | parametrix |
|---|---|---|---|---|---|
| t = 0, σ = 0 | 1.0 | 4 | 1 | 1 | 1 |
| t = 0.5 / 1 / 2, σ = 0 | 1.2 / 1.5 / 3.1 | >200 | 9 / 12 / 27 | **3 / 3 / 4** | 3 / 3 / 4 |
| t = 1, σ = 0, N = 256 / 1024 / 4096 | — | 256 / >300 / >300 | 12 / 12 / 12 | **3 / 3 / 3** | — |
| t = 2.5 (near caustic), σ = 0 | 10 | >200 | 41 | 10 | 10 |
| t = 3 (caustic), σ = 0 | 4.5e14 | >200 | 179, x err 9e-2 | 86, x err 9e-2 | 86 |
| t = 3, N = 256 / 1024 / 4096 (jacobian) | — | — | — | 40 / 86 / 106 | — |
| t = 1, σ = 1 / 3 / 6 | 4.1 / 29 / 580 | >200 | 36 / 187 / >200 | 25 / 162 / >200 | **3 / 3 / 3** |

(">200" means not converged at the cap. "x err" is ‖x − x_true‖/‖x_true‖, which is ≤ 3e-7 in every
converged non-caustic case.)
**Why.** Dense spectra at N = 512 (`results/gmres.json`): for t = 1, `jacobian` leaves only **8** eigenvalues
of KM with |λ−1| > 1e-3, while `adjoint` leaves all 511. At t = 2.5 / 2.6 / 3 the number of outliers grows
to 30 / 32 / 67. Without a preconditioner, the eigenvalues of K/√N spread around a circle centred at 0,
so GMRES needs ~N steps. It converges at N = 256 only because of finite termination (exactly 256 steps).

**Where it wins / fails.** It wins away from caustics (t ≤ 2): N-independent iteration counts, 3 with the
Jacobian-weighted adjoint and 9–27 with the plain adjoint. Each iteration is two butterfly passes, so the
solve costs O(N log N). It fails in two ways. (1) At a **caustic** the operator itself is near-singular
(cond 1e14). GMRES drives the residual to 1e-8, but the solution is wrong (error 9e-2), and the iteration
count grows with N. No preconditioner can fix this: the problem is ill-posed. (2) Under **damping**,
K*/N squares the amplitude, giving iterations ~160–200 at σ = 3. The plain adjoint degrades even though
cond(K) is only 29. Dividing by the known a(ξ)² restores 3 iterations up to σ = 6 (cond 580), with
solution error ≤ 3e-7.

## Claims

| claim | label |
|---|---|
| At error ≤ 1e-6 and 10 points per wavelength, the butterfly beats the low-rank ID for a separated far block above k ≈ 190–250, and stores 4–5× less at k ≈ 1600–2600 | computed |
| Low-rank ID rank grows ~k (slope 0.84 → 1); butterfly tile rank stays at 4; butterfly storage grows ~N log N | computed |
| Collinear (separable-phase) blocks: low rank wins at every k (5.5× less at k = 2574) | computed |
| Butterfly-compressed admissible blocks in H-matrices at high frequency | cited: Michielssen–Boag 1996; Liu et al. 2021 (SISC 43, arXiv:2002.03400) |
| Jacobian-weighted adjoint = FIO parametrix; KM has O(1) outlier eigenvalues (8 at N = 512, t = 1) | derived here (standard FIO parametrix idea) + computed |
| GMRES iterations are N-independent for t ≤ 2 (adjoint 12, jacobian 3 at t = 1, N = 256…4096) | computed |
| Caustic (t ≥ 2.55): K is near-singular, the solve is inaccurate whatever the preconditioner | computed (cond 4.5e14 at t = 3) |
| Damping breaks K*/N; dividing by the known amplitude fixes it | computed |

**Next.** (1) Insert butterfly far blocks into `rskel` for P3 at k ≥ 200. Not done: the factorization needs
the block's ID skeletons, which a butterfly does not provide; Liu et al. use a butterfly-LU, which is a
bigger change. (2) Use the parametrix in a hierarchical-butterfly LU preconditioner (Liu–Yang 2019,
arXiv:1901.11371). (3) At caustics, use Tikhonov regularisation or a splitting in t instead of a better
preconditioner.
