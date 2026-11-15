# E8: Fractional Laplacian and Riesz kernels in the unified harness

**Result.**
- The harness solves the fractional Poisson problem with **one dial changed**: dial 4, sampling.
- The stock proxy surface **fails** under strong admissibility.
  - It errs by 1e-4 to 4 in 2D and by 1e-2 to 0.16 in 1D, at every fractional order s.
  - It is exact only at the two exponents where the kernel *is* the Green's function of R^d.
- The fix is to **lift the proxy circle into one extra dimension**: proxy points go on a hemisphere in R^{d+1}.
  - It uses the fact that |x|^{-p} is the Laplace Green's function of R^{p+2} (the Caffarelli–Silvestre extension picture).
  - The change is about 20 lines: one `proxy_points` override, plus two lines that let the kernel accept points with an extra coordinate.
  - It reaches ε = 1e-8 or better for every s, both kernels, d = 1 and 2.
  - Cost: 2.3× the circle's proxy points (2 latitudes) to 6× (8 latitudes), and 30–40% more skeletons than the exact ID.
- A second, independent breakage is **conditioning, not sampling**.
  - The hypersingular matrix has cond(A) ~ N^{2s/d}.
  - Under *weak* admissibility the solve error is ≈ cond(A)·ε for *every* compressor, exact included. In 1D at s = 0.5, N = 8192 that is 1.5e-2 at ε = 1e-8.
  - Strong admissibility does not suffer from this.

**Follow-up (R6): equal rank vs equal tolerance.**
- When every compressor is given the **exact ID's per-box ranks**, every proxy loses 2–4 digits under strong admissibility (worst per-box far-field error 2e-6 to 2e-4). The 15–45% extra skeletons a proxy keeps at equal tolerance are therefore **needed**, not wasted.
- At equal rank in 2D, the lifted hemisphere is the most accurate proxy: 4e-8 … 1e-6, against 1e-7 … 1e-6 for the annulus and 4e-7 … 8e-6 for a Xing–Chow-style selected volume proxy. In 1D no proxy wins clearly: the volume proxies are mostly 1–3× better, except at Riesz s = 0.4.
- At equal tolerance all three proxies reach ε in 2D. Lifted needs 236 points and no selection step. Xing–Chow-style needs 93–150 points plus a one-time selection costing 2.3–11× all of its IDs in 2D (unoptimized).
- A second, more targeted literature search found no prior lifted / extension-based proxy surface; quotes are in (f).

Everything is reproduced by `.venv/bin/python -m unified.explore.e8_fractional.run`:
- it takes about 380 s on a laptop: 321 s for R1–R5 plus 56 s for R6 (`--quick` is faster; `--only r6` runs the follow-up alone);
- data: `results/r1..r5*.json`; figures: `results/r*.png`; log: `results/run_log.txt`.

Checks:
- `.venv/bin/python -m pytest -q unified/explore/e8_fractional`: 9 tests;
- `.venv/bin/mypy unified/explore/e8_fractional`: clean.

---

## (a) The physics problem, in plain words

**Ordinary diffusion.** Heat or a dye spreads by many tiny random steps (Brownian motion). Its density obeys the heat equation ∂u/∂t = Δu. The steady state with a source f solves Poisson's equation −Δu = f. The Laplacian Δ is *local*: Δu(x) only needs u near x.

**Anomalous diffusion.** Some particles occasionally make *long jumps*. The probability of a jump of length r decays only like a power, r^{-(d+2s)}, with 0 < s < 1. Such random walks are called Lévy flights. They model:
- foraging animals;
- tracer transport in turbulent and porous media;
- charge transport in disordered semiconductors;
- heavy-tailed price models in finance.

The generator of a Lévy flight is the **fractional Laplacian**:

  (−Δ)^s u(x) = C_{d,s} · PV ∫ (u(x) − u(y)) / |x − y|^{d+2s} dy.

Here PV ("principal value") means the integral is taken symmetrically around y = x. The constant C_{d,s} makes the operator act on Fourier modes as |ξ|^{2s}. At s = 1 it becomes the ordinary −Δ; as s → 0 it tends to the identity.

**Why it is nonlocal.** (−Δ)^s u(x) depends on u *everywhere*. Discretizing it on N points gives a **dense N×N matrix**: every point interacts with every other one. This is a volume problem, not a boundary problem: the unknowns fill the domain.

**The test problem.** The fractional Poisson problem (the mean time a Lévy flight takes to leave the ball):

  (−Δ)^s u = 1 in the unit ball B ⊂ R^d,  u = 0 outside B.

It has a known exact solution (Getoor 1961; Dyda 2012):

  u(x) = Γ(d/2) / (4^s Γ(1+s) Γ(d/2+s)) · (1 − |x|²)^s.

For d = 1, s = ½ this is exactly the semicircle √(1−x²). The solution behaves like (distance to boundary)^s, which is typical for fractional problems (Ros-Oton & Serra 2014). That boundary behaviour is what makes these problems numerically hard.

**The second kernel.** The **Riesz potential** |x − y|^{-(d−2s)} is the kernel of the *inverse* operator (−Δ)^{-s} on all of R^d. It is the "fractional Coulomb potential" of Riesz gases. We use it in a second-kind test operator A = I + h^d|x−y|^{-(d−2s)}, the analogue of SPEC P2. This is a compression probe, not a physical model.

Both kernels belong to one family, k_p(r) = r^{-p}:
- p = d + 2s for the fractional Laplacian (the "hypersingular" kernel);
- p = d − 2s for the Riesz potential.

**Discretization (`fractional.py`).** We use lattice points x_i = h·z_i inside B. A Riemann sum over the *whole* lattice is valid because u = 0 outside B. After dividing by C_{d,s}h^{-2s} it gives

  A = Z_d(s) I − [(h/|x_i − x_j|)^{d+2s}]_{i≠j},  b = h^{2s}/C_{d,s}.

- Z_d(s) = Σ_{z≠0} |z|^{-(d+2s)}. In closed form, Z₁ = 2ζ(1+2s) and Z₂ = 4ζ(1+s)β(1+s), with β the Dirichlet beta function.
- **Taylor correction.** The singular cell is corrected by adding −κ_d(s)·h²(−Δ_h), where Δ_h is the 5-point (2D) or 3-point (1D) discrete Laplacian. Here κ₁ = ζ(2s−1) and κ₂ = ζ(s)β(s) are zeta-regularized lattice sums. This lifts the local consistency error from O(h^{2−2s}) to roughly O(h).
- The correction changes only **nearest-neighbour** entries. All other off-diagonal entries are still kernel values, which the proxy compressor needs.
- A is symmetric and strictly diagonally dominant, hence SPD (proved: the row sums of the off-diagonal part are below Z_d).

## (b) Predictions before running: which dials move

The predictions are written out in `hypotheses.md`. That file has an honesty note: two smoke tests had been run before it was written.

| Dial | Prediction | Reason |
|---|---|---|
| 1 tree | unchanged | The kernel is translation-invariant and asymptotically smooth; a uniform quad/binary tree is right. |
| 2 admissibility | unchanged; both Weak and Strong usable | Far-field ranks of r^{-p} are bounded under strong and grow polylog/√N under weak, as for Laplace. |
| 3 representation (one ID, skeletons) | unchanged | The matrix is real and symmetric, so E5's complex-T instability cannot occur. |
| **4 sampling** | **must change** | The stock proxy (a circle in 2D, the two points c ± r in 1D) relies on Green's representation *in R^d*. r^{-p} solves no PDE in R^d unless p = 0 (2D) or p = −1 (1D). |
| 5 algebra (factor) | unchanged | The matrix is SPD and diagonally dominant, so pivots are benign. |

**Why lifting should work (dial 4).**
- Write m = p + 2 − d.
- For X = (x, t′) ∈ R^d × R^m, the function |X|^{-p} = |X|^{2−(p+2)} is the Laplace Green's function in R^{p+2}.
- It depends on t′ only through t = |t′|. So a sphere in R^{p+2} around a box reduces to a **hemisphere {(z, t) : |z − c|² + t² = r², t ≥ 0} in R^{d+1}**.
- Suppose a combination Σc_j|X − y_j|^{-p} of box sources vanishes on that sphere. Then it is harmonic outside the sphere, zero on it, and decays at infinity (p > 0). By the exterior maximum principle it vanishes everywhere outside, in particular at the far points, which sit at t = 0.
- So ID rows sampled on the hemisphere control the true far field.
  - **Proved** when m is an integer.
  - For fractional m this is the weighted operator div(t^{m−1}∇·) of the Caffarelli–Silvestre extension. **Conjecture** there, especially for the hypersingular kernel: its weight exponent m − 1 = 1 + 2s lies outside the A₂ class where the standard theory lives.
- To the kernel, a lifted point is simply a point with an extra coordinate t ≥ 0. Real points are embedded at t = 0.

## (c) The change and its size

| File | What | Lines | Dial |
|---|---|---|---|
| `proxies.py` `LiftedProxy.proxy_points` | hemisphere (2D) / semicircle (1D) proxy points | 18 executable (25 with docstring) | 4 |
| `fractional.py` `power_kernel` | pad the shorter point array with t = 0 (two `np.pad` lines) | 2 beyond a plain kernel | 4 (kernel interface) |
| core, drivers, tree, admissibility, ID, factor | none | 0 | 1, 2, 3, 5 |

- `LiftedProxy` subclasses the core `ProxyCompressor`. The driver's FMM-LU Q-set path calls `comp.proxy_points(node)` and `comp.kernel(Pp, X[box])`, so it works without modification.
- The rest of the directory is experiment code: physics, harness, sweeps, tests.
- `AnnulusProxy` (24 lines) is a comparison: the kernel-agnostic volume proxy.

## (d) Verification

**Against the exact solution (R1).** The table gives the relative 2-norm error of the dense solve, with u_exact = c(1−|x|²)^s. "fast vs dense" is the error of strong RS with the lifted proxy at ε = 1e-8, measured against the dense solve.

| d, s | N | raw scheme | corrected | corrected, \|x\| < ½ | fast vs dense |
|---|---|---|---|---|---|
| 1, 0.25 | 256 → 8192 | 4.7e-3 → 3.5e-4 | 5.9e-3 → 4.4e-4 | 1.7e-3 → 5.3e-5 | ≤ 1.6e-10 |
| 1, 0.5 | 256 → 8192 | 1.7e-3 → 1.2e-4 | 4.7e-3 → 1.8e-4 | 2.4e-3 → 7.5e-5 | ≤ 3.1e-9 |
| 1, 0.75 | 256 → 8192 | 7.0e-2 → 1.2e-2 | 2.8e-3 → **9.1e-5** | 2.0e-3 → 6.1e-5 | ≤ 2.4e-7 (cond-limited, see (e)) |
| 2, 0.25 | 253 → 4093 | 5.9e-2 → 2.3e-2 | 4.8e-2 → 2.0e-2 | 8.5e-4 → 9.2e-4 | ≤ 1.5e-12 |
| 2, 0.5 | 253 → 8197 | 1.2e-1 → 3.1e-2 | 6.0e-2 → 1.6e-2 | 1.2e-2 → 3.7e-3 | ≤ 2.7e-12 |
| 2, 0.75 | 253 → 4093 | 4.5e-1 → 1.9e-1 | 7.7e-2 → 2.2e-2 | 3.5e-2 → 1.1e-2 | ≤ 1.5e-12 |

**1D.** Over a 32× range of N, the corrected scheme converges at order 0.75 / 0.93 / 0.99 in h for s = 0.25 / 0.5 / 0.75.
- The raw scheme converges at order 0.50 for s = 0.75, which is h^{2−2s} as predicted.
- At s = 0.75 the correction gains a factor 130 at N = 8192.
- At s = 0.25 the raw error is slightly smaller in absolute value, with the same rate. There the singular cell is not the bottleneck; the boundary layer u ~ dist^s is.

**2D.** Convergence is slow and not monotone in N (about h^{0.8} for s = 0.5).
- The lattice "staircase" disk changes shape with h. That is a boundary-geometry error of size O(h^s)-ish (conjecture).
- The interior error (|x| < ½) is 3–10× smaller.
- The fast solver matches dense to ≤ 3e-12 in every 2D case. So whatever discretization error remains is physics, not compression.
- For N ≤ 1033 in 2D, "fast vs dense" is ~1e-16 because the tree is too shallow (depth 3) for strong admissibility to compress anything.

**Physical invariants.**
- A is SPD (test `test_matrix_is_spd`).
- The exact-solution prefactor is checked at d = 1, s = ½, where it equals 1 (`test_exact_solution_constant`).
- Z₂ in closed form matches the brute-force lattice sum to 5e-7 (`test_lattice_constants`).

## Main results: does the surface proxy fail, and does lifting fix it? (R2, N = 2048, ε = 1e-8)

Solve error against dense (worst value over the s sweep):

| kernel, d | s values | rule | exact | **plane (stock)** | **lifted (new)** | annulus (volume) |
|---|---|---|---|---|---|---|
| hyper, 1D | 0.1–0.9 | strong | 3e-10 … 3.7e-6 | **2.7e-2 … 1.6e-1** | **2.7e-11 … 2.7e-6** | 7.7e-9 … 2.2e-5 |
| hyper, 2D | 0.1–0.9 | strong | 9e-12 … 1.7e-10 | **1.5e-3 … 4.2e-3** | **3e-13 … 1.5e-12** | 6e-12 … 3.5e-11 |
| Riesz, 1D | 0.1–0.4 | strong | 7e-10 … 1.4e-9 | **1.5e-2 … 3.2e-2** | **3e-10 … 4e-10** | 3.8e-9 … 1.5e-8 |
| Riesz, 2D | 0.25–0.9 | strong | 4e-10 … 6e-10 | **3.5e-4 … 1.8e-2** | **5e-11 … 2.5e-10** | 1e-9 … 1.2e-9 |
| Riesz, 2D | 0.1 (indefinite A, cond 2e5) | strong | 6.5e-7 | **4.3** | 9.4e-9 | 4e-8 |
| hyper, 2D | 0.1–0.9 | weak | 2e-9 … 8e-7 | 2e-8 … 5e-7 | 5e-10 … 2e-7 | 1.6e-9 … 9e-7 |
| Riesz, 2D | 0.25–0.9 | weak | 1.5e-9 … 5.7e-9 | 2.8e-8 … 1.4e-7 | 7e-10 … 2.4e-9 | 1.6e-9 … 1.1e-8 |

Notes on the table:
- In the 1D hyper strong row, the upper ends at s = 0.9 are set by cond(A) ≈ 1e6, not by sampling; exact is equally affected.
- The weak 1D rows are cond-limited for every compressor and are omitted here; see (e).

Readings:
- **Plane proxy under strong admissibility fails everywhere, by 4–8 orders of magnitude** (prediction P1 confirmed).
  - In 1D it is catastrophic: only 2 proxy points, so the far-field rank is capped at about 2 per direction (P2 confirmed).
  - For Riesz in 2D the plane error *shrinks* as s → 1 (from 1.8e-2 to 3.5e-4), because p = 2 − 2s → 0 approaches the log kernel, for which the circle is exact.
- **Under weak admissibility the plane proxy is mildly wrong** (2e-8 to 1.4e-7 in 2D), as for the Gaussian in E3 (P3 confirmed): the touching neighbours lie inside the proxy circle and are kept explicitly.
- **Lifted proxy: ε reached in every case not limited by conditioning**, usually 10–100× below the exact compressor's error (P4 confirmed computationally, for all s ∈ [0.02, 0.98]).
  - That includes fractional "dimension" m = p + 2 − d, where the proof is only conjectural, and the hypersingular weight outside A₂.
- **Annulus volume proxy** (8 rings out to the domain diameter, 512 points) works in 2D. In 1D strong it misses ε for the hypersingular kernel (1e-6 to 2e-5 at s ≥ 0.5) with only 16 points (P7 partly confirmed).

**The two exact cases of the plane proxy (R5 power sweep, N = 2048).** These are exactly the two exponents where r^{-p} is the Green's function of R^d.
- 2D, p = 0 (log): plane error 4.9e-12, strong.
- 1D, p = −1 (|x|): plane error 4.0e-15. Here exact, plane and annulus all keep 256 skeletons in total.
- At every other p ∈ {−1, −0.5, −0.2, 0.2, 0.5} the plane error is 4e-4 to 1.6e-2 (strong).

**Cost (R2, 2D hyper s = 0.5, strong).**

| | rows of all ID matrices | max rank | total skeletons | root block |
|---|---|---|---|---|
| exact | 1 125 490 | 69 | 5920 | 913 |
| plane | 59 020 | 96 | 6285 | — (wrong answer) |
| lifted (8 latitudes, 398 points) | 253 080 | 139 | 7642 | 1768 |
| annulus (512 points) | 319 080 | 104 | 6958 | — |

At N = 8197 (R4), lifted uses 2.0× fewer ID flops than exact, but a 2.3× larger root block.

## Resolution of the lifted proxy (R3, strong, N = 2048)

| latitudes above the equator (2D) | 0 (= plane) | 1 | 2 | 3 | 4 | 6 | 8 | 12 |
|---|---|---|---|---|---|---|---|---|
| proxy points | 64 | 110 | 152 | 193 | 236 | 317 | 398 | 563 |
| hyper s = 0.5: error | 4.2e-3 | 1.8e-6 | **9.4e-9** | 3.0e-10 | 2.4e-11 | 6.9e-13 | 8.3e-13 | 1.1e-12 |
| hyper: max rank | 96 | 105 | 114 | 122 | 129 | 138 | 139 | 139 |
| Riesz s = 0.25: error | 1.8e-2 | 3.3e-5 | 4.5e-7 | **7.9e-9** | 3.5e-10 | 9.0e-11 | 9.0e-11 | 8.1e-11 |

**Two latitudes are enough** for the hypersingular kernel, and three for Riesz. That costs 2.4–3× the circle's 64 points, not the O(L²) of a fully resolved sphere (P5 refined).

**1D semicircle.** 9 points (8 intervals) reach 4.4e-8 for hyper and 3.2e-9 for Riesz; 17 points saturate.

**Radius factor.** Varying it from 1.75 to 3.0 trades rank (162 → 117) against nothing measurable: error stays at 1e-13 to 1e-12. Rank falls as the proxy moves out, as for Laplace.

**Lifted keeps more skeletons than exact** (+30% in total in 2D, +40% in 1D) (P6 confirmed). The ID is asked to reproduce every field harmonic in R^{p+2} outside the sphere, not only its trace on the plane t = 0. In R3 the rank grows with the number of latitudes until saturation, which shows that the off-plane directions really do cost skeletons.

*Correction after R6.* We first guessed that this surplus only buys accuracy that was not asked for. The equal-rank test below shows that most of it is **needed**: cut back to exact's ranks, the lifted ID loses 2–4 digits. Every proxy compressor pays a similar rank premium, because none of them sees the true far field.

## Rank vs s and vs N

**Rank vs s (R2, N = 2048, strong, exact compressor = intrinsic rank).**
- Hypersingular: the max rank is flat in s (2D: 66–72; 1D: 11).
- Riesz 2D: it *falls* from 62 (s = 0.1, p = 1.8) to 38 (s = 0.9, p = 0.2). The kernel approaches r⁰ = 1, which has rank 1, plus a small log part.
- So the rank depends on s mostly through how singular the kernel is, and only mildly (P8 confirmed, with the Riesz monotone decrease as a refinement).

**Rank vs N (R4, fractional Poisson s = 0.5).**

| | N | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|
| 1D strong, max rank (exact / lifted) | | 11 / 15 | 11 / 15 | 11 / 15 | 11 / 15 |
| 1D weak, max rank (exact) | | 21 | 22 | 24 | 24 |
| 2D strong, max rank (exact / lifted) | | 59 / 81 | 69 / 139 | 90 / 177 | 95 / 193 |
| 2D weak, top-level rank (exact) | | 138 | 198 | 282 | 401 |

- 1D strong: bounded. 1D weak: about log N. 2D weak: grows ×1.42 per doubling of N, i.e. √N, as for 2D Laplace (P9 confirmed).
- 2D strong: still creeping up at N = 8192 (95, still rising). The levels where the max is attained shift as N grows, so this is not a clean plateau yet (computed; the asymptotic boundedness is expected from smoothness but not demonstrated here).

## Follow-up R6: equal tolerance vs equal rank

**Setup.**
- N = 2048, ε = 1e-8, max_leaf 32. Hypersingular kernel at s ∈ {0.25, 0.5, 0.75}, Riesz 1D at s ∈ {0.1, 0.25, 0.4}, Riesz 2D at s ∈ {0.25, 0.5, 0.75}. Both rules.
- **Equal tolerance**: each compressor truncates its own ID at ε.
- **Equal rank**: the exact run's per-box ranks are copied box by box (`equalrank.fixed_ranks`); every ID is truncated at that rank instead of at ε.
  - The schedule transfers exactly, because equal child ranks give equal parent active sets.
  - `test_equal_rank_schedule_is_reproduced` checks that the ranks match box by box.
- **far err** is measured for every box against the **true** current far block of the Schur-complemented matrix W: ‖M_R − M_S T‖_F / ‖M‖_F with M = [W_FB; W_BFᵀ]. This is what the elimination actually drops.
- Compressors: exact; plane (stock); lifted (4 latitudes, 236 points in 2D, 65 in 1D); annulus (8 rings, 512 points in 2D, 16 in 1D); and **xc**.
- **xc** is a Xing–Chow-style selected volume proxy. Per tree level it takes 3000 random candidates in the far region (radii log-uniform from 2.5 half-widths to the domain diameter) and keeps the rows chosen by a pivoted QR of K(candidates, 400 box samples) at tolerance ε/10. This follows Xing & Chow 2020, Alg. 5.1, without their densification step.
- Data: `results/r6_equal_rank.json`; figure: `results/r6_equal_rank.png`.

**Strong admissibility (ranges over the 12 problems):**

| compressor | equal tol: solve err | equal tol: skeletons ÷ exact's | equal rank: solve err | equal rank: worst per-box far err | proxy points |
|---|---|---|---|---|---|
| exact, 1D | 2e-10 … 5e-7 (cond-limited at s = 0.75) | 1 | (same run) | 4e-9 … 2e-8 | — |
| exact, 2D | 9e-12 … 6e-10 | 1 | (same run) | 5e-9 … 2e-8 | — |
| plane, 1D | 1e-2 … 2e-1 | 0.67 … 0.78 | 1e-2 … 1e-1 | 1e-2 … 2e-1 | 2 |
| plane, 2D | 2e-3 … 2e-2 | 1.03 … 1.10 | 1e-3 … 6e-3 | 4e-3 … 1e-1 | 64 |
| **lifted, 1D** | **8e-11 … 5e-8** | 1.37 … 1.47 | 1e-6 … 6e-4 | 2e-5 … 2e-4 | 65 |
| **lifted, 2D** | **2e-11 … 5e-10** | 1.16 … 1.23 | **4e-8 … 1e-6** | 2e-6 … 6e-5 | 236 |
| annulus, 1D | 4e-9 … **1e-5** | 0.99 … 1.14 | 6e-7 … 2e-4 | 7e-6 … 5e-5 | 16 |
| annulus, 2D | 3e-11 … 1e-9 | 1.10 … 1.15 | 1e-7 … 1e-6 | 2e-6 … 8e-5 | 512 |
| xc, 1D | 5e-9 … 3e-6 | 1.05 … 1.19 | 1e-6 … 3e-4 | 7e-6 … 5e-5 | 9 … 12 |
| xc, 2D | 5e-11 … 5e-9 | 1.11 … 1.15 | 4e-7 … 8e-6 | 5e-6 … 2e-4 | 93 … 150 |

**Per-problem detail (2D hypersingular, s = 0.25 / 0.5 / 0.75, strong, equal rank).**
- lifted: 4e-8 / 4e-8 / 5e-8;
- annulus: 1e-7 / 1e-7 / 3e-7;
- xc: 5e-7 / 4e-7 / 8e-7;
- plane: 2e-3 / 2e-3 / 1e-3.

**Riesz 2D (s = 0.25 / 0.5 / 0.75, strong, equal rank).**
- lifted: 1e-6 / 7e-7 / 5e-7;
- annulus: 1e-6 / 8e-7 / 7e-7;
- xc: 8e-6 / 3e-6 / 1e-6.

**Cost at equal tolerance (strong, s = 0.5).** ID flops count 2·rows·|B|² over all boxes. "select" is xc's one-time pivoted-QR selection, summed over the tree levels.

| | exact | plane | lifted | annulus | xc (ID + select) |
|---|---|---|---|---|---|
| hyper 1D | 3.7e8 | 2.9e6 (wrong) | 3.6e7 | 9.3e6 (1e-6, misses ε) | 8.5e6 + 6.0e8 |
| hyper 2D | 1.6e9 | 4.4e8 (wrong) | 8.7e8 | 1.2e9 | 6.7e8 + 5.8e9 |
| Riesz 2D | 1.4e9 | 4.2e8 (wrong) | 8.2e8 | 1.2e9 | 5.4e8 + 5.8e9 |

**Readings.**
1. **The rank premium of proxies is real, and it is shared.** At exact's ranks, every proxy is 2–4 orders worse than exact under strong admissibility. Its worst box drops a far-field error of 1e-5 to 1e-4 instead of about 1e-8.
   - The exact ID chooses skeletons for the far field that actually exists. A proxy ID chooses them for a larger family of possible far fields: harmonic in R^{p+2} (lifted), or arbitrary kernel fields on a volume (annulus, xc). It needs 10–45% more skeletons for the same guarantee.
   - So the lifted proxy's surplus is the price of not reading A's far blocks, not waste (computed).
2. **At equal rank, lifted is the most accurate proxy in 2D for both kernels.** For the hypersingular kernel it is 2.5–6× better than annulus and 10–16× better than xc; for Riesz it ties the annulus (within 1.4×) and is 2–8× better than xc. In 1D it is not clearly best: annulus and xc are 1–3× better at equal rank, except at Riesz s = 0.4, where lifted is 2× better. Conjecture: the off-plane directions of the semicircle carry proportionally more information that the far field on the line never uses.
3. **At equal tolerance all three volume/lifted proxies are usable in 2D. In 1D the lifted proxy is the most robust.**
   - Lifted meets ε everywhere except hypersingular s = 0.75, where it gets 5e-8. That case is cond-limited: exact gets 5e-7 there.
   - Annulus misses ε for the hypersingular kernel at s = 0.25 / 0.5 / 0.75: 3e-8 / 1e-6 / 1e-5.
   - xc misses at hypersingular s = 0.5 / 0.75 (5e-8 / 3e-6) and at Riesz s = 0.1 (4e-8).
   - These are fixable by more points, but nothing in their construction says how many.
4. **Cost.**
   - Lifted: no selection step, a fixed point set independent of the domain size, 1.8–10× fewer ID flops than exact.
   - xc: the fewest proxy points (2D: 93–150 against 236) and the cheapest IDs, but its selection QR (3000 candidates × 400 box samples per level) costs 2.3–11× (2D) and 28–106× (1D) more than all its IDs together.
   - Our selection is unoptimized. For a homogeneous kernel it could be done once on a reference box, if the far region were scaled too (it is not here: r_max is the domain diameter).
5. **Weak admissibility** compresses the regimes together. In 2D every non-plane compressor stays within 1e-6 of dense at equal rank, and the skeleton premium at equal tolerance is only 1–4% (the touching neighbours are explicit rows). In 1D everything is cond-limited, as in (e).

## (e) Where it breaks

**1. Weak admissibility on an ill-conditioned kernel: error ≈ cond(A)·ε, independent of the compressor.** This was not predicted.

Our hypersingular matrix has cond(A) ≈ 50 / 2.0e3 / 6.3e4 in 1D at s = 0.25 / 0.5 / 0.75 (N = 2048), and 7 / 40 / 210 in 2D. R5 measured the ratio err / (cond·ε) for the lifted proxy over ε = 1e-6 … 1e-12:

| | weak | strong |
|---|---|---|
| 1D | 0.16 – 6 | < 1e-4 |
| 2D | 0.004 – 0.16 | < 1e-4 |

- In 1D, weak RS with the **exact** compressor at ε = 1e-8 gives 1.5e-4 (s = 0.5, N = 2048) and 1.5e-2 (N = 8192, R4).
- Mechanism:
  - Under weak admissibility the far block of a box contains the touching neighbour, whose entries are O(1).
  - So the relative ID tolerance ε becomes an absolute perturbation ε‖A‖, which the inverse amplifies by cond(A).
  - Under strong admissibility the dropped far entries are smaller by (h/dist)^{p}, and the same ε costs far less (computed; mechanism is the standard perturbation bound, a heuristic for this setting).
- Remedy: choose ε ≈ target / cond(A), or use strong admissibility. Weak at ε = 1e-12 restores 1e-9 at s = 0.5 in 1D.

**2. s → 1 (hypersingular).**
- 1D, s = 0.95 / 0.98, strong: errors 6e-7 / 2e-6 (lifted) and 5.5e-6 / 1.5e-5 (exact).
- 2D at the same s: ≤ 6e-10.
- These are conditioning effects of the same kind (cond grows like N^{2s/d}). Lifted is still 10× better than exact.

**3. A quadrature that destroys definiteness.**
- At Riesz 2D, s = 0.1 (p = 1.8), the zero-diagonal quadrature matrix I + h²r^{-1.8} is indefinite (λ_min = −0.083, cond 2.1e5).
- Exact reaches only 6.5e-7 at ε = 1e-8; lifted reaches 9.4e-9; at ε = 1e-10 both are fine.
- Again conditioning, not sampling.

**4. Growing kernels p ≤ 0, where the maximum-principle argument is not available.**
- Lifted still works: 1e-10 to 9e-9 for p ∈ {−1, −0.5, −0.2, 0} in both dimensions.
- But at p = −1 its margin is thin (5e-9 in 1D, 9e-9 in 2D, strong).
- At 1D p = −1 the kernel |x| is exactly rank 2 per direction, yet the lifted proxy keeps **5× more skeletons** (1294 vs 256), because the off-plane field sqrt((x−z)² + t²) is not linear.
- **The lifted proxy is wasteful when the kernel is genuinely harmonic in R^d**: use the plane proxy there (computed; why it works at p < 0 is conjecture).

**5. Nothing in the s sweep breaks the lifted proxy itself.** No hard failure was found for s ∈ [0.02, 0.98], ε ∈ [1e-12, 1e-6], radius factor ∈ [1.75, 3], N ≤ 8197. Every miss of ε was shared with, and smaller than, the exact compressor's miss.

## (f) Literature: what is known and what may be new

Two searches were run. The first was general; the second (follow-up) targeted lifted and extension-based proxies and was asked to quote sources. The quotes below were transcribed from the sources by the search agent; items marked "abstract only" were not read in full.

**Proxy surfaces are tied to Green's identity in the kernel's own dimension (cited, quoted):**
- Ho & Greengard, SISC 2012 (arXiv 1110.3105):
  > "any field induced by the distant sources induces a harmonic function inside Γ and can therefore be replicated by a charge density on Γ itself"
  > "Similar arguments hold for other kernels of potential theory including the heat, Helmholtz, Yukawa, Stokes, and elasticity kernels"

  No Riesz or fractional kernels.
- Ho & Ying, HIF-IE, CPAM 2016 (arXiv 1307.2666):
  > "By Green's theorem, all off-diagonal interactions with a given box B can be represented by its interactions with an artificial local proxy surface Γ"
- Ying, Biros & Zorin, kernel-independent FMM, JCP 2004:
  > "designed to generalize FMM to second-order constant coefficient non-oscillatory elliptic partial differential equations ... Such kernels satisfy the underlying PDE everywhere but the singularity"

  Its equivalent surfaces live in R^d.
- Xing & Chow, *Interpolative decomposition via proxy points for kernel matrices*, SIMAX 2020:
  > "The proxy surface method mainly works for kernel functions from potential theory, such as the Laplace and the Stokes kernel. The motivation ... [is] based on Green's identity"
  > "For Gaussian kernels, the proxy surface method works poorly ... a heuristic and effective variant method [32] is to select the proxy points Yp in an annulus"

  Their theory covers "any kernel function K(x, y) ... smooth in X×Y" with proxy points in a *volume* of R^d. Our `xc` compressor follows their Alg. 5.1.
- Ye, Xia & Ying (arXiv 1903.08821): proxy analysis by contour integration for K(x,y) = 1/(x−y)^d with x, y in the complex plane. This is the closest power-kernel result, but it is analytic in C, not |x−y|^{-p} in R^d.

**Fast methods that already handle Riesz or fractional kernels (cited):**
- Minden & Ying, SISC 2020 (arXiv 1802.03770): FFT matvecs and CG with a finite-difference Laplacian preconditioner. No hierarchical compression of the operator.
  > "We use Chebyshev polynomials for far-field compression in the vein of Fong and Darve [16], though we do not require the full FMM machinery"

  This is for a lattice sum. The extension appears only in the review:
  > "Other notable schemes ... include work based on the Caffarelli-Silvestre extension"
- Fong & Darve, black-box FMM, JCP 2009: Chebyshev interpolation; "Results for five kernels ... (2) the kernel 1/r^4". So power kernels are handled kernel-independently, without any PDE.
- Belevtsov & Lukashchuk, Itogi Nauki i Tekhniki 176 (2020); J. Math. Sci. 275 (2023) (abstract only):
  > "A multipole expansion of the fundamental solution of the fractional degree of the Laplace operator is constructed in terms of the Gegenbauer polynomials"

  These are analytic FMM expansions. Gegenbauer polynomials are the zonal harmonics of R^{p+2}, so this is the **analytic** counterpart of our lifted proxy.
- Ambikasaran et al., TPAMI 2016 (HODLR for Gaussian processes): "will work for any covariance kernel, but the scaling ... might not be optimal; for instance, if the covariance kernel has a singularity". No fractional or Riesz work by Ambikasaran was found.
- Massei, Mazza & Robol (arXiv 1804.05522): HODLR for 2D fractional diffusion.
- Zhao, Hu, Cai & Karniadakis, CMAME 2017: H-matrix FEM.
- Ainsworth & Glusa (arXiv 1708.01923): FEM assembly.
- **Not found:** a "Yu–Lai–Ying" or "Lai–Ying" fast fractional solver. Treat any such citation as unverified.

**The extension as a computational device (cited):**
- Caffarelli & Silvestre, Comm. PDE 32 (2007).
- Nochetto, Otárola & Salgado, FoCM 2015 (arXiv 1302.0698), and Banjai et al., FoCM 2019 (abstract only): they **solve** the extended problem with tensor FEM; they do not compress with it.
- Gatto & Hesthaven, J. Sci. Comput. 65 (2015) (abstract only): hp-FEM for the extension problem, posed in "a Sobolev space with a Muckenhoupt weight". No H-matrices seen.
- Karkulik & Melenk, Adv. Comput. Math. 2019 (arXiv 1808.04274): the only place the R^{d+1} extension underpins compression, and there only as a **proof device** for H-matrix approximability of the inverse:
  > "fractional powers of certain elliptic operators posed in Rd can be realized as the Dirichlet-to-Neumann maps for (degenerate) PDEs posed in Rd+1"
- A different kind of lifting: Goodwill, Greengard, Hoskins, Rachh & Wang (arXiv 2509.05458) embed complex point clouds "in R3 in the two-dimensional case and in R5 in the three-dimensional case" so that kernel-independent FMMs apply. This is coordinate embedding, not harmonic lifting of a power kernel.

**Explicit solution and regularity (cited):**
- Getoor, Trans. AMS 101 (1961).
- Dyda, FCAA 15 (2012). Our constant matches Dyda's formula at α = 2s.
- Ros-Oton & Serra, J. Math. Pures Appl. 101 (2014).
- Huang & Oberman (arXiv 1311.7691): 1D quadrature of order h^{3−2s}.

**What may be new.** Searched and not found (MathSciNet, Google Scholar and citation graphs were not accessible, so this is a negative search, not proof):
1. A proxy *surface* for Riesz and fractional kernels placed on a hemisphere of R^{d+1}, justified by "r^{-p} is harmonic in R^{p+2}", inside recursive skeletonization.
2. The measurement that it beats both an annulus and a Xing–Chow-style selected volume proxy at equal rank in 2D, while needing no selection step.
3. The equal-rank evidence that the proxy rank premium (15–45%) is required for accuracy, for all three proxy families alike.

The weak-admissibility cond(A)·ε effect is surely folklore; we only quantify it here.

## Claims

| # | Claim | Status |
|---|---|---|
| 1 | Only dial 4 must change; tree, admissibility, ID and factor are untouched (0 lines) | computed |
| 2 | The change is `LiftedProxy.proxy_points` (18 executable lines) + 2 padding lines in the kernel | computed (line count) |
| 3 | The stock plane proxy fails under strong RS for both kernels and all s: 2D 3.5e-4 … 4.3, 1D 1.5e-2 … 1.6e-1 (ε = 1e-8, N = 2048) | computed |
| 4 | The plane proxy is exact at p = 0 in 2D (4.9e-12) and p = −1 in 1D (4.0e-15), the Green's-function cases | computed; proved (Green's representation) |
| 5 | Lifted hemisphere proxy samples control the far field for r^{-p}, p > 0, integer m = p + 2 − d | proved (exterior maximum principle in R^{p+2}) |
| 6 | Same for fractional m (incl. hypersingular weight 1 + 2s outside A₂) | conjecture; computed to hold for s ∈ [0.02, 0.98] |
| 7 | Lifted reaches ε in every non-cond-limited case, error 3e-13 … 2.5e-10 under strong | computed |
| 8 | 2 latitudes (152 points) suffice for hyper 2D at 1e-8; 3 (193) for Riesz; 9 semicircle points in 1D | computed |
| 9 | Lifted keeps 30–40% more skeletons than exact; max rank about 2× in 2D strong; ID flops 2× fewer at N = 8197 | computed |
| 10 | Weak RS error ≈ (0.004–6)·cond(A)·ε for any compressor; strong ≤ 1e-4·cond·ε | computed; mechanism heuristic |
| 11 | Strong max rank flat in s for hyper (66–72 in 2D); Riesz 2D rank falls 62 → 38 as s: 0.1 → 0.9 | computed |
| 12 | Weak 2D top rank ~ √N (138, 198, 282, 401 for N ≈ 1k … 8k); 1D strong rank constant (11) | computed |
| 13 | The Taylor-corrected lattice scheme converges at order ≈ 0.75–1 in h in 1D; raw at h^{2−2s} for s = 0.75 | computed |
| 14 | 2D accuracy against (1 − \|x\|²)^s is limited by the staircase boundary (≈ 2e-2 at N ≈ 4–8k) | computed; mechanism conjecture |
| 15 | Lifted works also for growing kernels p ≤ 0 but wastes skeletons (5× at 1D p = −1) | computed; mechanism conjecture |
| 16 | The zero-diagonal Riesz quadrature is indefinite at s = 0.1 in 2D (λ_min = −0.083) | computed |
| 17 | No prior use of a lifted (d+1)-dim hemisphere proxy found, in two searches with quotes in (f); the closest prior work is Gegenbauer FMM expansions (Belevtsov–Lukashchuk) and the extension as a proof device (Karkulik–Melenk) | cited (negative search, not exhaustive) |
| 18 | At exact's per-box ranks, every proxy (lifted, annulus, xc) is 2–4 orders worse than exact under strong admissibility; its worst per-box far error is 2e-6 … 2e-4 against ≤ 2e-8 for exact | computed (R6) |
| 19 | So the proxy rank premium at equal tolerance (2D: ×1.10–1.23; 1D lifted: ×1.37–1.47) is needed for ε, not surplus | computed (R6); this corrects the earlier "unneeded accuracy" reading |
| 20 | At equal rank in 2D, lifted is the most accurate proxy: hyper 4e-8 … 5e-8 vs annulus 1e-7 … 3e-7 vs xc 4e-7 … 8e-7; Riesz ties the annulus. In 1D the volume proxies are mostly 1–3× better (lifted is 2× better at Riesz s = 0.4) | computed (R6); 1D mechanism conjecture |
| 21 | At equal tolerance in 1D, lifted misses ε only in the cond-limited hyper s = 0.75 case (5e-8, against exact's 5e-7); annulus misses at hyper s ≥ 0.25 (up to 1e-5), xc at hyper s ≥ 0.5 (up to 3e-6) and Riesz s = 0.1 (4e-8) | computed (R6) |
| 22 | xc uses the fewest proxy points (93–150 in 2D) but its selection QR costs 2.3–11× (2D) and 28–106× (1D) all its IDs; lifted needs no selection | computed (R6, unoptimized selection) |

## Next steps
- *(Done in R6: the equal-rank test.)*
- **Rank-matched hybrid.** Lifted proxy rows plus a few true far rows (E5's hybrid idea), to approach exact's ranks without reading every far block.
- **A trace-only lifted proxy.** Weight the off-plane rows by a decaying factor in t, to recover exact's rank.
- **3D volumes.** The hemisphere becomes a 3D half-ball surface (S³ ∩ {t ≥ 0}). Check whether "2–3 latitudes" still holds.
- **Better discretization.** A Huang–Oberman or Minden–Ying quadrature, or a conforming FEM, to remove the 2D staircase error so the solver can be tested at physics-limited accuracy.
