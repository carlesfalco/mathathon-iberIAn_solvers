# E9 — Lippmann–Schwinger: wave scattering in an inhomogeneous medium, through the unchanged harness

**One-line result.** The 2D Lippmann–Schwinger (LS) volume equation runs through `rskel_factor` with **zero
changes to core or drivers**. The kernel swap is ~40 lines of matrix assembly plus a ~25-line proxy kernel.
Fast solves match a dense solve to ε, and the dense solve converges to two independent references: the
analytic Bessel series for a disk (O(h)), and a radial-ODE solution for a smooth bump (O(h²)).

Three things are new compared with the boundary problems:
1. **Weak admissibility grows like √N on area data.** Strong admissibility stays flat in N.
2. **A contrast that varies inside a box costs up to 1.85× in skeleton size.** This happens because the
   contrast multiplies only one side of the kernel. Ho–Ying's symmetrization removes the penalty
   (−42% skeletons under strong admissibility).
3. **Evaluating the contrast at the proxy points silently breaks the proxy compressor** under strong
   admissibility. The error floor is ≈ 5e-7 whatever ε is.

The E5 instability (complex T with a Tᵀ row step) is **present in the pivots** (cond up to 1e7 vs 3e3) but
costs at most ~10× in accuracy, up to k·(box half-width) = 40.

Reproduce: `PYTHONPATH=. .venv/bin/python -m unified.explore.e9_lippmann_schwinger.run` (770 s on a laptop;
`--quick` ≈ 2 min). Data are in `results/r*.json`; figures are `results/r0_verify.png`,
`results/r1_ranks.png` and `results/r3_r4_sweeps.png`. The hypotheses were written down before any code ran:
`HYPOTHESES.md`.

---

## (a) The physics problem, in plain words

**The physical setup.** A sound or light wave of fixed frequency travels through a background medium, such
as water, air or vacuum. Its spatial frequency is the **wavenumber k**, so the wavelength is λ = 2π/k.

**The object.** Somewhere in the background sits an object in which the wave travels at a different speed.
Examples: a tumour in soft tissue (medical ultrasound), a salt body in rock (seismic imaging), a glass fibre
(optics). We describe the object by its **contrast** q(x). The squared refractive index is 1 + q(x), and
q = 0 outside the object.

**The equation.** The total wave u satisfies the variable-coefficient Helmholtz equation
Δu + k²(1 + q(x)) u = 0. Far away, u looks like the incoming wave u_inc plus an outgoing scattered wave.

**Why an integral equation.** Move the q-term to the right-hand side and invert (Δ + k²) with the free-space
outgoing Green's function G_k(x) = (i/4) H₀⁽¹⁾(k|x|). This gives the **Lippmann–Schwinger equation**

  u(x) − k² ∫ G_k(x − y) q(y) u(y) dy = u_inc(x),  for x in the object.

Two features make it attractive:
- The unknowns live **only where q ≠ 0**, which is an area, not a curve.
- The radiation condition is built in exactly, with no artificial absorbing boundary.

The price: the matrix is **dense**. Every point talks to every other point.

**Why it matters.** It is the forward model inside every iterative inverse-scattering or imaging loop
(ultrasound tomography, full-waveform seismic inversion, optical diffraction tomography). That loop solves
it many times, with many right-hand sides. That is exactly what a fast **direct** solver (factor once,
solve many times) is good at.

**Discretization (Richmond 1965).** We use cell centres x_j of a uniform grid with spacing h that lie inside
the object, each with weight h². On the cell that contains the singularity, ∫G is replaced by its exact
integral over the equal-area disk, s = (iπρ/2k) H₁⁽¹⁾(kρ) − 1/k² with ρ = h/√π. The result is

  A = I + K D,  K_ij = G_k(x_i, x_j) (i ≠ j),  K_ii = s/h²,  D = diag(d),  d_j = −k² q_j h².

The contrast multiplies the kernel on **one side only** (the columns).

**Test objects.**
- **disk:** constant q₀ on r < 1. Staircase grid; analytic solution below.
- **bump:** q₀(1 − r²)³ on r < 1. It is C², and it varies inside every box.
- **square:** constant q₀ on [−1, 1]², where the grid fits exactly.

## (b) Dial predictions, stated before running (`HYPOTHESES.md`)

| dial | prediction | reason | verdict |
|---|---|---|---|
| 1 clustering | unchanged | the quadtree already handles area points (P2) | ✔ unchanged (`max_leaf = 64`) |
| 2 admissibility | code unchanged; **behaviour** changes. Weak rank ∝ √N on area data, strong bounded in N and growing with k·(box width) | a box touches its neighbours along an edge densely filled with points | ✔ (r1): weak N^0.47–0.55; strong flat |
| 3 representation | unchanged (one stacked ID per box) | — | ✔ works, but pays a rank penalty when d varies in a box (H2, r2) |
| 4 sampling | exact: unchanged. **Proxy: the kernel must change.** Use G·d at unknowns but *unscaled* at proxy points | the Green representation is about G; d only rescales sources and targets | ✔ (r2); the naive kernel fails under strong admissibility |
| 5 algebra | unchanged | — | ✔ `rskel_factor` untouched |

H4 (E5 transfer) and H5 (verification) are below.

## (c) Size of the change

| piece | code lines (no comments or blank lines) | dial |
|---|---|---|
| LS matrix: grid, contrast, Richmond self-term, assembly | ~40 (`ls_problem.py: grid_nodes, contrast, self_term, build`) | the kernel itself |
| proxy kernel `ls_kernel(.., "two_sided")` | ~25 | 4 (sampling) |
| changes to `unified/core` or `unified/drivers` | **0** | — |
| *optional:* realified compressor (E5 fix, real T) | 8 | 4 |
| *optional:* symmetrized solve Â = I + D^½ K D^½ (removes the H2 penalty) | ~15 (`variants.symmetrized`) | change of variables, no dial |

The rest of `ls_problem.py` holds the reference solutions: the Bessel series, the radial ODE, Graf
coefficients and unitarity. The rest of `variants.py` and `run.py` is diagnostics.

## (d) Verification (r0; dense = LU of A; fast = `rskel_factor` at ε = 1e-8)

**References.**
- **Disk.** The Bessel series follows from continuity of u and ∂_r u at r = a. Inside,
  u = Σ c_n J_n(k₁r) e^{inθ} with k₁ = k√(1+q₀). Outside, u − u_inc = Σ b_n H_n(kr) e^{inθ}.
- **Bump.** For each mode, the radial ODE w'' + (2|n|+1)/r w' + k²(1+q) w = 0 is integrated with
  `solve_ivp` (DOP853, rtol 1e-12) and matched at r = 1. Run on the constant disk, this ODE solver
  reproduces the Bessel series to 4e-12.
- **Discrete b_n.** From the discrete solution via Graf's addition theorem:
  b_n = (i/4) k² Σ_j J_n(kρ_j) e^{−inφ_j} q_j h² u_j.
  By Parseval, ‖b − b_ref‖/‖b_ref‖ equals the relative L² far-field error.
- **Physical invariant.** Energy conservation for a lossless object: every partial-wave S-matrix entry
  s_n = 1 + 2 b_n i⁻ⁿ has |s_n| = 1. This holds to 1e-15 for both references.

| object | k | m (grid) | N | points per interior wavelength | far-field error (dense) | interior field error | unitarity defect |
|---|---|---|---|---|---|---|---|
| disk | 5 | 24 / 36 / 48 / 72 | 448 – 4060 | 11 – 32 | 3.6e-2 / 8.9e-3 / 1.2e-2 / 7.7e-3 | 4.8e-2 → 1.1e-2 | 2.0e-2 → 1.2e-2 |
| disk | 10 | 24 / 36 / 48 / 72 / 102 | 448 – 8184 | 5.3 – 23 | 2.2e-1 / 6.5e-2 / 5.2e-2 / 2.8e-2 / 8.9e-3 | 2.2e-1 → 9.7e-3 | 1.7e-1 → 1.7e-2 |
| bump | 5 | 24 / 36 / 48 / 72 | 448 – 4060 | 11 – 32 | 4.6e-3 / 2.0e-3 / 1.1e-3 / 5.1e-4 | — | 3.8e-4 → 4.8e-6 |
| bump | 10 | 24 / 36 / 48 / 72 / 102 | 448 – 8184 | 5.3 – 23 | 2.8e-2 / 1.2e-2 / 6.9e-3 / 3.0e-3 / 1.5e-3 | — | 1.2e-2 → 3.7e-5 |

- **Bump: clean O(h²).** Tripling h multiplies the error by 9.1. This is the expected order for
  trapezoid-type cells with a correct singular cell.
- **Disk: about O(h), not monotone.** The staircase boundary dominates and the cut-cell pattern changes
  with m.
- **Fast vs dense at N = 4060**, over all 12 runs (disk and bump, k ∈ {5, 10}, weak and strong,
  exact / realified / proxy2): ‖x_fast − x_dense‖/‖x_dense‖ ∈ [6.3e-10, 9.0e-9]. The far-field error of
  every fast solution equals the dense one to 3 digits. **The solver adds nothing visible on top of the
  discretization error.**

## (e) Results and where it breaks

### r1 — Ranks: weak grows like √N on area data; strong stays flat (bump, q₀ = 1, two-sided proxy, ε = 1e-8)

Max skeleton size on level-2 boxes. These boxes have a fixed physical size (a quarter of the domain width),
so the numbers are comparable across N. Only N ≥ 2048 is used for the fits, because below that, level 2
holds the leaves.

| k | rule | N = 2053 | 4060 | 8184 | fitted exponent | root size (8184) | storage (8184) | error vs dense |
|---|---|---|---|---|---|---|---|---|
| 2 | weak | 112 | 157 | 213 | **0.47** | 979 | 4.8e6 | 3.5e-8 |
| 2 | strong | 59 | 58 | 58 | **0.0** | 809 | 11.0e6 | 5.9e-9 |
| 10 | weak | 122 | 182 | 261 | **0.55** | 1227 | 6.9e6 | 5.0e-9 |
| 10 | strong | 72 | 77 | 74 | **0.0** | 1020 | 12.9e6 | 1.4e-9 |

**Dependence on k** at N = 4060, from k = 1 (160 points per wavelength) to k = 40 (4 points per wavelength):
- Strong level-2 skeletons grow steadily: 54 → 68 → 77 → 89 → 100 → 112. This is the k·(box width) term.
- Weak level-2 skeletons barely move after k = 5: 177 → 182 → 184 → 186 → 198. At these sizes the weak rank
  is dominated by the √N interface term, not by k.

**Honest negative.** At N ≤ 8192, **strong storage is about 2× weak storage**. Strong stores dense blocks
from each box to its 8 neighbours, and its level-1 boxes cannot be compressed (all four touch), so the root
holds about 800–1000 unknowns. Both storages grow like N^1.13–1.24 over N = 2k–8k. The asymptotic advantage
of strong (O(N) vs O(N^{3/2})) has not been reached at these sizes. This agrees with E2's observation that
weak ranks grow on area data, and it quantifies it for Helmholtz LS.

**Exact, proved.** For a *constant* contrast, the weak-admissibility skeleton counts do not depend on q₀
(r4: 8961 at every q₀ from 0.1 to 8). The reason:
1. In weak RS, the far blocks of the working matrix are never modified. Schur complements stay inside the
   box, and the sparsification only touches redundant rows and columns.
2. So W[F, B] is a submatrix of A[F, B] = q₀-scaled K[F, B].
3. The ID rank of a uniformly scaled block does not change.

Strong RS varies slightly (5814 → 5959), because modified near-field blocks enter through the Q set.

### r2 — The diagonal contrast multiplies only one side (k = 10, N ≈ 2048)

The driver's single ID compresses M = [A_FB; A_BFᵀ] = [K_FB D_B; D_F K_FB]. The two blocks need different
column relations:
- K_FB·D_B needs the relations of a column-scaled block.
- D_F·K_FB needs the relations of K_FB (row scaling changes nothing).

The two coincide when d is constant on the box.

| object | rule | level | rank of A_FB | rank of A_BFᵀ | rank of the **stack** | max ratio over boxes |
|---|---|---|---|---|---|---|
| disk / square (d constant per box) | weak, strong | 2, 3 | — | — | = one-sided | **1.00** (all boxes) |
| bump | weak | 3 | 28.9 | 29.4 | 34.1 | 1.31 |
| bump | weak | 2 | 61.5 | 63.5 | 91.0 | 1.48 |
| bump | strong | 3 | 17.7 | 18.2 | 30.3 | **1.78** |
| bump | strong | 2 | 19.2 | 20.1 | 35.5 | **1.85** |

**H2 holds, and more strongly than predicted.** The predicted ratio was 1.1–1.5; under strong admissibility
it is 1.8–1.85, nearly the worst case of 2.

**Fix: Ho–Ying symmetrization.** Solve Â y = √d ∘ f with Â = I + D^½ K D^½. This matrix is complex
symmetric, so A_BFᵀ = A_FB and the stack equals the one-sided block exactly. Then recover u = f − K(√d ∘ y)
with one kernel apply; there is no division by √d, so a contrast that goes to 0 at the edge of the bump is
safe. Measured, exact ID, bump:

| ε | rule | skeletons, plain A | skeletons, symmetrized Â | error (sym) | skeletons, proxy2 on A |
|---|---|---|---|---|---|
| 1e-6 | strong | 2416 | **1458 (−40%)** | 1.9e-8 | 3248 |
| 1e-8 | strong | 3074 | **1789 (−42%)** | 1.8e-10 | 3864 |
| 1e-10 | strong | 3528 | **2251 (−36%)** | 2.8e-12 | 4302 |
| 1e-8 | weak | 4116 | **3169 (−23%)** | 5.2e-9 | 4227 |

On the constant disk, the symmetrized and plain variants produce identical skeleton counts (3275 / 1956),
as they must.

### r2 — The proxy with a one-sided contrast

**Naive kernel.** Writing kernel(x, y) = G(x, y)·(−k² q(y) h²) is the "obvious" thing. It evaluates q at the
proxy points as well. Wherever the proxy circle leaves the object, q = 0, so those rows of the transposed
block are zero, and the ID loses the relations of D_F K_FB.

**Two-sided kernel.** It uses d_j at grid nodes and 1 at proxy points. Then kernel(X_B, P)ᵀ = G(P, X_B)
carries exactly the relations the Tᵀ row step needs.

**Far-field fit.** Max over boxes of ‖M[:, R] − M[:, S] T‖/‖M‖, with M the true stacked far field and T
taken from the compressor under test:

| object | rule | exact | two-sided proxy | naive proxy |
|---|---|---|---|---|
| disk, square | weak, strong | ≤ 5e-9 | ≤ 3e-9 | ≤ 4e-11 – 2e-9 (fine: d constant per box) |
| bump | weak | ≤ 3e-9 | ≤ 5e-9 | ≤ 6e-9 (fine: rescued by the explicit rows inside the circle) |
| bump | strong | ≤ 7e-9 | ≤ 8e-9 | **4.1e-5 – 7.5e-5** |

**Solve error under strong admissibility, bump:**

| ε | two-sided proxy | naive proxy |
|---|---|---|
| 1e-6 | 1.6e-7 | 7.0e-7 |
| 1e-8 | 3.3e-10 | **2.8e-7** |
| 1e-10 | 3.0e-12 | **4.8e-7** |

The naive kernel leaves an error **floor** of about 5e-7 that tightening ε does not lower. H3 holds for
strong admissibility. For weak admissibility it is refuted: there the far points inside the proxy circle
(all the neighbours) are explicit rows of A and carry the missing relations.

### r3 — Does the E5 Tᵀ vs Tᴴ issue appear? (bump, q₀ = 1, N = 4060, ε = 1e-8)

Herm(A) here is the Hermitian part of A, (A + Aᴴ)/2.

| k | points per wavelength | cond(A) | λ_min Herm(A) | weak: exact / realified / proxy2 error | strong: exact / realified / proxy2 error | strong cond(X_RR): exact / realified |
|---|---|---|---|---|---|---|
| 5 | 32 | 4.4 | +0.13 | 2.3e-9 / 1.4e-9 / 5.8e-9 | 6.3e-10 / 7.2e-10 / 7.7e-10 | 1.7e3 / 2.0e3 |
| 10 | 16 | 9.6 | −0.80 | 5.0e-9 / 4.0e-9 / 2.5e-9 | 1.5e-9 / 1.5e-9 / 6.6e-10 | 1.4e4 / 1.3e3 |
| 20 | 8 | 28 | −2.7 | 2.6e-8 / 1.4e-8 / 8.9e-9 | 5.5e-9 / 6.3e-9 / 1.5e-9 | 6.8e3 / 2.1e3 |
| 30 | 5.3 | 58 | −4.6 | 3.4e-8 / 3.7e-8 / 3.1e-8 | 1.7e-8 / 9.0e-9 / 2.2e-9 | 1.7e5 / 3.1e3 |
| 40 | 4 | 101 | −6.6 | 6.6e-8 / 5.2e-8 / 4.9e-8 | 3.9e-8 / 1.2e-8 / 6.4e-9 | 8.7e5 / 1.2e3 |
| 60* | 2.7 | 279 | −10.4 | 1.1e-7 / 5.8e-8 / 4.0e-8 | **3.7e-7** / 2.8e-8 / 1.7e-8 | **1.8e6** / 4.8e3 |
| 80* | 2 | 1.0e3 | −14.3 | 3.8e-7 / 7.1e-7 / 1.1e-6 | **2.9e-6** / 5.4e-7 / 6.3e-7 | **1.0e7** / 2.8e3 |

\* Under-resolved (fewer than 4 points per wavelength): the physics is wrong there, but the algebraic
question (does the fast solve match the dense solve?) is still well posed.

**What we see.**
- **The E5 signature is present.** With the exact (complex T, Tᵀ) variant, the Hermitian part of the pivot
  blocks is strongly indefinite at every k: λ_min Herm(X_RR) ranges from −2.6 to −118. With real T
  (realified) or the proxy it stays positive (+0.02 to +0.09) up to k = 40. The worst cond(X_RR) under
  strong admissibility grows from 1.7e3 to 1.0e7 for exact, but stays at 1.2e3–4.8e3 for realified.
- **The accuracy impact is mild.** Every variant stays within ε·cond(A) (dotted line in
  `r3_r4_sweeps.png`). Exact is worse only under strong admissibility at k ≥ 60, by 5–13× (3.7e-7 vs
  2.8e-8; 2.9e-6 vs 5.4e-7).
- **Unlike E5, tightening ε does not hurt exact.** Weak, k = 80, exact vs realified:

  | ε | exact | realified |
  |---|---|---|
  | 1e-6 | 1.6e-4 | 2.0e-4 |
  | 1e-10 | 2.2e-8 | 2.7e-8 |
  | 1e-12 | 1.2e-10 | 1.3e-10 |

  The cascade E5 saw on the curve, where weak error reached 2.6e-4 at ε = 1e-12, does not develop here,
  even though the largest box has k·hw = 40 (E5's curve broke at k·hw ≳ 30).
- **Conjecture for why.** The tree has only 4 levels, and in area problems the Schur-complement updates
  per box are spread over more unknowns, so fewer levels compound. Not tested.
- **E5's theorem does not apply here.** It needs Herm(A) ⪰ μ > 0, but λ_min Herm(A) is already negative
  at k = 10 (strong scattering). Even so, real-T pivots stayed definite up to k = 40. They turn indefinite
  (−0.3 to −2.3) only at k ≥ 60, or at q₀ ≥ 2 (r4), and cond(X_RR) stays moderate there.
- **Recommendation.** Use real T or Tᴴ (E5) anyway. It costs about 4% more skeletons (weak) or 4–20% more
  (strong), and it removes a 1e7 pivot.

### r4 — Contrast strength (constant disk, k = 10, N = 4060): the discretization breaks before the solver

| q₀ | 0.1 | 0.5 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|---|
| points per interior wavelength | 22 | 18 | 16 | 13 | 10 | 7.5 |
| cond(A) | 1.8 | 10 | 36 | 444 | 1.1e3 | 1.3e4 |
| λ_min Herm(A) | +0.70 | −0.51 | −2.0 | −5.0 | −11 | −23 |
| far-field error vs Bessel series (dense) | 6.6e-3 | 1.0e-2 | 2.8e-2 | 4.9e-2 | **0.17** | **0.57** |
| worst fast-vs-dense error, all 6 variants | 1.3e-9 | 4.9e-9 | 9.0e-9 | 2.8e-8 | 2.9e-8 | 2.2e-7 |

- **The fast solvers track ε·cond(A)** across a 7000× range of cond(A). Every error is between 0.02 and
  0.6 of ε·cond(A).
- **The Richmond staircase discretization fails first.** It gives 17% and 57% far-field error at
  q₀ = 4 and 8, because the interior wavelength shrinks and near-resonant modes inside the disk are
  sensitive to the staircase boundary.
- **So for strong contrasts the bottleneck is quadrature, not compression.** Higher-order LS quadratures
  exist: Duan–Rokhlin 2009; Ambikasaran et al. 2016.

## (f) Literature: what is known, and what is new

A sub-agent checked the papers below; FT means it read the full text. We have not ourselves read
Corona–Martinsson–Zorin or Engquist–Zhao in full.

- **Ho & Ying 2016, HIF-IE** (CPAM 69(8); arXiv:1307.2666; FT).
  - Their model equation a(x)u + b(x)∫K c(y)u dy = f includes diagonal scalings.
  - Example 3 is **2D LS** with a Gaussian contrast and 32 points per wavelength. They **symmetrize**
    with u = √ω σ: our r2 fix is theirs.
  - Their RS top-level skeletons double when κ and √N double.
  - Their proxy (Lemma 3.3) puts b and c at the box's own points, never at proxy points, so it implicitly
    avoids our naive-kernel pitfall. **cited**
- **Minden, Ho, Damle, Ying 2017, strong RS** (MMS 15(2); arXiv:1609.08130; FT).
  - Mentions LS as in scope, but has **no Helmholtz or LS example**.
  - Their proxy derivation is done for b = c = 1 only. **cited**
- **Corona, Martinsson, Zorin 2015** (ACHA 38(2); arXiv:1303.5466; partial FT). Weak (HBS) skeletons on
  plane volume problems grow like O(n^{1/2}), giving O(N^{3/2}) for the classical scheme. Our r1 exponent of
  0.47–0.55 agrees. **cited**
- **Gopal & Martinsson 2022** (Adv. Comput. Math. 48:42; arXiv:2007.12718; FT).
  - HBS for (I + BG)q = f, with B = diag(κ²b). They **compress G alone** with a proxy ring and reuse
    column skeletons as row skeletons, so compression is independent of the contrast.
  - Rank grows like √(box size). This is the closest prior work. **cited**
- **Ambikasaran, Borges, Imbert-Gérard, Greengard 2016** (SISC 38(3); arXiv:1505.07157). High-order LS with
  a HODLR solver, O(N^{3/2}). **cited**
- **Also relevant:**
  - Chen 2002 (Adv. Comput. Math. 16), a direct LS solver.
  - A butterfly-accelerated HOD-BF volume integral equation solver (arXiv:2111.01730), the high-k route.
  - FMM-LU (Sushnikova et al. 2023): boundary only. **cited**
- **Classics:**
  - Richmond 1965, IEEE TAP 13(3):334–341 (pulse basis, equal-area-disk self-term).
  - Vainikko 2000 (FFT-based LS).
  - Duan–Rokhlin 2009, JCP 228.
  - Bessel-series solution for a dielectric cylinder: Bohren & Huffman 1983, Ch. 8. **cited**
- **Rank theory:** Engquist & Zhao 2018 (CPAM 71(11)) give separability bounds versus k. We only read the
  abstract. **cited**

**What is new here, as far as the sub-agent's search found:**
1. **The stacked-ID penalty from a one-sided contrast, quantified.** It is 1.31–1.85× per box and nearly 2×
   under strong admissibility; it is exactly 1 for piecewise-constant contrast, and the same mechanism shows
   why the symmetrization pays off (−42% skeletons). Prior work avoids the issue by symmetrizing (HIF-IE)
   or by compressing G alone (Gopal–Martinsson), but we found no measurement of what happens when you do
   neither. **computed**
2. **The naive-proxy pitfall.** Evaluating the contrast at proxy points kills the transposed-direction rows,
   which leaves a ~5e-7 error floor under strong admissibility; weak admissibility is rescued by the explicit
   rows. We found it stated nowhere. **computed**
3. **E5's indefinite-pivot mechanism transfers to LS** (cond(X_RR) up to 1e7 with Tᵀ vs ≤ 5e3 with real T),
   but does not cascade into large errors here, and tightening ε does not hurt. **computed**; the reason is
   a conjecture.
4. **Weak skeleton counts for a constant contrast are independent of q₀.** **proved** (short argument in
   r1); also computed.

Everything else (√N weak growth, the feasibility of RS for LS, symmetrization) is known.

## Claims

| # | Claim | Status |
|---|---|---|
| 1 | LS runs through `rskel_factor` and the core with 0 changed lines. The kernel swap is ~40 lines of assembly plus a ~25-line proxy kernel. | computed (line counts) |
| 2 | The dense LS solve converges to the Bessel series (disk, about O(h), staircase) and to the radial-ODE reference (bump, O(h²): ratio 9.1 for 3× h). The ODE reproduces the Bessel series to 4e-12. Unitarity defect → 0 (bump: 3.8e-4 → 4.8e-6). | computed |
| 3 | Fast and dense solutions agree to 6e-10 – 9e-9 at ε = 1e-8 (N = 4060). Far-field errors are identical to 3 digits. Across sweeps, error ≤ 0.6·ε·cond(A). | computed |
| 4 | Weak level-2 skeletons grow like N^0.47 (k = 2) and N^0.55 (k = 10). Strong skeletons are flat in N (58–77) and grow with k (54 → 112 for k = 1 → 40). | computed; √N agrees with Corona–Martinsson–Zorin 2015 and HIF-IE (cited) |
| 5 | At N ≤ 8192, strong storage is about 2× weak storage (dense neighbour blocks, uncompressible level 1). The asymptotic advantage of strong is not reached. | computed |
| 6 | Stacked/one-sided rank ratio: 1.00 for d constant per box; 1.31–1.48 (weak) and 1.78–1.85 (strong) for the bump. | computed; ratio ≤ 2 is proved (rank of a 2-block stack ≤ sum of ranks) |
| 7 | Symmetrization (Ho–Ying) cuts skeletons by 36–42% (strong) and 23% (weak) on the bump, at equal accuracy. | computed; the method is cited |
| 8 | The naive proxy kernel (contrast at proxy points) gives a far-field fit of 4e-5 – 7.5e-5 and a solve error floor of 3e-7 – 7e-7 under strong admissibility for a variable contrast. It is fine for weak admissibility or constant contrast. The two-sided kernel fixes it. | computed |
| 9 | With complex T (Tᵀ) the pivots are indefinite (λ_min Herm(X_RR) between −2.6 and −118), and strong cond(X_RR) reaches 1e7 at k = 80, vs ≤ 5e3 with real T or proxy. The accuracy loss is ≤ 13× and only at k ≥ 60 under strong admissibility. Tightening ε does not hurt exact (unlike E5). | computed |
| 10 | Real-T pivots stay Hermitian-definite up to k = 40 even though λ_min Herm(A) < 0 (outside E5's theorem). They turn indefinite at k ≥ 60 or q₀ ≥ 2. | computed; the reason is a conjecture |
| 11 | For a constant contrast, weak RS skeleton counts do not depend on q₀ (8961 for q₀ from 0.1 to 8). | proved (weak RS never modifies far blocks); computed |
| 12 | For strong contrast (q₀ ≥ 4 at k = 10) the Richmond discretization fails (17–57% far-field error) while the fast solver still matches the dense solve to 3e-8 – 2e-7. | computed |

## Files
- `ls_problem.py`: LS assembly, the proxy kernels, the Bessel series, the radial ODE, Graf coefficients
  and unitarity.
- `variants.py`: compressor variants, symmetrized solve, pivot diagnostics, stacked ranks.
- `run.py`: experiments r0–r4.
- `test_e9.py`: 12 tests.
- `HYPOTHESES.md`: the predictions, written before running.
