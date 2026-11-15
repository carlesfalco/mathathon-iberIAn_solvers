# E11: Matérn kernels (Gaussian processes, kriging) — breaking "the kernel solves a PDE outside the box"

**Result in five lines.**
- The stock proxy circle **fails** on Matérn covariance matrices under strong admissibility for every smoothness ν > 0.
  The normwise backward error is 4e-6 … 4e-4 against 1e-11 for the exact ID; at ν = 2.5 the solve is useless
  (error 2.7 against dense, log-det wrong by 6.3). It is exact only for the ν = 0 control (K₀, Yukawa).
- **The coordinator's B2 count is right only for second-order lifts.** C_ν is not a Yukawa kernel of a higher
  dimension. But it *is* the restriction to R² of the Green's function of the **order-2β** operator (κ² − Δ)^β in
  R^D with D = 2(β − ν), for any integer β ≥ ν + 1 [proved]. Half-integer ν (0.5, 1.5, 2.5) lifts to D = 3.
- An order-2β operator needs β boundary data, so the fix is a **lifted multi-kernel proxy**: β kernel orders
  r^{ν−m}K_{ν−m}(κr), m = 0…β−1, sampled on E8's hemisphere. It reaches ε = 1e-8 at every ν tested, including the
  fractional-D case ν = 0.7 (D = 2.6). Integer ν needs no lift: multi-kernel on the plain circle is exact.
- **It is not the cheapest fix.** It keeps 1.6× the exact ID's skeletons. A Xing–Chow-style selected volume proxy
  (~105 points) and the annulus keep about 1.3× and reach ε too (the selected proxy sometimes within 2–7× of ε).
  At *equal rank* the volume proxies are 2–10× more accurate than the lift, the reverse of E8's finding for power laws.
- On the statistics task (GP regression, N ≤ 8192), the fast solver reproduces the dense posterior mean, log
  determinant and log marginal likelihood to the accuracy the conditioning allows. The log-det comes free from the
  factorization (proved), with absolute error ≤ 3e-9 (lifted, strong) at cond up to 7e6.

Reproduce: `PYTHONPATH=. .venv/bin/python -m unified.explore.e11_matern.run` (≈ 6 min; `--quick` ≈ 2 min;
`--only r1,r3`). Data `results/*.json`, figures `results/*.png`, log `results/run_log.txt`.
Checks: `.venv/bin/python -m pytest -q unified/explore/e11_matern` (20 tests);
`.venv/bin/mypy --strict unified/explore/e11_matern` (clean). Predictions written before any run:
[`HYPOTHESES.md`](HYPOTHESES.md).

---

## (a) The problem in plain words

**Gaussian processes / kriging.** You measured a quantity y (rainfall, ore grade, temperature) at N scattered
places x_i, with noise. You model the unknown field as a random function whose values at two places are correlated
by a *covariance kernel* C(|x − x′|): close places are similar, far places are nearly independent. Prediction at a
new place x* (the *posterior mean*) and the fit quality (the *log marginal likelihood*, used to tune the kernel's
parameters) both need the N × N covariance matrix A = C + σ²I:
- solve A α = y; the posterior mean is m(x*) = Σ_i C(x*, x_i) α_i;
- log p(y) = −½ yᵀα − ½ log det A − (N/2) log 2π (Rasmussen–Williams eq. 2.30).

A is dense. σ² (the *nugget*) is the measurement-noise variance; it is what keeps A invertible.

**The Matérn family** (the default kernel of spatial statistics): with smoothness ν, length scale ℓ and κ = √(2ν)/ℓ,
C_ν(r) = 2^{1−ν}/Γ(ν) (κr)^ν K_ν(κr), C(0) = 1. ν = ½ is the exponential kernel e^{−r/ℓ}; ν → ∞ the Gaussian.
Samples of the field are ⌈ν⌉ − 1 times differentiable.

**Why it breaks a hypothesis.** The proxy surface (core `ProxyCompressor`) replaces "all far points" by points on a
circle around a box. That is exact when the kernel solves a PDE outside the box: then values on the circle determine
the field outside (uniqueness of the exterior problem). The Matérn kernel is the Green's function of the SPDE operator
(κ² − Δ)^{ν+d/2} (Whittle 1954; Lindgren–Rue–Lindström 2011). In d = 2 that is (κ² − Δ)^{ν+1}: a **local** operator
only for integer ν, and even then of order 2ν + 2 > 2.

## (b) Theory, checked before and during the run

**Dimension counting.** The Green's function of L^β = (κ² − Δ)^β in R^D is ∝ r^{β−D/2}K_{β−D/2}(κr)
[cited: standard Fourier transform of (κ² + |ξ|²)^{−β}; e.g. Lindgren et al. 2011 §2].
- β = 1 (the Yukawa / second-order lift that B2 uses): matching r^νK_ν needs D = 2 − 2ν < 2. **B2 is right that no
  Yukawa lift exists** [proved].
- Any integer β ≥ ν + 1: D = 2(β − ν) ≥ 2 [proved]. ν = 0.5, 1.5, 2.5 → D = 3 (β = 2, 3, 4); ν = 1 → D = 2 (β = 2,
  no lift, but order 4); ν = 0.7 → D = 2.6 (fractional, E8's situation).

**Why one kernel on the sphere is not enough, and what is.** If u = Σ c_j C(· − y_j) vanishes on a sphere, an
order-2β equation does *not* force u = 0 outside (the exterior Dirichlet problem is not unique for β ≥ 2). If also
L^m u = 0 on the sphere for m = 1…β−1, then v = L^{β−1}u solves Lv = 0 outside, vanishes on the sphere and at
infinity, so v ≡ 0 (maximum principle for κ² − Δ); induct downwards to u ≡ 0 [proved for integer D; for fractional D
via the weighted extension, conjecture as in E8]. L^m C_ν ∝ r^{ν−m}K_{ν−m}(κr). So the proxy rows are the β kernel
orders ("multi-kernel proxy", the §B3 idea) on a sphere of R^D, which reduces to E8's hemisphere in (x, y, t).

**A subtlety found by the tests** (`test_family_identity_any_dimension`, `test_chain_terminates_only_in_lift_dimension`).
The identity (κ² − Δ_D)[r^μK_μ] = c_D r^{μ−1}K_{μ−1} holds for r > 0 in *every* D; only the constant depends on D. What
singles out D = 2(β − ν) is that the chain **terminates**: the last order μ = 1 − D/2 is annihilated by κ² − Δ_D only
there. In the plane (D = 2) the chain never terminates for non-integer ν, so the multi-kernel rows on a circle carry
L^m u data for an operator that has no finite order [proved by the identity; computed in the tests].

## (c) What was changed (dial 4 only)

| file / object | role | code lines |
|---|---|---|
| `proxies.MaternProxy` | proxy geometry: circle, hemisphere, annulus, Xing–Chow-style selection (one class, overrides only `proxy_points`) | 52 |
| `matern.StackedKernel.__call__` | stacks β kernel orders along the proxy axis (data points 2 columns, proxies 3) | 17 |
| `matern.multi_orders`, `lift_order` | β and D from ν | 6 |
| `matern._pairdist` | distances with the extension coordinate t | 7 |
| `harness.logdet` | log-det read-out from the stored LU blocks (algebra read-out, no new factorization) | 7 |
| **dial total** | (lifted-multi path alone: 43) | **89** |
| physics/statistics set-up | Matérn formula, GP problem, posterior mean, likelihood, kernel formulas | **60** |
| core, drivers, tree, admissibility, ID, factor | — | **0** |

Counted by `unified.showcase.reproduce.code_lines` (non-blank, non-comment, no docstrings); `results/lines.json`.
The driver's FMM-LU Q-set path calls `comp.proxy_points` and `comp.kernel(Pp, X[box])`; since the stacked kernel
recognises proxy points by their third coordinate, the multi-kernel rows reach that path too, with no driver change.

**log det from the factorization** [proved]: the sparsifying row/column operations are unit triangular (det 1) and
block elimination gives det A = Π det X_RR · det(root). `harness.logdet` sums log|diag| of the stored LU factors.

## (d) Compressors compared

exact (true far block) · rand (E3's randomized ID, imported read-only) · **plane1** (stock circle, 64 points) ·
**planeM** (circle, β orders) · **lift1** (E8 hemisphere, 4 latitudes, 236 points, one order) · **liftM** (hemisphere,
β orders; falls back to the circle when D = 2) · annulus (8 rings to the domain diameter, 512 points) · xc
(Xing–Chow 2020 Alg. 5.1-style selection from 3000 far candidates, ~100 points; re-implemented from E8's recipe
because E8's directory was still being edited).

**Error measures.** A = C + σ²I is ill-conditioned when σ² is small (cond up to 7e6 here). As in E8, compression and
conditioning are reported separately:
- **bwd** = ‖Ax − y‖ / (‖A‖₂‖x‖ + ‖y‖), the normwise backward error, which is cond-independent and ≈ ε for a sound
  compression;
- **far** = worst per-box ‖M_R − M_S T‖_F/‖M‖_F against the *true* current far block M = [W_FB; W_BFᵀ] (strong
  runs);
- **err** = ‖x − x_dense‖/‖x_dense‖ ≈ cond × bwd.

## (e) R1 — does the stock circle fail, and does lifting fix it? (N = 2048, ℓ = 0.1, σ² = 1e-2, ε = 1e-8, strong)

Entries: **bwd / far / total skeletons** (`results/r1_table.json`, `r1_table.png`).

| ν (β, D) | cond | exact | rand | plane1 (stock) | planeM | lift1 | **liftM** | annulus | xc |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 (2, 3) | 8e3 | 3e-11/8e-9/3289 | 6e-10/2e-7/3138 | **4e-6/1e-2**/3742 | 3e-8/1e-4/4407 | 2e-11/3e-8/5035 | **3e-12/3e-9**/5284 | 4e-11/2e-8/4283 | 7e-11/4e-8/4273 |
| 0.7 (2, 2.6) | 1e4 | 3e-11/8e-9/3308 | 2e-9/8e-7/3174 | **5e-6/2e-2**/3736 | 2e-8/7e-5/4393 | 3e-11/6e-8/5067 | **3e-12/2e-9**/5290 | 3e-11/1e-8/4296 | 7e-11/4e-8/4265 |
| 1 (2, 2) | 1e4 | 3e-11/7e-9/3069 | 3e-10/7e-8/2970 | **5e-6/2e-2**/3591 | **8e-12/1e-8**/4191 | 4e-11/1e-7/5081 | = planeM | 2e-11/1e-8/3973 | 2e-11/1e-8/3948 |
| 1.5 (3, 3) | 1e4 | 2e-11/1e-8/3368 | 5e-10/1e-7/3248 | **1e-5/3e-2**/3744 | 3e-10/7e-7/4862 | 5e-11/3e-7/5089 | **8e-13/2e-9**/5452 | 2e-11/3e-8/4352 | 3e-11/7e-8/4343 |
| 2.5 (4, 3) | 1e4 | 2e-11/1e-8/3438 | 8e-10/2e-7/3304 | **4e-4/4e-2**/3743 | 4e-12/1e-8/5178 | 9e-11/7e-7/5100 | **4e-13/5e-10**/5522 | 2e-11/1e-7/4426 | 3e-11/7e-8/4447 |
| K₀ control (ν = 0) | 1.0 | 3e-13/5e-9/2139 | 2e-12/4e-8/2088 | 1e-12/2e-7/3069 | = plane1 | 6e-15/7e-10/5180 | = lift1 | 1e-12/2e-7/3053 | 7e-13/9e-8/3004 |
| RTE control e^{−μr}/r | 1.6 | 1e-12/1e-8/3473 | 4e-11/3e-7/3332 | **1e-6/1e-2**/3446 | = plane1 | 4e-11/4e-6/5294 | = lift1 | 5e-11/4e-6/4481 | 2e-11/7e-7/4489 |

Readings:
- **plane1 fails for every ν > 0**, and gets worse as ν grows: the far error is 1e-2 … 4e-2, and at ν = 2.5 the solve
  is wrong by 270% (err 2.7, log-det off by 6.3; `r1_table.json`). P1 is confirmed, with direction: smoother is worse,
  because more boundary data are missing (β grows). [computed]
- **The K₀ control is exact on the circle** (bwd 1e-12; it is the Yukawa Green's function of R²). [computed]
- **planeM (multi-kernel ring, B3's idea)**: exact for integer ν = 1 (P3 confirmed). For half-integer ν it is
  approximate: 1e-4 far error at ν = 0.5, 7e-7 at ν = 1.5, and *within ε at ν = 2.5* (4 orders). That the plane ring
  improves with more orders is **computed, unexplained** (conjecture: the non-terminating remainder of the chain
  shrinks with β).
- **lift1 (one kernel, E8's hemisphere)** works for β = 2 (ν = 0.5, 0.7: within 3–6× of ε in far error, bwd ≈ exact) but
  misses by 30–70× at β = 3, 4 (far error 3e-7, 7e-7). This is P2 partly *refuted*: at β = 2 the missing L u data cost
  little in practice. Warning sign for interpretation: lift1 also does fairly well at **ν = 1, where the lift has no
  justification at all** (D = 2: far error 1e-7). Off-plane points evidently act partly as generic extra samples
  (lifted points see the box at varying distances, like a volume proxy) [conjecture].
- **liftM reaches ε (far ≤ 3e-9, bwd ≤ 3e-12) at every ν**, including the fractional dimension ν = 0.7 (P4 confirmed,
  computationally for D = 2.6). [computed]
- **Radiative-transfer control (§B2's remark)**: circle fails (far 1e-2); the single-kernel hemisphere (D = 3, β = 1,
  proved) gives bwd 4e-11 = exact's level (P6 confirmed). Its worst per-box far error is 4e-6, but annulus and xc show
  the same: the proxy rows are larger than the true far block, so a relative ε on the proxy rows is looser on the far
  block. The solve is unaffected. [computed]
- **Weak admissibility**: every compressor, plane1 included, gives bwd 1e-10 … 2e-10 and far ≤ 2e-8. The touching
  neighbours sit inside the circle and are explicit, so the missing data hardly matter (the E3/E8 pattern). Randomized
  is the exception at 2e-6 far error (its adaptive stopping is loose). [computed]
- **Randomized ID** misses ε by 10–100× in far error under both rules (oversampling 10, relative stopping); bwd stays
  at ≤ 2e-9. [computed]

## (f) R2 — how many latitudes, and one kernel vs β kernels (strong, N = 2048)

Worst per-box far error / total skeletons, by latitudes above the equator (0 = circle; points 64, 110, 152, 236, 398).
`r2_latitudes.json/png`.

| ν | proxy | 0 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|---|
| 0.5 | lift1 | 1e-2 / 3742 | 2e-4 / 4293 | 4e-6 / 4637 | 3e-8 / 5035 | 7e-9 / 5099 |
| 0.5 | liftM | 1e-4 / 4407 | **3e-8** / 5038 | 3e-9 / 5252 | 3e-9 / 5284 | 2e-9 / 5322 |
| 0.7 | lift1 | 2e-2 | 4e-4 | 7e-6 | 6e-8 | 8e-9 |
| 0.7 | liftM | 7e-5 | **3e-8** / 5046 | 2e-9 | 2e-9 | 3e-9 |
| 1.5 | lift1 | 3e-2 | 1e-3 | 2e-5 | 3e-7 | 3e-8 |
| 1.5 | liftM | 7e-7 | **4e-9** / 5352 | 1e-9 | 2e-9 | 1e-9 |
| 2.5 | lift1 | 4e-2 | 2e-3 | 6e-5 | 7e-7 | 1e-7 (never reaches ε) |
| 2.5 | liftM | 1e-8 | 7e-10 | 7e-10 | 5e-10 | 2e-10 |

**One latitude (110 points) suffices for liftM**; lift1 needs 4–8 latitudes and still misses at β ≥ 3. The skeleton
count rises with latitudes until saturation, as in E8: off-plane directions cost rank. [computed]

## (g) R3 — equal rank (every ID truncated at the exact ID's per-box rank; strong, N = 2048)

Worst per-box far error / bwd (`r3_equal_rank.json`):

| ν | exact (ref.) | rand | plane1 | planeM | lift1 | liftM | annulus | xc |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 8e-9 / 3e-11 | 3e-8 / 1e-10 | 1e-2 / 3e-6 | 3e-4 / 7e-8 | 8e-5 / 6e-8 | 2e-4 / 1e-7 | **1e-5 / 7e-9** | 2e-5 / 8e-9 |
| 0.7 | 8e-9 | 3e-8 | 2e-2 | 2e-4 | 2e-4 | 2e-4 | **3e-5** | 6e-5 |
| 1.5 | 1e-8 | 3e-8 | 3e-2 | 4e-4 | 3e-4 | 3e-4 | **5e-5** | 1e-4 |
| 2.5 | 1e-8 | 2e-8 | 4e-2 | 2e-4 | 6e-4 | 2e-4 | **1e-4** | 9e-5 |

- Every proxy loses 3–4 digits at equal rank: the extra skeletons a proxy keeps at equal tolerance are **needed**
  (E8's R6 result holds here too). Randomized (true far field, sketched) loses only a factor 2–4.
- **Unlike E8**, the lifted proxy is *not* the most accurate at equal rank: the volume proxies (annulus, xc) are 2–10×
  better. For Matérn the lift is a correct proxy, but not the most rank-efficient one. [computed]

## (h) R4 — the GP task: ν × ℓ × nugget (N = 2048)

ν ∈ {0.5, 0.7, 1.5, 2.5}, ℓ ∈ {0.03, 0.1, 0.3}, σ² ∈ {1e-1, 1e-4}; strong (exact, liftM, xc) and weak (exact, liftM).
24 problems, 120 factorizations (`r4_gp_sweep.json`, `r4_conditioning.png`).

| | worst over the 24 problems |
|---|---|
| cond(C + σ²I) | 1.3e2 (ℓ = 0.03, σ² = 0.1) … **6.6e6** (ν = 2.5, ℓ = 0.3, σ² = 1e-4) |
| bwd, strong exact / liftM / xc | 8e-11 / 1e-11 / 1e-10: **compression is at ε everywhere** |
| bwd, weak exact / liftM | 8e-10 / 5e-10 |
| err vs dense, strong exact / liftM / xc | 1e-4 / 2e-6 / 7e-5 (all at cond 6.6e6) ≈ cond × bwd |
| err vs dense, weak exact / liftM | 4e-4 / 2e-4 |
| posterior-mean error, strong liftM / exact / xc | 2e-6 / 5e-5 / 1e-4 |
| \|log det_fast − log det_dense\|, strong liftM / exact / xc | **3e-9** / 1e-5 / 7e-6 |
| \|log det error\|, weak exact | 3e-4 |

- Errors track cond × bwd (`r4_conditioning.png`): the small-nugget problem is a **conditioning** problem, not a
  compression problem (P8 confirmed). The well-conditioned cases (σ² = 0.1, ℓ = 0.03) give err ≤ 1e-10.
- **Rank vs ℓ** (strong exact skeletons): 3460 (ℓ = 0.03) → 3283 (0.1) → 2934 (0.3) at ν = 0.5; similar for every ν.
  Rank falls with ℓ (P7 confirmed), but only by 15–20% across a 10× range. Rank vs ν at fixed ℓ is flat (3289 … 3438 in
  R1); P7's "rank decreases with ν" is *refuted* at these sizes. [computed]
- Weak is 10× worse than strong in forward error at small σ²: its larger ranks come with a larger compression error
  amplified by the same cond.

## (i) R5 — N scaling to 8192 (strong, ℓ = 0.1, σ² = 1e-2)

| ν, compressor | N = 2048 / 4096 / 8192: max rank at level 2 | total skeletons at 8192 | bwd at 8192 | log-det err at 8192 | factor s (dense solve s) at 8192 |
|---|---|---|---|---|---|
| 0.7 exact | 52 / 68 / 79 | 12521 | 3e-11 | 1.4e-9 | 3.4 (4.7) |
| 0.7 liftM | 108 / 134 / 155 | 18841 | 5e-12 | 5.5e-11 | 8.0 |
| 0.7 xc | 74 / 91 / 106 | 15067 | 6e-11 | 9.8e-11 | 3.6 |
| 1.5 exact | 54 / 69 / 78 | 12438 | 4e-11 | 2.8e-8 | 3.5 (3.9) |
| 1.5 liftM | 116 / 143 / 160 | 19385 | 4e-12 | 1.1e-10 | 4.2 |
| 1.5 xc | 78 / 90 / 104 | 14932 | 7e-11 | 4.0e-9 | 1.8 |

Ranks still creep up with N for every compressor (about +15% per doubling); as in E8's 2D strong case this is not a
clean plateau yet [computed]. Wall times are secondary: the driver keeps a dense working matrix, so the fast and the
dense solve cost the same at these sizes.

## (j) R6 — the statistics task: log marginal likelihood vs ℓ (ν = 1.5, σ² = 1e-2, N = 2048)

| ℓ | 0.03 | 0.08 | 0.18 | 0.4 | 0.9 |
|---|---|---|---|---|---|
| dense | −604.857168 | 688.762923 | 1317.513156 | 1596.420402 | 1684.301361 |
| strong RS, liftM | −604.857168 | 688.762922 | 1317.513156 | 1596.420401 | 1684.301361 |
| strong RS, xc | −604.857168 | 688.762922 | 1317.513153 | 1596.420371 | 1684.301320 |

liftM matches dense to ≤ 1.6e-6 absolute over the grid; xc to ≤ 4e-5, with errors growing with ℓ (larger cond). The
curve is still increasing at ℓ = 0.9 on this synthetic data, so the grid does not contain the maximiser; the check is
the agreement, not the optimum. [computed]

## (k) Where it breaks, and verdicts

| prediction | verdict |
|---|---|
| P1 stock circle fails for ν > 0, exact at ν = 0 | **confirmed**; failure grows with ν (far 1e-2 → 4e-2; solve useless at ν = 2.5) |
| P2 one kernel on the hemisphere is not enough for β ≥ 2 | **partly refuted**: fine at β = 2 (ν = 0.5, 0.7); 30–70× short at β = 3, 4 |
| P3 multi-kernel ring: exact for integer ν, fails for half-integer | **confirmed at ν = 1, 0.5, 1.5**; *unexpectedly within ε at ν = 2.5* |
| P4 lifted multi-kernel reaches ε for all ν | **confirmed** (incl. fractional D = 2.6), with 1 latitude |
| P5 volume proxies reach ε with more points | **refuted on cost**: xc uses ~105 points vs 110–236, and fewer skeletons |
| P6 RTE control liftable with one kernel | **confirmed** |
| P7 rank decreases with ν and ℓ | ℓ: **confirmed (weakly)**; ν: **refuted** (flat) |
| P8 conditioning separate from compression; log det accurate | **confirmed** |
| Coordinator's B2: "Matérn cannot be lifted" | **refuted as stated**: no *Yukawa* lift (correct), but an order-2β lift with β kernel orders exists and works |

**Where it still breaks.**
- The lifted multi-kernel proxy costs **+60% skeletons** over exact and +25% over xc, and a 2.3–2.6× larger root block.
- At equal rank it is not the best proxy.
- ν = 0.7 is supported only by computation (fractional D).
- The randomized ID needs tighter stopping to reach ε.
- Small nuggets make the *problem* ill-conditioned; no compressor fixes that.

## (l) Literature and novelty (searched 2026-11-14; standard web searches, not exhaustive)

- Fast GP / kriging solvers with hierarchical compression are a crowded area: HODLR for GPs (Ambikasaran,
  Foreman-Mackey, Greengard, Hogg, O'Neil 2016) [cited]; skeletonization for spatial GP maximum likelihood with a proxy
  annulus and log-det (Minden, Damle, Ho, Ying 2017, arXiv:1603.08057) [cited]; H2Pack / Xing–Chow kernel-independent
  proxy points (2020) [cited]; sparse Cholesky for Matérn as Green's functions (Schäfer–Sullivan–Owhadi) [cited];
  SKI/KISS-GP, Vecchia approximations, and the SPDE/GMRF approach (Lindgren–Rue–Lindström 2011) [cited, not used].
- The searches found **no** use of (i) a proxy surface justified by an order-2β operator, (ii) multi-order Matérn
  kernels as proxy rows, or (iii) a lifted (extension-dimension) proxy for Matérn. That matches the B3 note's one
  earlier search. **Conjecture-level novelty only**: kernel-independent FMMs with 3D equivalent surfaces may embody
  similar information implicitly, and the measured advantage over a selected volume proxy is negative. So the
  contribution is the *explanation* (why the circle fails, why a volume proxy works, which ν are exactly liftable),
  not a better compressor.

## (m) Claim table

| # | claim | label |
|---|---|---|
| 1 | C_ν (d = 2) is the Green's function of (κ² − Δ)^β in R^{2(β−ν)}, any integer β ≥ ν + 1; no β = 1 lift for ν > 0 | proved |
| 2 | β kernel orders on a sphere of R^D determine the exterior field (integer D) | proved; fractional D conjecture |
| 3 | (κ² − Δ_D)[r^μK_μ] ∝ r^{μ−1}K_{μ−1} in every D; the chain terminates only in D = 2(β − ν) | proved + computed (tests) |
| 4 | Stock circle fails for ν > 0 (far 1e-2 … 4e-2, strong) and is exact for K₀ | computed (R1) |
| 5 | Lifted multi-kernel proxy reaches ε at ν = 0.5, 0.7, 1, 1.5, 2.5 with ≥ 1 latitude | computed (R1, R2) |
| 6 | It keeps ≈ 1.6× exact's skeletons; xc/annulus ≈ 1.3× | computed (R1) |
| 7 | At equal rank, volume proxies beat the lift by 2–10× in far error | computed (R3) |
| 8 | Multi-kernel ring within ε at ν = 2.5 | computed; explanation conjecture |
| 9 | log det = Σ log\|det X_RR\| + log\|det root\| | proved |
| 10 | log det, posterior mean and likelihood match dense to cond-limited accuracy, N ≤ 8192 | computed (R4–R6) |
| 11 | Forward error ≈ cond × backward error; backward error ≈ ε for every sound compressor | computed (R4) |
| 12 | No prior lifted/multi-order proxy for Matérn found | cited search, weak (standard searches only) |
