# E7: the screened (Yukawa / modified Helmholtz) kernel K0(κr), and the heat equation by Rothe's method

**Bottom line.**
- Swapping the Laplace kernel for K0(κr) is almost free.
  - The kernel-only swap (Y1) is 1 changed line plus a 9-line kernel.
  - The physical problem (Y2, a Dirichlet BVP solved with a double layer) adds about 50 lines of kernel and geometry code. Of those, 2 lines are the only change the proxy needs.
  - No driver line changes. Fast and dense solves agree to ≤ 3e-8 at every κ from 1e-12 to 1e4, apart from the breakdowns below.
- A heat-equation solver follows with one factorization reused for every time step. It is backward-Euler exact against an analytic semi-discrete solution and O(dt) against the continuous solution.
- **The rank question has a sharper answer than "ranks fall with κ".** Whether ranks fall depends on dial 4's *tolerance type*:
  - With the harness's relative ID tolerance (ε·|R₁₁|), ranks first **rise** (×1.2–1.7 at κ·hw ≈ 1–15) and only then fall.
  - With an absolute tolerance, ranks fall to 0. Strong-admissibility far ranks are 0 for every box once κ·hw ≳ 10.
  - The relative tolerance also **breaks down** when a whole far block falls into the subnormal floating-point range (κ ≳ 1000 here). The absolute tolerance fixes that.
- Two things limit how far the cost can fall:
  - The cost does not go to zero: it hits a floor set by the blocks kept exact. Under *strong* admissibility that floor is 10× higher than under *weak*, because strong keeps the touching neighbours dense even when they are numerically negligible.
  - The *answer* (not the solver) degrades as κh grows, like ≈ 0.02 κ²h³. This is a quadrature limit.
- **Follow-up (r7, section (g)): near-field truncation removes strong's floor.** `NearTruncated` is a 25-line admissibility wrapper. It demotes touching neighbours to the far field.
  - With the requested **norm** criterion (‖block‖_F ≤ τ), it only removes the floor once κh ≳ 5–20, i.e. where the discretization is already under-resolved (the answer is off by ~1e-3 or more). Adjacent points on a resolved grid are never negligible.
  - With a **rank** criterion (absolute-ε rank ≤ ¼ of the box), it removes the floor at every κ. On the volume problem it beats both weak and strong at κ = 100–300.
  - Solve error and heat-equation accuracy are unchanged in every case.
  - Caveat: the rank test's SVDs are not counted in the flop estimate, and they dominate wall time.

Reproduce: `PYTHONPATH=. .venv/bin/python -m unified.explore.e7_yukawa_heat.run` (r1–r5 take ≈ 12 min, r6 takes 20 s, r7 takes 200 s; `--quick` takes ≈ 1 min; `--only r6` or `--only r7` runs one part). Data are in `results/r*.json`, figures in `results/r*.png`, logs in `results/run_log*.txt`.
Tests: `.venv/bin/python -m pytest -q unified/explore/e7_yukawa_heat` (15 tests).

---

## (a) The physics, in plain words

**The equation.** Take −Δu + κ²u = 0. Compared with Laplace's equation, it has one extra term, κ²u. That term makes influence die off exponentially with distance, with decay length 1/κ. The free-space response to a point source (the Green's function) is G_κ(x, y) = K0(κ|x−y|)/(2π). Here K0 is the modified Bessel function: it behaves like −log r near the source (just like Laplace) and like e^{−κr}/√r far away.

**Where it appears.**
1. **Screened electrostatics (Debye–Hückel).** In salt water, mobile ions crowd around a charge and cancel its field beyond the Debye length 1/κ. This is used for proteins and colloids in solution.
2. **Yukawa potential.** The same kernel appears as the potential of a massive force carrier.
3. **Implicit time stepping of diffusion: the use we exploit.**
   - For the heat equation u_t = Δu, one backward-Euler step of size dt reads (u^n − u^{n−1})/dt = Δu^n, that is, (−Δ + κ²)u^n = κ²u^{n−1} with **κ² = 1/dt**.
   - So every implicit time step is a screened-Poisson problem.
   - A small time step means strong screening: in one step, heat only travels about √dt = 1/κ.
   - Stiff diffusion solvers (heat, Navier–Stokes viscous terms, Allen–Cahn, Cahn–Hilliard) solve this equation thousands of times.

**The boundary-integral formulation (problem Y2).**
- We solve the interior Dirichlet problem (−Δ + κ²)u = 0 in the starfish domain, with u = g on its boundary Γ.
- We write the solution as a double-layer potential, u = D_κ[σ]: a density σ of dipoles on Γ, with kernel D_κ(x, y) = ∂_{n_y} G_κ = −κK1(κr)(y−x)·n_y/(2πr).
- The jump relation turns the boundary condition into a second-kind equation, (−½I + D_κ w)σ = g. Here w are trapezoid weights.
- This equation is uniquely solvable for **every** κ > 0, because −Δ + κ² is a positive operator. So there are no resonances, unlike Helmholtz.
- **Y1** is the toy that SPEC's P1 uses, A = I + S_κ w (single layer), with only the kernel swapped.

**The heat equation with boundary unknowns only (r5).**
- Each Rothe step has a *volume* source κ²u^{n−1}. Normally that needs a volume potential, which in 2D means quad-tree work.
- Instead we write u^n = Σ_{j≤n} L_{n−j}[σ_j], with history layers L_p = (−1/dt)^p/p! · ∂^p_{κ²} D_κ.
- Differentiating (Δ − κ²)D_κ = 0 p times with respect to κ² gives (Δ − κ²)∂^pD = p ∂^{p−1}D. Hence (I − dtΔ)L_p = L_{p−1}, and (I − dtΔ)u^n = u^{n−1} holds automatically (**proved**, two lines).
- In closed form, L_p(x, y) = −(κ/2π) (z/2)^p/p! K_{p−1}(z) (y−x)·n_y/r, with z = κr. This uses (d/dκ²)^p K0(κr) = (−r/2κ)^p K_p(κr), which follows from the standard identity (z⁻¹d/dz)^m[z^{−ν}K_ν] = (−1)^m z^{−ν−m}K_{ν+m}.
- Each step solves **the same** matrix: (−½I + D_κ w)σ_n = g_n − Σ_{j<n} L_{n−j}w σ_j. So it is factor once, then solve every step.
- This is the Chapko–Kress idea (exterior problem; cited), written here for the interior problem.

**Analytic checks.**
- **BVP.** We use g = the trace of three screened point charges placed outside Γ, then evaluate u at five interior points.
- **Heat, semi-discrete.** A unit heat source switched on at t = 0 at x0, just outside Γ, has the exact backward-Euler solution u^n = Σ_{p<n} M_p(|x−x0|), with M_p = (z/2)^p K_p(z)/(2πp!). We verified (I − dtΔ)M_p = M_{p−1} numerically in a test.
- **Heat, continuous.** The same source has the continuous solution u = E1(|x−x0|²/4t)/(4π).

## (b) Predictions, written before running (frozen in `PREDICTIONS.md`), with verdicts

| Dial | Prediction | Verdict |
|---|---|---|
| 1 tree | unchanged | **held**: same `Tree`, max_leaf 32 |
| 2 admissibility | Weak/Strong unchanged and correct. A screening truncation (drop far blocks below ε) is valid once κ·dist ≳ log(1/ε) | **held**, plus a new finding: at large κ, *Strong is worse than Weak* (its exact near field is a cost floor, see (e)) |
| 3 representation | unchanged | **held** |
| 4 sampling | Exact unchanged. Proxy valid with a monopole ring at every κ; for the double layer the proxy *sources* must be monopoles (~10-line wrapper) | **held for κ ≥ 0.1 (2 lines needed).** **Failed** in two places I did not predict as stated: κ → 0 in weak RS loses digits (fixed with a centred ring, 13 lines), and κ ≳ 1000 gives subnormal ring rows and a NaN breakdown (P5 predicted waste, not a crash) |
| 5 algebra | unchanged; factor reused for all time steps | **held** |
| E5 complex-T instability | absent | **held**: K0 is real, so T is real (proved; cond(A) ≤ 3.5) |
| P1 relative-ε rank grows like √κ | | **Partly refuted.** It grows only for κ·hw ≈ 1–15, by ×1.2–1.7, then falls. The √κ asymptotics assumed box points on a circle at fixed radius; real boxes have a dynamic range e^{κ·hw} across their own points, so the relative ID drops the far-away columns too |
| P2 absolute-ε ranks → 0, cost falls with dt | | **Held for ranks.** Cost falls only to a floor (leaf blocks for Weak, dense near field for Strong) |
| P3 κ → 0 harmless (DLP); Y1 gains a rank-one constant | | DLP exact-ID harmless. **Y1 cond grows like log(1/κ)** (1.7 → 43). The monopole proxy is *not* harmless in weak RS (r6) |
| P4 κh ≳ 1: fast = dense, but the answer degrades | | **Held.** Fast = dense at every κ. Analytic error ≈ 0.02 κ²h³ (O(h³) at fixed κ); it is ~1e-3 near κh ≈ 1 and above 0.1 when κh ≳ 20 |
| P5 proxy wastes rank at large κ | | **Held, and worse.** Strong/proxy keeps 2× to 13× more skeletons than exact-abs at κ = 30–1000, and breaks at κ = 3000 |
| P6 heat: one factorization, backward-Euler exact, O(dt) | | **Held** |

## (c) Size of the change (non-blank code lines, docstrings excluded)

| Piece | Lines | Dial touched |
|---|---|---|
| Y1: SPEC's P1 with `kernels.laplace2d` → `slp_kernel(κ)` | **1 changed + 9** (the kernel) | none (kernel only) |
| Y2 physics: double-layer kernel 19 + starfish normals/curvature 19 + matrix 7 + target evaluation 4 | 49 | none (kernel/problem) |
| …of which the proxy needs: off-curve columns → monopoles | **2** | 4 (sampling), compatibility only |
| `AbsTolExact` + `abs_id` (absolute ID threshold) | 12 | 4 (tolerance) |
| `Truncated` + `cutoff_radius` (screening truncation) | 31 | 2 (admissibility, as a compressor wrapper) |
| `CenteredProxy` (κ → 0 fix) | 13 | 4 |
| `NearTruncated` (follow-up: demote negligible / low-rank touching neighbours) | 25 | 2 (admissibility wrapper; `rskel` unchanged) |
| Heat: history kernel 3 + `rothe_heat` 19; analytic solutions 18 (test only) | 22 (+18) | 5 (reuse of one factor) |
| `unified/core`, `unified/drivers` | **0** | — |

## (d) Verification

### r2: κ sweep, N = 4096, ε = 1e-8, Y2 (double layer). Total skeletons are summed over levels; err = ‖x − x_dense‖/‖x_dense‖

| κ | weak: rel / abs / proxy skeletons | strong: rel / abs / proxy skeletons | max fast err (non-breakdown) | analytic err (fast, exact-rel) | cond A |
|---|---|---|---|---|---|
| 1e-3 | 2059 / 1678 / 2695 | 1970 / 1594 / 2762 | 1.4e-7 (weak proxy, see r6); others ≤ 2.9e-8 | 1.3e-10 | 5.1 |
| 1 | 2480 / 1775 / 2970 | 1989 / 1576 / 3035 | 2.7e-8 | 1.5e-10 | 3.5 |
| 30 | 3455 / 2357 / 3960 | 2221 / **993** / 3752 | 1.9e-8 | 1.7e-7 | 1.1 |
| 100 | 3303 / 2217 / 3982 | 1869 / **302** / 4058 | 9.0e-9 | 2.0e-6 | 1.09 |
| 1000 | 1732 / 1090 / 1732 | **BREAKDOWN** / 365* / 2277 | 7.2e-10 | 1.8e-4 | 1.01 |
| 3000 | 1072 / 538 / 1072 | **BREAKDOWN** / 365* / **BREAKDOWN** | 1.7e-10 | 8.1e-4 | — |
| 1e4 | 516 / **22** / 516 | 365* / 365* / 319 | 2.3e-10 | (u underflows) | 1.004 |

\*Strong never compresses the level-1 boxes, because their far field is empty. The driver reports them at full size, so 365 means "every compressible box has rank 0". At κ = 1e4 all three strong variants are rank 0 everywhere.

**Accuracy.**
- Every variant that does not break down meets ε against dense, up to about 3ε.
- The absolute-tolerance variants sit at 1–3e-8, because τ = ε·½ is applied per column. Use τ = ε/10 for a strict ε.
- The exception is weak/proxy at κ ≤ 0.1, at 1–1.4e-7. That is the κ → 0 defect, fixed in r6.

**Analytic error.**
- The fast solver gives the same digits as dense at every κ (r3: identical to 2 digits, apart from ε-level differences at small κ).
- So the analytic error is pure discretization.

**Y1 (the kernel-only swap).** Weak and strong, with exact and proxy compressors, all reach ≤ 1.2e-9 for κ ≤ 1000. Strong breaks at κ = 3000 with *both* exact-rel and proxy (see (e)).

### r5: the heat equation (N = 1024, 16 steps, strong admissibility with exact-abs + truncation)

| dt | κ | fast vs semi-discrete exact | dense vs exact | fast vs dense | max principle 0 ≤ u ≤ max g | factor time | march time (16 steps) |
|---|---|---|---|---|---|---|---|
| 1e-2 | 10 | 3.2e-7 | 3.2e-7 | 6.5e-10 | yes | 0.04 s | 6.1 s |
| 1e-3 | 31.6 | 1.2e-6 | 1.2e-6 | 7.2e-10 | yes | 0.02 s | 4.8 s |
| 1e-4 | 100 | 1.2e-5 | 1.2e-5 | 9.2e-13 | yes | 0.01 s | 4.1 s |
| 1e-5 | 316 | 4.2e-4 | 4.2e-4 | 5.0e-17 | yes | 0.01 s | 3.9 s |

- The error against the exact semi-discrete solution equals the BVP quadrature error at that κ. That error grows like κ²h³, and it is the same for the dense solve.
- So the Rothe history layers are exact, and the fast factorization contributes nothing measurable.
- The march time is dominated by building the dense history matrices (O(n_steps·N²)), not by the solver. Fast history summation is the next bottleneck (see the literature section).

**O(dt) convergence to the continuous solution E1(r²/4t)/4π at T = 0.04.**

| steps | 4 | 8 | 16 | 32 |
|---|---|---|---|---|
| error | 5.81e-3 | 3.19e-3 | 1.68e-3 | 8.59e-4 |

The ratios are 1.82, 1.90 and 1.96, approaching 2. That is first order, as expected for backward Euler (**computed**).

## (e) Where it breaks (parameter sweeps)

### 1. Ranks versus κ (r1, N = 4096, mean over boxes per level, ε = 1e-8)

| level (points per box) | κ·hw ≈ 0 | κ·hw ≈ 3 | κ·hw ≈ 30 | κ·hw ≈ 300 |
|---|---|---|---|---|
| strong far, rel / abs | 12.3 / 11.5 (lvl 2) | 13.7 / 0–6 | 9 / **0** | 0.6 / 0 |
| weak far, rel / abs | 16 / 15.7 (lvl 2) | 21.5 / 17.7 | 14.5 / 9.5 | 4.9 / 2.7 |
| proxy ring (128 points), rel | 13.8 (lvl 2) | 19–20 | **22–23** | 5 |

(`r1_ranks.png` has all levels.)

**Mechanism (proved / heuristic).**
- *Absolute* rank of a separated block is 0 once max|entry|·√(size) < τ, that is, once κ·dist ≳ log(1/ε) + O(log κ). For the strong far field, dist ≥ 2·hw, which gives the drop near κ·hw ≈ 8–10.
- *Relative* rank first grows. The Graf terms I_m(κρ)K_m(κR) decay in m more slowly once κρ ≳ 1, so for intermediate κ more modes sit above ε·(largest).
- It then falls, because the columns of a box differ in size by e^{κ·(spread of distances)}. When that spread exceeds log(1/ε), the relative ID simply drops the shielded columns.
- My √κ prediction modelled a circle of box points all at the same radius, which a real box is not.

### 2. Breakdown of the relative ID in the subnormal range (r2, κ ≥ 1000; strong only)

- Strong far blocks begin two boxes away. At κ = 1000–3000, all their entries are about e^{−κ·2hw} ≈ 1e-300, which is subnormal.
- `interp_decomp` then tests |R_kk| > ε·|R₁₁| with ε·|R₁₁| = 0, so it keeps garbage pivots. Solving with them gives T = inf/NaN, and `lu_factor` raises.
- Measured cases:
  - Y2: strong/exact-rel breaks at κ = 1000 and 3000; strong/proxy-rel breaks at 3000.
  - Y1: both strong variants break at 3000.
  - Localised once (κ = 3000, N = 2048): M has |M| ≤ 9.6e-304 and its smallest nonzero entry is 5e-324.
- At κ = 1e4 the entries underflow to exactly 0, rank 0 is returned, and everything works again.
- The absolute tolerance (`AbsTolExact`, 12 lines) never breaks. That makes it the right default for any decaying kernel. A one-line core fix would be "rank 0 if |R₁₁| < tiny·‖A‖". This is a harness bug that the Laplace and Helmholtz kernels never exercise (**computed**).

### 3. The cost floor, and why Strong loses to Weak at large κ (r2 flops; dense LU = 4.6e10 at N = 4096)

| κ | weak exact-abs flops | strong exact-abs flops |
|---|---|---|
| 1 | 5.3e6 | 4.6e7 |
| 100 | 4.8e6 | 4.0e7 |
| 1e4 | 4.1e6 | 3.9e7 |

- The cost does fall with dt, but only by 20% (weak) or 15% (strong) between κ = 1 and 1e4.
- At large κ, every far rank is 0, so the factorization becomes a block-sparse LU of what is kept exact.
  - **Weak:** the leaf diagonal blocks (2/3·32³ per leaf).
  - **Strong:** the leaf blocks *plus dense Schur updates onto all touching neighbours*. Those blocks are numerically negligible (e^{−κh}), but the strong rule keeps them exact by definition.
- So for strongly screened kernels the "near field" should itself be truncated (a sparse near field). Under the present dials, **Weak is the better admissibility for κ·hw ≫ 1** (10× fewer flops). The follow-up in section (g) implements and measures this.
- Further reduction needs smaller leaves. At κh ≫ 1 the matrix is essentially −½I plus a band.

**Screening truncation (`Truncated`).**
- It gives the *same* skeleton counts as exact-abs in every r2 row. That is consistent with the bound it was built on: dropped entries have row sums ≤ ε·½.
- It cuts factor time 10× at large κ, because the far blocks are never formed. r4, N = 4096, κ = 300: 0.04 s vs 0.47 s for strong; 0.06 s vs 0.47 s for weak.
- With a proxy base it changes nothing, because the ring still stands for "everything outside".

### 4. Factor cost versus N (r4; flops, N = 1024 → 8192)

| configuration | κ = 1 | κ = 300 |
|---|---|---|
| weak/exact-rel | 2.9e6 → 1.2e7 | 3.9e5 → 1.1e7 |
| strong/abs+trunc | 5.9e6 → 8.8e7 | 2.4e6 → 7.9e7 |

- Roughly linear growth (exponent 0.7–1.3), with all fast errors ≤ 3.2e-8.
- Wall-clock factor times at N = 8192 are 0.13–1.6 s. Building the dense matrix dominates, because the driver is dense by design.

### 5. The κ → 0 (Laplace) limit (r6, N = 2048)

| κ | 1 | 1e-1 | 1e-3 | 1e-6 | 1e-9 | 1e-12 |
|---|---|---|---|---|---|---|
| weak/proxy-rel err | 7.7e-9 | 3.1e-8 | 1.0e-7 | 2.5e-7 | 4.1e-7 | **5.5e-7** |
| weak/centred-proxy err | 2.8e-9 | 6.1e-9 | 6.0e-9 | 6.0e-9 | 6.0e-9 | 6.0e-9 |
| exact-rel and strong/proxy err | ≤ 1.3e-8 at every κ | | | | | |
| Y1 cond(I + S_κ w) | 1.70 | 4.48 | 11.3 | 21.7 | 32.3 | 43.0 |

**Mechanism (proved).**
- K0(κr) = −log r + c(κ) + o(1), with c(κ) = log(2/κ) − γ. Every monopole ring row therefore carries the same large constant, so the relative ID resolves the useful −log r variation only to about ε·c(κ).
- **Fix (`CenteredProxy`):**
  - subtract from each row its value at the box centre;
  - add one constant row.
- The row space then contains the original one *exactly*. The result is κ-independent 6e-9, at +20–30% skeletons (1600 vs 1239–1499). The plain proxy had been *under*-ranking.
- Strong/proxy is immune. Its far field is farther away, so the constant matters less. Y1's conditioning grows like c(κ) because S_κ picks up the rank-one term c(κ)·(1·wᵀ). The *double-layer* matrix is unaffected (cond 5.1 at κ = 1e-12, same as Laplace).

### 6. Large κ: the answer degrades, the solver does not (r3, analytic error, fast = dense)

| κ | N = 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|
| 10 | 8.9e-6 | 1.1e-6 | 1.4e-7 | 4.5e-8 |
| 100 | 9.3e-4 | 1.2e-4 | 1.6e-5 | 2.0e-6 |
| 1000 | **0.31** | 2.5e-3 | 1.1e-3 | 1.8e-4 |
| 3000 | **1.2** | 0.16 | 2.3e-3 | 8.1e-4 |

- The error converges at O(h³) for fixed κ: the ratios are ≈ 8 per doubling of N.
- An empirical fit is err ≈ 0.02 κ²h³ while κh ≲ 5. It becomes O(1) at κh ≳ 20.
- **Cause (proved).** κK1(κr) = 1/r + (κ²r/2)log(κr/2) + …, so D_κ contains a κ²r²log r term. The trapezoid rule is then only third order, whereas it is spectral for Laplace. And once κh > 1, the kernel's boundary layer 1/κ is narrower than the grid spacing.
- **Fix.** This is not a harness issue. Standard remedies are Kapur–Rokhlin or Alpert end corrections, or Kress product quadrature. They only modify near-diagonal entries, which is near field, so all five dials remain unchanged (**conjecture**, not implemented).
- For heat this means **small dt needs boundary resolution h ≲ √dt** (κh ≲ 1).

## (g) Follow-up: near-field truncation to remove strong admissibility's cost floor (r7)

**What was built.** `NearTruncated(base=Strong(), A, tau, rank_frac)` is an `Admissibility` wrapper, 25 lines, in `yukawa.py`. It returns Strong's touching neighbours *minus* those judged dispensable on the original matrix blocks M = [A[N,B]; A[B,N]ᵀ]:
- **nt-norm** (the requested criterion): demote if max(‖A[B,N]‖_F, ‖A[N,B]‖_F) ≤ τ = ε·½.
- **nt-rank** (added after the first test, see below): demote if the absolute-ε rank #{σ_i(M) > τ} ≤ ¼·min(|B|, |N|).

A demoted neighbour joins the far field, so the box is *compressed* against it rather than Schur-coupled to it.

**Why accuracy is safe by construction (proved).**
- The compressor is `AbsTolExact`. It reads the current working matrix W, so it sees the demoted neighbour's actual entries, including any Schur fill-in, and keeps whatever rank those need at the absolute tolerance.
- The heuristic therefore only decides *what stays exact* (cost), never what is dropped.
- With every neighbour demoted the rule is literally Weak; with none demoted it is literally Strong.
- `rskel` is unchanged. All runs use the screening truncation of the far field as well (`Truncated`).

**Not predicted in advance (honest).** I expected the norm criterion to remove the floor at large κ. The first test (κ = 3000, N = 4096) demoted only 308 of 824 neighbour pairs. The reason:
- Touching boxes share an edge, and the points on either side of it are one spacing h apart. The block norm is therefore ≳ G_κ(h)·w ~ e^{−κh}, which only falls below τ once κh ≳ 20.
- In the resolved regime (κh ≲ 1) only *corner* neighbours whose points happen to be far apart get demoted.
- The edge-coupling blocks are, however, **low-rank**: only points within ~log(1/ε)/κ of the shared edge couple. That motivated the rank criterion. The norm criterion is its rank-0 special case.

### r7a: Y2 curve (double-layer BVP), N = 4096; dense LU = 4.6e10 flops and 1.68e7 stored numbers

**Factor flops (estimate)**

| κ (dt = 1/κ²) | κh | weak | strong | nt-norm | nt-rank |
|---|---|---|---|---|---|
| 1 (1) | 0.002 | 5.3e6 | 4.6e7 | 4.6e7 | 7.6e6 |
| 30 (1.1e-3) | 0.07 | 5.0e6 | 4.3e7 | 4.3e7 | 1.0e7 |
| 300 (1.1e-5) | 0.66 | 4.5e6 | 3.9e7 | 3.4e7 | 5.9e6 |
| 1000 (1e-6) | 2.2 | 4.2e6 | 3.9e7 | 2.5e7 | 4.8e6 |
| 3000 (1.1e-7) | 6.6 | 4.2e6 | 3.9e7 | 2.3e7 | 4.3e6 |
| 1e4 (1e-8) | 22 | 4.1e6 | 3.9e7 | 7.2e6 | 4.1e6 |
| 3e4 (1.1e-9) | 66 | 4.1e6 | 3.9e7 | 4.1e6 | 4.1e6 |

**Neighbour pairs demoted, out of 824**

| κ | 1 | 30 | 300 | 1000 | 3000 | 1e4 | 3e4 |
|---|---|---|---|---|---|---|---|
| nt-norm | 0 | 18 | 226 | 284 | 308 | 802 | 824 |
| nt-rank | 620 | 570 | 686 | 742 | 776 | 824 | 824 |

**Stored numbers (memory)**

| κ | weak | strong | nt-norm | nt-rank |
|---|---|---|---|---|
| 1 | 2.0e5 | 5.9e5 | 5.9e5 | 2.4e5 |
| 1000 | 1.7e5 | 5.2e5 | 4.0e5 | 1.8e5 |
| 3e4 | 1.5e5 | 5.2e5 | 1.5e5 | 1.5e5 |

**Accuracy (all κ).**
- Solve error against dense: ≤ 2.9e-8 for every rule.
- The truncated rules are no worse than the strong/weak baselines: nt-norm ≤ 2.9e-8, nt-rank ≤ 2.6e-8.
- Analytic BVP error is identical across rules to 2–3 digits (e.g. 1.95e-6 at κ = 100 and 8.1e-4 at κ = 3000 for all four), so truncation adds nothing visible to the discretization error.

### r7b: Y3 volume problem (P2 with the screened kernel, A = I + K0(κr)/(2πN), N = 4096 random points; h ≈ N^{−1/2} = 0.0156)

**Factor flops (estimate)**

| κ | κh | weak | strong | nt-norm | nt-rank |
|---|---|---|---|---|---|
| 1 | 0.016 | 1.4e8 | 3.8e8 | 3.8e8 | 1.6e8 |
| 30 | 0.47 | 1.2e8 | 2.9e8 | 2.9e8 | **1.2e8** |
| 100 | 1.6 | 8.2e7 | 6.9e7 | 6.9e7 | **5.8e7** |
| 300 | 4.7 | 1.5e7 | 3.9e7 | 2.4e7 | **9.3e6** |
| 1000 | 16 | 8.8e5 | 3.9e7 | 3.4e6 | **8.8e5** |
| 3000 | 47 | 8.5e5 | 3.9e7 | 8.8e5 | 8.5e5 |

**Neighbour pairs demoted, out of 2376**

| κ | 1 | 30 | 100 | 300 | 1000 | 3000 |
|---|---|---|---|---|---|---|
| nt-norm | 0 | 0 | 0 | 696 | 2026 | 2362 |
| nt-rank | 314 | 418 | 996 | 2118 | 2376 | 2376 |

**Stored numbers (memory)**

| κ | weak | strong | nt-norm | nt-rank |
|---|---|---|---|---|
| 300 | 3.4e5 | 5.5e5 | 4.2e5 | 2.8e5 |
| 1000 | 7.8e4 | 5.5e5 | 1.4e5 | 7.8e4 |

**Accuracy.** Solve error against dense is ≤ 3.8e-9 for every rule and κ.

**Reading.**
- **Volume.** Here the κ-dependence is genuine. At small κ, touching blocks in 2D have Laplace-like rank (~√(box)), so few are demoted. As κ grows, the coupling concentrates in a strip of width ~log(1/ε)/κ along the shared edge, and the rank criterion demotes progressively more neighbours.
  - At κ = 100–300, nt-rank is **cheaper than both** weak and strong: 5.8e7 vs min(8.2e7, 6.9e7), and 9.3e6 vs min(1.5e7, 3.9e7). It keeps the neighbours that strong needs and drops the ones weak can compress.
  - At κh ≳ 15, strong's floor of 3.9e7 falls to weak's 8.8e5 (45×).
- **Curve.** Touching blocks are low-rank at every κ (quasi-1D geometry), so nt-rank ≈ weak throughout. There, near truncation mainly confirms that weak was already the right rule for curves.
- **nt-norm.** It removes the floor only where κh ≳ 5 (volume) or ≳ 20 (curve), i.e. where the trapezoid discretization is under-resolved (r3: error ≳ 1e-3). **In the regime where the answer is accurate, norm-based near truncation saves 0–37% (curve κ ≤ 1000) and nothing on the volume problem for κ ≤ 100.** Negative result.

### r7c: the heat equation through the truncated factorizations (N = 1024, 16 steps)

| dt | error vs semi-discrete exact (all 4 rules, = dense) | fast vs dense: weak / strong / nt-norm / nt-rank | flops: weak / strong / nt-norm / nt-rank |
|---|---|---|---|
| 1e-2 | 3.2e-7 | 2.7e-10 / 6.5e-10 / 6.5e-10 / 1.2e-9 | 1.6e6 / 4.3e6 / 4.3e6 / 3.8e6 |
| 1e-3 | 1.2e-6 | 6.0e-10 / 7.2e-10 / 7.2e-10 / 1.4e-9 | 7.3e5 / 3.0e6 / 3.0e6 / 2.4e6 |
| 1e-4 | 1.2e-5 | 1.2e-10 / 9.2e-13 / 9.6e-12 / 1.3e-10 | 4.7e5 / 2.4e6 / 2.4e6 / 9.5e5 |
| 1e-5 | 4.2e-4 | 6.5e-11 / 5.0e-17 / 5.0e-17 / 6.5e-11 | 3.1e5 / 2.4e6 / 1.6e6 / 4.5e5 |

- The time stepping still matches the analytic semi-discrete solution to the same digits as the dense solve, under every rule.
- With the nt-rank factor, the error against the continuous solution E1(r²/4t)/4π at T = 0.04 is 5.81e-3, 3.19e-3, 1.68e-3 and 8.59e-4 for 4, 8, 16 and 32 steps. That is identical to r5, so it is still O(dt).
- With dt fixed, the factor is built once and reused for every step. At dt = 1e-5, nt-rank cuts strong's factor flops 5.4× and its memory proportionally.

**Caveats.**
1. **Classification cost.**
   - The flop estimate counts the factorization only. The nt-rank test runs one SVD per neighbour pair at every level, and the coarse-level boxes are large.
   - In wall time it dominates: 1.1–1.9 s against 0.02–0.9 s for the factorization at N = 4096. nt-norm adds about 0.15 s.
   - Cheaper tests are needed for this to pay in wall time: e.g. reuse the ID the compressor computes anyway, use a randomized rank estimate, or test only the leaves and inherit upward. **Not implemented** (conjecture that it can be made O(N)).
2. The rank threshold ¼ is a single untuned choice.
3. Wall times are single runs on a shared machine. They are secondary evidence.

## (f) Literature: what is known and what (if anything) is new

**Known (cited).**
- **Kropinski & Quaife:**
  - "Fast integral equation methods for the modified Helmholtz equation", J. Comput. Phys. 230 (2011) 425–434 (arXiv:1006.0008);
  - "Fast integral equation methods for Rothe's method applied to the isotropic heat equation", Comput. Math. Appl. 61 (2011) 2436–2446 (arXiv:1007.5335).
  - Together they cover modified-Helmholtz BIEs plus FMM, used for heat via backward Euler, *with volume potentials* for the source. Quaife's SFU thesis (2011) extends this to forced heat and Allen–Cahn.
- **R. Chapko & R. Kress**, "Rothe's method for the heat equation and boundary integral equations", J. Integral Equations Appl. (DOI 10.1216/jiea/1181075987). They treat the exterior problem and reduce the Rothe sequence to boundary integral equations with recursively defined kernels, *without volume integrals*. Our L_p recursion is that idea, written as κ²-derivatives of D_κ for an interior problem.
- **Cheng, Huang & Leiterman**, "An adaptive fast solver for the modified Helmholtz equation in two dimensions", J. Comput. Phys. 211 (2006): volume FMM for this kernel. Greengard & Huang (2002) give the 3D screened-Coulomb FMM.
- **Fryklund, Kropinski & Tornberg**, Adv. Comput. Math. (2020): forced heat equation via modified Helmholtz, with layer and volume potential splitting.
- Graf's addition theorem for K0 (Abramowitz & Stegun 9.1.79 analogue for K_ν; Watson) and the Bessel derivative identities are textbook.
- Exponential decay making H-matrix far blocks negligible is folklore in the H-matrix community (e.g. Börm's work on dissipative Helmholtz kernels). I did not find a precise reference stating the *relative versus absolute tolerance* distinction for screened kernels; treat it as possibly known but unpublished.

**What may be new (to my knowledge; computed in this repo, not claimed as a literature first).**
1. The precise interplay of *tolerance type × admissibility* for screened kernels:
   - the relative-ε rank bump at κ·hw ≈ 1–15;
   - the absolute-ε collapse to 0;
   - the subnormal breakdown of relative-ε IDs;
   - Strong's near-field cost floor making Weak preferable at large κ.
2. The monopole-proxy precision loss as κ → 0 (∝ log(1/κ)) in weak RS, and the exact centred-ring fix.
3. A fast-direct-solver Rothe heat solver with boundary unknowns only: one rskel factorization reused for all steps, history via the closed-form L_p kernels. Chapko–Kress used dense Nyström; Kropinski–Quaife used FMM plus GMRES with volume potentials.

## Claims

| # | Claim | Status |
|---|---|---|
| 1 | Y1 needs 1 changed line + a 9-line kernel; Y2 needs 49 lines, 2 of them for the proxy; 0 driver/core lines | computed (line counts) |
| 2 | (−½I + D_κw)σ = g is uniquely solvable for every κ > 0 (no resonances) | cited / standard (positivity of −Δ+κ²) |
| 3 | A monopole proxy ring spans the regular solutions inside it for every κ > 0 (Graf; I_m K_m > 0) | proved (Graf) |
| 4 | T is real, so the E5 non-congruence instability cannot occur | proved; computed cond(A) ≤ 5.1, fast err ≤ 3e-8 |
| 5 | Fast solver = dense solve (≤ 3e-8) for κ ∈ [1e-12, 1e4] except in claims 8 and 10 | computed (r2, r3, r6) |
| 6 | Absolute-ε strong far ranks are 0 for κ·hw ≳ 10; weak far ranks reach 0 at κ·hw ≳ 1000 | computed (r1) |
| 7 | Relative-ε ranks rise ×1.2–1.7 for κ·hw ∈ [1, 15], then fall; the √κ growth prediction is refuted | computed (r1) |
| 8 | The relative-ε ID breaks (NaN T) on subnormal far blocks: strong at κ = 1000–3000; absolute ε fixes it | computed (r2, test) |
| 9 | Factor flops floor: weak ≈ 4e6, strong ≈ 4e7 at N = 4096; strong's floor is the dense near field | computed (r2); mechanism proved by counting |
| 10 | Monopole proxy in weak RS loses digits ∝ log(1/κ) (5.5e-7 at κ = 1e-12); the centred ring restores 6e-9 | computed (r6, test); mechanism proved |
| 11 | Y1 cond grows like log(1/κ); the double-layer cond does not | computed (r6) |
| 12 | Analytic BVP error ≈ 0.02κ²h³ (O(h³) at fixed κ), O(1) at κh ≳ 20; cause is the κ²r²log r term | computed (fit, r3); cause proved |
| 13 | (I − dtΔ)L_p = L_{p−1}, so the Rothe history needs no volume potential | proved; computed (r5 error = quadrature error) |
| 14 | Rothe + one fast factorization matches dense to ≤ 7e-10 and is O(dt) against the continuous solution (ratios 1.82 → 1.96) | computed (r5) |
| 15 | Screening truncation never changes the skeleton choice and cuts factor time ~10× at large κ | computed (r2, r4) |
| 16 | High-order near-diagonal quadrature corrections would remove the κ²h³ limit without touching any dial | conjecture |
| 17 | ~~A sparse (truncated) near field would remove Strong's cost floor~~. Now tested: rank-based near truncation removes it at every κ; norm-based truncation only at κh ≳ 5–20 | computed (r7) |
| 18 | `NearTruncated` with `AbsTolExact` is accurate to the absolute tolerance whatever neighbours it demotes (the ID sees the current W, including fill-in) | proved; computed (fast vs dense ≤ 2.9e-8 curve, ≤ 3.8e-9 volume, all κ) |
| 19 | Norm-based near truncation cannot demote edge-sharing neighbours while κh ≲ 1 (adjacent points h apart give block norm ≳ e^{−κh}); savings in the accurate regime are 0–37% | proved (bound); computed (r7a/b) |
| 20 | Rank-based near truncation (abs-rank ≤ ¼ box) is cheaper than both weak and strong on the screened volume problem at κ = 100–300 (5.8e7 / 9.3e6 flops vs 6.9e7 / 1.5e7) and reaches weak's floor at κh ≳ 15 | computed (r7b) |
| 21 | Heat time stepping through any truncated factor matches the dense solve (≤ 1.4e-9) and the analytic semi-discrete solution (3.2e-7 … 4.2e-4 = dense), and stays O(dt) | computed (r7c) |
| 22 | The rank classification can be made O(N) (e.g. reuse the ID, or randomized rank) so the flop savings also show in wall time | conjecture |
