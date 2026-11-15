# E6 — Stokes flow: rigid particles in a viscous fluid, through the unchanged harness

Reproduce everything with `PYTHONPATH=. .venv/bin/python -m unified.explore.e6_stokes.run` (156 s on a laptop, 3.8 GB peak; `--quick` takes 14 s). The data are in `results/r*.json` and the figures in `results/r*.png`. The full log is `results/run_log.txt`. Tests: `.venv/bin/python -m pytest -q unified/explore/e6_stokes` (6 pass). Type check: `.venv/bin/python -m mypy --strict unified/explore/e6_stokes` (clean).

Every claim is tagged **[computed]** (measured by `run.py`), **[proved]** (a short argument is given here), **[cited]** (from the literature) or **[conjecture]**.

---

## (a) The physics problem, in plain words

**Stokes flow** is the motion of a fluid so viscous, or at so small a scale, that inertia does not matter. Examples are bacteria and sperm swimming, red blood cells in capillaries, colloids settling in a jar, and microfluidic chips. The fluid velocity **u** and pressure p obey

  −Δ**u** + ∇p = 0 and ∇·**u** = 0 (viscosity set to 1).

The equations are linear and have no time in them. The flow at each instant is decided entirely by where the particles are and by the forces acting on them. A simulation of a suspension therefore solves one such problem per time step. That problem is a **mobility problem**:

> Given rigid particles B_1,…,B_K and the force F_k and torque T_k on each (e.g. gravity), find each particle's velocity U_k and spin Ω_k.

A second version is the **free particle in a background flow**. A particle carries no force and no torque, and sits in a shear flow **u**∞ = (γy, 0). We ask how fast it spins. For an ellipse the answer is Jeffery's formula from 1922 [cited, Jeffery 1922]:

  Ω = −γ (a² sin²φ + b² cos²φ)/(a² + b²).

**How it becomes a matrix.** Only the boundary curves of the particles are discretized. The flow is written as a *double layer*: a smooth layer of "stresslet" sources μ spread over each boundary curve, plus a point force G·F_k and a point torque R·T_k at each particle's centre. The double-layer kernel is

  D(x, y) = (1/π) (r·n_y) r rᵀ/|r|⁴, with r = x − y.

The condition "the fluid sticks to the rigid particle" gives one linear equation per boundary point and velocity component:

  (½ I + K + P) μ = −**u**∞ − Σ_k (G(·, c_k) F_k + R(·, c_k) T_k), and the rigid motion of body k is −(Pμ)|_k.

Here:
- K is the double-layer matrix on the curves. It is a **2 × 2 block per pair of points**, because velocity has two components.
- P is the projector onto the 3 rigid motions of each body (two translations and one rotation), with rank 3 per body. It removes the 3-dimensional null space of ½I + K.

This is the "completed double layer" second-kind formulation [cited: Power & Miranda 1987 in 3D; Rachh & Greengard 2016 in 2D]. Second kind means the matrix is identity-like and well conditioned: cond(A) ≈ 2 for one circle and 4–40 for the suspensions here [computed].

**Why it matters for this project.** It is a different PDE family from Laplace and Helmholtz:
- It is vector-valued.
- Pressure is coupled to velocity.
- The kernel is a 2 × 2 matrix.
- It carries a low-rank non-kernel term P.

If the harness handles it by swapping the kernel, the "one harness" thesis extends to systems of PDEs.

**Analytic checks** (all derived in `stokes.py`):
- **Translating or rotating circle of radius a** [proved; matches Pozrikidis 1992 ch. 2, cited]:
  - U = (½ − log a) F/(4π) and Ω = T/(4π a²).
  - Derivation: the exact flow is a Stokeslet plus a potential dipole, and its value on |x| = a is the constant (½ − log a)F/(4π).
  - In 2D the velocity of a forced body depends on the length unit through log a. This is the **Stokes paradox**: there is no decaying flow past a cylinder.
- **Jeffery's rate** for a free ellipse in shear [cited].
- **Reciprocity**: the 3K × 3K mobility matrix is symmetric (Lorentz reciprocal theorem) [cited, Happel & Brenner].
- **Positive dissipation**: F·U ≥ 0. In 2D this holds only for **zero net force**, because log-growing flows have infinite dissipation [proved by the Stokes-paradox argument; checked in r7].

---

## (b) Prediction before running: which dials stay and which must change

These were written down before any experiment; the verdicts come from the runs.

| Dial | Prediction | Reason | Verdict |
|---|---|---|---|
| 1 clustering/ordering | **unchanged** | Interleave the dofs (2i → u_x, 2i+1 → u_y at point i). Build the same `Tree` over the dof coordinates with every point listed twice. Both components then always share a box. | ✅ 0 lines changed. One `np.repeat` builds the tree points. |
| 2 admissibility | **unchanged** | The Stokeslet decays like log r and the stresslet like 1/r, the same smoothness class as Laplace. Weak and strong rules both apply. | ✅ Both work, to error ≤ 4e-11 at N_dof = 8192. |
| 3 representation (ID, skeletons, T) | **unchanged** | The ID is purely algebraic. | ✅ `core.interp_decomp` is untouched. |
| 4 sampling, exact | **unchanged** | `ExactCompressor` reads matrix entries and needs no kernel knowledge. | ✅ It works as is. Kernel swap only. |
| 4 sampling, proxy | **must change** | (i) Each proxy point needs a 2 × 2 block, plus the source normals of the box. (ii) The matrix is the non-symmetric double layer, so the two directions need different proxy "sources". (iii) P is not a kernel value, so a pure Green's representation cannot see it. (iv) The strong driver gives its Q-set (modified-entries) fix only to `ProxyCompressor` subclasses. | Partly right: (i), (ii) and (iv) are real. **(iii) is refuted**, because the explicit near rows already carry P (r5). |
| 4 proxy validity | **holds**, with one risk | Stokes has a Green's representation: velocity on a circle determines a decaying exterior Stokes flow. **Risk:** Stokeslet-only proxies on a circle of radius R = √e lose the constant fields, the 2D Stokes "degenerate scale". | Degeneracy **real** (proved, computed), but **masked** in the solver (r4, r4b). |
| 5 algebra | **unchanged** | `rskel_factor` only ever sees index sets. | ✅ 0 lines changed. |
| rank vs Laplace | **≈ 2× Laplace** at every level | A 2D Stokes velocity is built from **two** analytic functions (the Goursat / Sherman–Lauricella representation [cited, Greengard–Kropinski–Mayo 1996]); Laplace needs one. Same expansion order means twice the coefficients. | ✅ Ratio 2.00–2.2 for every N, ε, rule, compressor and geometry (r2, r3). |

---

## (c) Size of the change

| What | Lines (non-blank, no comments) | Dial |
|---|---|---|
| Stokeslet and stresslet kernels, interleaved 2 × 2 blocks (`stokes.py`: `_interleave`, `stokeslet`, `stresslet_dl`) | ≈ 28 | kernel only |
| Nyström assembly: ½I + KW, curvature diagonal limit −(κ/2π) t tᵀ, rigid projector P | ≈ 40 | kernel / physics, not a harness dial |
| Tree over the dofs | 1 (`np.repeat(X, 2, axis=0)`) | dial 1: none |
| `rskel_factor`, `Tree`, `Weak`/`Strong`, `interp_decomp`, `ExactCompressor` | **0** | 1, 2, 3, 5 |
| `BlockProxyCompressor` (`compressor.py`) | ≈ 60 | **dial 4 only** |
| Geometry, right-hand sides, analytic answers (`stokes.py`) | ≈ 110 | problem set-up, not the solver |

So **"kernel swap + one sampling dial"** is literally true:
- With exact sampling, Stokes needs **zero** harness changes.
- With proxy sampling, the sampler must be rewritten (≈ 60 lines), because the core `ProxyCompressor` cannot run on block kernels.
- **r0** [computed] shows the failure: plugging the stresslet into `core.ProxyCompressor` raises `ValueError: ... array at index 0 has size 80 and ... index 2 has size 40`. The proxy block has 2 columns per box point; the explicit block has 1 per dof.

**What the new sampler does.**
1. **Column side**, A[far, box] (the far velocity made by the box's densities): evaluate the double layer *at* n_p proxy points, giving 2 n_p rows. Each uses the box's own normals and quadrature weights.
2. **Row side**, A[box, far]ᵀ (the velocity in the box made by far densities): place proxy *sources* on the circle. These can be Stokeslets ("sl"), stresslets with the circle normal ("dl"), or both. Rows are scaled by the proxy arc length.
3. **Explicit rows**: far dofs inside the proxy circle, read from the current matrix.
4. **Strong admissibility only**: the active far dofs in all same-level boxes at Chebyshev distance 2. This is a conservative superset of FMM-LU's Q set, because Q only contains neighbours of neighbours. It replaces the driver's Q hook, which is reserved for `ProxyCompressor` subclasses and assumes a scalar kernel.
5. **Optional rigid rows**: V_box and (W V)_box, which span P's far blocks.
6. The same `interp_decomp` is applied to the stack.

---

## (d) Verification

### r1: analytic solutions, through the fast solver (weak RS, block proxy "sl", ε = 1e-12, n = 1024 per curve)

| Test | Cases | max abs error vs closed form | solve error vs dense |
|---|---|---|---|
| Circle mobility, U and Ω | a ∈ {0.3, 1, √e, 3} | **3.3e-16** | ≤ 1.2e-14 |
| Jeffery Ω, free ellipse in shear | b/a ∈ {1, .5, .2, .1} × φ ∈ {0, .7, 1.4} | **5.0e-14** | ≤ 1.3e-12 |

Both are [computed]. At a = √e the predicted U is exactly 0 and is reproduced; the 2D log scale causes no trouble here. The translational velocity of the free ellipse is ≤ 5e-17, as symmetry requires.

### r2: Stokes vs Laplace on the starfish (P1 curve), ε = 1e-10, max_leaf = 32 points (64 Stokes dofs)

Both use the same geometric boxes, with the tree depth taken from the point tree. The Laplace analogue is ½I + K_L W + (constants projector), the 2D *elastance* problem (Rachh–Greengard 2016).

| N points (Stokes dofs) | rule | exact: err (L / S) | exact: Stokes/Laplace skeleton per level | proxy-sl: err (L / S) | root size L / S (exact) |
|---|---|---|---|---|---|
| 1024 (2048) | weak | 5.2e-12 / 1.3e-11 | 2.03, 2.05, 2.06, 1.98 (levels 4→1) | 1.0e-11 / 8.9e-12 | 172 / 341 |
| 1024 (2048) | strong | 2.3e-12 / 7.8e-12 | 2.04, 2.08, 2.07, 2.07 | 1.3e-12 / 2.0e-12 | 200 / 413 |
| 4096 (8192) | weak | 2.8e-12 / 7.3e-12 | 2.03 (mean over 5 levels) | 2.6e-11 / 1.3e-11 | 168 / 341 |
| 4096 (8192) | strong | 1.9e-12 / 6.9e-12 | 2.06 (mean) | 2.1e-12 / 2.0e-12 | 199 / 413 |

The ratio **(mean over levels) is 2.01–2.11 for every N ∈ {512…4096}, both rules and both compressors** [computed]; see `r2_ranks.png`, middle panel. Over the ε sweep 1e-4…1e-12 it is 2.0–2.2. The ratio rises slightly at loose ε, and is a bit higher for proxy (2.07–2.21) than for exact (2.01–2.12) [computed].

Skeleton sizes per level are flat in N, for Laplace and Stokes alike. That is the usual O(N) behaviour for a curve. **Stokes changes the constant, not the complexity** [computed].

### r3: suspensions of K random ellipses (n = 128 per body, gap ≥ 0.05, area fraction fixed), ε = 1e-10

| K | N_dof (Stokes) | rule / comp | err vs dense LU | root | storage / N² | factor s | dense LU s |
|---|---|---|---|---|---|---|---|
| 16 | 4096 | weak / proxy | 2.0e-11 | 783 | 0.094 | 0.4 | 0.5 |
| 32 | 8192 | weak / exact | 1.8e-11 | 930 | 0.054 | 5.2 | 3.5 |
| 32 | 8192 | weak / proxy | 3.9e-11 | 1113 | 0.065 | 1.4 | 3.5 |
| 32 | 8192 | strong / exact | 7.1e-12 | 1100 | 0.086 | 5.6 | 3.5 |
| 32 | 8192 | strong / proxy | 1.8e-12 | 1621 | 0.134 | 2.2 | 3.5 |

These are [computed]. Over all K, rules and compressors:
- The skeleton ratio Stokes/Laplace is 1.99–2.07.
- The root ratio is 1.94–2.19.

Wall times are secondary evidence. The driver keeps a dense working matrix (O(N²) memory, as its docstring says), so it is not a fair speed comparison.

### r7: one factorization, the whole mobility matrix, and two physical invariants (K = 24 ellipses, N_dof = 6144)

One weak-RS factorization (0.50 s) followed by 72 solves (0.56 s) gives all 3K × 3K mobilities [computed]:

| | value |
|---|---|
| ‖M_fast − M_dense‖/‖M_dense‖ | **1.1e-11** |
| asymmetry ‖M − Mᵀ‖/‖M‖, fast / dense | 1.5e-11 / 8.9e-16 (reciprocity holds to solver accuracy) |
| eigenvalues of sym(M) | **−1.36 … 1.61 (indefinite)** |
| smallest eigenvalue of sym(M) restricted to Σ_k F_k = 0 | **+0.0138 (positive)** |

The indefiniteness is the **2D Stokes paradox**, not a bug:
- log-growing flows from a net force have no finite dissipation, so the 2D mobility depends on the length unit;
- once the net force is zero, F·U ≥ 0 is restored [computed; explanation proved in (a)].

The figure `r7_suspension.png` shows the settling velocities.

---

## (e) Where it breaks: sweeps and negative results

### r4: proxy sources and the degenerate scale

**Proved.** For the Stokeslet G = (1/4π)(−log r I + r rᵀ/r²), a constant density e on a circle of radius R produces, at every point inside,

  ∫ G(x, p) e ds_p = (R/2)(½ − log R) e.

This vanishes at **R = √e ≈ 1.6487**, so Stokeslet-only proxy sources on that circle cannot produce a uniform flow. `test_degenerate_scale_single_layer_vanishes_inside` checks this to 1e-14. The Laplace analogue, log on a circle of radius R = 1, is classical [cited, Jaswon–Symm; Hsiao–Wendland]. E3 found the Helmholtz analogue for Y0.

We rescale the starfish so that the leaf-level proxy radius sweeps 0.8–1.25 × R* (`r4_degenerate.png`):

| leaf R / R* | Stokes: constant field on box **curve points**, SL residual | Stokes: constant field on **area points** in the box, SL | Stokes: solve err sl / dl / both | Laplace: area-points residual at R = 1 |
|---|---|---|---|---|
| 0.9 | 1.0e-11 | 1e-15 | 2.3e-10 / 1.9e-11 / 1.6e-10 | 1e-15 |
| **1.0** | **6.3e-11** | **1.9e-2** | 2.1e-10 / 1.9e-11 / 2.3e-10 | **0.57** |
| 1.1 | 2.1e-11 | 1e-15 | 3.4e-10 / 1.9e-11 / 3.1e-10 | 1e-15 |

All values are [computed]. The findings:
- **The degeneracy is real and sharp.** Measured on points that fill the box, the Stokeslet proxy span loses the constant fields (residual 2e-2 for Stokes, 0.57 for Laplace) exactly at R*, and nowhere else (±2% is already 1e-15).
- **On a boundary curve it is masked.** The box's unknowns lie on a 1D arc, and on an arc a constant can be imitated by non-constant Stokes or harmonic fields. The residual rises only from 1e-11 to 6e-11.
- Stokes is more exposed than Laplace here (6e-11 vs 4e-13 on curve points).
- No proxy choice changes the solve error by more than 2× across the sweep.

The same sweep was repeated on an **area** (volume-sum) analogue of P2 with the Stokeslet, A = I + G w (**r4b**, N = 1024 points, proxy radius at level 2 swept through R*). At exactly R*:
- weak, sl: err **8.1e-10**, against about 1e-11 elsewhere (an 80× bump, still < 100ε);
- dl and both: ≤ 1.2e-10;
- strong: ≤ 4e-12 for every proxy choice;
- Laplace at R = 1: no visible bump.

So **the degeneracy does not break this harness** [computed]. What carries the lost direction are the explicitly sampled rows: far points inside the proxy circle in weak mode, and the conservative distance-2 Q-superset in strong mode. Both sample the constant field directly.

[conjecture] A proxy compressor *without* explicit rows, such as a kernel-independent FMM with Stokeslet equivalent densities, would fail at R*. The fix is to use stresslet ("dl") or both kinds of proxy sources, or a radius away from √e (in whatever length unit the kernel uses).

**DL-only proxies are the most accurate for this matrix** [computed]. They give 1.9e-11 against 2e-10 for SL across the whole sweep, at the same ranks. The far sources are double layers, so double-layer proxies are the natural basis for the row side.

**Ring size** (starfish, N = 1024 points, weak) [computed]:
- radius 2.5·hw: n_proxy = 8 already gives 8e-11.
- radius 1.5·hw: n_proxy = 8 fails (1.4e-7). Correct digits need n_proxy ≥ 32 there.

A larger radius gives smaller skeletons: 3369 total at 4·hw against 4987 at 1.5·hw. This matches the E3 finding for Laplace.

### r5: does the proxy see the rigid projector P? Hypothesis (iii) refuted

| geometry | projector weights | weak: with rigid rows / without | strong: with / without |
|---|---|---|---|
| starfish | smooth (= quadrature) | 8.9e-12 / 1.3e-11 | 2.0e-12 / 2.0e-12 |
| starfish | rough (× U[0.2, 1.8]) | 1.4e-11 / 9.9e-12 | 1.6e-12 / 1.6e-12 |
| 8 ellipses | smooth | 9.3e-12 / 9.1e-12 | 9.2e-13 / 1.0e-12 |
| 8 ellipses | rough | 1.3e-11 / 1.1e-11 | 9.4e-13 / 8.7e-13 |

The extra rigid rows are not needed, even with rough weights that make P's rows non-smooth [computed]. Mechanism [proved]:
- P[far, box] = V_far Z_box has row space span(Z_box). Any explicit row of the *same body* contains exactly Z_box, provided that row comes from a far dof inside the proxy circle, or from the distance-2 ring in strong mode.
- A resolved boundary curve that leaves the box always crosses that annulus or ring. A body lying entirely inside box ∪ near has no far coupling through P at all.
- So P is carried for free.

Direct check [computed, during development]: ‖P_far,box (I − Π_proxy)‖/‖P‖ = 1e-15 at the leaves.

[conjecture] The rigid rows would be needed only for under-resolved bodies that skip the annulus, or for a sampler with no explicit rows. They cost 6 rows per body and are kept on by default as a safety net.

### r6: what breaks first as the geometry gets hard

**Two settling circles approaching contact** (n = 128 per circle, h = 0.025) [computed]:

| gap | discretisation error, n vs 4n | fast (weak) vs dense, same n | fast (strong) vs dense | weak root |
|---|---|---|---|---|
| 0.5 | 8e-16 | 2e-13 | 9e-14 | 193 |
| 0.1 | 3.6e-12 | 1.6e-12 | 4e-14 | 277 |
| 0.05 | 4.8e-6 | 1.2e-12 | 5e-14 | 286 |
| 0.02 | **8.2e-2** | 1.4e-12 | 6e-14 | 287 |
| 0.005 | **0.27** | 3.8e-12 | 7e-14 | 281 |

**Physics and quadrature break first; the fast solver does not.** Plain trapezoid Nyström fails once the gap is about the panel size h. The compressed solve stays at 1e-12 with flat ranks (root 280–290) down to gap 0.005.

The remedy is a close-evaluation quadrature [cited: Barnett–Wu–Veerapaneni 2015 for 2D Stokes], which changes the near-field entries. With strong admissibility those entries are kept exactly anyway. [conjecture] Strong RS is therefore the natural host for such corrected near blocks. Weak RS would compress the corrected touching-neighbour blocks, which stay smooth enough here.

**Thin ellipse in shear** (n = 1024, ε = 1e-10) [computed]:

| b/a | cond(A) | dense vs Jeffery | fast weak vs dense (density) | fast weak: Jeffery error | weak skeletons |
|---|---|---|---|---|---|
| 1 | ≈ 2 | 4e-16 | 6e-13 | 1e-13 | 3312 |
| 0.1 | — | 4e-16 | 6e-11 | 3e-11 | 2469 |
| 0.03 | 1.2e4 | 2e-15 | 5.9e-9 | 1.6e-11 | 4700 |
| **0.01** | **3.3e5** | 2.7e-8 | **3.1e-7 (> 100ε)** | 2.7e-8 | 4566 |

**This is a genuine break of the ε target, and it is conditioning, not compression.** The evidence:
- The tips of the ellipse are under-resolved: the radius of curvature b²/a = 1e-4 is about the point spacing. That drives cond(A) to 3e5.
- The density error follows ε·cond(A).
- Tightening ε to 1e-12 restores 1.7e-9 (proxy weak) and 8e-11 (proxy strong).
- The rigid-body answer (Jeffery Ω) is still as accurate as the dense solve. The error sits in density components that carry no force.
- Exact sampling is **worse** than proxy here (7.2e-6 vs 3.1e-7 weak, 2.6e-7 vs 1.4e-8 strong). This echoes E3/E5: the exact ID's truncation is less benign than proxy's.
- Weak skeletons grow by about 1.9× from b/a = 0.1 to 0.03. The two sides of the ellipse (distance 2b) fall into touching boxes, and weak admissibility must compress that near-singular interaction. Strong admissibility keeps it exact and grows less (2476 → 4117 with proxy, 2694 with exact).

---

## (f) Literature: what is known and what, if anything, is new

**Known [cited]:**
- **Integral equations:**
  - Second-kind BIEs for 2D Stokes mobility and resistance, including the completed double layer (Power & Miranda 1987; Rachh & Greengard 2016, SINUM 54, which also treats the Laplace elastance problem used here as the scalar twin).
  - The Sherman–Lauricella / Goursat formulation (Greengard, Kropinski & Mayo 1996).
- **Fast direct solvers and proxies for Stokes:**
  - Fast direct (HBS / recursive-skeletonization) solvers for 2D Stokes BIEs with proxy sources: Marple, Barnett, Gillman & Veerapaneni 2016 (SISC 38), who invert the channel-wall Stokes operator once by RS and use proxy sources for periodization.
  - The Ho–Greengard 2012 RS paper and its slides state that HBS covers "Laplace, Stokes, low-frequency Helmholtz".
  - Kernel-independent FMM uses Stokeslet equivalent and check surfaces (Ying, Biros & Zorin 2004).
  - So **"RS / FMM-LU works for Stokes" is not new.**
- **Analytic results:** Jeffery 1922 (the ellipse rate); the circle mobility and the Stokes paradox (Pozrikidis 1992; Happel & Brenner); degenerate scales of log-kernel single layers (Jaswon–Symm for Laplace; Chen, Kao, Lee and others for elasticity and Stokes, which shows the scale depends on the kernel).

I did **not** find a source stating the value √e for the 2D Stokeslet circle. It follows from the one-line integral in r4 [proved], and it depends on the normalization of the log term.

**What this experiment adds (modest, mostly harness-level):**
1. A measured "cost of a new PDE family" in a unified RS harness:
   - **0 lines** in tree, admissibility, ID and driver;
   - ≈ 28 lines of kernel and ≈ 60 lines of block proxy sampler;
   - the core scalar proxy sampler fails to run on block kernels (r0) [computed].
2. A clean **2.0× rank law** Stokes/Laplace, flat in N (512–4096), ε (1e-4–1e-12), rule, sampler and geometry (r2, r3). [computed; the mechanism (two analytic functions vs one) is cited and the factor 2 is a conjecture-level heuristic.]
3. The Stokes degenerate scale R = √e **is visible** in the proxy span on area points and **masked** on curve points and by the explicit rows the RS proxy keeps (r4, r4b). This is not an error-control guarantee [computed]. It explains why E3's single-ring Y0 failure has no Stokes counterpart in this harness.
4. The non-kernel completion term P is carried by the explicit near rows for any connected, resolved body (r5) [computed + proved argument].
5. **A driver hook gap.** `rskel_factor` passes the strong-admissibility Q set only to `core.ProxyCompressor` subclasses, through a scalar kernel call. A block sampler must reconstruct a Q superset itself, here the distance-2 ring. That is correct but costs rank: strong proxy root 1621 vs exact 1100 at K = 32.
   - Suggestion for the core (not done; outside my directory): pass `q_extra` to every compressor through the protocol.

---

## Claim table

| # | Claim | Tag |
|---|---|---|
| 1 | Swapping the kernel to the 2 × 2 stresslet (interleaved dofs, tree over duplicated points) and using `ExactCompressor` solves the Stokes mobility problem with **zero** harness changes; err ≤ 2e-11 at N_dof = 8192, weak and strong | computed |
| 2 | `core.ProxyCompressor` cannot run with a block kernel (shape error) | computed |
| 3 | A ≈ 60-line block proxy sampler (dial 4) restores proxy compression; err ≤ 4e-11 at N_dof = 8192 | computed |
| 4 | Circle mobility U = (½ − log a)F/4π and Ω = T/4πa² reproduced to 3e-16; Jeffery's rate to 5e-14 | proved / cited + computed |
| 5 | Stokes/Laplace skeleton ratio 2.0–2.2 at every level, every N, ε, rule and sampler; root ratio 1.94–2.19 | computed |
| 6 | Mechanism for (5): two analytic functions (Goursat) vs one | cited (representation), conjecture (that it fixes the factor at exactly ~2) |
| 7 | A Stokeslet single layer with constant density on a circle of radius √e vanishes inside (degenerate scale) | proved + computed (1e-14) |
| 8 | At R = √e the SL proxy span loses constants on area points (residual 2e-2; Laplace at R = 1: 0.57); on curve points only 1e-11 → 6e-11 | computed |
| 9 | Solver accuracy across the degenerate radius changes by ≤ 80× (area problem) and ≤ 2× (curve problem), always < 100ε, because the explicit rows sample the constant field | computed |
| 10 | Without explicit rows (KIFMM-style), SL-only proxies would fail at R = √e; DL or both kinds of sources avoid it | conjecture |
| 11 | DL-only proxies give about 10× lower solve error than SL-only for the double-layer matrix at equal ranks | computed |
| 12 | The rank-3-per-body projector P needs no extra proxy rows: same-body explicit rows contain its row space | computed + proved (argument in r5) |
| 13 | Reciprocity: the fast mobility matrix is symmetric to 1.5e-11; sym(M) is indefinite (Stokes paradox) but positive on Σ F = 0 | computed; explanation cited/proved |
| 14 | Near contact, Nyström discretisation fails (error 0.27 at gap 0.005) long before the fast solver (1e-12 to 4e-12; ranks flat) | computed |
| 15 | Thin ellipse b/a = 0.01: fast error 3e-7 > 100ε, from cond(A) = 3e5 (under-resolved tips). ε = 1e-12 fixes it; exact sampling is worse than proxy there | computed |
| 16 | The strong Q-set hook in `rskel_factor` is scalar-proxy-specific; block samplers need their own Q superset (cost: +47% root vs exact at K = 32) | computed (root sizes) / observation about code |
| 17 | RS / HBS / FMM-LU-type solvers and proxy surfaces for 2D Stokes are already in the literature | cited (Marple et al. 2016; Ho–Greengard 2012; Ying–Biros–Zorin 2004) |

## Files
- `stokes.py`: geometry, kernels, Nyström matrices, right-hand sides, read-out and analytic answers.
- `compressor.py`: `BlockProxyCompressor`, the dial-4 change.
- `harness.py`: one factor-and-solve measurement against dense LU.
- `run.py`: experiments r0–r7.
- `test_e6.py`: 6 tests.
- `results/`: JSON data, PNG figures and `run_log.txt`.
