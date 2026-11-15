# E12: Oseen flow — breaking "separation is symmetric, so ball-shaped admissibility is enough"

**Result in five lines.**
- The scalar Oseen kernel has a parabolic wake, but **per-block ε-ranks barely depend on direction**. Every block is
  a diagonal scaling of the matching Yukawa block [proved], so only *absolute* ranks are directional: upstream
  blocks drop to rank 0. The driver's one-ID-for-rows-and-columns design sees the wake on **both** sides of a box: its
  rank depends on the stream *axis*, not the sign, and is about **2× the single-block rank** for this nonsymmetric
  kernel.
- **A stream-aware (stretched) admissibility pays only at high Péclet number.** With the exact ID at λ = 320, the
  stretched rule dir(3,3,1) keeps 43% fewer skeletons than strong and a 3× smaller root block, for +6% storage and
  +13% flops; a cross-stream control of the same kind does nothing. At λ ≤ 80 the stretched rule saves 8–17%
  skeletons, but so does the cross-stream control, and storage (+20–35%) and flops (+55–95%) go up: no net win.
- With a proxy compressor the near list does not matter by itself: **the proxy curve must change with the rule**. A
  stretched ellipse proxy (14 lines) recovers part of the gain (−11 … −17% skeletons at λ = 320).
- The proxy circle still works (Oseen is a PDE), but **n_proxy must grow like √(λr)**: 64 points are enough to λ = 80,
  128 at λ = 320, 256 at λ = 1280. No E5-type instability appears: the matrix is real, so the Tᵀ row step is a
  congruence; worst pivot cond ≤ 2.1e3.
- Physics check: the tensor Oseen single-layer BIE converges spectrally in n and matches Lamb's (1911) drag to 2.4e-4
  at ka = 0.01. The deviation grows like (ka)² log(1/ka). The fast solver reproduces the dense drag to ≤ 5e-9
  relative under every rule.

Reproduce: `PYTHONPATH=. .venv/bin/python -m unified.explore.e12_oseen.run` (≈ 2 min; `--quick`; `--only r2,r6`).
Data `results/*.json`, figures `results/r1_block_ranks.png`, `r2_admissibility.png`, `r6_drag.png`, log
`results/run_log.txt`. Checks: `.venv/bin/python -m pytest -q unified/explore/e12_oseen` (10 tests);
`.venv/bin/mypy --strict unified/explore/e12_oseen` (clean). Predictions written before any run:
[`HYPOTHESES.md`](HYPOTHESES.md).

---

## (a) The physics in plain words

**Oseen flow.** A body sits in a slow, viscous stream of speed U (a fibre in a slow river, a sedimenting particle, a
small swimmer). Very slow flow is Stokes flow, which is symmetric up- and downstream. Oseen (1910) kept the leading
effect of inertia by linearising the convection term about the uniform stream:
μ(−Δu + 2k ∂ₓu) + ∇p = f, div u = 0, with k = U/(2ν) (ν the kinematic viscosity). The result is a **wake**: disturbances
are swept downstream, decaying slowly behind the body and quickly in front of it. Lamb (1911) used it to resolve
Stokes' paradox (no Stokes solution exists for a cylinder in 2D) and found the drag per unit length
F = 4πμU / (½ − γ − ln(ka/2)) for a cylinder of radius a.

**Scalar model.** The same wake appears in steady advection–diffusion of a passive scalar (heat or dye from sources in
a stream): −Δu + 2λ∂ₓu = δ, with Green's function **G(x) = e^{λx₁}K₀(λ|x|)/(2π)** [proved: substitute u = e^{λx₁}v,
then (λ² − Δ)v = δ]. G decays like r^{−1/2} along +x and like e^{−2λr} along −x. The test matrix A = I + G/N on N random
points in the unit square is the SPEC-P2 analogue.

**Tensor model.** The 2D Oseen tensor ("Oseenlet") in closed form (derived in `oseen.py`, checked against the PDE in
`test_oseen_tensor_pde`): with E = e^{kx₁}K₀(kr)/(2π) and Φ = log r/(2π),
S₁₁ = (E − ∂₁(Φ+E)/(2k))/μ, S₁₂ = −∂₂(Φ+E)/(2kμ), S₂₂ = ∂₁(Φ+E)/(2kμ). As k → 0 it tends to the Stokeslet plus a constant.
Near r = 0, S = Stokeslet + diag(−c₀, −1−c₀)/(4πμ) + O(r log r), with c₀ = log(k/2) + γ (derived by series, checked to
O(r log r) in `test_near_constant`).

## (b) The structural facts that decide everything

- **F1** [proved]. G(x − y) = e^{λx₁} · K₀(λ|x − y|)/(2π) · e^{−λy₁}. Every block of the scalar Oseen matrix is a
  diagonal scaling of the matching Yukawa block. Exact ranks are therefore direction-independent; only ε-ranks can
  depend on direction.
- **F1′** [proved]. Consequently A = I + G/N = D(I + K₀/N)D^{−1} with D = diag(e^{λx₁}). The Oseen system is *similar*
  to a symmetric Yukawa system, which suggests "factor the Yukawa matrix and rescale". R5 measures why that fails:
  cond(D) = e^{λ·width}.
- **F2.** The driver compresses rows and columns with one ID of M = [A_FB; A_BFᵀ]. A_FB is large when F is downstream
  of B, A_BFᵀ when F is upstream (G(x_B − y_F) needs x_B downstream of y_F). So the stacked ID sees the wake on both
  sides, and a symmetric stretch along the axis is the natural rule.

## (c) What changed (dials 2 and 4)

| object | dial | code lines |
|---|---|---|
| `directional.Directional(a_down, a_up, a_cross)` — near = same-level boxes with −a_up ≤ Δx ≤ a_down, \|Δy\| ≤ a_cross | 2 admissibility | 19 |
| `harness.EllipseProxy` — proxy ellipse enclosing B and its stretched near field (only `proxy_points`, plus a radius that makes the driver's inside test a superset) | 4 sampling | 14 |
| `harness.TensorProxy` — 2×2 Oseenlet proxy on interleaved dofs, with a conservative Q-set superset (near-of-near boxes), after E6 | 4 sampling (tensor) | 22 |
| **dial total** | | **55** |
| scalar run with the stock `ProxyCompressor` | — | **0** |
| physics set-up (`oseen.py`: kernels, tensor, near-field constant, cylinder quadrature, Lamb, area problem) | — | **82** |
| core, drivers, tree, ID | — | 0 |

Counted with `unified.showcase.reproduce.code_lines` (`results/lines.json`). The probe, harness and run code are
experiment code and are not counted. The rskel driver accepted a near list larger than the touching neighbours
without change: its own Q-set bookkeeping records fill among B and *all* its near boxes.

## (d) R1 — far-block rank vs direction and λ (box width w = 1/8, 300 random points per box, ε = 1e-8)

θ is the direction from source box B to target box F (0° = downstream), distance D = 2w. The first three columns are
the *relative* ε-rank of K(F, B) at θ = 0°, 90°, 180°; the next three the *absolute* rank (threshold ε × the downstream
block's norm); the last three the driver's stacked-ID rank. From `results/r1_block_ranks.json`.

| λ | rel 0° / 90° / 180° | abs 0° / 90° / 180° | stacked 0° / 90° / 180° | ‖block‖ 180° ÷ 0° |
|---|---|---|---|---|
| 0 (Laplace) | 11 / 12 / 11 | 11 / 12 / 11 | 13 / 13 / 14 | 0.98 |
| 1 | 11 / 11 / 11 | 11 / 11 / 11 | 22 / 22 / 21 | 0.61 |
| 5 | 12 / 12 / 12 | 12 / 11 / 11 | 23 / 23 / 23 | 0.11 |
| 20 | 13 / 12 / 12 | 13 / 10 / 7 | 21 / 24 / 21 | 7e-4 |
| 80 | 14 / 10 / 11 | 14 / 5 / **0** | 16 / 20 / 16 | 5e-11 |
| 320 | **19** / 7 / 11 | **19** / 0 / 0 | **21 / 12 / 21** | 7e-38 |

At D = 4w and λ = 320: relative 12 / 4 / 8, absolute 12 / 0 / 0, stacked 14 / 8 / 14.

- **P1 confirmed.** Relative ranks are nearly direction-free up to λ = 20 (F1). At λ ≥ 80 the downstream block's rank
  grows (11 → 19 at λ = 320) while cross-stream and upstream blocks fall. Absolute ranks are strongly directional:
  upstream → 0. [computed]
- **The stacked ID doubles the rank** for λ ≥ 1 (22 vs 11 at λ = 1; 13 vs 11 for Laplace). Rows and columns are scaled
  by e^{−λy_B} and e^{+λx_B} respectively, so their spans differ. The cost of the one-ID-for-both-sides design is
  specific to nonsymmetric kernels [computed; explanation from F1]. At λ = 320 the stacked rank is axis-dependent
  (21 along, 12 across), as F2 predicts.

## (e) R2 — weak / strong / directional rules on A = I + G_λ/N (N = 2048, ε = 1e-8)

Total skeletons, and storage and flops relative to strong (both with the exact ID). All runs: err vs dense ≤ 2e-11,
backward error ≤ 8e-12, except the weak proxy at ≤ 8e-9. `results/r2_admissibility.json/png`.

| λ | strong (exact) skel | dir(2,2,1) | dir(3,3,1) | dir(3,1,1) (downstream only) | **dir(1,1,2) (cross control)** | weak |
|---|---|---|---|---|---|---|
| 0 | 1932 | 1740 (−10%), stored ×1.29, flops ×1.74 | 1682 (−13%) | 1816 | **1741 (−10%)** | 3347 |
| 5 | 2972 | 2675 (−10%), ×1.21, ×1.57 | 2604 | 2797 | **2655** | 4153 |
| 80 | 2500 | 2187 (−13%), ×1.21, ×1.56 | 2079 (−17%), ×1.36, ×1.97 | 2254 | **2172 (−13%)** | 4108 |
| **320** | 2499 (root 444) | **1887 (−24%)**, ×1.03, ×1.10 | **1427 (−43%, root 151)**, ×1.06, ×1.13 | 1828 (−27%) | **2493 (−0%)** | 3944 |

- **At λ ≤ 80 the stream direction is irrelevant.** Stretching across the stream saves as many skeletons as stretching
  along it; any larger near field does. The extra near-field fill costs more than it saves (storage +21–36%, flops
  +56–97%). P2 ("modest win at large λ") is refuted at λ ≤ 80. [computed]
- **At λ = 320 (λw ≈ 40 at level 3) the wake is real.** Stretching along the stream cuts skeletons by 24–43% and the
  root block by 3×, for +3–6% storage and +10–13% flops; the cross control does nothing. That is the first regime in
  this repo where a non-ball admissibility wins on skeletons and root size at nearly equal cost [computed]. The
  crossover λw ≈ 10–40 is a rough bracket from two points [conjecture].
- **Proxy compression needs a matching proxy curve.** With the stock circle, every directional rule gives *identical*
  skeletons to strong (e.g. 3774 at λ = 80, 4049 at λ = 320): the circle, not the near list, decides what is compressed.
  The ellipse proxy recovers part of the gain: −5 … −17% at λ ≤ 80, −11 … −17% at λ = 320 (3590 and 3344 vs 4049).
  It still costs more storage than the exact-ID route because the proxy keeps 1.4–2.4× more skeletons. [computed]
- **R3 (N scaling, `r3_scaling.json`).** At λ = 0 and 80 the pattern holds to N = 8192: dir(2,2,1) with the exact ID
  saves 6–9% skeletons and costs +30–37% storage and +75–95% flops. Errors stay ≤ 5e-12.

## (f) R4 — does the proxy circle still work? (strong, N = 2048)

err vs dense, by circle size:

| λ \ n_proxy | 16 | 32 | 64 (stock) | 128 | 256 |
|---|---|---|---|---|---|
| 20 | 8e-8 | 7e-12 | 1e-13 | 1e-13 | 1e-13 |
| 80 | 6e-6 | 7e-8 | **2e-13** | 3e-14 | 3e-14 |
| 320 | 2e-5 | 4e-6 | 2e-8 | **2e-14** | 1e-14 |
| 1280 | 1e-5 | 7e-6 | 2e-6 | 2e-8 | **6e-15** |

P3 is confirmed. The proxy is valid: the exterior Dirichlet problem for −Δ + 2λ∂ₓ and for its adjoint is unique by the
maximum principle [proved], so the representation argument holds. It must resolve the wake on the circle:
the smallest sufficient n_proxy (64, 128, 256 at λ = 80, 320, 1280, with leaf proxy radius r = 0.156) fits
n_proxy ≈ 18√(λr): a fixed number of points per wake width √(r/λ) around a circle of radius r. That is the analogue of
Helmholtz's points per wavelength [computed on three points; scaling conjecture].

## (g) R5 — the similarity route (factor the symmetric Yukawa matrix, rescale by D)

| λ | 1 | 5 | 10 | 20 | 30 | 40 | 80 |
|---|---|---|---|---|---|---|---|
| cond(D) = e^{λ·width} | 2.7 | 1.5e2 | 2.2e4 | 4.8e8 | 1.0e13 | 2.3e17 | 5.2e34 |
| err vs dense | 1e-12 | 1e-12 | 9e-13 | 7e-12 | **9e-9** | **8e-8** | **232** |

The similarity transform is free and accurate while e^{λL} < ~1e10. Beyond that, the information it hides (the wake)
costs every digit. This is the exponential dichotomy the wake encodes: one cannot simply rescale it away
[computed; mechanism proved by F1′].

## (h) R6 — tensor Oseen: drag on a cylinder vs Lamb (1911), and the fast solver

**Dense convergence and Lamb** (`r6_cylinder.json`, `r6_drag.png`). n = 64 … 512 boundary nodes, a = μ = U = 1:

| ka | dense drag (n = 512) | change n 64 → 512 | Lamb | dense ÷ Lamb − 1 |
|---|---|---|---|---|
| 0.01 | 2.4062672147 | 5e-9 | 2.4068427187 | −2.4e-4 |
| 0.03 | 3.0431654571 | 5e-8 | 3.0482481228 | −1.7e-3 |
| 0.1 | 4.2505369249 | 1e-7 | 4.3057389419 | −1.3e-2 |
| 0.3 | 6.3985512837 | 7e-7 | 6.9049622424 | −7.3e-2 |
| 1 | 11.6831966819 | 6e-6 | 20.4022205308 | −0.43 (Lamb is an asymptotic small-ka formula) |

- Discretization converges fast in n. The quadrature is a punctured trapezoid with a diagonal-only correction from the
  exact circle identity ∫log|x − y| ds = 2πa log a and the near-field constant C [verified by the convergence].
- The deviation from Lamb shrinks as ka → 0 at a rate ≈ (ka)² log(1/ka): ratios 7.0× and 7.7× per 3× step in ka.
  That is consistent with Lamb being the leading term of the Oseen solution (Tomotika–Aoi 1950 give the full series
  [cited, not re-derived]). P5 confirmed for ka ≤ 0.03; at ka = 0.1 the deviation is 1.3%, not < 1%. [computed]
- The single-layer matrix has a near-null vector, the normal density (a uniform normal force on a closed curve is
  balanced by pressure; true for Stokes and for Oseen) [proved by the divergence theorem]. cond grows like n³
  (1e6 → 6e8). Dense and fast densities may differ along it without effect on the flow or the drag, so errors are
  reported modulo that vector.

**Fast solver** (n = 1024, 2048 dofs, ε = 1e-8, tree over interleaved dofs):

| k | rule, compressor | drag rel. err vs dense | density err (mod null) | backward | skeletons | pivot cond / min Herm |
|---|---|---|---|---|---|---|
| 0.1 | strong, exact | 4e-12 | 5e-7 | 5e-10 | 1593 | 83 / +1.2e-2 |
| 0.1 | strong, TensorProxy | 5e-11 | 2e-6 | 1e-9 | 2520 | 220 / +4.5e-3 |
| 0.1 | weak, TensorProxy | 4e-10 | 9e-6 | 7e-9 | 2744 | 329 / +3.0e-3 |
| 1 | strong, exact / proxy | 3e-11 / 1e-10 | 2e-7 / 1e-6 | 5e-10 / 2e-9 | 1713 / 2644 | 121 / 90 |
| 10 | strong, exact / proxy | 1e-10 / 8e-11 | 4e-8 / 7e-8 | 5e-10 / 7e-10 | 1886 / 2882 | 61 / 344 |
| 10 | dir(2,2,1), exact | 1e-10 | 4e-8 | 4e-10 | 1730 | 83 |

On a single cylinder (a curve), the directional rule changes little (−8% skeletons with the exact ID). The wake
matters for *area* or *multi-body* problems spread along the stream; a row of bodies is the natural follow-up.

## (i) E5 check: does the Tᵀ row step break anything here?

**No** [proved + computed]. E5's failure needs a complex T: then Πᵀ W Π is not a congruence and the Hermitian part of
the pivot can turn indefinite. Here every matrix is real, so T is real and Πᵀ = Πᴴ: the row step *is* a congruence.
For the scalar area matrix and the tensor BIE, the Hermitian part of every pivot X_RR stayed positive. Its worst
min-eigenvalue ÷ ‖X_RR‖ is ≥ 4.8e-4, and the worst cond(X_RR) is ≤ 2.1e3, over all λ and rules (R2, R6). The
non-symmetry costs rank (the 2× stacked rank in R1), not stability.

## (j) Verdicts

| prediction | verdict |
|---|---|
| P1 relative ranks direction-free; absolute ranks directional | **confirmed** (plus: the stacked ID doubles the rank) |
| P2 directional rule: modest win at large λ only | **refuted at λ ≤ 80** (any larger near field helps equally, cost goes up); **confirmed and stronger than predicted at λ = 320** (−43% skeletons, 3× smaller root, +6% storage) |
| P3 proxy still works if resolved; n_proxy grows with λr | **confirmed**, but as √(λr), not λr (64 → 128 → 256 points at λ = 80 → 320 → 1280) |
| P4 no E5 instability | **confirmed** |
| P5 Lamb agreement ≲ 1% for ka ≤ 0.05; fast = dense | **confirmed** (2.4e-4 at ka = 0.01, 1.7e-3 at 0.03); fast drag error ≤ 5e-9 |
| (new) similarity route | works for e^{λL} ≲ 1e10, fails beyond (err 232 at λ = 80) |
| (new) with a proxy compressor, the admissibility rule and the proxy curve must change together | computed |

## (k) Literature (searched 2026-11-14; standard web searches, not exhaustive)

- **Oseen BIE/BEM**: the 2D Oseen fundamental solution and BEM go back to Oseen (1910) and Lamb (1911), with BEM work by
  Bush and by Hsiao–Kress / Fischer–Hsiao–Wendland (1985) [cited]; method of fundamental solutions by
  Karageorghis–Lesnic (eprints.whiterose.ac.uk/146897) [cited]. The search found no fast-multipole or skeletonization
  solver specialised to the Oseen kernel. Kernel-independent FMMs cover it in principle [cited, general]. This repo's
  closed-form tensor matches the standard structure (K₀, K₁ and a potential part) and was checked directly against
  the PDE.
- **Direction-dependent admissibility for convection.** Prior work exists, but for *inverses of FEM matrices*, not
  integral kernels. Le Borne (2003, *Computing*; 2005) modified the H-matrix partition and admissibility along a
  constant (then variable) convection direction because standard admissibility fails for convection-dominated
  problems. Saunier et al. (arXiv:2512.04824, Dec 2025) use "convection tubes" [cited-abs]. So the *idea* of
  stream-aligned admissibility is not new. New here, as far as this search shows: (i) the measurement that for the
  integral kernel it pays only at high Péclet number (λw ≳ 40 here), (ii) the 2× stacked-ID rank cost of
  nonsymmetric kernels under a one-T driver, (iii) the coupling "stretched rule ⇒ stretched proxy curve", and (iv) the
  failure threshold of the similarity route. [conjecture-level novelty]

## (l) Claim table

| # | claim | label |
|---|---|---|
| 1 | G = e^{λx₁}K₀(λr)/(2π) solves −Δu + 2λ∂ₓu = δ; blocks are diagonal scalings of Yukawa blocks; A is similar to I + K₀/N | proved (+ tests) |
| 2 | Closed-form 2D Oseen tensor; near-field constant diag(−c₀, −1−c₀)/(4πμ) | derived here; computed (PDE and limit tests) |
| 3 | Relative block ranks nearly direction-free (λ ≤ 20); absolute ranks directional | computed (R1) |
| 4 | Stacked-ID rank ≈ 2× single-block rank for λ ≥ 1; axis-dependent at λ = 320 | computed (R1) |
| 5 | Directional rule: no advantage over an equally enlarged cross-stream rule at λ ≤ 80; −24 … −43% skeletons, 3× smaller root, ≤ +6% storage at λ = 320 (N = 2048) | computed (R2) |
| 6 | With a proxy compressor the rule alone changes nothing; an ellipse proxy recovers part of the gain | computed (R2) |
| 7 | The stock proxy remains valid (maximum principle); it needs n_proxy ≈ 18√(λr) | proved (validity) + computed (resolution) |
| 8 | Similarity route fails once e^{λL} ≳ 1e10 | computed (R5) |
| 9 | No E5 instability: real T makes the row step a congruence; pivots stay well conditioned | proved + computed |
| 10 | Dense drag converges spectrally; matches Lamb to 2.4e-4 at ka = 0.01; deviation ∝ (ka)² log(1/ka) | computed; rate conjecture; Lamb cited |
| 11 | Fast drag = dense drag to ≤ 5e-9 under weak/strong/directional, exact or tensor proxy | computed (R6) |
| 12 | Stream-aligned admissibility exists for FEM-inverse H-matrices (Le Borne 2003/05; Saunier et al. 2025) | cited |

**Scope not covered.** Tensor block ranks vs direction were not probed (the tensor has a potential part ∂Φ that
decays like 1/r in *all* directions, so the wake effect should be weaker; conjecture). Multi-body / area tensor
problems, where a directional rule should matter, were not run.
