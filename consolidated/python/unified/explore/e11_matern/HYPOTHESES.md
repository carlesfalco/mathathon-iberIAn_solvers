# E11 — Matérn kernels: predictions written BEFORE any run (kept verbatim; verdicts in findings.md)

**Honesty note.** No code in this directory had been run when this file was written. The coordinator's note on
`research/sota/15_next_targets.md` §B2/§B3 had been read.

## Setup (planned)
Matérn covariance with smoothness ν, length scale ℓ, κ = √(2ν)/ℓ:
C_ν(r) = 2^{1−ν}/Γ(ν) (κr)^ν K_ν(κr), C(0) = 1. GP matrix A = C + σ² I on N random points in the unit square.
Control kernels: ν = 0 (K₀, the Yukawa Green's function of R², not a covariance: A = I + K₀/N like SPEC P2) and
the radiative-transfer kernel e^{−μr}/r (the 3D Yukawa kernel restricted to the plane).

## The hypothesis this experiment breaks
"The surface proxy needs the kernel to solve a PDE outside the box." For d = 2, C_ν is the Green's function of
(κ² − Δ)^{ν+1} in R²: a *local* operator only when ν is an integer.

## Dimension counting (checked by hand before running; the coordinator's B2 claim)
Green's function of (κ² − Δ)^β in R^D is ∝ r^{β−D/2} K_{β−D/2}(κr). Matching r^ν K_ν:
- **Second-order lift (β = 1, Yukawa)**: β − D/2 = ν needs D = 2 − 2ν < 2. B2 is right *for this route*: a
  Matérn kernel with ν > 0 is not a restricted Yukawa kernel of a higher dimension. (The match via K_ν = K_{−ν}
  gives r^{−ν}K_ν, a different function.)
- **Higher-order lift (missed by B2)**: any integer β ≥ ν + 1 gives D = 2(β − ν) ≥ 2. Then C_ν is the restriction
  to R² of the Green's function G_β of the *local* operator L^β = (κ² − Δ_D)^β in R^D.
  - half-integer ν (0.5, 1.5, 2.5) with β = ν + 3/2: **D = 3, an integer**. Exponential kernel e^{−κr} = Green's
    function of L² in R³.
  - integer ν: β = ν + 1, D = 2: no lift, but order 2β > 2.
  - ν = 0.7: β = 2, D = 2.6: fractional dimension (E8's situation; weighted extension, conjecture).
- **But order 2β needs β boundary data.** If u = Σ c_j G_β(· − y_j) vanishes on a sphere, it does NOT follow that
  it vanishes outside (exterior Dirichlet for L^β, β ≥ 2, is not unique). If also L^m u = 0 on the sphere for
  m = 1 … β−1, then v = L^{β−1}u solves Lv = 0 outside, v = 0 on the sphere, v → 0: v ≡ 0 (maximum principle),
  and downward induction gives u ≡ 0 outside. L^m G_β = G_{β−m} ∝ r^{ν−m} K_{ν−m}(κr). So the proxy rows should be
  the **β kernels G_β, G_{β−1}, …, G_1 on a (lifted) sphere** ("multi-kernel proxy", B3's idea), with the lift
  needed whenever ν is not an integer. [proved for integer D; conjecture for fractional D]

## Predictions (P = prediction, to be marked confirmed / refuted)
- **P1** Stock circle proxy (one kernel, plane): exact only at ν = 0 (K₀). Fails under strong admissibility for
  ν > 0 with error 1e-6 … 1e-3 (E3-Gaussian / E8-like), *worse* for smoother kernels? Uncertain direction: the
  missing data (L^m u) matter more as β grows, but smoother kernels have lower rank so the ID may capture the
  far field anyway. Weak admissibility: milder failure (neighbours inside the circle are explicit).
- **P2** Lifting with ONE kernel (E8 hemisphere, G_β rows only): improves on the circle but does NOT reach ε in
  general for β ≥ 2 (missing L^m data). If it does reach ε, the non-uniqueness is harmless in practice because the
  ID is asked to match a discrete set, and we report that.
- **P3** Multi-kernel (β kernels) on the circle: exact for integer ν (1, 2); fails for half-integer ν.
- **P4** Lifted multi-kernel (hemisphere in R³, β kernels): reaches ε for half-integer ν (proved case) and probably
  for ν = 0.7 (fractional D = 2.6, conjecture). Costs about (β × E8's 2.3–6×) the circle's proxy rows.
- **P5** Annulus / Xing–Chow-style volume proxy (kernel-agnostic) reach ε at every ν, with more points than the
  lifted proxy. Exact ID and randomized ID reach ε (randomized with 2× sketch matvecs).
- **P6** B2's radiative-transfer control e^{−μr}/r: circle fails, single-kernel hemisphere (D = 3, β = 1) reaches ε.
- **P7** Ranks: decrease with ν and with ℓ (smoother, longer-range covariance is globally lower rank); grow
  with κ·box size when ℓ is small (exponentially decaying far field still compressed to relative ε).
- **P8** Conditioning: cond(C + σ² I) ≈ λ_max/σ² ~ N·(fraction)/σ²; with σ² = 1e-4 cond ≳ 1e6, so error vs dense
  ≈ cond·(residual) and must be reported separately from the compression residual. Log-determinant from the
  factorization (Σ log|det X_RR| + log|det root|) matches slogdet to ≈ N·ε relative-ish (absolute error ≲ 1e-5).
- **P9** Novelty: fast GP solvers with skeletonization/proxies exist (Minden–Damle–Ho–Ying 2017 annulus proxy;
  H2Pack; Ambikasaran et al. 2016 HODLR). A lifted multi-kernel proxy for Matérn is probably not published; check.
