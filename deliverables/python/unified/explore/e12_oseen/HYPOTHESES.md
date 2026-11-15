# E12 — Oseen flow: predictions written BEFORE any run (kept verbatim; verdicts in findings.md)

**Honesty note.** No code in this directory had been run when this file was written.

## Setup (planned)
- **Scalar model** (advection–diffusion / scalar Oseen): −Δu + 2λ ∂ₓu = δ in R², stream along +x.
  G_λ(x) = e^{λx₁} K₀(λ|x|)/(2π) [proved: substitute u = e^{λx₁}v, then (λ² − Δ)v = δ].
  Area test matrix A = I + G_λ/N on N random points in the unit square (the SPEC P2 analogue).
- **Tensor model**: the 2D Oseen tensor (derived in `oseen.py`), single-layer first-kind BIE on a circular cylinder
  of radius a, drag compared with Lamb (1911): F = 4πμU / (½ − γ − ln(ka/2)), k = U/(2ν).

## The hypothesis this experiment breaks
"Separation is symmetric, so ball-shaped (weak/strong) admissibility is enough." G_λ has a parabolic wake
along +x: it decays like r^{−1/2} downstream and like e^{−2λr} upstream.

## Structural facts I expect to matter (derived before running)
- **F1 (proved).** G_λ(x − y) = e^{λx₁} · K₀(λ|x − y|)/(2π) · e^{−λy₁}: every block of the Oseen matrix is a
  diagonal scaling of the same block of the Yukawa (K₀) matrix. The *exact* rank of every block is
  therefore direction-independent. Only *ε-ranks* can depend on direction, through the scaling.
- **F2.** The driver compresses rows and columns with ONE ID of M = [A_FB; A_BFᵀ]. A_FB is large when F is
  downstream of B, A_BFᵀ when F is upstream. So the stacked ID sees the wake on BOTH sides: rank depends on the
  stream AXIS, not on the sign of the direction.

## Predictions
- **P1** Per-block ε-rank (relative tolerance), box width w, displacement at distance D and angle θ to the stream:
  roughly direction-independent (F1: a separable factor times a near-Gaussian), growing like w√(λ/D) at large λw.
  With an absolute tolerance (ε × the largest entry of the matrix), blocks upstream and across the stream fall
  to rank ≈ 0 at large λ, while downstream blocks keep rank ≈ w√(λ/D): the absolute ε-rank is strongly
  direction-dependent. At λ → 0 everything tends to the Laplace (Stokes) ranks.
- **P2** Strong RS with stock proxy and stock ID: ranks grow with λ (at fixed N) above the Laplace values; error
  stays at ε. Directional (stretched along the stream axis) admissibility lowers the max rank at large λ (≥ 20 on
  the unit square) but raises near-field cost; net skeleton total roughly unchanged. Honest expectation: a
  modest win at large λ only.
- **P3** The proxy circle still works for the scalar kernel (exterior Dirichlet uniqueness holds for
  −Δ + 2λ∂ₓ by the maximum principle, both for the operator and its adjoint), provided the circle is resolved:
  n_proxy must grow with λ·r (needed when λ r ≳ 10).
- **P4** No E5 instability: the matrix is real, so T is real and the Tᵀ row step IS a congruence; the symmetric part
  of the Oseen single layer is the (positive) viscous dissipation. Cond of pivots stays small.
- **P5** Drag: the dense solve converges to the Oseen drag; Lamb's formula agrees to O((ka)² log) relative,
  i.e. ≲ 1% for ka ≤ 0.05; the fast solve matches dense to ≈ ε·cond.
- **P6** Literature: Oseen boundary integral methods exist (Hsiao–Kress 1985, Fischer–Hsiao–Wendland 1985,
  Olmstead–Gautesen); kernel-independent FMMs cover Oseen. Directional admissibility for advection–diffusion
  kernels: I expect little; directional H²/DH² exists for Helmholtz (Börm, Engquist–Ying), not wakes.
