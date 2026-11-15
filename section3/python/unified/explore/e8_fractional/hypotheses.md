# E8 — predictions written before the sweeps (kept verbatim; verdicts are in findings.md)

**Honesty note.** These predictions were formed before any run. They were written to this file only after
two smoke tests: the physics discretisation, and one N = 2048, s = 0.5 compressor table. That table already
showed P1 and P4 holding. A quick p sweep (N = 1024) also showed the plane proxy exact at p = 0 (2D) and
p = −1 (1D), and the lifted proxy working at p ≤ 0. That table also and it revealed the weak-admissibility cond(A)·ε effect, which is not predicted
below. Every other prediction was untested when written.

- **Dials 1, 2, 3, 5 unchanged.** The tree, the Weak/Strong rule, the one-ID skeleton representation and the
  factor algebra (`rskel_factor`) do not care which kernel fills the matrix. Fractional matrices are real
  and symmetric, so E5's complex-T pathology cannot occur.
- **Dial 4 must change.** The core proxy (circle in 2D, the two points c ± r in 1D) is a Green's
  representation of the *2D Laplace / 1D |x|* kernel. |x|^{-p} solves no PDE in R^d, so:
  - P1: the plane proxy fails under strong admissibility, for every s, with E3-like errors (1e-6 … 1e-3).
    It is exact only at p = 0 in 2D (log) and p = −1 in 1D (|x|).
  - P2: in 1D it fails catastrophically, because the two proxy points cap the far-field rank at 2.
  - P3: under weak admissibility the damage is milder, because the touching neighbours sit inside the
    proxy radius and are kept explicit.
- **Modified proxy (the one-line idea).** |X|^{-p} is the Laplace Green's function in R^{p+2}. Restricted to
  functions that are axisymmetric in the p+2−d extra coordinates, the sphere of R^{p+2} becomes a
  hemisphere in (z, t) ∈ R^{d+1}.
  - P4: proxy points on that hemisphere restore ε accuracy for every s.
    - This is proved for integer p+2−d by the exterior maximum principle (it needs p > 0 for decay).
    - It is conjecture for fractional dimension; the hypersingular weight 1+2s lies outside A₂.
  - P5: the lifted proxy costs more proxy points than the plane: O(L^d) on the hemisphere vs O(L) on the
    circle.
  - P6: it may keep more skeletons than exact, because it represents every field that is "harmonic in
    R^{p+2}", not only its trace on the plane.
- **Volume proxy (annulus, Minden et al. 2017 / Xing–Chow 2020).**
  - P7: it works too, but needs rings out to the domain diameter.
- **Rank vs s.**
  - P8: strong ranks are bounded in N and grow only mildly with p = d ± 2s (Gegenbauer coefficients grow
    polynomially in the degree).
  - P9: weak ranks grow like log N in 1D and like √N in 2D, as for Laplace (P2).
- **Physics.**
  - P10: the lattice-quadrature solution converges to (1−|x|²)^s slowly, O(h^{2−2s}) from the singular
    cell; the Taylor-corrected scheme converges faster in 1D.
  - P11: in 2D the staircase approximation of the disk limits accuracy.
