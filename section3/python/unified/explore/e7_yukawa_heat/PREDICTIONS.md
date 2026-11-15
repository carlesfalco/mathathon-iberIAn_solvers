# E7 predictions, written 2026-11-14 14:30 before any experiment was run

(This file is frozen; `findings.md` quotes it and gives the verdicts.)

Kernel: G_κ(x, y) = K0(κ|x − y|) / (2π), the free-space Green's function of −Δ + κ².
Starfish perimeter L ≈ 9.02, so the point spacing is h = L/N (h ≈ 0.0044 at N = 2048).

| Dial | Prediction | Why |
|---|---|---|
| 1 tree / ordering | unchanged | geometry only |
| 2 admissibility | Weak/Strong unchanged and still correct. A *truncation* (drop far blocks whose entries are all below ε) is valid and becomes active once κ·(box separation) ≳ log(1/ε) | K0(z) ≤ sqrt(π/2z) e^{−z} |
| 3 representation (ID / skeletons) | unchanged | kernel is smooth away from the diagonal |
| 4 sampling | Exact: unchanged. Proxy: still valid with a ring of plain monopoles K0 (no combined-field fix needed at ANY κ), because the Graf coefficients I_m(κρ)K_m(κR) never vanish. For the double-layer (physical) matrix the proxy *sources* must be monopoles, not dipoles with a made-up normal: ~10-line kernel wrapper | Graf addition theorem for K0 |
| 5 algebra | unchanged; factor reused for every time step when dt is fixed | Rothe: same operator each step |
| E5 instability (complex T, Tᵀ row step) | absent at every κ | K0 is real, so T is real and the row step is a congruence |

Rank predictions (the key question of the task):
- **P1** With the harness's RELATIVE ID tolerance (ε·|R11|), skeleton counts do NOT drop with κ. For a box
  of radius ρ against sources at distance R, the Graf terms behave like e^{−m²(1/ρ − 1/R)/(2κ)} relative to
  m = 0, so the relative-ε rank grows like sqrt(κ) for large κρ (conjecture from the asymptotics of I_m, K_m).
- **P2** With an ABSOLUTE tolerance (ε times the matrix scale, ≈ ½ here), ranks fall with κ and reach 0 once
  κ·dist ≳ log(1/ε) + O(log κ). Then the factorization degenerates into a block-sparse (banded) LU and the
  factor cost falls with dt.
- **P3** κ → 0: the double-layer kernel tends to the Laplace double layer smoothly, nothing breaks. The
  single-layer toy A = I + S_κ w picks up a constant −log(κ)/(2π) (a rank-one term): +1 rank per box at most.
- **P4** κ → ∞: the fast solver keeps matching the dense solve (algebra is unaffected), but the ANSWER loses
  accuracy against the analytic solution once κh ≳ 1 (≈ κ > 200 at N = 2048): the kernel's boundary layer
  1/κ is not resolved by the trapezoid rule. This is a quadrature limit, not a fast-algorithm limit.
- **P5** Proxy at large κ: ring rows are ~e^{−1.5κ·hw} small; when a box has no explicit (inside-ring) far
  points, the relative ID fits noise-level rows at full rank (wasted rank), until they underflow to 0.
- **P6** Heat (Rothe/backward Euler, κ² = 1/dt): with the history written as layer potentials of κ²-derivatives
  of the double layer (Chapko–Kress style), only boundary unknowns are needed, one factorization serves every
  step, and the result is backward-Euler exact (error = quadrature level) against an analytic semi-discrete
  solution, and O(dt) against the continuous solution.
