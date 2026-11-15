# E10 hypotheses, written before any experiment was run (kept verbatim; findings.md grades them)

Dials expected to stay unchanged:
- D2 admissibility: Complementary (butterfly) pairing, unchanged. Reason: for fixed m, P_l^m(cos θ) ≈
  amplitude(θ,l) · cos((l+½)θ − π/4 + phase correction) in the oscillatory region, i.e. a 1D Fourier-
  integral kernel with phase ≈ l·θ, which is exactly the class the complementary low-rank property covers.
- D3 representation: column ID + nested interpolation matrices, unchanged.
- D4 sampling: exact ID on kernel entries (all rows of a tile), unchanged. A proxy/Green representation
  does NOT apply: P_l^m(x_j) is not a Green's function of a PDE in (θ, l) jointly; there is no
  "radiating outside a box" structure in the degree variable. So the proxy compressor is out.
- D5 algebra: apply. Analysis needs Aᵀ, which is the transposed butterfly (same factors), not a solve,
  because Gauss quadrature makes Aᵀ W A = I exactly. No factorization/inversion needed.

Dial expected to need care:
- D1 clustering/ordering: the trees must cut the colatitude θ, not x = cos θ (Gauss nodes cluster in x near
  the poles; the phase is linear in θ, not in x). Expectation: θ-tree gives max rank O(1) bounded in n;
  x-tree gives higher ranks (by ~1.5–2×) but still works. Bit reversal (E2's DFT trick, rank 1 for the
  DFT) is expected to FAIL for Legendre: the amplitude and the −π/4 + O(1/l) phase corrections break the
  exact character identity, so tiles become full rank as for the FIO with t > 0 in E1/E2.

Rank scaling expectations:
- m = 0: max tile rank bounded in n (like B1 at t = 0..1), ~10–15 at eps 1e-10, ~7–10 at eps 1e-6.
- growing m: ranks do NOT grow; the turning curve (l+½) sin θ = m is a fold of the WKB phase, but the
  function itself is smooth (Airy-like) there with lower local frequency, so I expect straddling tiles to
  have rank ≤ oscillatory tiles. Contrast with E4: there the caustic broke the SOLVE (cond 1e14), not the
  butterfly. Here there is no solve, so I expect no breakage at turning points.
- One expected waste: in the evanescent region tiles are exponentially small but the driver's ID uses a
  RELATIVE tolerance per tile, so those tiles still get rank > 0. Zeroing entries below eps·max|A|
  (kernel-only change) should cut storage substantially for m ≳ n/4.
- Crossover vs dense in storage: below n ≈ 512 at eps 1e-6, 1024–2048 at eps 1e-10 (B1 had nnz ≈ 0.75 N²
  at N=1024 eps 1e-10 per the run above... i.e. crossover in storage only in the thousands).
  Wall time crossover (Python sparse vs BLAS dense) later than storage, possibly not reached at n ≤ 4096.
