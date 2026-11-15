# E2 — Replace the butterfly by FFT-based algorithms

**Swap.** We keep the B1 operator `K(x, ξ) = exp(2πi (x ξ + t c(x)|ξ|))` (x = j/N, ξ = k − N/2, ε = 1e-6) and
replace the butterfly in two ways: (a) **exactly**, by reordering so the ID butterfly *becomes* an FFT, and
(b) **approximately**, by a low-rank phase split plus R FFTs (the NUFFT-style method of H. Yang,
arXiv:1803.04128). Reproduce with `run.py` (≈13 min for N ≤ 4096; `--quick` for N ≤ 1024; `--reuse` redoes
the crossovers and figures from the JSON). Code: `fftswap.py`. Tests: `test_e2.py`.

**Cost unit.** We count complex multiply-adds per apply. For a butterfly this is its nnz. For one FFT it is
the nnz of the rank-1 butterfly from (a), `2N log₂N + 2N`, so the two methods are measured in the same unit.
R-FFT apply cost = `R (2N log₂N + 4N)`, which includes the two diagonal scalings per term. R-FFT storage =
`R (N + N/2 + 1)` (a_r on x; b_r on the distinct |ξ|).

## Hypotheses (stated before running)
H1. In bit-reversed ordering the DFT tiles are rank 1, so the ID butterfly recovers the FFT
(nnz ≈ 2N log₂N).
H2. R grows like t · (variation of c) · N. A small t or a nearly constant c means few FFTs, which win.
A large t or a strongly varying c means the butterfly wins.

## (a) The FFT is the rank-1 butterfly — H1 confirmed
Proof: with j = j₀ + 2^l a (row box at depth l, bit-reversed) and k = k₀ + 2^{L−l} m (column box at depth
L − l, bit-reversed), jk/N = (j₀k₀ + 2^{L−l} j₀ m + 2^l a k₀)/N + a·m. The last term is an integer, so each
tile is u(a)·v(m)ᵀ. The driver needed no change: a wrapper (`permuted_kernel`) gives the coordinate tree
virtual coordinates u = 0..N−1.

| N | ordering (t = 0) | max rank | ID nnz | SVD-ref nnz | 2N log₂N + 2N | rel. err (ID) |
|---|---|---|---|---|---|---|
| 256 | natural | 8 | 51,968 | 89,600 | 4,608 | 3.2e-7 |
| 256 | bit-reversed, both | **1** | **4,608** | 4,608 | 4,608 | 2.6e-14 |
| 1024 | natural | 8 | 348,049 | 559,104 | 22,528 | 4.9e-7 |
| 1024 | bit-reversed, both | **1** | **22,528** | 22,528 | 22,528 | 1.0e-13 |
| 1024 | bit-reversed rows only (or cols only) | 32 (full) | 1,144,828 | 2,795,505 | — | 8e-16 |
| 4096 | natural | 8 | 1,975,395 | — | 106,496 | 5.4e-7 |
| 4096 | bit-reversed, both | **1** | **106,496** | — | 106,496 | 4.1e-13 |

- The bit-reversed ID butterfly has **exactly** the FFT's nnz, and every G_l entry has modulus 1 (deviation
  ≤ 2.7e-15): these are the radix-2 twiddle factors. Natural ordering costs **11× (N=256) to 18.5× (N=4096)**
  more, because its tiles have ε-rank ≈ 8 instead of 1.
- **The trick is fragile.** Both orderings must be bit-reversed; with only one, the tiles are full rank
  (√N). For t = 1 the bit-reversed tiles are also full rank (32 at N = 1024, nnz 1.10M). Natural ordering
  gives the same rank (8–9) at t = 0 and t = 1, so it is the robust choice for a general FIO.

## (b) Low-rank phase + R FFTs vs the butterfly
`M = exp(2πi t c(x)|ξ|) ≈ Σ_{r<R} a_r(x) b_r(ξ)` (SVD, σ_R ≤ ε σ_1), and then
`K g = (−1)^j Σ_r a_r ∘ N·ifft(b_r ∘ g)`. Media: **constant** c = 1/8; **weak** c = (20 + sin 2πx)/160
(Δc = 1/80); **sine** (the SPEC default, Δc = 1/8); **lens** c = (1 + 2e^{−((x−½)/0.05)²})/16 (Δc = 1/8, 5× steeper).
All errors are measured against the exact K g. R-FFT: 2e-14 to 9.7e-7. ID butterfly: 2.3e-7 to 6.0e-7.
So both methods meet ε.

| N = 4096 | t | B = tΔcN/2 | R | R-FFT cost | ID-BF nnz (max r) | cost ratio | R-FFT storage |
|---|---|---|---|---|---|---|---|
| constant | any | 0 | **1** | 0.11M | 1.97M (8) | **0.06** | 6.1k |
| weak | 0.1 | 2.6 | 10 | 1.15M | 1.97M (8) | 0.58 | 61k |
| weak | 1 | 25.6 | 38 | 4.36M | 1.95M (8) | 2.2 | 234k |
| sine | 0.01 | 2.6 | 10 | 1.15M | 1.98M (8) | 0.58 | 61k |
| sine | 0.1 | 25.6 | 38 | 4.36M | 1.95M (8) | 2.2 | 234k |
| sine | 1 | 256 | 275 | 31.5M | 1.94M (9) | **16** | 1.69M |
| sine | 3 | 768 | 789 | 90.5M | 1.88M (11) | 48 | 4.85M |
| lens | 1 | 256 | 270 | 31.0M | 2.01M (12) | 15 | 1.66M |
| lens | 3 | 768 | 389 | 44.6M | 2.34M (18) | 19 | 2.39M |

**Crossover** (`results/crossover.json`; t* is interpolated on the t grid; B* = t* Δc N/2):

| metric | N=256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|
| cost: break-even R* = nnz_BF / (FFT + 2N) | 10 | 12 | 14 | 16 | 17 |
| cost: B* (sine / lens / weak) | 2.2 / 2.7 / 2.2 | 3.9 / 3.9 / 3.9 | 4.9 / 5.5 / 4.9 | 6.1 / 6.3 / 6.1 | 7.1 / 7.8 / 7.1 |
| cost: t* for sine (Δc = 1/8) | 0.13 | 0.12 | 0.076 | 0.048 | 0.028 |
| storage: B* for sine | — | — | — | 245 | 294 |

- **H2 mostly confirmed, with a refinement.** R depends on t, c and N only through **B = t Δc N/2**: the
  number of wavelengths by which the phase perturbation varies. Fit: R ≈ B + O(log 1/ε), with R ≈ 3–7 for
  B < 1, 10 at B = 2.6, 38 at B = 26 and 275 at B = 256 (`fig_b_rank.png`; weak, sine and lens collapse onto
  one curve). The *smoothness* of c does not matter for R; only the *range* of its values does. Sine and
  lens have equal R up to B ≈ 30. Lens with large B saturates below R = B (R = 389 vs 789 at B = 768).
- **Who wins on apply cost.** R-FFT wins iff R < R* ≈ 10–17, i.e. **B ≲ 2–8 wavelengths**. Examples:
  constant c for any t (R = 1, 17× cheaper at N = 4096); the weak medium for t ≲ 0.3–1.3; the sine and lens
  media for t ≲ 0.03–0.17. For fixed t > 0 and non-constant c, B ∝ N, so R-FFT costs O(N² log N) and the
  butterfly (O(N log N), rank ≈ 8–12) wins for large N. At N = 4096, t = 1 the R-FFT method is 16× costlier.
- **Storage and wall time move the crossover far to the right.** The FFT stores nothing, so R-FFT storage
  (1.5RN) beats butterfly nnz until R ≈ 320, i.e. B* ≈ 245–295 for sine at N ≥ 2048. In wall time (a
  batched numpy FFT vs a scipy.sparse matvec, on a shared machine, so noisy) the crossover is at B* ≈ 18–60.
- **The lens hurts both methods at large t.** At t = 3 the butterfly's max rank grows from 11 (sine) to 18
  (lens), and its nnz by 25%. This is the "strongly varying medium" effect H2 predicted, but it shows up
  in the butterfly, not in R.
- Build cost (secondary): SVD of N × (N/2+1) is 2.4 s at N = 4096; the O(N²) ID butterfly takes ≈ 6 s.

## Claims
| claim | label |
|---|---|
| Both-bit-reversed DFT tiles are exactly rank 1; the ID butterfly has nnz 2N log₂N + 2N and unimodular twiddles | proved + computed |
| Natural-order DFT butterfly needs 11–18.5× more nnz (rank ≈ 8 at ε = 1e-6) | computed |
| One-sided bit reversal, or t > 0 with bit reversal, gives full-rank tiles | computed (N ≤ 1024) |
| Constant medium: K = DFT·diag(e^{2πi t c|ξ|}) → R = 1, one FFT, any t | proved + computed |
| R ≈ B + O(log 1/ε) with B = t Δc N/2, independent of the shape of c | computed (fit), consistent with Yang 2018 [cited] |
| Cost crossover at R* ≈ 10–17 (B* ≈ 2–8), growing slowly with N | computed |
| Storage crossover at B* ≈ 250–300 (N ≥ 2048) | computed |
| Lens R saturation at large B: rows under-resolve the phase (t·max c′/2 ≈ 3 > 1 wavelength per row) | conjecture |
| NUFFT-type (low-rank phase) beats the butterfly iff the phase is close to linear | cited: Yang, arXiv:1803.04128 |
| Butterfly ranks O(1) for FIOs in natural ordering | cited: Candès–Demanet–Ying 2009; Li et al. 2015 |

## What to try next
1. **Local R-FFT ("windowed NUFFT")**: split x into blocks on which c varies little, use R_block FFTs per
   block. This interpolates between the two methods and is the butterfly's top levels in disguise.
2. A hybrid butterfly that uses rank-1 FFT stages wherever a level's tiles are exactly rank 1 (for example
   the t-independent ξ-sign split), and ID stages elsewhere.
3. Chebyshev interpolation in c in place of the SVD, so construction is O(R N log N) and R can be predicted
   from B a priori (Yang's O(N) selector).
