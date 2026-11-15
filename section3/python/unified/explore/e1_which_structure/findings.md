# E1: which structure for which kernel (swap: admissibility × kernel)

**Question.** Fix one kernel matrix. Compress it three ways, using the admissibility rules from `unified/core`, the
same trees, and the same ε-rank rule: keep σ_i > ε σ_1 per block, with ε = 1e-6.
- **Weak (HODLR):** every off-diagonal box pair is compressed.
- **Strong (H):** only boxes that do not touch are compressed. Touching leaf pairs stay dense.
- **Complementary (butterfly):** at every stage l, tiles pair a row box at depth l with a column box at depth L−l.

Which rule stores the fewest numbers, and how does storage grow with N?

**Hypothesis (stated before running).** For non-oscillatory kernels, strong is best and complementary is
wasteful. For oscillatory kernels, only complementary keeps bounded ranks.

**What was run.** The script is `run.py`, which takes 330 s on 4 workers. Results are in
`results/sweep.json` and `results/ordering.json`. The figures are `heatmap.png`, `storage_vs_N.png` and
`ordering.png`. N = 256…4096.

**How storage is counted.**
- Weak and strong: flat, non-nested H storage, Σ min(r(m+n), mn) + dense blocks. The ranks come from
  SVD singular values.
- Complementary: the entries of the L+2 factors of a nested-basis SVD butterfly. This generalises
  `code/butterfly.py` to any pair of trees, including quadtrees.
- Each structure gets the best of two leaf sizes: {16, 64} for H; {1, 8} for the 1D butterfly and
  {4, 16} for the 2D butterfly.

**Accuracy achieved.**
- H: relative Frobenius error, computed exactly from the discarded singular values. It stays at
  ≤ 1e-6 everywhere.
- Butterfly: estimated with 4 Gaussian probes. It stays at ≤ 1e-6, with one exception: the Gaussian with
  h = 0.05 at N = 4096 reaches 2e-6, because projection errors add up over L stages.

**Kernels.**
- Laplace 2D, on the starfish curve and on random points in the unit square.
- Helmholtz on the starfish (perimeter 9.02):
  - fixed k = 1, 10, 50, 150, using only N with ≥ 10 points per wavelength;
  - a series with 10 points per wavelength, where k ∝ N and k runs from 18 to 285.
- The 1D FIO `fio1d(t)` for t = 0 (the DFT), 1 and 3.
- The Gaussian RBF exp(−|x−y|²/h²) on random points in the unit square, with h = 0.5 and h = 0.05.

## Results at N = 4096

Each cell gives storage/N, the max rank, and the fitted exponent α in storage ∝ N^α (fit over N = 256–4096).

| kernel | weak | strong | complementary | best |
|---|---|---|---|---|
| Laplace curve | **200, r17, 1.19** | 276, r6, 1.20 | 434, r141, 1.31 | weak |
| Laplace area | **519, r52, 1.42** | 784, r9, 1.44 | 772, r86, 1.39 | weak |
| Helmholtz k=1 / 10 / 50 | **189 / 216 / 292**, α≈1.12–1.19 | 263 / 303 / 367 | 420 / 451 / 510 | weak |
| Helmholtz k=150 (one N, 16 ppw) | **441, r47** | 466, r19 | 589, r141 | weak (by 6%) |
| Helmholtz 10 ppw (k ∝ N) | 624, r18→74, 1.52 | **566, r8→27, 1.40** | 666, r141, 1.40 | strong |
| DFT (FIO t=0) | 2886, r1045, 1.90 | 1543, r275, 1.70 | **629, r18, 1.26** | complementary |
| FIO t=1 / t=3 | 2885 / 2890, α 1.90 / 1.92 | 1539 / 1544, α 1.70 / 1.74 | **630 / 665, r22 / 29, α 1.28 / 1.30** | complementary |
| Gauss h=0.5 | 256, r19, 1.16 | 779, r10, 1.39 | **117, r10, 0.95** | complementary |
| Gauss h=0.05 | 855, r95, 1.77 | 958, r21, 1.72 | **753, r141, 1.66** | complementary (narrowly) |

### Ordering test

Butterfly tiles at leaf size 1 and N = 4096. The table gives the max tile rank and storage/N for each
ordering of rows and columns (natural or bit-reversed).

| kernel | nat/nat | bitrev/bitrev | mixed |
|---|---|---|---|
| DFT | r 7, 742 | **r 1 at every stage, 26 = 2L+2** | r 64 = √N, 10922 |
| FIO t=1 | r 8, 758 | r 64, 10519 | r 61–64 |
| FIO t=3 | r 10, 751 | r 64, 10440 | r 44–64 |

## What the data say

1. **The oscillatory half of the hypothesis holds for the 1D DFT and FIO.**
   - Complementary keeps its max rank bounded: 18 to 29, and flat in N. Storage grows like N log N
     (α ≈ 1.26–1.30).
   - Weak ranks grow like N/4: r = 1045 at N = 4096, so α ≈ 1.9, which is nearly dense.
   - Strong ranks grow like N^0.9, so α ≈ 1.7.
   - At N = 4096 the butterfly stores 2.5× less than strong and 4.6× less than weak. The gap widens with N.

2. **Helmholtz on a curve falsifies "only complementary stays bounded".**
   - With 10 points per wavelength, no structure stays bounded. Strong ties complementary in exponent
     (1.40 each) and has lower storage (566 vs 666 per point).
   - The butterfly's max rank is 141 at every N. That is the full rank of the tiles that contain the
     diagonal. The rank per stage, 10, 20, 40, 80, 141, 80, …, doubles toward the middle stage, so these
     tiles are full rank: the log singularity sits inside them.
   - Strong ranks grow slowly: 8, 10, 13, 18, 27.
   - So on boundary-integral Helmholtz kernels, complementary admissibility is wrong near the diagonal. It
     fits only far blocks. This matches the H-BF / HODBF design (Liu et al. 2021).

3. **The non-oscillatory half is only partly right: strong is never best in flat storage at N ≤ 4096.**
   - On curves, weak (HODLR) wins by 25–40%. Its ranks stay flat (14 → 17) on this 1D geometry. Strong pays
     for dense neighbour blocks.
   - On the area problem, weak still wins at 4096. But its rank grows (20 → 52), while strong stays at 9.
     The exponents are close (1.42 vs 1.44), so the crossover is beyond 4096. This is a conjecture: the
     fitted exponents are inflated by changes of leaf size and by the log factor.
   - Complementary is indeed wasteful on curves, at 1.6–2.2× the best storage. On the area problem it is
     about equal to strong.

4. **Smooth global kernels favour complementary, and the reason is nesting, not admissibility.**
   - For the Gaussian with h = 0.5, every tile has rank ≤ 10. The nested butterfly bases then give
     storage/N that falls slightly (α = 0.95).
   - Flat weak and strong storage store a basis per block. A nested weak format (HBS/H²) would also be
     O(N). This comparison is therefore biased toward the butterfly.
   - For the Gaussian with h = 0.05, h is fixed while the point density grows. So rows hold Θ(N)
     significant entries and every format tends toward N² (α ≈ 1.7).

5. **Ordering.**
   - Bit-reversing both the rows and the columns makes every DFT tile *exactly* rank 1. The reason:
     ω^{(a+2^l m)(b+2^{L−l} n)} separates. The storage of 26N = (2L+2)N is exactly the radix-2 FFT.
   - Mixing the orders gives rank √N, which is worse than dense.
   - The FIO loses this property. With t ≥ 1, bit-reversed tiles are full rank (64). Only natural ordering
     keeps rank 8–10.
   - So exact rank 1 is algebraic (it comes from the group structure of the DFT). The butterfly's bounded
     rank is geometric (it comes from the phase-space cells).

## Claims

| claim | status |
|---|---|
| DFT and FIO: complementary max rank ≤ 29, flat in N; weak/strong ranks ∝ N^0.9–1; storage α 1.26–1.30 vs 1.70–1.92 | computed |
| DFT, bit-reversed rows and columns: every tile has rank 1, storage (2L+2)N; mixed orders give rank √N; FIO with t ≥ 1 has full-rank tiles | computed + proved (one-line identity, tested) |
| Helmholtz at 10 ppw on a curve: strong ≤ complementary (566 vs 666 per point, same α = 1.40); butterfly diagonal tiles are full rank | computed |
| Laplace and low-k Helmholtz on curves: flat weak storage < strong < complementary at N ≤ 4096 | computed |
| Area Laplace: strong overtakes weak for N > 4096 | conjecture (weak rank 20 → 52, strong 9) |
| Gaussian h = 0.5 butterfly advantage comes from nested bases, not from the admissibility rule | conjecture (nested-weak storage not measured) |
| Butterfly bounded rank for FIOs (Candès–Demanet–Ying 2009; Li et al. 2015); weak ranks of HF kernels grow like k (Engquist–Zhao 2018); butterfly restricted to far blocks for BIEs (Liu et al. 2021) | cited |

## Next steps

- Hybrid structure: use the strong H partition, and replace each far block with a butterfly. The
  prediction is that this beats both pure structures on Helmholtz at 10 ppw.
- Measure nested storage (HBS/H²) for weak and strong, so that point 4 compares like with like.
- Extend Laplace area to N = 16384 to locate the weak/strong crossover.
