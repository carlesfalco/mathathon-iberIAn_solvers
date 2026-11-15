# E5 — Why exact-ID recursive skeletonization breaks down at high k, and how to fix it

**Setup.** The problem is P3: the 2D Helmholtz equation on the starfish curve, A = I + S_k w, with N = 2048. A is well conditioned (cond(A) ≈ 1.05). We use `rskel_factor` with ε = 1e-8 and max_leaf 32. We measure:
- **err**: ‖x − x_dense‖/‖x_dense‖ for one complex random right-hand side;
- **cond**: the worst cond(X_RR) over all eliminated pivot blocks;
- **skel**: the total number of skeletons.

Reproduce with `PYTHONPATH=. .venv/bin/python -m unified.explore.e5_highk_instability.run` (≈ 6 min; `--quick` is faster). Data are in `results/r1..r9*.json`; figures are `r1_ksweep.png`, `r2_fine.png` and `r4_boxes.png`.

## Mechanism (one sentence)
The driver compresses rows and columns with **one** ID of the stack M = [A_FB; A_BFᵀ] and applies the row operation with **Tᵀ**. When T is complex, the sparsification W → LWU uses L = Uᵀ ≠ Uᴴ, so it is **not a congruence**. The pivot X_RR = Πᵀ W_BB Π (with Π = [I; −T]) then loses the positive-definite Hermitian part that A has (Herm(A) ⪰ 0.98 I at every k tested). Near-singular pivots appear, and their Schur complements feed the next level, so the damage cascades up the tree.

**Smallest example.** Take W = I and T = i (the imaginary unit). Then Πᵀ Π = 1 + i² = 0: even the identity has a zero bilinear pivot. The Hermitian pivot Πᴴ Π = 2 does not.

## Hypotheses, stated before running, and verdicts
- **H-a (E3's conjecture).** The exact ID marks nearly non-radiating combinations as redundant. **Refuted at the leaves.**
  - At the leaves, the redundant subspaces of the exact and proxy IDs nearly coincide: the largest principal angle has sin 0.04–0.06 at every k.
  - Both kinds of redundant combinations radiate off the curve at the 1e-9 level (r5).
  - Leaf pivots are fine in weak RS (λ_min Herm ≥ 0.99).
  - The breakdown starts at level 3 or above, on the Schur-complemented matrix W, which is far from I (‖W_BB − I‖ ≈ 1).
- **H-b.** Resonances of the boxes. **Refuted.**
  - In the step-1 sweep over k = 40…100, log₁₀ cond grows smoothly at about 0.072 decades per unit of k.
  - The largest local jump is 1.5 decades (at k = 81), with no clustering.
  - Error first exceeds ε at k = 54 (r2).
- **H-c.** The cause is the non-congruent (Tᵀ) row operation when T is complex. **Confirmed**; see the table and claims 3–5.

## Localisation (r4; weak, k = 80, worst box per level)

| level | k·hw | exact: cond / min Herm(X_RR) / ‖W_BB − I‖ / ‖Im T‖/‖T‖ | realified (real T): cond / min Herm(X_RR) |
|---|---|---|---|
| 4 (leaves) | 6.2 | 43 / +0.91 / 0.013 / 0.07 | 40 / +1.00 |
| 3 | 12.5 | 110 / **−2.6** / 0.99 / 0.19 | 77 / +0.14 |
| 2 | 24.9 | 1.1e5 / −3.4 / 1.5 / 0.48 | 190 / +0.07 |
| 1 | 49.8 | **2.5e10** / −60 / **2.1e3** / 0.60 | 97 / +0.06 |

- The share of T that is imaginary grows with box size times k.
- The Hermitian part of X_RR turns negative one level before cond blows up. In strong RS it is already negative at the leaves (−0.32 at k = 80, −5.8 at k = 100).
- cond(X_RR) does not track the skeleton fraction (`r4_boxes.png`, middle panel).
- Proxy T is almost real (‖Im T‖/‖T‖ ≤ 0.15). This is because the proxy rows are closed under conjugation up to signs: the Graf modes m and −m both appear. That is why proxy works.

## Fixes (r6; ε = 1e-8; the same pattern holds at k = 100 and 120)

| k = 80 | weak err / cond / skel | strong err / cond / skel |
|---|---|---|
| exact (baseline) | 5.7e-5 / 3.3e9 / 2916 | 2.3e-7 / 4.1e10 / 2366 |
| proxy | 5.3e-11 / 73 / 3088 | 1.4e-11 / 44 / 2791 |
| **realified**: ID of [Re M; Im M], so T is real; driver unchanged | **3.0e-11 / 54 / 3071** | **1.5e-11 / 30 / 2736** |
| **congruent**: ID of [A_FB; A_BFᴴ], row operation Tᴴ (`congruent.py`) | **3.3e-11 / 72 / 3046** | **1.9e-11 / 32 / 2690** |
| (a) hybrid: exact rows + proxy rows | 2.0e-11 / 68 / 3097 | 1.3e-11 / 41 / 2770 |
| (b) oversample: rank + 8 skeletons | **1.5e-2** / 4.9e12 / 3697 | 1.6e-6 / 4.4e10 / 3240 |
| (c) strong RRQR (\|T_ij\| ≤ 1.01) | 6.0e-7 / 1.1e9 / 2919 | 1.7e-7 / 9.4e9 / 2369 |
| (d) larger far field (weak's far field ⊃ strong's) | fails in both columns | — |
| (e) absolute tolerance | 1.2e-7 / 4.8e8 / 2448 | 2.1e-8 / 7.7e8 / 1966 |

**What works.**
- The fixes that restore congruence, or make T nearly real, all work: realified, congruent, hybrid and proxy.
- Over k = 10…120 (r1), realified and congruent stay at err ≤ 9e-11 with cond ≤ 146, in both weak and strong RS.
- On the ε sweep at k = 80 (r3), they reach the target:

  | ε | 1e-6 | 1e-8 | 1e-10 | 1e-12 |
  |---|---|---|---|---|
  | realified err | 3.4e-9 | 3.0e-11 | 2.3e-13 | 3.9e-15 |
  | exact err | 2.1e-7 | 5.7e-5 | 7.0e-5 | 2.6e-4 |

**Cost.**
- Weak: +4.5% skeletons (congruent) or +5.3% (realified) compared with exact.
- Strong: +14% (congruent) or +16% (realified).
- Both are below proxy (+6% weak, +18% strong).
- Realified doubles the rows of the ID; congruent costs the same flops as exact.

**What does not work.**
- The fixes that add skeletons *uniformly* or change the pivot rule, but leave T complex, all fail.
- Oversampling is *worse*: error 2.2 at k = 100 and 25 at k = 120, both weak.
- This matches E3's observation that tightening ε makes Exact worse.

Whether the per-box *rank allocation* matters is checked at equal rank below (r8).

## Reviewer diagnostics

### (1) As ε tightens, does the rank grow, and does that make T worse? (r7; k = 80, weak, worst box = largest cond(X_RR))

| ε | 1e-4 | 1e-6 | 1e-8 | 1e-10 | 1e-12 |
|---|---|---|---|---|---|
| exact: total skeletons | 2058 | 2498 | 2916 | 3284 | 3635 |
| exact: worst-box rank / \|B\| (level 1) | 72/107 | 83/127 | 94/145 | 106/162 | 116/180 |
| exact: worst-box cond(X_RR) | 2.9e7 | 1.0e9 | 2.5e10 | 7.1e10 | 1.9e11 |
| exact: worst-box cond(T) | 2.2e9 | 3.5e10 | 1.4e10 | 7.1e8 | 2.8e7 |
| exact: worst-box max \|T_ij\| | 1.39 | 1.48 | 1.48 | 1.60 | 1.32 |
| exact: worst-box \|Im T\|/\|T\| | 0.71 | 0.63 | 0.66 | 0.63 | 0.61 |
| realified: worst-box cond(X_RR) | 78 | 130 | 190 | 270 | 150 |
| realified: max cond(T) over all boxes | 4e6 | 5e9 | 6e11 | 1e16 | 1e14 |
| realified: max \|T_ij\| over all boxes | 1.59 | 1.77 | 1.71 | 1.82 | 1.69 |

- **Yes, the rank grows** as ε tightens.
- **No, T does not get worse.**
  - max |T_ij| stays between 1.3 and 1.8 for every compressor and every ε.
  - cond(T) is very large (1e7–1e16) for *all* compressors, including the stable ones. For exact's worst box it is not even monotone in ε.
- So neither the size nor the conditioning of T predicts cond(X_RR). What separates the two rows is the imaginary share of T (0.6–0.7 for exact, 0 for realified), equivalently the sign of Herm(X_RR).

### (2) Equal rank (r8; weak; ranks copied box by box from a reference run)

| k | ranks taken from | exact: err / cond | randomized: err / cond | proxy: err / cond | realified: err / cond |
|---|---|---|---|---|---|
| 80 | exact (2916) | 5.7e-5 / 3.3e9 | 1.6e-7 / 2.1e9 | 3.0e-7 / 54 | 1.9e-7 / 54 |
| 80 | proxy (3088) | 1.7e-11 / **1.8e5** | 2.9e-11 / 2.4e6 | 5.3e-11 / 73 | 2.7e-11 / 110 |
| 120 | exact (3342) | 5.2e-4 / 1.2e12 | 6.9e-5 / 3.0e11 | 1.2e-5 / 140 | 3.6e-6 / 65 |
| 120 | proxy (3555) | 2.4e-11 / **1.4e5** | 3.8e-11 / 2.9e4 | 9.5e-11 / 65 | 3.0e-11 / 72 |

**At exact's own ranks**, proxy and realified win *without* being more accurate:
- Their worst error against the true far field is **larger**: 1e-4 to 3.5e-3 in a few boxes, versus ≤ 1.5e-8 for exact and randomized.
- They fit a nearly conjugation-closed space with too few skeletons. Their solve error, 3e-7 to 1e-5, comes from that truncation.
- Their pivots stay definite: cond ≤ 140 and min Herm(X_RR) ≥ −0.26.
- Exact and randomized at the same ranks have indefinite pivots, with min Herm(X_RR) between −60 and −900.

**At proxy's (or realified's) per-box ranks**, which are about 6% larger:
- Exact and randomized also reach err ≤ 2e-10.
- **But their pivots stay indefinite**: min Herm(X_RR) between −10 and −540, and cond 3e4–7e6, versus 55–180 for real T.

**Conclusion: both rank and subspace matter.**
- The proxy rank allocation damps the cascade by about 5 orders of magnitude.
- Only a real T (or the Tᴴ row step) removes the indefiniteness.
- Simply *more* rank does not help: uniform +8 and tighter ε both make exact worse.
- Why proxy's specific per-box ranks help exact is open. A conjecture: they match the dimension of the conjugation-closed far space.

### (3) The proxy hypothesis, stated precisely, and the test (r9; W = A, weak far field)

**Hypothesis (Green's representation).**
- Take x in box B, and y outside the proxy circle Γ (radius 2.5·hw).
- The function u = G(·, x) is a radiating Helmholtz solution outside Γ. Therefore
  G(y, x) = ∫_Γ [u ∂_n G(y, ·) − ∂_n u · G(y, ·)] ds.
- Consequently, the row space of A[F_out, B] (and of A[B, F_out]ᵀ, since the kernel is symmetric) is contained in the row space of [K(Γ, B); ∂_n K(Γ, B)]. Far points inside Γ are kept as explicit rows.

**Single-layer-only proxies.**
- The core proxy uses only K(Γ, B). This suffices when H_m(kR) ≠ 0 for all m (E3, claim 2: true for H0^(1)).
- FMM-LU's combined-field proxy (∂_n G + iηG) is the variant that is safe for every kernel and every k.

**Measured.**
- Containment: ‖M_exact (I − P_proxy)‖/‖M_exact‖ ≤ 1.6e-8 at levels 2–4, for k = 20 and 80 (with n_proxy = 1.5kR + 16). **The hypothesis holds.**
- Rank: proxy ≥ exact (46.8 vs 43.4 at level 2, k = 80).

**Consistency with our mechanism.**
- The proxy hypothesis is about *approximation*, and it holds. It does not by itself guarantee stable pivots.
- By Graf, the proxy-ring row space is span{J_m(kr) e^{imθ}, |m| ≤ M}. Because J₋ₘ = (−1)ᵐ Jₘ, this span is closed under complex conjugation. Measured conjugation gap (sine of the largest angle between the row space and its conjugate): 0.006–0.06.
- The exact far row space is a non-closed subspace of that space. Its gap is 0.16 at the leaves and grows to 0.95–1.0 at level 2–3.
- The explicit rows for far points inside Γ partly break the closure (gap 0.02–0.30 for proxy with all rows). That matches proxy's T being *nearly* real (‖Im T‖/‖T‖ ≤ 0.17) rather than exactly real.
- **So the mechanism is consistent with the proxy hypothesis.** The realified ID compresses against the conjugation closure of the exact row space, which (to ε) lies inside the proxy space. It therefore inherits proxy's stability while using only the true far field.

## Claims

| # | Claim | Status |
|---|---|---|
| 1 | Reproduces E3. Exact weak at k = 80: err 5.7e-5; ε = 1e-6/1e-8/1e-12 gives 2.1e-7/5.7e-5/2.6e-4. cond grows smoothly, about 10^(0.072k), with no resonance clustering. | computed |
| 2 | The breakdown is a cascade: leaf pivots are fine in weak RS, and cond grows level by level (k = 80, weak: 43 → 1.1e2 → 1.1e5 → 2.5e10) while ‖W_BB − I‖ reaches 2e3. | computed |
| 3 | **Theorem.** Suppose Herm(W) ⪰ μI with μ > 0, and the row operation is Uᴴ (i.e. T is real for the driver's Tᵀ, or Tᴴ in `congruent.py`). Then, up to the O(ε) dropped R–F blocks: (i) Herm(X_RR) ⪰ μI, so ‖X_RR⁻¹‖ ≤ 1/μ; (ii) the next working matrix W′ satisfies Herm(W′) ⪰ μ/(1 + ‖T‖²) I. | **proved.** Herm(UᴴWU) = Uᴴ Herm(W) U, plus Lin's Schur-complement lemma Re(A/A₁₁) ⪰ (Re A)/(Re A₁₁) [Lin 2014, arXiv:1410.5140, Lemma 2.5], plus min over x of ‖x‖² + ‖y − Tx‖² ≥ ‖y‖²/(1 + ‖T‖²). Checked numerically on every box: min over boxes of [λ_min Herm(X_RR) − λ_min Herm(W_BB)] = +5.7e-5 ≥ 0. The bound (ii) compounds pessimistically over levels; measured min Herm(X_RR) ≥ 0.05. |
| 4 | With complex T and plain Tᵀ there is no such bound. Πᵀ W Π can vanish even for W = I (T = i), and measured Herm(X_RR) goes negative at level 3 (weak) or at the leaves (strong). | proved (example); computed |
| 5 | Realified and congruent RS restore err ≈ 1e-11 and cond ≤ 150 for k ≤ 120, at +5% (weak) or +15% (strong) skeletons. Uniform oversampling, strong RRQR, absolute tolerance and a larger far field do not help. | computed |
| 5b | At equal per-box rank, real or proxy T keeps pivots definite (cond ≤ 180). Exact or randomized T at proxy's ranks reaches err ≤ 2e-10 but with indefinite pivots (cond 3e4–7e6). At exact's ranks, proxy and realified are *less* accurate on the far field (1e-4–3.5e-3) yet still stable. Rank allocation damps the cascade; the realness of T removes it. | computed |
| 5c | Tightening ε grows the rank, but max \|T_ij\| (1.3–1.8) and cond(T) do not track cond(X_RR); the imaginary share of T does. | computed |
| 6b | Green's-representation containment holds (≤ 1.6e-8). The proxy-ring row space is nearly conjugation-closed (gap ≤ 0.06); the exact far row space is not (gap 0.16–1.0). | computed; closure under conjugation follows from Graf (proved for exact Bessel modes) |
| 6 | Proxy works because its row set is nearly conjugation-closed, so T is nearly real (‖Im T‖/‖T‖ ≤ 0.15 vs 0.6 for exact). | computed; mechanism (Graf ±m symmetry) conjecture |
| 7 | E3's "non-radiating redundant subspace" conjecture is false at the leaves: exact and proxy subspaces agree (sin of the largest angle ≤ 0.06), and their off-curve field is ~1e-9. | computed |

## Literature
These are full-text checks of five papers by a sub-agent, plus Lin 2014; we have not read the Martinsson 2019 book or Xing–Chow 2020.

**Which row transpose each paper uses:**
- FMM-LU (Sushnikova–Greengard–O'Neil–Rachh 2023, arXiv:2201.07325) uses one T for A_FB and A_BFᵀ, i.e. **Tᵀ**. That is our failing setup. The paper notes that this works "when the kernel … is symmetric" and *assumes* X_RR is invertible.
- Minden–Ho–Damle–Ying 2017 (arXiv:1609.08130) and HIF-IE (Ho–Ying 2016) use **T\***, i.e. our "congruent" variant. They also only assume the pivot block is invertible.
- Ho–Greengard 2012 use separate row and column IDs. They warn that block elimination "may be numerically unstable" without pivoting.
- RSRS (Yesypenko–Martinsson, arXiv:2311.01451, Remark 1) reports that cond(A_RR) and the modified block differ only slightly in practice.

**Not found.** We found no paper that describes this high-k breakdown, or that connects the choice between Tᵀ and T\* to pivot stability. To our knowledge both are new (a ~15-minute search).

**Recommendation.** For complex, non-Hermitian kernels, use the Tᴴ convention, or a real T. It costs nothing when the kernel's far field is closed under conjugation.
