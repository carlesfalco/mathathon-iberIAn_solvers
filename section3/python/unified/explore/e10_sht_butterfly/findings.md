# E10: the fast spherical harmonic transform as a butterfly kernel swap

**Verdict.**
- **Kernel swap.** The associated-Legendre half of the spherical harmonic transform (SHT) runs through the
  unmodified `drivers/butterfly_id.py`. The only new parts are a table kernel, zero-padding, and the
  transposed factors. No driver line changed.
  - Accuracy: synthesis and analysis both meet the target ε for every order m and every n up to 8192.
  - Physics checks: an analytic plane-wave expansion and a full-sphere Parseval check pass at 1e-11 and 2e-14.
- **Ranks.** Tile ranks are bounded in n: the maximum is 13 at ε = 1e-6 and 18 at ε = 1e-10, for n = 256…8192.
- **Storage.** Storage beats dense from n ≈ 256 (ε = 1e-6) or n ≈ 512 (ε = 1e-10). At n = 8192 it is 12% / 20% of dense.
- **Turning points do not break the butterfly.** Tiles that straddle the turning curve have *lower* rank
  than oscillatory tiles.
- **What breaks is numerical.** Deep below the turning curve the driver runs a per-tile *relative* ID on
  subnormal (≈1e-310) entries and returns **NaN**. A one-line kernel fix removes this: zero every entry
  below 1e-16·max.
- **Ordering dial.** Bit reversal, the E2 trick that turns the DFT into an FFT, gives rank 4 for the
  Chebyshev transform (proved) but full rank for Legendre.

Reproduce (≈ 8 min; `--quick` ≈ 20 s):
```
PYTHONPATH=. .venv/bin/python -m unified.explore.e10_sht_butterfly.run
.venv/bin/python -m pytest -q unified/explore/e10_sht_butterfly
```
Data are in `results/r0…r6*.json`. Figures: `r1_crossover.png`, `r2_msweep.png`, `r4_turning_map.png`. The
log is `results/run_log.txt`. The hypotheses were written before any run, in `hypotheses_prerun.md`.

---

## (a) The physics problem, in plain words

Many quantities live on a sphere:
- the temperature and pressure of the atmosphere (weather and climate models such as ECMWF's IFS, which is spectral);
- the Earth's gravity and magnetic fields (geodesy and geophysics);
- the temperature of the cosmic microwave background (CMB).

Just as a periodic signal is written as a sum of sines and cosines, a function on the sphere is written as
a sum of **spherical harmonics**:

  f(θ, φ) = Σ_{l=0}^{n−1} Σ_{m=−l}^{l} a_lm · P̃_l^m(cos θ) · e^{imφ}.

Here θ is the colatitude (0 at the north pole, π at the south pole) and φ is the longitude.

- **Integer degree l.** It sets the spatial scale: about l wavelengths around a great circle.
- **Order m.** It counts the oscillations along a latitude circle.
- **P̃_l^m.** The *associated Legendre function*, normalised so that ∫_{−1}^{1} P̃_l^m(x)² dx = 1.
- **Why it matters.** The coefficients a_lm are what spectral weather models time-step. They are also what
  CMB analyses turn into the angular power spectrum C_l. Going from grid values to coefficients
  (**analysis**) and back (**synthesis**) is the SHT.

**Where the cost is.**
- The φ direction is an ordinary FFT.
- After the FFT, each order m leaves a dense matrix

    A^(m)[j, l] = P̃_l^m(cos θ_j),  j = 0..n−1 (latitudes),  l = m..n−1 (degrees).

- Synthesis is f_m = A^(m) a_m. Applying all n matrices densely costs ~n³/2 flops, against n² log n for the
  FFT part, so the Legendre part dominates at high resolution (n in the thousands).
- We put the latitudes at the **Gauss–Legendre nodes**, with weights w_j. Then analysis is just the
  transpose: a_m = A^(m)ᵀ diag(w) f_m.
  - This is exact because P̃_l^m P̃_{l′}^m is a polynomial of degree l + l′ ≤ 2n − 2, and n-point Gauss
    quadrature integrates every polynomial of degree ≤ 2n − 1 exactly. [proved]
  - So the inverse needs no linear solve, unlike E4.

**Why a butterfly can work.** Fix m and look at P̃_l^m(cos θ) in the (θ, l) plane. It has two regions,
split by the **turning curve** (l + ½) sin θ = m:
- **Oscillatory region (near the equator).** P̃_l^m ≈ amplitude(θ, l) · cos(phase(θ, l)), with phase ≈ (l + ½)θ
  for m = 0. This is a Fourier-integral kernel, the same class as B1. [cited: WKB / Olver asymptotics]
- **Evanescent region (near the poles).** P̃_l^m decays like exp(−l·g(θ)) and quickly becomes smaller than
  any floating-point number.

So A^(m) is a B1-like oscillatory kernel glued to an exponentially small one along a fold. That fold is the
"caustic-like" region we probe below.

## (b) Which dials should stay and which must change (written before running)

Full text: `hypotheses_prerun.md`. Summary, with grades:

| dial | prediction | outcome |
|---|---|---|
| 1 clustering / ordering | must change: the trees should cut θ, not x = cos θ | **Partly right.** θ wins on the ranks at the ends of the factorization. The x-tree leaves at the poles have many points, so the final stages have rank 20–26 against 2–9. Total storage is about the same (0.683 vs 0.701 of dense, n = 1024). |
| 1, bit reversal | helps Chebyshev, fails Legendre | **Confirmed.** Chebyshev tiles have rank 4 (proved), storage 0.129 of dense. Legendre tiles are full rank (32). |
| 2 admissibility | Complementary, unchanged | **Confirmed.** |
| 3 representation | column ID + nested interpolation, unchanged | **Confirmed.** |
| 4 sampling | exact ID; proxy/Green does not apply | **Confirmed as stated.** The per-tile *relative* tolerance has a defect, see (e). |
| 5 algebra | apply; analysis = transposed factors, no solve | **Confirmed.** Round-trip error ≈ 1–2× the synthesis error. |
| ranks | bounded in n; ~10–15 at 1e-10, ~7–10 at 1e-6 | **Bounded: confirmed. Constants underestimated:** 18 and 13. Reason (proved below): a real cosine has two phase branches. |
| turning points | no rank increase | **Confirmed.** Straddling tiles: mean rank 10.3–10.8, against 13.6–14.9 for oscillatory tiles. |
| evanescent region | zeroing entries < ε·max cuts storage "substantially" for m ≳ n/4 | **Refuted.** Storage drops by only 0–3%. An absolute-tolerance ID saves 7–28% at ε = 1e-10 (vs raw). The cost is spread over all tiles. Separately, a NaN failure that I had *not* predicted was found. |
| storage crossover | n ≈ 512 at 1e-6; 1024–2048 at 1e-10 | **Earlier than predicted:** ≈ 256 at 1e-6 and between 256 and 512 at 1e-10. |
| time crossover | later than storage, maybe not reached by 4096 | **Inconclusive:** the ratio is 0.5–2.7 at n = 4096–8192 and noisy under shared load. |

**Why no proxy surface?**
- The proxy trick needs a Green's function: entries outside a box must be fields radiated by sources inside it.
- P̃_l^m(cos θ) solves an ODE in θ for each l, the associated Legendre equation, which is the angular part
  of the Laplacian on the sphere.
- But the *column* variable l is a spectral index, not a point in space. There is no surface in (θ, l)
  through which the "far field" must pass.
- The butterfly's complementary low-rank property is the right structure, not the FMM's. [conjecture /
  folklore; consistent with all the butterfly SHT literature]

## (c) The change and its size

The driver and core are unchanged (0 lines). Everything is in `sht.py`.

| piece | lines | dial |
|---|---|---|
| table kernel `SHTProblem.kernel`: maps the tree coordinates the driver passes back to row and column indices of a precomputed A^(m) | 12 | kernel swap |
| `synth`: pads to len(x), because the driver sizes W by len(x) and A^(m) is n × (n − m) | 5 | glue (rectangular) |
| `adjoint` / `analyze`: Wᵀ G₀ᵀ…G_{L−1}ᵀ Uᵀ, then diag(w) | 16 | 5 (algebra: transpose apply) |
| `build`: two equal-depth `Tree`s over (θ, l), then `butterfly_id(..., row_tree, col_tree)` | 6 | 1 (coordinate choice) |
| *physics, not harness:* P̃_l^m by three-term recurrence with a per-node exponent; Gauss nodes polished by Newton | ≈ 70 | — |
| *optional:* `clip` (one `np.where`), and `absolute_tolerance` (a 25-line context manager that rebinds the driver's `interp_decomp` at runtime) | 1 / 25 | kernel / 4 |

So the swap is **≈ 39 lines of glue plus a 70-line special-function evaluator**. Driver lines touched: 0.

**Caveats on the kernel.**
- It reads a precomputed table rather than evaluating P̃_l^m at arbitrary (θ, l). The recurrence needs all
  lower degrees, so evaluating a single entry costs O(l).
- This is the same O(n²)-per-m precomputation that the driver's construction already costs; IDBF-style row
  sampling would cut it.
- Every published butterfly SHT (Tygert 2010, Seljebotn 2012) also feeds the compressor from the recurrence. [cited]

## (d) Verification

**R0, the kernel** (`r0_validation.json`):
- P̃_l^m agrees with `scipy.special.lpmv` (with its normalisation) to 1.4e-13 for l ≤ 60.
- ‖AᵀWA − I‖_max ≤ 3.6e-12 for n ≤ 4096, m ∈ {0, n/4, n/2}.
- Newton polishing uses the correct sign
  \(dP_n/d\theta = n(xP_n-P_{n-1})/\sin\theta\), checked by an explicit one-step regression test.
  Orthogonality is assessed from the resulting node/weight set rather than the recurrence alone.
- A dedicated test checks the rescaled recurrence where P̃_m^m underflows but P̃_l^m returns to O(1) past
  the turning point (m = 1500, θ = 0.3, l up to 6000). [computed]

**R1, m = 0, all results agree with the dense matrix** (random complex coefficients; "rt" = analysis(synthesis(a)) − a):

| ε | n | max rank | nnz / dense | synthesis err | round-trip err | build (s) |
|---|---|---|---|---|---|---|
| 1e-6 | 256 | 12 | 0.947 | 1.9e-7 | 3.0e-7 | 0.2 |
| 1e-6 | 1024 | 12 | 0.491 | 3.9e-7 | 5.5e-7 | 0.7 |
| 1e-6 | 4096 | 13 | 0.196 | 6.2e-7 | 7.4e-7 | 4.7 |
| 1e-6 | 8192 | 13 | **0.117** | 7.0e-7 | 8.5e-7 | 13.6 |
| 1e-10 | 256 | 16 | 1.087 | 1.3e-11 | 2.2e-11 | 0.2 |
| 1e-10 | 512 | 16 | 0.924 | 2.3e-11 | 3.2e-11 | 0.3 |
| 1e-10 | 1024 | 18 | 0.701 | 3.1e-11 | 4.6e-11 | 0.8 |
| 1e-10 | 2048 | 18 | 0.493 | 3.8e-11 | 5.3e-11 | 1.8 |
| 1e-10 | 4096 | 18 | 0.323 | 4.6e-11 | 6.6e-11 | 5.6 |
| 1e-10 | 8192 | 18 | **0.202** | 4.6e-11 | 9.7e-11 | 16.0 |

- nnz/dense falls by ≈ 0.63× per doubling of n, consistent with nnz ≈ c·r·n·log₂n. [computed]
- Errors grow slowly with n (×3–5 from 256 to 8192): the usual accumulation over L = log₂ n stages.
- **The Chebyshev transform** (cos lθ_j at Chebyshev points, the "cleanest case") gives *identical* ranks
  and storage to Legendre in natural order, to three digits: 0.701 / 0.493 / 0.323 at ε = 1e-10.
  - So the Legendre amplitude factor and the O(1/l) phase corrections cost nothing extra.
  - The rank is that of a real cosine.
- **Wall time** (secondary evidence: scipy CSR against BLAS dgemv, on a machine shared with other agents):
  - butterfly/dense is 3–11× at n ≤ 1024, 2–5× at 2048, and 0.5–2.7× at 4096–8192 across two runs.
  - So the time crossover lies around n ≈ 4096–8192 in this implementation, and is not reliably measured.

**Why rank 18 and not 12 like B1.** [proved, short]
- A tile of B1 holds one phase-space cell of exp(2πi xξ).
- A Legendre tile of the same index size (Δθ = π/2^s, Δl = n/2^{L−s}) holds a phase range Δθ·Δl = π: half as much.
- But cos φ = (e^{iφ} + e^{−iφ})/2 is a sum of two exponentials, one per phase branch ±lθ. Each branch is a
  B1-like tile, so

    rank(Legendre tile) ≤ 2 · rank(exp tile with half the phase range).

- The measured values are 18 ≈ 2 × 9 at 1e-10 and 12–13 ≈ 2 × 6 at 1e-6. The Chebyshev case has exactly the same ranks.
- Tygert (2010) and Seljebotn (2012) both use the parity splitting P_l^m(−x) = ±P_l^m(x). That halves the
  number of latitudes and degrees but keeps the two-branch structure. [cited; not run here]

**R5, physics checks** (`r5_physics.json`):

*(i) Analytic: plane-wave expansion* (Jacobi–Anger + Rayleigh / addition theorem). A plane wave
e^{ik k̂·r} with k̂ at polar angle α, restricted to the unit sphere, has m-th Fourier mode

  f_m(θ) = i^m J_m(k sin α sin θ) e^{ik cos α cos θ},

and its exact coefficients are

  a_lm = 2 i^l j_l(k) P̃_l^m(cos α). [cited identity; checked numerically by the dense transform]

At n = 2048 and k = 512, butterfly analysis of f_m matches these coefficients:

| m | butterfly vs exact | dense vs exact |
|---|---|---|
| 0 | 4.0e-11 | 6.8e-12 |
| 128 | 3.2e-11 | 7.5e-13 |
| 256 | 3.3e-11 | 5.5e-13 |

This is the incident-field expansion used in Mie and T-matrix scattering, so it is a physically meaningful test.

*(ii) Full sphere.* A random complex band-limited field with l_max = 255 on 256 Gauss latitudes × 512
longitudes, built with all 256 butterflies (one per |m|) plus the FFT:
- round-trip coefficient error 1.3e-11;
- **Parseval**, ∫|f|² dΩ (quadrature) = 2π Σ|a_lm|², holds to 2.1e-14 relative. [computed]

At this small size the total storage is 0.93 of dense, and building all 256 butterflies takes 23 s. This
check is about correctness, not speed.

## (e) Where it breaks

**R2, sweep over the order m** (n = 2048, plus n = 4096 for the zero-below-ε variant only). ε = 1e-10:

| m / n | raw: max r / nnz ratio / err | zero-below-1e-16 | zero-below-ε (err of the zeroed kernel) | absolute-tol ID (R6) |
|---|---|---|---|---|
| 0 | 18 / 0.493 / 3.8e-11 | 18 / 0.493 | 18 / 0.493 (0) | 16 / 0.460 / 8.8e-11 |
| 1/16 | 18 / 0.484 | 30 / 0.482 | 32 / 0.484 (2e-11) | — |
| 1/8 | 18 / 0.470 | 28 / 0.463 | 32 / 0.465 | 18 / 0.459 |
| 1/4 | 19 / 0.450 | 29 / 0.437 | 32 / 0.439 | 19 / 0.435 |
| 1/2 | 18 / 0.426 | 30 / 0.403 | 25 / 0.404 | 18 / 0.397 |
| 3/4 | **NaN** | 16 / 0.350 / 1.8e-11 | 16 / 0.349 | 16 / 0.338 |
| 7/8 | **NaN** | 16 / 0.296 | 16 / 0.288 | 13 / 0.276 |

At ε = 1e-6 the pattern is the same: NaN at m ≥ 3n/4 with the raw kernel, and nnz ratio 0.318 → 0.245
across the sweep.

1. **Subnormal tiles break the ID (a real failure).**
   - For m ≥ 3n/4 (n = 2048), and already at m = n/2 for n = 1024, A^(m) holds subnormal numbers down to
     5e-324. Near the poles the true values are below 1e-300.
   - `interp_decomp` truncates *relative to the tile's largest pivot* (`d > eps·d[0]`). On a tile whose
     largest entry is 7e-311 it accepts pivots that are numerical garbage. The triangular solve for T then
     overflows, and NaN propagates through the whole apply.
   - Fix A: zero the entries below 1e-16·max|A|, which changes A by 2e-17 in norm.
   - Fix B: use an absolute tolerance (eps·max|A|).
   - Either restores ε-accuracy for every m.
   - This is the exact analogue of what libsharp and Wavemoth do on purpose: skip values below
     1e-30 / "below accuracy" near the poles. [cited]
   - Lesson for the harness: **dial 4 should use a tolerance relative to the global operator, not to the
     tile.** E5 reached the same conclusion from a different direction (absolute tolerance was one of its
     partial fixes).
2. **Relative tolerance wastes rank in the evanescent region, but it is not where the storage goes.**
   - Tile maps (R4, n = 1024, middle level, `r4_turning_map.png`): evanescent tiles have relative-tolerance
     rank up to 26, above every oscillatory tile (max 16). Measured against ‖A‖, their rank is ≤ 6 (mean 0.4–0.9).
   - Zeroing below 1e-16 or below ε makes it worse: the cut creates a sharp edge inside tiny tiles, and max
     rank rises to 28–32 at n = 2048 and to 56 at n = 4096. Total storage still *drops* slightly.
   - So max rank is the wrong metric here: the inflated ranks sit in a few tiny tiles.
   - The absolute-tolerance ID gives the lowest storage at every m: 7% below raw at m = 0, 28% at m = 7n/8.
   - **Catch at m = 0:** with scale = max|A| = 31.6, which is set by the pole spike P̃_l(1) = √(l + ½), the
     error rises from 3.8e-11 to 8.8e-11 at ε = 1e-10, still within target. At ε = 1e-6 it rises from 4.6e-7
     to 1.5e-6, which misses the target by 1.5×.
     ε·‖A‖₂ or a per-column scale would be the principled choice. [computed]
3. **Turning points are benign.** Tile classes (n = 1024, middle level, ε = 1e-10, rank relative to ‖A‖):

   | m | oscillatory: count / max / mean | straddling: count / max / mean | evanescent: count / max / mean |
   |---|---|---|---|
   | 0 | 1024 / 14 / 12.7 | — | — |
   | 128 | 772 / 16 / 14.5 | 92 / 14 / 10.3 | 160 / 6 / 0.9 |
   | 256 | 652 / 16 / 14.5 | 86 / 15 / 10.7 | 286 / 7 / 0.7 |
   | 512 | 464 / 16 / 13.6 | 84 / 15 / 10.3 | 476 / 6 / 0.4 |

   At ε = 1e-6 the straddling mean is 7.4–8.0 and the oscillatory mean 9.9–10.1.
   - **Why.** Across the turning curve the WKB phase has a square-root fold, but the function itself is a
     smooth Airy profile whose local frequency drops to zero. A tile that straddles the fold is partly
     oscillatory and partly decaying, which leaves fewer effective oscillations. [conjecture, consistent
     with the data]
   - **Contrast with E4.** At the FIO caustic, the *butterfly* stayed fine and the *inverse problem*
     became ill-posed (cond 4.5e14). Here the fold is a turning point of a self-adjoint problem: A^(m) has
     orthonormal columns under the Gauss weights, and analysis is the transpose.
   - So there is **no caustic-type failure at all**, neither in compression nor in inversion.
   - The only "caustic-region" pathology is the floating-point one in item 1.
4. **Ranks against m.**
   - Max rank stays 16–19 at ε = 1e-10 for every m under raw and absolute tolerance.
   - Storage ratio *falls* with m (0.49 → 0.28), because the evanescent triangle becomes a growing share of
     the dense n(n − m).
   - Seljebotn (2012) reports average rank *growing* with m for his scheme. We see no growth in the maximum,
     but our metric differs: leaf size 1 against his blocked tiles, and max rather than average. [cited / computed]
5. **Ordering dial** (R3, n = 1024, ε = 1e-10, zero-below-1e-16 kernel):

   | kernel | θ-tree | x = cos θ tree | bit-reversed indices | random |
   |---|---|---|---|---|
   | Legendre m = 0 | 18 / 0.701 | 26 / 0.683 | 32 (full) / 1.033 | 32 / 1.092 |
   | Legendre m = n/4 | 17 / 0.625 | 18 / 0.648 | n/a (n − m not a power of 2) | 24 / 0.961 |
   | Chebyshev (DCT-II) | 18 / 0.701 | 26 / 0.682 | **4 / 0.129** | 32 / 1.092 |

   - **Chebyshev, bit-reversed: rank ≤ 4.** [proved]
     - Write a row box at depth s in bit-reversed order as j = j₀ + 2^s a, and a column box as
       l = k₀ + 2^{L−s} b.
     - Then e^{iπ l(j + ½)/n} = (const) · α(a) · β(b) · (−1)^{ab}.
     - The sign matrix [(−1)^{ab}] = 1 − 2·[a odd][b odd] has rank 2, and taking the real part of
       cos = Re e^{i·} doubles it: rank ≤ 4.
   - **Legendre, bit-reversed:** this identity fails, and every tile is full rank. Bit reversal is a
     property of exact characters, not of oscillatory kernels in general. That matches E1/E2 for the FIO
     with t > 0.
   - **x-tree:** about the same total storage, but much higher end-stage ranks, because the leaves near
     the poles are crowded. The natural choice is θ.

## (f) Literature: what is known and what is new

All entries were checked against abstracts or full text by a literature pass; unverified details are flagged.

- **Tygert, "Fast algorithms for spherical harmonic expansions, III"**, J. Comput. Phys. 229 (2010), arXiv:0910.5435.
  - Method: exactly this one, an ID-based butterfly in the O'Neil–Woolfe–Rokhlin style, with columns from
    the three-term recurrence and exponents tracked explicitly against underflow.
  - Precomputation: O(n²) per m.
  - Ranks: observed average ≈ 65–83 and maximum 110–379, roughly independent of m and n. Tygert notes that
    the compression is not proved.
  - No discussion of turning points. [cited]
- **Tygert, Part II**, JCP 227 (2008): a stable transform costing ~l² ln l. **Rokhlin & Tygert**, SISC 27
  (2006): O(N² log N log 1/ε). [cited]
- **O'Neil, Woolfe, Rokhlin**, ACHA 28 (2010): the butterfly for special-function transforms. Whether it
  treats P_l^m specifically is unverified. [cited]
- **Seljebotn, "Wavemoth"**, ApJS 199:5 (2012), arXiv:1110.4874. Butterfly SHT for the CMB.
  - Speed: 2× faster than libpsht at low and medium L, 3–6× at L ≈ 4000.
  - It drops values below 1e-30 by moving the start index in l, rather than compressing them.
  - The rank property for P_l^m is "not proven", and the average rank grows with m.
  - Precomputation is O(L³) and needs tens of GiB. [cited]
- **Reinecke & Seljebotn, libsharp**, A&A 554:A112 (2013): "polar optimisation" skips negligible
  terms near the poles, and uses an extended exponent in the recursion. **Fukushima**, J. Geod. 86 (2012):
  extended-exponent ("X-number") P_l^m for arbitrary degree. [cited]
- **Bremer, Chen & Yang**, arXiv:2004.11346: SHT via IDBF, O(N² log³ N) including precomputation.
  **Pang, Ho & Yang**, IDBF, arXiv:1809.10573. **Li & Yang**, IBF, SISC 39 (2017). [cited]
- **Other routes:**
  - Driscoll & Healy (1994) and Healy et al. (2003): divide-and-conquer FLT, needing stabilisation heuristics.
  - Hale & Townsend (2014): Chebyshev–Legendre via asymptotics.
  - Townsend, Webb & Olver (2018): Toeplitz/Hankel.
  - Slevinsky (2019): spherical harmonics ↔ bivariate Fourier via butterflied connection matrices.
  - All of these avoid the turning-point question altogether. [cited]

**What is *not* new:** that the SHT is butterfly-compressible, the method itself, the need for exponent
handling, and dropping the polar region.

**What is, as far as we found, new or at least not in these papers** (all small, all computed here):
1. A tile-resolved measurement that **turning-point tiles have lower rank than oscillatory tiles**. No
   paper we found states a rank result near turning points.
2. The **two-branch explanation** of the rank constant: Legendre ranks equal the Chebyshev ranks to three
   digits, and are about 2× the rank of a complex exponential tile.
3. The specific **failure mode of a per-tile relative ID on subnormal tiles (NaN)**, and that the
   absolute-tolerance fix beats clipping. Clipping lowers storage but raises max rank, through the edge it creates.
4. The bit-reversed DCT-II rank-4 identity, with proof.
   - Probably folklore, since the FFT is the bit-reversed rank-1 butterfly, but we found no citation.
   - Bit reversal does *not* carry over to Legendre.

**Comparison with Tygert's ranks.** Our maximum rank (18 at 1e-10) is not comparable with his (110–379).
He uses large leaf blocks and a different precision. Ours have one point per leaf, so "rank" means
skeletons per complementary tile.

## Claims

| # | claim | status |
|---|---|---|
| 1 | Gauss quadrature makes analysis = Aᵀ diag(w); no solve is needed | proved (polynomial degree ≤ 2n − 2) |
| 2 | The unmodified butterfly driver plus ≈ 39 lines of glue gives ε-accurate synthesis and analysis for every m, n ≤ 8192, once subnormal entries are zeroed | computed |
| 3 | Max tile rank is bounded in n: 12–13 at 1e-6 and 16–18 at 1e-10, for n = 256…8192 at m = 0 | computed |
| 4 | Storage nnz/dense: 0.117 (1e-6) and 0.202 (1e-10) at n = 8192; crossover ≈ 256 (1e-6) and 256–512 (1e-10) | computed |
| 5 | Wall-time crossover against BLAS around n ≈ 4096–8192 in this Python implementation | computed, noisy (shared machine) |
| 6 | Legendre ranks equal Chebyshev ranks (natural order); rank ≤ 2 × the rank of a single exponential tile | computed; the bound is proved (cos = sum of two exponentials) |
| 7 | Bit-reversed DCT-II tiles have rank ≤ 4; measured exactly 4 | proved + computed |
| 8 | Bit-reversed Legendre tiles are full rank; bit reversal does not help | computed |
| 9 | Straddling tiles have lower mean rank than oscillatory tiles (10.3–10.8 vs 13.6–14.9 at 1e-10) | computed |
| 10 | The reason for 9: Airy-type smoothness and falling local frequency at the fold | conjecture |
| 11 | A per-tile relative ID on subnormal tiles yields NaN (m ≥ 3n/4 at n = 2048; m = n/2 at n = 1024) | computed |
| 12 | Zeroing entries < 1e-16·max fixes 11 at an operator change of 2e-17 | computed |
| 13 | An absolute-tolerance ID gives the lowest storage (−7% to −28% vs raw), but with scale = max\|A\| it misses ε = 1e-6 at m = 0 by 1.5× (synthesis error 1.5e-6) | computed |
| 14 | Turning points produce no E4-type failure: no compression blow-up, and no ill-posed inverse | computed (compression); proved (orthogonality) |
| 15 | Butterfly SHT, its O(n²) precomputation, and polar skipping are known (Tygert 2010; Seljebotn 2012; Reinecke–Seljebotn 2013) | cited |
| 16 | No proxy/Green representation applies (the column variable is spectral) | conjecture |
| 17 | Plane-wave coefficients reproduced to 4e-11; full-sphere Parseval to 2e-14 | computed |

## Harness lessons (for the thesis)

- **"Kernel swap only" holds for the SHT,** given one coordinate choice (dial 1: θ, not x) and a transpose
  apply (dial 5).
- **One harness fix is suggested by this experiment:** dial 4's ID should accept an absolute or global
  tolerance. A tile-relative tolerance is (i) unsafe on subnormal data and (ii) wasteful on exponentially
  small tiles. E5 points the same way for the solver drivers.
- **One improvement is left untested:** a rectangular-aware driver. Today W is sized by len(x), so we pad.
