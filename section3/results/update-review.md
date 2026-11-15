# Post-audit review of E6–E12 and the ten-model showcase

## 0. Outcome

The update substantially strengthens the project as an **experimental compatibility map for dense integral-operator methods**. It does not strengthen the claim that one universal solver follows from a few independent dials. The most useful formulation remains:

> Choose the partition, interaction pattern, representation, sampling construction, and algebraic assembly; then certify approximation error, numerical stability, and total cost.

The new work is at its best when it distinguishes those certificates:

- E5 identifies a real coupling between representation and elimination stability.
- E6, E7, and E9 show cases where the fast solve agrees with dense while the discretization or quantity of interest is the limiting factor.
- E8 and E11 turn proxy failure into constructive sampling experiments.
- E10 separates tile rank from transform-level storage and wall-clock performance.
- E12 supplies a useful negative cost result and a clean functional-versus-density conditioning example.

The ten-model showcase is good breadth evidence: its quick reproduction passed all ten agreement checks. It is not evidence of a universal speedup. At the recorded sizes the fast path was slower than dense in five of the ten shipped timings, and the drivers still retain dense work arrays.

This document consolidates eight targeted audits. The supplied `source/` and `updated/` trees were not edited. Detailed reports and reproduction scripts are listed in §8.

## 1. Corrections needed before an external presentation

### 1.1 New or newly measured issues

1. **Butterfly storage:** replace the stale `$0.18\times$ dense at $N=4096$` claim. Direct driver runs give **$0.116\times$ at $\varepsilon=10^{-6}$** and **$0.150\times$ at $\varepsilon=10^{-8}$**.
2. **Test counts:** `263` is a collection count, not a pass count. The full updated-tree suite was not completed. Seven failures were reproduced locally: six missing-reference `test_butterfly_id` cases and one platform-dependent E10 assertion. Other updated-tree failures are not excluded.
3. **“Core unchanged”:** the core files are unchanged, but the Helmholtz showcase row uses the E5 driver fork and six rows change the sampling layer. Say exactly which interface changed rather than “no code changes.”
4. **E12 Oseen cost:** the $N=2048$ point does not generalize. At $N=4096$, directional admissibility reduces skeletons by $21.6\%$ but increases storage by $28.6\%$ and elimination flops by $67.3\%$. Keep the approximately threefold root reduction; drop any total-cost-win wording.
5. **E8 coverage:** “reaches $\varepsilon$ for every $s$” is false in 1-D. The lifted strong-admissibility errors include $5.4\times10^{-8}$ at $s=0.75$ and $2.7\times10^{-6}$ at $s=0.9$. The exact local-lift argument covers integer codimension—essentially $s=1/2$ for the families swept—not the full fractional sweep.
6. **E11 lift rule:** a finite integer-dimensional local lift in the stated construction requires $2\nu\in\mathbb Z$. A value such as $D=2.6$ is formal arithmetic, not a Euclidean space in which a local Green's-function proof has been supplied. Fractional cases must be labelled numerical truncations/evidence.
7. **E10 SHT:** fix the wrong-sign Gauss-node Newton step and the test that requires a platform-dependent NaN. Report per-order and all-order costs separately; the per-order storage ratios do not compose into a transform-level win at the tested sizes.
8. **E6 Stokes:** do not present `r7_suspension.png` as physical settling: every $y$-velocity opposes the applied downward load because the nonzero-net-force 2-D log sector is convention-dependent. Quote the rank result as approximately $2\times$ per box, $1\times$ per dof, and $4\times$ storage at equal geometry.
9. **Flop labels:** the E7/E12 `factor_cost` figures count elimination arithmetic only. Measured compressor work in E7 was $17$–$63$ times larger than the plotted elimination count. Proxy construction and selection costs also need explicit treatment.
10. **Stacked-ID claims:** compare SVD ranks with SVD ranks and QR skeleton counts with QR skeleton counts. There is no supported universal “exactly $2\times$” or `max rank + 1` law. The safe general bound is $r_{AB}\le r_A+r_B$; overlap and actual storage must be measured.

### 1.2 Older corrections still absent from the update

The update incorporates the pivot-stability and prior-work discussions, but several factual corrections from the first review remain outside the new experiment files:

- `updated/md/wideband_fmm.md` still uses $7^d-3^d$ for the standard interaction list; it should be $6^d-3^d$ ($27$ in 2-D, $189$ in 3-D).
- The FMM-LU cost-table derivation still contains inconsistent active/skeleton counts and the unsupported $N^3/16$ row. It needs replacement, not a single-number patch.
- The butterfly notes still infer a general $O(\log(1/\varepsilon))$ rank result from one mixed-derivative estimate. State the stronger joint-regularity assumptions and the cited theorem's actual conclusion.
- Caustics, phase regularity, complementary low rank, and inverse conditioning are still conflated in older prose. E10 now gives a direct counterexample to a blanket “caustics raise tile rank” narrative: turning-point-straddling tiles had lower mean rank.

## 2. Verification performed and its limits

| Check | Result |
|---|---|
| Ten-model showcase, quick no-write reproduction | **10/10 agreement checks passed**, approximately 239 s in the audit environment |
| Per-directory tests reported by the experiment audits | Showcase 24/24; E6 6/6; E7 15/15; E8 9/9; E9 12/12; E11 20/20; E12 10/10; E10 8/9 |
| DFT generic-ID butterfly, independent rerun | $N=256,512,1024,2048$: every tile rank one; `nnz` $=2N\log_2N+2N$ exactly; apply error $2.6\times10^{-14}$ to $2.0\times10^{-13}$ |
| E12 rank table, independent rerun | Recorded relative-rank and stacked-QR tables reproduced at $\lambda=0,5,80,320$ |
| E1 saved storage ratios | DFT strong/complementary $=2.45$, weak/complementary $=4.59$ at $N=4096$ reproduced |
| E12 $N=4096$, $\lambda=320$, independent dense-reference run | Strong: 7007 skeletons, root 485, storage 3,471,408, elimination flops $1.6979\times10^9$. Directional: 5490, 163, 4,463,489, $2.8406\times10^9$ |
| Complex congruence examples | `python3 results/examples.py` passes; 2,000 random complex cases gave minimum transpose Hermitian eigenvalue $-15.540$ and adjoint minimum $1.000$ |
| E11 saved-table audit | 81 comparisons made; 74 met the script's strict numerical tolerance and the remaining seven were documented one-significant-digit printing differences. Additional controls tested circle refinement, radius, kernel-order count, and underflow |

The original source-tree suite had **119 passed, 7 failed**, all seven caused by the absent `source/code/butterfly.py`. The updated full suite exceeded a 2400 s limit, so no complete updated-tree pass count is claimed. The updated butterfly benchmark still fails on the same missing reference dependency. Production-size runs, original-laptop wall-clock claims, and every item of cited literature were not independently exhaustively checked.

## 3. Experiment-by-experiment verdicts

### E6 — Stokes

**Supported.** With exact sampling, Stokes uses the unchanged core/driver and reaches errors at most about $4\times10^{-11}$ at $N_{\rm dof}=8192$. The block sampler makes proxy compression possible. The Stokes/Laplace per-box skeleton ratio is consistently about two, while the ratio per dof is about one and equal-geometry factorization storage is about four times larger. The projector and degenerate-scale observations are useful examples of non-kernel low-rank content being carried by explicit rows.

**Corrections.**

- Suspension conditioning ranges from about $20$ to more than $200$ in the tested multi-body cases, not $4$–$40$; one circle has condition number two.
- The near-contact experiment supports “quadrature, not compression, is the bottleneck,” but its global-normalized `0.27` is not a componentwise relative error. At gap $0.005$ and $n=128$, the angular-velocity discrepancy against the finer reference is about $198\%$, while translational velocity is much more accurate.
- The claimed robust $10\times$ double-layer advantage at equal rank was not established; the $Q$-superset was also wrongly blamed for a $47\%$ root increase that is proxy overhead.
- The suspension figure's nonzero-net-force sector is not a convention-free physical mobility. Use a zero-net-force dipole or impose and state an outer scale.

**Presentation value.** Lead with the unchanged-interface result and the three rank/cost ratios together. The near-contact example is the clearest “the fast solver is not the physics bottleneck” exhibit.

### E7 — Yukawa and Rothe heat

**Supported.** The screened-kernel normalization, history-kernel recurrences, and Rothe identity were checked independently. On the resolved branch ($\kappa\gtrsim10$ and $\kappa h\lesssim3$), the measured solve-plus-near-evaluation error is approximately $0.017\kappa^2h^3$; $0.02$ is a reasonable conservative fit there, not a universal bound. A $\kappa\to0$ control recovers spectral convergence. Absolute-tolerance compression avoids the relative-tolerance rank inflation and subnormal failure regime.

The boundary-only Rothe statement is scoped to **zero initial data and no interior source**. Fast and dense agree closely at fixed time step; convergence to continuous-in-time data is a separate experiment and should not be conflated with the semidiscrete exactness check.

**Corrections.**

- Use $\kappa(A)\lesssim5.2$ over the sweep, not $3.5$.
- The truncation speedup is size-dependent: measured gains at $N=1024$ were about $1.2$–$1.75\times$, not a universal $10\times$.
- The spatial accuracy law implies $h\propto(\eta/0.017)^{1/3}\,\Delta t^{1/3}$ at target accuracy $\eta$, not a general $h\lesssim\sqrt{\Delta t}$ rule.
- Exact diagonal panel integrals did not fix the $h^3$ defect; a banded kernel-split quadrature is the appropriate follow-up.
- Label all plotted flops **elimination-only** and include ID/QR/build work before drawing end-to-end cost conclusions.

**Presentation value.** The $\kappa^2h^3$ plot with the $\kappa\to0$ control, and the relative-versus-absolute tolerance failure, are more informative than another scaling table.

### E8 — Fractional/Riesz kernels

**Supported.** Flat-circle proxy failure is structural rather than a point-count issue. A hemisphere with several latitudes can work well, and the equal-rank experiment shows that the proxy rank premium buys required representation accuracy rather than pure waste.

The new per-box control is the key mechanism result:

- Increasing circle points from 64 to 256 left errors essentially unchanged.
- One off-plane ring improved the error by at most about a factor four and could be worse at high latitude.
- Several latitudes reduced the error by orders of magnitude.
- An in-plane annulus also reached the target regime.

Thus “fill a sufficiently rich two-dimensional sample region” is a more accurate conclusion than “lifting one dimension is necessary.” The hemisphere is a cheap selection-free construction, not the unique fix. Enlarging proxy radius is not a clean control because the core then adds interior far points explicitly, silently moving content into the exact part.

**Corrections.** Restrict the “every $s$” headline as described in §1. Do not claim weak-admissibility 1-D compressor orderings where every method is conditioning-limited. The $N=8192$ 1-D weak error outlier deserves a rerun before any predictive $\kappa(A)\varepsilon$ law is quoted. Correct the saved-table transcription from rank 95 to 105 where applicable. Treat the exact compressor as an oracle rank reference, not an implementable competitor.

**Presentation value.** Use the latitude/point-count control as the figure. It explains the mechanism better than an eight-panel sweep.

### E9 — Lippmann–Schwinger

**Supported.** The singular self-term is the exact equal-area-disk integral, dense results match two independent references, and the fast solve adds little to discretization error. The naive proxy floor reproduces: evaluating the contrast on the wrong side of the two-sided kernel breaks row relations, while the correct two-sided construction converges with tolerance.

**Corrections.**

- The claimed $0.02$–$0.6\,\kappa(A)\varepsilon$ error band is not in the saved data; retain conditioning as a bound, not a fitted law.
- Strong-admissibility rank growth in the recorded curve is partly proxy inflation. The “only four tree levels” explanation for weaker E5 severity is false; E5 has four levels too.
- The maximum $1.85\times$ stacked/one-sided ratio is primarily a **smoothness penalty**. Constant-contrast boxes away from interfaces gave $1.00$, interface boxes about $1.20$, and a smooth bump about $1.67$ mean, with $1.85$ as the worst case.
- E5 severity is regime-dependent. In the added strong-admissibility $k=80$, $\varepsilon=10^{-6}$ run, the exact/realified error ratio was $135\times$, not at most $13\times$. High-contrast realified pivots can also be indefinite; do not recommend realification unconditionally.
- The staircase error is under-resolution that decreases with refinement, not a compression failure.

**Presentation value.** Show the naive-proxy floor next to the correct two-sided curve and report discretization error beside fast-versus-dense error.

### E10 — Spherical harmonic butterfly

**Supported.** Per-order associated-Legendre butterflies reproduce the recorded ranks and accuracy over the stated coverage. Gauss analysis is a transpose application, not an inverse solve. Turning-point-straddling tiles have lower rather than higher mean rank; their rank tracks the nonseparable mixed-phase variation. The bit-reversed DCT-II rank-four result is sound.

**Corrections.**

- The Newton derivative sign is wrong. On current SciPy nodes the existing polishing step worsens orthonormality by about fourfold; it should be fixed or removed and the old justification withdrawn.
- Subnormal failure is tile-, platform-, and library-dependent. The test must assert the guard/accuracy property, not require a NaN at one $(n,m)$ point.
- Coverage is $m=0$ through $n=8192$ and the order sweep through $n=4096$, not every order through $n=8192$.
- Per-order ratios of $0.117$ and $0.202$ of dense at $n=8192$ are real, but summing the recorded sweep at $n=2048$ gives about $0.30$ and $0.44$ of dense storage at $\varepsilon=10^{-6}$ and $10^{-10}$—roughly 20 and 30 GB. The $n=8192$ all-order estimate is about 0.8 TB and many hours of build time; label that estimate as extrapolation.
- Use the arithmetic crossover near $n\approx300$ rather than a load-sensitive wall-clock crossover claim.

**Presentation value.** The turning map plus mixed-phase explanation is the strongest figure. Do not claim superiority to production SHT implementations from these timings.

### E11 — Matérn kernels

**Supported.** The stock circle's failure is structural: refining from 64 through 512 points leaves the far-field errors at the same roughly $10^{-2}$ scale for the tested $\nu$ values. Multi-order rows fix the representation issue. Integer $\nu=2,3$ controls support the terminate-versus-truncate distinction, and the saved GP tables show why compression error, conditioning, posterior means, and likelihood errors must be reported separately.

For $f_\mu(r)=(\kappa r)^\mu K_\mu(\kappa r)$,

$$
(\kappa^2-\Delta_D)f_\mu=\kappa^2(2\mu+D-2)f_{\mu-1},\qquad r>0.
$$

With $\beta=\lceil\nu+1\rceil$ and $D=2(\beta-\nu)$, the finite integer-dimensional local-lift construction applies when $2\nu\in\mathbb Z$. This normalized form matters: using unscaled $r^\mu K_\mu(\kappa r)$ changes the displayed factor from $\kappa^2$ to $\kappa$.

**Corrections and new practical conclusion.**

- Do not call fractional $D$ a proved local Euclidean lift. Keep that case empirical.
- At $N=2048$, a plain ring with one extra kernel order matched the lifted proxy at similar skeleton counts and **$1.6$–$2.4\times$ fewer ID rows** in the tested $\nu=0.5,1.5$ cases. Accordingly, the current data support the multi-order idea, not a claim that the lift is always necessary or cheapest.
- Cost must include more than skeleton count. At $N=8192$, lift skeletons were about $1.5\times$ exact while ID flops were $0.57$–$0.76\times$ exact. Also, the four-latitude configuration charges about $1.9\times$ the ID rows of the one-latitude configuration that the resolution study found sufficient.
- At the shortest length scale tested, underflow/relative tolerance made the proxy keep nearly full rank. Report rank in that sweep and use an absolute or globally scaled tolerance guard.
- The equal-max scaling of stacked kernel-order blocks is a free convention that affects QR stopping and must be disclosed.

**Presentation value.** Combine the circle-refinement control, an error-versus-ID-work frontier, and the short Bessel-chain identity. Avoid a large raw table of conditioning-limited solve errors.

### E12 — Oseen

**Supported.** The scalar kernel satisfies

$$
G_\lambda(x)=e^{\lambda x_1}\frac{K_0(\lambda|x|)}{2\pi},
\qquad (-\Delta+2\lambda\partial_1)G_\lambda=\delta,
$$

so every block is a diagonal scaling of a Yukawa block. This proves invariance of **exact algebraic rank** under the corresponding nonsingular scalings; diagonal scaling alone does not prove invariance of a spectral $\varepsilon$-rank. The measured relative ranks and the stronger directionality of downstream-scaled ranks should therefore be reported as measurements, with their tolerance conventions.

The cylinder quadrature converges at third order against an analytic single-layer control. The normal density is an exact continuum null direction in the checked formulation. The drag deviation from Lamb's formula scales as $(ka)^2\log(1/ka)$ with measured coefficient about $-0.51$; this coefficient was measured, not checked against the full published expansion.

**Corrections.**

- The independently reproduced deeper-tree cost table is:

| $N$, depth | Skeleton change | Storage ratio | Elimination-flop ratio | Root |
|---|---:|---:|---:|---:|
| 2048, depth 3 | $-42.9\%$ | $1.062\times$ | $1.126\times$ | $444\to151$ |
| 4096, depth 4 | $-21.6\%$ | $1.286\times$ | $1.673\times$ | $485\to163$ |

  The robust positive statement is a roughly threefold root reduction at high $\lambda$, not a total-cost win. A tested level-dependent hybrid did not recover the depth-three cost.
- The scalar area matrix is close to the identity ($\kappa_2(A)\approx1.004$ at $\lambda=80$). The dramatic diagonal-similarity failure measures the coordinate change, not intrinsic Oseen ill-conditioning.
- SVD and QR rank conventions explain part of the supposed stacked-rank doubling. At low/moderate $\lambda$ like-for-like stacked/single ratios are around $1.7$ in the examples; at high along-stream $\lambda$ there is no doubling.
- “Fast matches dense” is true for drag, not raw density. Raw density errors reach $2.14$, while the drag functional's condition number is only about $1$–$2.3$ and the matrix condition number grows like $n^3$. Report both errors and name the quantity of interest.
- The three-point proxy ladder is consistent with $n_{\rm proxy}\sim\sqrt{\lambda r}$ but does not identify that exponent precisely.

**Presentation value.** Lead with directional rank structure and the cross-stream control, then show the honest cost table. The drag plot should include the raw-density caveat.

## 4. Cross-cutting conclusions

### 4.1 The five dials are useful only with three certificates

| Certificate | Minimum evidence | Recurrent failure prevented |
|---|---|---|
| Approximation | Achieved error against an independent reference, at stated norm and tolerance | Assuming low nominal rank implies an accurate solve |
| Stability | Conditioning, residual/backward error, pivot health, and error in the actual quantity of interest | E5 pivot loss; E12 accurate drag but unresolved density |
| Total cost | Stored entries, construction/ID/QR work, near field, root, apply/solve work, and wall clock under controlled conditions | E6 “$2\times$” becoming $4\times$ storage; E12 fewer skeletons at higher cost; E10 per-order versus all-order confusion |

### 4.2 Recurring design lessons

1. **Relative tolerance needs a global/absolute floor.** E7, E10, and E11 all encounter rank inflation or non-finite behavior when entire tiles become tiny or subnormal. A guarded absolute/global tolerance option is a framework-level fix, not three unrelated hacks.
2. **Proxy radius changes more than resolution.** If points inside the proxy surface are made explicit, a radius sweep simultaneously changes the exact/compressed split. Report the non-explicit far-block error and the number of explicit points.
3. **Exact ID is an oracle.** It measures intrinsic block compressibility but reads the true far block. It is not a fair practical algorithm to compare against proxy build cost without saying so.
4. **One shared ID is not always the same as two one-sided IDs.** E6, E9, and E12 motivate measuring the overlap term and testing separate row/column bases. No fixed penalty law has been established.
5. **Dense agreement is not physical validation.** E6 near contact, E7 boundary quadrature, and E9 volume discretization all admit a highly accurate solve of an inaccurate discrete problem.
6. **A functional can be accurate when the density is not.** E12 makes this quantitative with matrix versus functional conditioning. E11 likewise shows why posterior mean and likelihood belong beside compression error.
7. **Negative results are part of the map.** The honest E12 cost result and the E8/E11 proxy failures are more useful than forcing every row into a success story.

## 5. Recommended presentation

### 5.1 Thesis

A defensible one-sentence thesis is:

> Dense integral-operator multiplication, transforms, and direct solves can be compared in one hierarchical low-rank language; successful choices are governed by kernel–geometry–sampling compatibility, and every choice requires separate approximation, stability, and total-cost certificates.

The contribution is the measured compatibility table, the failure mechanisms and controls, and the small shared interfaces—not derivation of every published algorithm from one unchanged program.

### 5.2 A 17-slide core talk, with details in an appendix

| Act | Slides | Content |
|---|---:|---|
| Why | 1–3 | Three dense tasks; the log-kernel separable example; numerical rank and why rank is not cost or stability |
| Map | 4–6 | Five dials; three certificates; compatibility table shown before the experiments |
| Decisions | 7–11 | E1 geometry; E2 exact DFT count; E3 proxy/PDE compatibility; E5 transpose versus adjoint; E4 far-block butterfly and preconditioning, with scope stated |
| Breadth and falsification | 12–15 | Ten-model showcase with timing caveat; E8/E11 sampling controls; E12 honest negative; E10 turning-point result |
| Claims and next steps | 16–17 | Known versus measured versus plausibly new; the two next experiments that would change a design decision |

Move paper-by-paper derivations, full sweeps, raw tables, and additional E6–E12 setup to the appendix. The current deck's compatibility table should be revisited after each experiment rather than displayed once and abandoned.

### 5.3 Figures and numbers worth keeping

- **Exact DFT representation:** $2N\log_2N+2N$ entries and rank-one tiles, checked at four sizes.
- **Stokes rank/cost:** approximately $2\times$ per box, $1\times$ per dof, $4\times$ storage—show all three or none.
- **Butterfly storage:** $0.116\times$ and $0.150\times$ dense at $N=4096$ for the two stated tolerances.
- **Proxy mechanism:** the E8 latitude control or the E11 64-to-512-point circle control, not a dense parameter sweep.
- **Stability:** the complex congruence counterexample and the measured E5 repair, with the E9 regime-dependence caveat.
- **Oseen honesty:** the two-depth cost table and approximately threefold root reduction.
- **Showcase breadth:** ten agreement checks pass, paired with the recorded timing reality and the dense-work-array caveat.

Every figure caption should state $N$, $\varepsilon$, admissibility, compressor, norm, whether the error is forward/residual/discretization, and whether cost includes construction. Several dozen PNGs exist, but the Markdown deck currently does not embed them; assets alone do not make a presentation-ready argument.

### 5.4 Claim language

Use three labels consistently:

- **Proved here:** an algebraic identity or a calculation with hypotheses displayed, such as the DFT residue-class rank-one factorization or the adjoint congruence identity.
- **Measured here:** a finite-size result with its norm, implementation, and parameter range attached.
- **Plausible/not found in our search:** a novelty hypothesis, never a proof of prior-art absence.

The ingredients of the proxy lifts, equivalent-source constructions, butterfly SHT, and modular hierarchical formats have substantial prior art. The potentially distinctive contribution is the particular transfer, quantitative compatibility measurements, and failure controls. No exhaustive novelty determination was made in this audit.

## 6. Research and engineering priorities

The ranking below separates work that clarifies/simplifies the framework from work that could yield a genuinely new result. Effort estimates are approximate and assume the existing harness.

| Priority | Project | Effort | Value | Type and expected payoff |
|---:|---|---|---|---|
| 1 | **Underflow-safe absolute/global-tolerance ID** | Small: days | High | Engineering plus robustness. One implementation addresses E7, E10, and E11. Re-run rank and error tables, not just pass/fail tests. |
| 2 | **Price the compatibility table and add claim provenance** | Medium: 1–3 weeks | High | Calibration/infrastructure. At common achieved error, record storage, ID/QR, proxy selection, near field, root, apply/solve, and wall time. Link every displayed number to a JSON key. This directly prevents most misleading headlines found here. |
| 3 | **A proxy deficiency/design law** | Small–medium: days to 2 weeks | High | Best candidate for an easy, interesting quantitative result. Measure projection/deficiency spectra versus latitude count, kernel order, $\kappa h$, and $\varepsilon$ in E8/E11. Aim to predict required samples/orders rather than add more isolated sweeps. |
| 4 | **One ID or two?** | Medium: 1–2 weeks | High | New measurement plus a possible driver improvement. Compare stacked and separate row/column IDs across E6/E9/E12 at equal achieved error. Measure overlap, stored entries, interpolation work, and pivot stability. |
| 5 | **Reproduce one published table exactly** | Small: days once the reference is restored | High credibility | Calibration, not new theory. Restore or replace the missing butterfly SVD reference and pin versions/BLAS settings. This is the shortest route to external reproducibility. |
| 6 | **Parameter-robust quadrature for the three bottleneck cases** | Medium | High capability | Apply known kernel-split/close-evaluation/boundary-aware ideas to E6 near contact, E7 screened layers, and E9 high-contrast volume scattering. The solver evidence says quadrature is where accuracy is currently lost. |
| 7 | **Row-sampled SHT construction and analytic barrier truncation** | Medium | High for E10 | Addresses the all-order build/storage wall instead of reporting another per-order ratio. Compare against a production transform at equal accuracy. |
| 8 | **A well-posed completed Oseen BIE, then two tandem bodies** | Medium–high: 1–3 weeks | High | New physics capability. First fix the singular first-kind density problem, then test whether directional admissibility pays for wake shielding where it should matter physically. |
| 9 | **Compress the Rothe history** | Medium | High if successful | The current history uses dense $N\times N$ blocks per step. Exploit the kernel-order recursion or a transform representation to avoid the emerging $O(n_{\rm steps}N^2)$ march. |
| 10 | **A fractional-lift approximation theorem** | High: weeks | High risk/high theory value | Close the gap between integer-codimension proofs and fractional numerical evidence, or provide a clear counterexample. Do not start by claiming a fractional-dimensional local PDE. |
| 11 | **An ALF mixed-phase rank bound or frequency-uniform pivot theorem** | High: months | Highest theory value, highest risk | These are genuine numerical-analysis projects, not quick harness extensions. The E10 mixed-phase measurements and E5/E9 pivot data supply targets, not proofs. |

**Do next:** (1), then (2) and (3) together. The first removes a common failure class; the second makes cost claims auditable; the third has the best chance of producing a small, useful design rule quickly.

**Not worth prioritizing:** more parameter values without a new control, larger-$N$ wall-clock studies while dense work arrays remain, additional proxy-radius sweeps that change the explicit/far split, or stronger novelty language based only on another negative search.

## 7. What incorporation changed in the review set

- `results/framework-explained.md` now includes the exact DFT storage certificate, the E8/E11 proxy-sampling lesson with the normalized Bessel identity, and the general complex-congruence check.
- `results/examples.py` now checks 2,000 random complex congruence instances in addition to the original scalar counterexample; `results/example-output.txt` records the passing output.
- `results/framework-review.md` links to this post-update verdict and updates the high-wavenumber research assessment without erasing the original audit.
- `results/README.md` is the entry point and states the updated test/reproduction limits.

## 8. Detailed evidence

The full audit reports and their scripts are retained at:

- E6: `.ultramath/smwh5ojey/m1/report.md`, with `checks/`.
- E7: `.ultramath/smwh5ojey/m2/report.md`, with `s1`–`s9` scripts and JSON.
- E8: `.ultramath/smwh5ojey/m3/report.md`, with per-box controls and `evidence.md`.
- E9: `.ultramath/smwh5ojey/m4/report.md`, with `repro/`.
- E10: `.ultramath/smwh5ojey/m5/audit.md`, with node, rank, and transform-cost checks.
- E11: `.ultramath/smwh5ojey/m6/report.md`, with table/theory checks and additional runs.
- E12: `.ultramath/smwh5ojey/m7/report.md`, with physics, conditioning, and cost scripts.
- Cross-document synthesis: `.ultramath/smwh5ojey/m8/report.md`, with `checks/`.

The coordinator's rerunnable checks, saved DFT/E1/E12 output, and independent E12 deeper-tree JSON are preserved in [verification/](verification/README.md). These scripts are evidence, not a replacement for a cleaned, one-command project reproduction target.
