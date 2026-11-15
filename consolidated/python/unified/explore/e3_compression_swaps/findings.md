# E3 — Swapping the compression engine of recursive skeletonization

**The swap.** `rskel_factor` (weak = HBS, strong = FMM-LU) calls one compressor per box: `comp(box, far, node) -> (skel, red, T)`.
We compare three compressors, all at the same ε = 1e-8, max_leaf 32, N = 2048 unless stated otherwise:

- **Exact**: an interpolative decomposition (ID) of the true far block `[A_FB; A_BFᵀ]`.
- **Proxy**: 64 points on a circle of radius 2.5·(box half-width), plus explicit rows for far points that fall inside that circle.
- **Randomized** (new; `compressors.py`): sketch the far block with Gaussian vectors, Y = A_BF G₂ + A_FBᵀ G₁. This needs p restricted products with A and p with Aᵀ. Then one ID of the p×|B| sketch. p doubles until rank ≤ p − 10.

We also tried **Proxy2**: two proxy rings, with n_proxy ≥ 1.5·kR + 16. It is the cheapest fix for the proxy failure found in E2.

**Metrics.**
- **err**: ‖x − x_dense‖/‖x_dense‖ for one random right-hand side. The residual ‖Ax−b‖/‖b‖ agrees with err to 2 digits in every run.
- **Skeleton sizes** per level.
- **Compression flops** (cost model in `compressors.py`), reported in units of one dense matvec (2N²).
- **cond(X_RR)**: the worst condition number over the eliminated pivot blocks.

**Hypotheses, stated before running.**
- H1: Proxy needs the kernel to satisfy a PDE outside the box (Green's representation).
- H2: Randomized is kernel-agnostic but costs more than proxy.
- H3: The single-layer proxy fails near interior Dirichlet eigenvalues of the proxy circle.

Reproduce: `.venv/bin/python -m unified.explore.e3_compression_swaps.run` (≈ 4 min on a laptop; `--quick`, `--only e1,..`).

## Results

| Input (N = 2048) | Rule | Exact err / cost | Proxy err / cost | Randomized err / cost |
|---|---|---|---|---|
| P1 Laplace curve | weak | 1.8e-10 / 71 | 9.5e-11 / **9.6** | 6.3e-10 / 59 |
| P2 Laplace area | strong | 7.6e-12 / 68 | 3.1e-13 / **26** | 1.8e-11 / 53 |
| P3 Helmholtz H0, k = 20 | strong | 3.5e-11 / 50 | 2.2e-11 / **5.0** | 1.2e-10 / 42 |
| Laplace double layer (nonsymmetric) | strong | 1.1e-10 / 45 | 4.4e-12 / **5.2** | 3.7e-10 / 37 |
| Advection–diffusion Green's fn, b = 20 (nonsymmetric) | strong | 2.4e-12 / 130 | 4.0e-14 / **46** | 4.7e-11 / 96 |
| Gaussian RBF σ = 0.2 (no PDE) | strong | 1.5e-11 / 113 | **4.3e-5** / 22 | 2.1e-10 / 89 |
| Cauchy 1/(1+r²/c²), c = 0.2 (no PDE) | strong | 1.5e-11 / 133 | **2.8e-6** / 38 | 4.1e-10 / 96 |
| skew-Cauchy (nonsymmetric, no PDE) | strong | 1.6e-11 / 141 | **2.8e-7** / 45 | 1.7e-10 / 100 |
| P3 H0, k = 80 | weak | **5.7e-5** / 110 | 5.3e-11 / 23 | **3.2e-6** / 100 |
| Y0 area kernel at k·R₂ = y₀,₁ | strong | — | **6.0e-4** / 28 | 2.1e-11 / 55 |

Cost is compression flops in dense-matvec units. Full data: `results/e1..e6*.json`. Figures: `e2_resonance.png`, `e3_ksweep.png`, `e4_nonpde.png`, `e5_cost.png`.

**Proxy fails on non-PDE kernels (H1 confirmed, E1/E4).**
- Under **strong** admissibility, Proxy misses ε for every Gaussian width σ ∈ [0.05, 1] and every Cauchy width c ∈ [0.05, 1]: errors range from 1.7e-7 to 4.3e-5, i.e. 17× to 4300× ε.
- Under **weak** admissibility the damage is mild: 1e-10 to 4e-8. Every touching neighbour lies inside the proxy circle and is kept as explicit rows. Under strong, no far point lies inside the circle (far boxes start at 3 half-widths, the circle is at 2.5), so the proxy carries the whole far field.
- Randomized and Exact stay at ≤ 5e-10 on these kernels.
- All PDE kernels are fine with Proxy, including the two nonsymmetric ones (double layer and advection–diffusion). What matters is a PDE outside the box, not symmetry.

**Proxy failure at resonances (H3 refined, E2/E3/E6).**
- The failure is set by zeros of the *proxy kernel's own* radial factor, not by Dirichlet eigenvalues as such. By Graf's addition theorem, the proxy rows carry mode m with weight Z_m(kR), where Z is the radial function of the kernel family.
  - **H0^(1) (the SPEC P3 kernel) cannot fail**, because H_m^(1) has no real zeros. Measured: at the J₀(kR_l) = 0 wavenumbers k = 3.09, 6.18, 12.35, every compressor reaches ≤ 6.4e-10.
  - **Real kernels Y0 and J0 do fail.** Under strong admissibility, with k·R = y₀,₁ or j₀,₁ exactly, the error is 6e-4 (Y0, level 2), 8e-5 (Y0, level 3), 6e-4 (J0, level 2) and 2e-4 (J0, level 3). Randomized reaches ≤ 2e-11 on the same inputs.
- The failure band is extremely narrow. For Y0, err ≈ 2e-22/δ², where δ is the relative detuning of k from the zero: it falls 100× per decade of δ, from 2.2e-6 at δ = 1e-8 to 2.2e-12 at δ = 1e-5. It exceeds ε only for δ ≲ 1.5e-7. A wavenumber chosen at random almost never hits the band, but hitting it exactly is catastrophic.
- Two proxy rings remove the failure: 2e-13 for Y0 and 3e-10 for J0, at the cost of 25–30% more proxy flops.
- Weak admissibility shows no resonance failure. On the curve, Y0 at its zeros does not fail even under strong (E6, ≤ 4.8e-10). Conjecture: points on a curve make the missing Bessel mode linearly dependent on the other modes.

**Exact and Randomized break down at high k on P3; Proxy does not (E3/E6). This was unexpected.**
- This is not a truncation error:
  - cond(A) ≈ 1.05.
  - With Exact, the worst eliminated block X_RR has cond ≈ 1e2 at k ≤ 10 and grows exponentially, to 1e10–1e11 at k = 90–100.
  - The error crosses ε at k ≈ 60–70 and reaches 6e-5 at k ≥ 80.
  - At k = 80, *tightening* ε makes Exact worse: ε = 1e-6/1e-8/1e-12 gives 2e-7/6e-5/2.6e-4. The same ε values give Proxy 6e-9/5e-11/1e-14.
- With Proxy, cond(X_RR) stays ≤ 150 at every k. Proxy keeps about 6% more skeletons (3088 vs 2916 at k = 80).
- Randomized tracks Exact. Both see the same far-field information, so this is not a sketching artefact.
- Conjecture: an ID against the true far field marks "nearly non-radiating" source combinations as redundant. Restricting the indefinite operator I + S_k w to that subspace has near-zero pivots. The proxy circle, at 2.5 half-widths, demands a slightly richer skeleton that avoids them.

**Cost (H2 partly confirmed, E5).**
- Proxy is the cheapest everywhere, 3–13× fewer flops than Exact, and its advantage grows with N:
  - P2 strong at N = 1024 → 4096: Exact 69 → 99, Proxy 29 → 18, Randomized 51 → 75 dense-matvec units.
- In this dense-access model Randomized costs 0.73–0.97× Exact. It saves only where the active set |B| is much larger than the rank. The ID cost is O(p|B|²) instead of O(|F||B|²), but the sketch itself still touches every far entry.
- As a black-box matvec method, per-box restricted sketches need 2Σp ≈ 3.6N vector products at N = 4096. That is not competitive. Real matvec-only constructions share random vectors across boxes: peeling (Lin–Lu–Ying 2011; Martinsson 2011) or RSRS (Yesypenko–Martinsson 2023).
- Randomized accuracy: 3–10× worse than Exact at the same ε. Two runs land just above ε (2.2e-8 on RBF, weak, N = 1024; 1.6e-8 on P3, k = 60), so ε should be set with about one digit of safety.

**Where each compressor wins.**
- **Proxy** wins on any PDE kernel (symmetric or not, Laplace / Helmholtz H0 / advection–diffusion), on cost, and on high-k stability.
- **Randomized (or Exact)** wins on non-PDE kernels under strong admissibility, by 10³–10⁵ in error, and on real-valued wave kernels sitting exactly at a proxy zero.
- **Randomized never beats Proxy on cost** in our setting.

## Claims

| # | Claim | Status |
|---|---|---|
| 1 | Proxy error under strong admissibility on Gaussian / Cauchy kernels is 1.7e-7 to 4.3e-5 (ε = 1e-8); Randomized ≤ 5e-10 | computed |
| 2 | Single-ring proxy for H0^(1) cannot lose a mode at any real k, because H_m^(1)(x) ≠ 0 for x > 0 | proved (Graf's addition theorem), computed |
| 3 | Y0/J0 proxies lose mode m when Y_m(kR) = 0 / J_m(kR) = 0; error ≈ 2e-22/δ² (fails ε only for δ ≲ 1.5e-7); two rings fix it | computed (mechanism: Graf, cited [K]) |
| 4 | Exact/Randomized RS on P3: cond(X_RR) grows exponentially in k, and error does not improve with ε; Proxy stays at cond ≤ 150 | computed |
| 5 | Mechanism of 4: near-non-radiating redundant subspace plus an indefinite operator | conjecture |
| 6 | Curve geometry masks the Y0 proxy-zero failure | computed (E6); mechanism conjecture |
| 7 | Randomized compression flops ≈ 0.73–0.97× Exact; Proxy 3–13× cheaper than Exact | computed (flop model) |
| 8 | The proxy argument requires the Green's representation, not symmetry | computed (double layer, advection–diffusion); cited: Ho–Greengard 2012, FMM-LU (research/02_fmm_lu.md), Xing–Chow 2020 |
| 9 | FMM-LU avoids resonances with a combined-field proxy; RSRS is "robust to indefiniteness" | cited (research/02_fmm_lu.md; Yesypenko–Martinsson 2023 via research/sota/11) |

**Next steps.**
- Test whether a hybrid ID (Exact rows plus proxy rows) cures the high-k pivot growth at Exact's skeleton size, i.e. whether the cure comes from *where* the far field is sampled or from *how many* skeletons are kept.
- Try a pivot-aware ID: reject skeletons that make X_RR ill-conditioned.
- Replace per-box sketches with one shared Gaussian sketch per level (peeling / RSRS) to obtain a real black-box cost.
