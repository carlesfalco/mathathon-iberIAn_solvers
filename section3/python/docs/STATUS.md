# Current status and limits

## What the Python package includes

- Shared tree, weak/strong/complementary interaction rules, column-pivoted-QR ID, exact and circle-proxy compressors.
- Recursive-skeletonization factorization and solve, for real and complex arrays.
- One-dimensional complex interpolative butterfly with sparse factors, exact DFT correspondence and an FIO example.
- Ten-model showcase and twelve research experiment directories with saved results.
- Test code and canonical parameter files for reproducing results, not just screenshots or prose.

## Verified before this cleanup

- Ten showcase models passed their quick agreement checks in the audited environment.
- The exact DFT complementary factorization has one skeleton per tile and $2N\log_2N+2N$ stored entries in the tested bit-reversed cases.
- The independent DFT/E1/E12 checks matched the saved output.
- The experiment audits found green per-directory suites for the showcase and E6–E9, E11, E12; E10 had one nonportable assertion requiring a NaN.
- The full 264-test upstream tree was **not** completed within the earlier 2400 s limit. Six missing-reference failures plus one platform-dependent E10 assertion were confirmed locally; this is not a complete tally of all remaining tests.

## Changes made for this submission

- Separated the clean Python package from supplied `source/` and `updated/`, which remain the audit inputs.
- Added `pyproject.toml`, a lightweight `.gitignore`, and stable wheel/install commands.
- Replaced the stale entry README with a runnable quick start, correct package paths and explicit scope.
- Preserved the original project description as a clearly marked legacy document instead of presenting it as the current run guide.
- Changed the E10 test to assert that clipping is accurate, rather than requiring the current numerical backend to emit a NaN. This corrects a platform-dependent test assumption; it does not disable the useful raw-path experiment.
- Restored the reference-only SVD helper as `unified.reference.butterfly`, with comparison loaders updated. All seven previously failing reference-comparison tests pass locally. The completed post-cleanup core-suite outcome is recorded below rather than inferred from a collection count.

## Verification completed by the cleanup

- `scripts/test-core.sh`: **126 passed in 58.26 s**, including all seven reference SVD comparisons.
- `scripts/test-showcase.sh`: **24 passed in 10.88 s**.
- E10: **10 passed**, including the corrected Newton-sign regression test; at $n=4096$, corrected-node orthonormality is $6.85\times10^{-13}$.
- Test collection: **264 tests**, consistent with the original update inventory; that count was not used as a claim that the longer E1--E12 experiment suite had completed.

## What this cleanup deliberately does not claim

- **No unified Python/Rust feature parity.** Rust covers representative slices; Python contains much more physics and more experiment families.
- **No numerical redesign.** The reference algorithms, dense working matrices and documented sampling conventions remain intentionally visible.
- **No universal speedup.** Accuracy and apply scaling can improve while construction, memory traffic or solve time remain expensive.
- **No universal proof of proxy validity.** E8 and E11's exact integer-codimension lift arguments do not cover every fractional case; multi-order construction may be empirically excellent without being necessary.
- **No production 3-D high-order quadrature implementation, no adaptive production PDE solver, and no exhaustive novelty proof.**
- **No publication-quality prior-art replacement.** Refer to the checked literature in `../../results/update-review.md` and `../../results/framework-review.md`.

## Key review corrections to remember

E6's Stokes cost is approximately $2\times$ per box, $1\times$ per dof, and $4\times$ storage at equal geometry. The suspension "settling" figure is convention-dependent and is not presented as physical velocity. E7/E12's elimination flop models exclude substantial construction work. E10's per-order SHT ratios do not establish an all-order transform win. E12's directional Oseen admissibility reduces the top-level/root block, but at $N=4096$ its storage and elimination flops increase by $28.6\%$ and $67.3\%$, respectively.

These are not minor editorial disagreements: they determine which evidence supports which claim. See the complete audit for the exact numbers and caveats.
