# Results and verification annex

This directory contains the data behind the report's measured claims, the independently maintained audit summaries, and reproduction receipts. Experiments themselves live under `../python/unified/explore/`.

## Start with the final receipts

- **[MANIFEST.json](MANIFEST.json)** — required deliverable files and test outcomes, machine-readable.
- **[verification-python-core.txt](verification-python-core.txt)** — 126 passed in 58.26 s, including the seven previously missing SVD comparisons.
- **[verification-python-showcase.txt](verification-python-showcase.txt)** — 24 passed in 10.88 s.
- **[verification-rust.txt](verification-rust.txt)** — main crate and companion crate build/test outcome; 21 main-crate checks.
- **[verification-output.txt](verification-output.txt)** — main Rust sweep plus the reference-core golden harness, including its intentional 17 tolerance-boundary differences.
- **[core-prototype-summary.json](core-prototype-summary.json)** — 72 exact golden keys, 17 documented differences and error-based acceptance.
- **[python-e10-tests.txt](python-e10-tests.txt)** — 10 spherical-harmonic checks pass after Gauss-node sign correction.
- **[e10-gauss-node-correction.json](e10-gauss-node-correction.json)** — corrected orthonormality at $n=4096$: $6.85\times10^{-13}$.
- **[python-svd-reference-tests.txt](python-svd-reference-tests.txt)** — 7 restored SVD-vs-ID comparison cases passed.

## Mathematical and experiment audits

- [framework-explained.md](framework-explained.md) — beginner-level proofs and vocabulary, including the exact DFT and Matérn results.
- [update-review.md](update-review.md) — E6–E12 validation, corrections and research agenda.
- [framework-review.md](framework-review.md) — original scope and literature-placement audit.
- [comparison-butterfly-explanation.md](comparison-butterfly-explanation.md) — paper organization versus the harness's derivation order.
- [comparison-wideband-explanation.md](comparison-wideband-explanation.md) — wideband representations versus teaching order.
- [audit-sources/](audit-sources/) — separable-log examples, small exact DFT checks and the tile-rank dataset used in the report.
- [verification/](verification/) — original independent DFT/E1/E12 scripts, saved outputs and testing notes.

## Data used by the submitted plots

- `butterfly-discovery-ranks.json`: fixed versus complementary tile ranks at tolerance $10^{-6}$. Source of `latex/figures/tile-ranks.png`.
- `matern-proxy-controls.json`: E11 circle refinement 64 through 512 points: why the proxy failure is structural rather than a point-count problem.
- `../rust/results/summary.json`: 12 kernel sweeps, five direct-FFT comparisons, exact DFT-output schema and timing protocol.
- `../python/unified/showcase/results.json`: recorded ten-model showcase parameters, accuracy metrics and full fast/dense timing columns.
- `../python/unified/explore/*/results/`: each experiment's saved JSON and figures, as referenced in the experiment annex.

## Important reading conventions

The Markdown files `framework-explained.md`, `update-review.md` and `framework-review.md` refer back to the original supplied `source/` and `updated/` trees when explaining how the audit was performed. For the current conventional paths, use the [`deliverables/README.md`](../README.md): `python/` replaces the supplied Python tree and `rust/` replaces the original Rust work area. Historical comparisons' recorded filenames and numeric evidence are preserved rather than rewritten to imply unsupported results.

Do not use a test **collection count** as a pass count. The longer E1–E12 suite may be run separately with `../python/scripts/test-all.sh`; its saved evidence is inspectable without that heavy computation. The packaged core, showcase and E10 outcomes above are completed trees reported by pytest.

For performance, distinguish at least storage, sparse reconstruction error, construction time, repeated apply time, dense/solve time, and peak RSS. Use `../rust/BENCHMARKING.md` for matched-backend comparison rules.
