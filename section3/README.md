# A common block strategy — main deliverables

Reference PDFs plus Python/Rust implementations, figures, data, and verification results for a shared hierarchical block framework. Report and slide LaTeX sources are included alongside the compiled reference documents.

## Start here

| What you need | Where to go |
|---|---|
| **Readable report with derivations** | **[main.pdf](main.pdf)** — 31-page self-contained exposition, algorithm map, twelve problem-oriented experiments and reproduction appendices |
| **Eight-minute slides** | **[slides.pdf](slides.pdf)** — pages 1–10 core talk, pages 11–13 backup |
| **Plain-language speaker notes** | **[speaker-notes.md](latex/speaker-notes.md)** — 7–7.5 minutes of script + pauses/silence |
| **Editable LaTeX report and slides** | [`latex/report.tex`](latex/report.tex), [`latex/slides.tex`](latex/slides.tex), [`latex/sections/`](latex/sections/) |
| **Python implementation** | [`python/README.md`](python/README.md) — core, recursive solver, butterfly, showcase, twelve experiments and saved results |
| **Rust implementation** | [`rust/README.md`](rust/README.md) — lean QR/apply crate, exact DFT butterfly, separate reference-core prototype |
| **Experiment/verification annex** | [`results/README.md`](results/README.md) — raw checks, audit trail and measured performance logs |

## What the framework actually says

The shared strategy is **partition → choose compressible interactions → build short representations → obtain them by a stated sampling method → assemble by suitable algebra**. The five design choices are partition, interactions, representation, sampling and algebra.

- **FMM** adds analytic expansions and translation operators.
- **Wideband FMM** changes expansions and translation schemes across scales.
- **HBS/recursive skeletonization/FMM-LU** turns local skeletons into sparsification and elimination.
- **Butterfly** uses two trees, complementary tiles, and nested transfers to obtain fast application.
- **FFT** makes complementary DFT tiles exactly rank one in both bit-reversed orderings, and adds Cooley--Tukey recursion.
- **The twelve experiments** make the required extra structure concrete for complex elimination, block physics, screening, fractional/nonlocal problems, volume scattering, transforms, covariance and wakes.

That is the organizing explanation; it is not a claim that one preset program derives every algorithm or every PDE result.

## Quick reproduction

From this directory, dependencies installed:

```bash
# Python: no editable install is necessary for the wrappers
scripts/check_python.sh

# If you do want the package installed
cd python && python3 -m pip install -e '.[test]' && cd ..

# Rust: main tests, reference golden check, and saved output validation
scripts/check_rust.sh

# Check links, included data, required files, and recorded test outcomes
python3 scripts/verify_package.py
```

`python/scripts/test-all.sh` additionally includes the **longer** twelve research suites. Their saved plots and data are included already; reproducing them in full is a separate computational job.

For the longer experiment suite's exact behavior at each parameter, start with `python/unified/explore/e*/findings.md` and the guide in Appendix B of the report.

## Verified status in this package

- Python **126 core tests passed**, including all seven reference SVD butterfly comparisons.
- Python showcase **24 tests passed**.
- Python spherical-harmonic checks **10 passed**, including the corrected Gauss-node sign regression; orthonormality error is $6.85\times10^{-13}$ at $n=4096$.
- Rust **21 checks passed**, including all 14 DFT invariants.
- Rust reference core preserves **72 exact golden values plus 17 documented tolerance-boundary rank/count differences**; acceptance is based on achieved errors, all below $10^{-8}$ in the recorded fixture, not on identical QR pivot counts.
- The exact DFT-butterfly at $N=1024$ stores **22,528 = $2N\log_2N+2N$** entries with relative error $1.83\times10^{-16}$ versus an independent FFT.
- Rust apply at $N=2048$ achieves **3.6–7.7% dense storage**, **11.8–14.7× faster apply**, while the current prototype's construction costs about 0.96 seconds. These are apply/storage measurements, not an end-to-end production solver claim.

See [`results/MANIFEST.json`](results/MANIFEST.json) for machine-checkable receipt paths and [`results/verification-output.txt`](results/verification-output.txt), plus the Python logs for the corresponding suites.

## Package boundaries

The Python package contains considerably more physics and special-function code than the Rust prototype. The Rust `core-prototype` demonstrates tree, admissibility, exact/proxy compression, complex FIO and recursive LU solve; a production unified API/backend and the complete physics experiment suite remain future work. Python's reference recursive solver retains its dense working matrix. These facts are stated rather than hidden in performance claims.

All original experiment results and report figures are included for inspection without lengthy recomputation. Cache directories, Python bytecode, LaTeX temporaries and target build directories are not part of the distribution. The original supplied `source/` and `updated/` directories remain untouched by the packaging work.

## Merge with team material

Recommended integration order:

1. Read the **abstract and §§1–3** in `main.pdf` for the common framework.
2. Use **§4** for the butterfly construction and nesting argument.
3. Use **§5 and Appendix B** for experiment-specific structure and links into each problem study.
4. Run the eight-minute deck with `latex/speaker-notes.md`.
5. Link `python/` and `rust/` as separate implementation packages, preserving their own READMEs and test wrappers.

Nothing here requires access to the project's `.ultramath/` working records. The independent mathematical audit and experimental-review PDFs in the original project are represented by the local `results/` Markdown files and direct checks.
