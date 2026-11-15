# From Green Functions to Fast Factorizations

Consolidated Mathathon project on making fast numerical-PDE algorithms easier to understand without hiding the mathematics that makes them effective.

## Read first

- **[Report PDF](report.pdf)** — the current consolidated report.
- **[Single-file report source](report.tex)** — includes appendices and references directly; no `\input`, `\include`, or external `.bib` file.
- **[Verification results](results/local/)** — locally generated numerical and test receipts.
- **[Consolidated release status](RELEASE.md)** — overall release requirements (outside this project directory).

## The six-part story

| Section | Role in the argument | Main evidence |
|---|---|---|
| 1. PDEs and Green functions | Turn a PDE into an interaction matrix; distinguish apply from solve | Dirichlet Green identity and a reaction–diffusion integral equation |
| 2. FFT | Show exact diagonalization *and* sparse basis changes; explain why global low rank is the wrong explanation | Proofs plus locally rerun periodic-solver and sparse-stage checks |
| 3. Structured factorizations | Introduce blocks and the five choices: partition, interactions, representation, sampling, algebra | Logarithmic-block proof, three assembly grammars, literature-backed algorithm map |
| 4. Butterfly mechanism | Derive centred phase subtraction, complementary boxes, and nested compressed potentials | Mixed-difference bound, cited analytic rank theorem, DFT and non-DFT figures |
| 5. Numerical linear algebra | Separate local rank, nesting, construction, apply, elimination, and physical accuracy | Exact DFT residue identity, regenerated tables, selected PDE diagnostics, Rust/Python evidence |
| 6. Synthesis | State what was simplified, limitations, contribution, and conclusion | Explicit scope boundaries rather than a universal-solver claim |

The central transition is **FFT exact sparse factors → complementary approximate blocks → nested sparse factors**. The twelve broader experiments support the diagnostic framework; they do not replace the main derivation.

## Project layout

```text
report.tex / report.pdf       single-file official-report candidate
figures/                     figures used by the report
examples/fft_periodic_solver.py
examples/butterfly/          minimal phase-based butterfly, tests, data, plots
python/                      larger experimental Python package
rust/                        apply benchmarks, DFT invariants, reference core
results/local/              receipts generated in this environment
results/package/            historical review notes and receipts from Section 3
scripts/                    build and verification entry points
```

The original section drafts remain outside this directory. They are source material, not parallel official reports.

## Build and verify

From `consolidated/`:

```bash
scripts/build_report.sh
python3 scripts/verify_project.py
```

The first command compiles `report.pdf` from exactly `report.tex` and then runs the structural checker. The checker enforces the single-file rule, figure existence, unique labels, clean LaTeX layout/reference logs, and key local receipt markers.

## Reproduce the numerical evidence

Small examples only:

```bash
python3 examples/fft_periodic_solver.py
cd examples/butterfly
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 test_butterfly.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 compare.py
```

Core supplement checks (about one minute on the assembly VM, plus the initial Rust build):

```bash
scripts/check_local.sh
```

That wrapper runs the FFT example, four butterfly tests and figure regeneration, 126 Python core tests, 24 showcase tests, all 21 Rust unit/integration checks, saved Rust-data validation, and the reference-core golden controls. It **does not** rerun the full twelve research suites; `python/scripts/test-all.sh` is the separate longer job.

## Verified in this pass

- FFT algebra and PDE-error checks generated under Python 3.12.14 / NumPy 2.5.3.
- Four butterfly tests passed; both comparison figures and their CSV data regenerated.
- Python: **126 core + 24 showcase tests passed**; 264 total tests collected, not all run.
- Rust: **7 unit + 14 DFT-invariant checks passed**.
- Saved Rust data: 12 compressed, five FFT, and one exact DFT-butterfly record validated.
- Reference core: 72 exact golden values and 17 documented tolerance-boundary count differences; all 11 achieved errors below $10^{-8}$ and maximum LU unit residual $1.72\times10^{-15}$.
- Report: 25-page A4 PDF, with appendices and bibliography inline.

Rust wall-time values quoted in the report are preserved historical measurements, not timings regenerated in this pass. No Lean verification is claimed.

## Release work still needed

1. Replace the title-block placeholder with team identity, public repository URL, and full commit hash; confirm whether the submission declares one or both tracks.
2. Finish the separate 20-minute slide deck from this report, remove its old placeholders, and ensure every slide claim appears here.
3. Decide which longer research suites must be rerun for the final release and preserve fresh receipts for those runs.
4. Reconcile the AI-use appendix with any team/tool disclosures absent from the supplied drafts, then perform a team mathematical review.
5. Update the root release checklist and avoid presenting the older per-section reports as competing official entry points.
