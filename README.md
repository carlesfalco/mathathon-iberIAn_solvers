# Numerical PDE demonstrations and figures

Supplementary code, numerical experiments, figures, data, and verification results for Team iberIAn_solvers' work on **Problem 29: Simplifying PDE Algorithms**.

The current numerical package is in [`consolidated/`](consolidated/). The current slide deck is [`presentation.pdf`](presentation.pdf), with [`presentation.tex`](presentation.tex) included as its LaTeX source snapshot. The **Final Report** chat remains authoritative for the report and final slide wording; the report source and PDF in `consolidated/` are repository snapshots. See [`SUBMISSION.md`](SUBMISSION.md) for the package boundary and release checklist.

## Consolidated package

| Path | Contents |
|---|---|
| [`consolidated/examples/`](consolidated/examples/) | Periodic FFT and phase-butterfly teaching implementations, plus target-only QR/IDBF comparisons |
| [`consolidated/python/`](consolidated/python/) | Larger Python experiment package and problem-specific research suites |
| [`consolidated/rust/`](consolidated/rust/) | Rust apply benchmarks, DFT invariants, and reference-core prototype |
| [`consolidated/figures/`](consolidated/figures/) | Figures generated from the included numerical data |
| [`consolidated/results/`](consolidated/results/) | Raw results, local verification receipts, and historical package evidence |

The older standalone section files, `section3/`, and the duplicate root butterfly demo have been removed. Their retained implementations and evidence are organized under `consolidated/`.

## Reproduce the core checks

```bash
cd consolidated
scripts/check_local.sh
```

The wrapper runs the FFT and framework examples, butterfly checks and figure regeneration, target-only comparisons, Python core/showcase tests, and Rust unit/integration and saved-data checks. It does **not** rerun the full twelve longer research suites.

For individual examples, historical-versus-local measurement scope, parameters, and result locations, see [`consolidated/README.md`](consolidated/README.md). To check the consolidated report structure, assets, and saved receipts without rerunning the numerical suite:

```bash
python3 consolidated/scripts/verify_project.py
```

Session records, attachments, LaTeX intermediates, caches, build directories, and local archives are excluded by `.gitignore`.
