# Simplify Numerical PDE Algorithms

Working repository for the Mathathon problem on making fast numerical-PDE algorithms easier to understand and, where possible, simpler while remaining effective. The current narrative develops PDE and Green-function motivation, the FFT, structured factorisations, butterfly algorithms, numerical linear algebra, and limitations.

> **Draft status:** this is a working repository, not a final submission. The current slide source still contains marked placeholders.
>
> **Current report:** [`report.tex`](report.tex) and [`report.pdf`](report.pdf) identify Team iberIAn_solvers and Problem 29, and merge Sections 1, 2, and 4 into one self-contained 14-page document. The remaining sections will be added in later versions.

## Repository map

| Path | Role |
|---|---|
| `report.tex` / `report.pdf` | Current self-contained interim report: Sections 1, 2, and 4 |
| `section1-motivation.tex` / `.pdf` | Standalone Section 1 draft: PDEs and Green-function integrals |
| `section4-butterfly.tex` / `.pdf` | Standalone Section 4 draft: oscillatory integrals and butterfly factorisation |
| `presentation.tex` / `.pdf` | Working 20-minute presentation; unfinished sections are explicitly marked |
| `butterfly-demo/` | Minimal Python butterfly/FFT demonstration, tests, and generated comparisons |
| `section3/` | Standalone common-block-strategy report, slides, implementations, and evidence (formerly `deliverables/`) |
| `consolidated/report.tex` | **Work-in-progress** integrated single-file report source; not the final report |
| `consolidated/examples/` | Teaching examples used by the integrated report |
| `consolidated/python/` / `rust/` | Supporting implementations and experiments copied for integration checks |
| `consolidated/results/` | Local and previously recorded verification results |
| `SUBMISSION.md` | Final repository layout, release blockers, and submission checklist |

The current report entry point is **one self-contained `report.tex`** at the repository root, accompanied by `report.pdf`. The source and PDF both contain only Sections 1, 2, and 4 at present; the official `slides.pdf` is still pending. The older `section3/latex/report.tex` is not that file: it uses `\input` and therefore does not satisfy the report-source requirement in its present form.

Session records, attachments, LaTeX intermediates, caches, build directories, and local ZIP exports are excluded by `.gitignore`. In particular, `.scratch/`, `.ultramath/`, and `attachments/` are working materials rather than submission content. The tracked directories, not `sec3.zip`, are the source of truth.

## Verification

Check the existing supplementary package without rebuilding its documents:

```bash
python3 section3/scripts/verify_package.py
```

This checker verifies required paths, local links, selected result data, PDF page counts, and recorded test receipts. It **does not rerun the full Python or Rust test suites** and does not validate the root report.

Build the current root report without placing LaTeX intermediates in the repository:

```bash
mkdir -p .scratch/report-build
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=.scratch/report-build report.tex
cp .scratch/report-build/report.pdf report.pdf
```

The compiled report was also checked for resolved references/citations, valid figure paths, and overfull/underfull boxes. Run the small butterfly demonstration's tests with:

```bash
cd butterfly-demo
python3 test_butterfly.py
```

See `section3/README.md` for the larger implementations' reproduction commands. All numerical and performance claims in the final report must identify their supporting computation and parameter range.

## Before release

Follow `SUBMISSION.md`. At minimum, finish the report and slides, remove placeholders, flatten the report into one `.tex` file, state how every main claim is supported, document AI use and event work, run the relevant checks, and commit the exact version intended for judging.
