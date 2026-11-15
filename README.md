# Simplify Numerical PDE Algorithms

Working repository for the Mathathon problem on making fast numerical-PDE algorithms easier to understand and, where possible, simpler while remaining effective. The current narrative develops PDE and Green-function motivation, the FFT, structured factorisations, butterfly algorithms, numerical linear algebra, and limitations.

> **Draft status:** this is a working repository, not a final submission. The global report has not been assembled, and the current slide source contains marked placeholders. The official report should not be compiled until the team requests it.

## Repository map

| Path | Role |
|---|---|
| `section1-motivation.tex` / `.pdf` | Standalone Section 1 draft: PDEs and Green-function integrals |
| `section4-butterfly.tex` / `.pdf` | Standalone Section 4 draft: oscillatory integrals and butterfly factorisation |
| `presentation.tex` / `.pdf` | Working 20-minute presentation; unfinished sections are explicitly marked |
| `butterfly-demo/` | Minimal Python butterfly/FFT demonstration, tests, and generated comparisons |
| `deliverables/python/` | Larger Python implementations and experiments |
| `deliverables/rust/` | Rust implementations and reference-core prototype |
| `deliverables/results/` | Numerical results, verification receipts, and audit notes |
| `deliverables/latex/` | An older common-framework report and slide package, retained as source material |
| `SUBMISSION.md` | Final repository layout, release blockers, and submission checklist |

The official report will eventually be written as **one self-contained `report.tex`** at the repository root, accompanied by `report.pdf` and the final `slides.pdf`. The older `deliverables/latex/report.tex` is not that file: it uses `\input` and therefore does not satisfy the report-source requirement in its present form.

Session records, attachments, LaTeX intermediates, caches, build directories, and local ZIP exports are excluded by `.gitignore`. In particular, `.scratch/`, `.ultramath/`, and `attachments/` are working materials rather than submission content. The tracked directories, not `sec3.zip`, are the source of truth.

## Verification

Check the existing supplementary package without rebuilding its documents:

```bash
python3 deliverables/scripts/verify_package.py
```

This checker verifies required paths, local links, selected result data, PDF page counts, and recorded test receipts. It **does not rerun the full Python or Rust test suites** and does not validate the future global report.

Run the small butterfly demonstration's tests with:

```bash
cd butterfly-demo
python3 test_butterfly.py
```

See `deliverables/README.md` for the larger implementations' reproduction commands. All numerical and performance claims in the final report must identify their supporting computation and parameter range.

## Before release

Follow `SUBMISSION.md`. At minimum, finish the report and slides, remove placeholders, flatten the report into one `.tex` file, state how every main claim is supported, document AI use and event work, run the relevant checks, and commit the exact version intended for judging.
