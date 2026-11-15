# Mathathon submission checklist

This file separates repository preparation from the later content-release step. The integrated report remains **under review and is not approved as the final submission**.

## Required final repository layout

At release, the repository root must unambiguously provide:

- `report.tex` — the complete report source in one file;
- `report.pdf` — compiled from that exact source;
- `slides.pdf` — the final presentation;
- supplementary code, results, and documentation needed to support report claims.

`report.tex` must include its appendix and bibliography entries directly. Do not use `\input`, `\include`, or a separate `.bib` file for the official report. Draft and legacy report sources must not be confused with this official entry point.

The repository URL and full commit hash requested by the organizers belong in the report title block and submission metadata. Names, affiliations, and repository links must appear only in the title block, which the organizers remove before LLM review.

## Report-content checks

- [ ] Begin with a short abstract and define notation early.
- [ ] Explain the problem, known algorithms, motivation, contribution, and main arguments accessibly.
- [ ] Make clear what work was done during the 40-hour event.
- [ ] For every main claim, state its support: proof in the report, literature citation, computation with methods and parameter ranges, Lean verification if applicable, or an explicitly labelled conjecture.
- [ ] Keep rigorous arguments separate from heuristic explanations and state limitations.
- [ ] Describe every key figure in words; links and images are not substitutes for explanation.
- [ ] Credit sources and closely followed arguments.
- [ ] Document which AI tools were used, what they were used for, and how their output was checked.
- [ ] Check that every team member can explain the submitted material.
- [ ] Remove length-oriented padding; length itself earns no credit.

## Slides and presentation checks

- [ ] Replace every `TO BE COMPLETED` and draft placeholder in `presentation.tex`.
- [ ] Ensure everything in the slides also appears in the report.
- [ ] Rehearse within the 20-minute presentation limit and prepare for up to 10 minutes of questions.
- [ ] Prepare answers on definitions, the key step, novelty versus known work, AI use and understanding, and confidence in each claim.
- [ ] Export the final deck as root `slides.pdf`.

## Current release blockers

- [ ] Review and approve the complete report before release.
- [x] Produce the self-contained root `report.tex` and compile `report.pdf`; the current 14-page version merges Sections 1, 2, and 4 and has no unresolved references or layout warnings.
- [ ] Integrate the remaining sections into the root report and rebuild the PDF from the resulting source.
- [ ] Complete the team's mathematical and editorial review of the root report.
- [ ] Finish the current slide placeholders and export root `slides.pdf`.
- [ ] Reconcile or remove alternative draft reports/decks so reviewers cannot mistake them for the official submission.
- [ ] Confirm the track designation and complete team/repository metadata.

## Verification and release

1. Run `python3 section3/scripts/verify_package.py` for the standalone supplementary package and `python3 consolidated/scripts/verify_project.py` for the integrated candidate.
2. Run the relevant Python and Rust test suites rather than relying only on their saved receipts; record any tests not rerun.
3. Build the final report and slides from a clean checkout and inspect the PDFs, logs, references, and figures.
4. Search the official sources for placeholders, unresolved comments, broken references, and unsupported claims.
5. Confirm that no `.scratch/`, `.ultramath/`, attachment, cache, or build-output directory is tracked.
6. Commit all release files, publish the repository so the organizers can access it, and submit the full hash of the exact version selected for judging.
7. Check that the GitHub submission contains both required PDFs and the self-contained report source before the deadline.

An optional Lean project is not currently part of this repository. If one is added, its version, declarations, axioms, build result, and relationship to report claims require a separate compliance check.
