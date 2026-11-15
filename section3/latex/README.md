# LaTeX report, slides and notes

- `report.tex` + `sections/`: editable article source for the full derivation.
- `main.pdf`: compiled, 31-page self-contained report includes the algorithm map, twelve problem-oriented experiment explanations and reproduction appendix.
- `slides.tex`: editable Beamer deck; 10 core frames + 3 backup frames.
- `slides.pdf`: compiled deck; deliver pages 1–10 for the eight-minute talk.
- `speaker-notes.md`: plain-language narration, timings, transition cues and backup answers.
- `figures/`: report figures generated from included numerical data plus the two-tree structural diagram.

## Reading order for a non-expert

Start with the abstract, then §1 for block compression, §2 for five choices, §3 for FMM, wideband FMM, recursive skeletonization/FMM-LU, butterfly and FFT. §4 supplies the butterfly worked example. §5 presents the experiments by problem, §6 the implementations, and §7 the conclusion. Appendices contain the experiment guide and exact reproduction commands.

## Rebuild from this delivery

From `section3/`, run:

```bash
scripts/build_documents.sh
```

That script regenerates figures and compiles both documents in an external temporary build directory. It does not place LaTeX's `.aux` files or temporary files in the submission.

The direct commands are also valid inside `latex/`:

```bash
python3 ../scripts/make_figures.py
latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error slides.tex
```

`main.pdf` is a convenience copy of the compiled report. The text and plots are fully editable, with no proprietary slide/report tool required.
