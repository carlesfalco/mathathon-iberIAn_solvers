#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
WORKSPACE=$(cd "$ROOT/.." && pwd)
OUT="$WORKSPACE/.scratch/deliverable-documents"
mkdir -p "$OUT/report" "$OUT/slides" "$OUT/texmf-var"
export TEXMFVAR="$OUT/texmf-var"

cd "$ROOT"
python3 scripts/make_figures.py
cd latex
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir="$OUT/report" report.tex \
  > "$OUT/report-build.log" 2>&1
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir="$OUT/slides" slides.tex \
  > "$OUT/slides-build.log" 2>&1
cp "$OUT/report/report.pdf" "$ROOT/main.pdf"
cp "$OUT/report/report.pdf" "$ROOT/latex/report.pdf"
cp "$OUT/slides/slides.pdf" "$ROOT/slides.pdf"
cp "$OUT/slides/slides.pdf" "$ROOT/latex/slides.pdf"
printf 'Wrote %s and %s\n' "$ROOT/main.pdf" "$ROOT/slides.pdf"
printf 'Build logs: %s\n' "$OUT/report-build.log and $OUT/slides-build.log"
if rg -n 'Overfull|Fatal error' "$OUT/report-build.log" "$OUT/slides-build.log"; then
  echo 'LaTeX layout warning(s) detected; please inspect them before release.' >&2
  exit 1
fi
