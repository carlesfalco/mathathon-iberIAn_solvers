#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
mkdir -p .build
latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex \
  > .build/latexmk-console.txt 2>&1
python3 scripts/verify_project.py
