#!/usr/bin/env python3
"""Check the consolidated report structure, local assets, and saved receipts."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
required = [
    "README.md", "report.tex", "report.pdf",
    "examples/fft_periodic_solver.py",
    "examples/butterfly/butterfly.py", "examples/butterfly/test_butterfly.py",
    "figures/two-trees.png", "figures/butterfly-dft.png",
    "figures/butterfly-non-dft.png", "python/README.md", "rust/README.md",
    "results/local/fft_checks.json", "results/local/python-core.txt",
    "results/local/python-showcase.txt", "results/local/rust-tests.txt",
]
missing = [p for p in required if not (ROOT / p).is_file()]
assert not missing, f"Missing files: {missing}"

source = (ROOT / "report.tex").read_text()
for pattern in (r"\\input\s*\{", r"\\include\s*\{", r"\\bibliography\s*\{",
                r"\\addbibresource\s*\{"):
    assert not re.search(pattern, source), f"Non-self-contained report command: {pattern}"
assert source.count(r"\begin{document}") == 1
assert source.count(r"\end{document}") == 1
assert "\\begin{thebibliography}" in source
labels = re.findall(r"\\label\{([^}]+)\}", source)
assert len(labels) == len(set(labels)), "Duplicate LaTeX labels"
for graphic in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", source):
    assert (ROOT / graphic).is_file(), f"Missing figure: {graphic}"

for p in (ROOT / "report.log",):
    if p.exists():
        log = p.read_text(errors="replace")
        assert not re.search(r"(?:Overfull|Underfull) \\[hv]box", log), "Layout warning remains"
        assert "undefined references" not in log.lower(), "Unresolved references remain"

for filename, marker in (("python-core.txt", "126 passed"),
                         ("python-showcase.txt", "24 passed"),
                         ("rust-reference-core-check.txt", "72 exact + 17 documented")):
    assert marker in (ROOT / "results/local" / filename).read_text(), filename
rust = (ROOT / "results/local/rust-tests.txt").read_text()
assert "running 7 tests" in rust and "running 14 tests" in rust
assert "FAILED" not in rust and "0 failed" in rust
fft = json.loads((ROOT / "results/local/fft_checks.json").read_text())
for key in ("max_unitary_error", "max_sparse_stage_error", "max_recursive_fft_error"):
    assert fft[key] < 1e-12, (key, fft[key])

info = subprocess.check_output(["pdfinfo", str(ROOT / "report.pdf")], text=True)
pages = int(re.search(r"^Pages:\s+(\d+)", info, re.M).group(1))
assert pages >= 20
print(f"Consolidated project verified: self-contained report, figures, local receipts, and {pages}-page PDF.")
print("Scope: 126 core + 24 showcase Python tests and 21 Rust checks; full twelve research suites not rerun.")
