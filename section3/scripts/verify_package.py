#!/usr/bin/env python3
"""Verify bundle paths, essential evidence and document/test receipts."""
from __future__ import annotations
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
required = [
    "README.md", "main.pdf", "slides.pdf",
    "latex/speaker-notes.md", "python/README.md", "python/pyproject.toml",
    "python/unified/reference/butterfly.py", "rust/README.md", "rust/Cargo.toml",
    "rust/results/summary.json", "rust/results/core-prototype-summary.json",
    "results/MANIFEST.json", "results/update-review.md", "results/framework-explained.md",
    "scripts/check_python.sh", "scripts/check_rust.sh", "scripts/make_figures.py",
]
missing = [name for name in required if not (ROOT / name).exists()]
assert not missing, f"Missing required files: {missing}"

# These are hand-written current guides, not preserved historical documents.
guides = [
    "README.md", "python/README.md", "python/docs/STATUS.md", "python/docs/TESTING.md",
    "python/docs/PROVENANCE.md", "python/docs/RESULTS_GUIDE.md", "rust/README.md",
    "rust/PORT_STATUS.md", "rust/BENCHMARKING.md", "rust/core-prototype/README.md",
    "latex/README.md", "latex/speaker-notes.md", "results/README.md",
]
for relative in guides:
    text = (ROOT / relative).read_text()
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
        if "://" in target or target.startswith("#"):
            continue
        target_path = (ROOT / relative).parent / target
        if not target_path.exists():
            raise AssertionError(f"Broken local Markdown link: {relative}: {target}")

expected = {
    "verification-python-core.txt": r"126 passed in",
    "verification-python-showcase.txt": r"24 passed in",
    "python-e10-tests.txt": r"10 passed in",
    "python-svd-reference-tests.txt": r"7 passed in",
}
folder = ROOT / "results"
for filename, marker in expected.items():
    candidates = [folder / filename, folder / "verification" / filename]
    found = next((p for p in candidates if p.exists()), None)
    assert found is not None, f"No Python test receipt for {filename}"
    assert re.search(marker, found.read_text()), f"Receipt does not match: {found}"
if (folder / "verification-rust.txt").exists():
    text = (folder / "verification-rust.txt").read_text()
    assert "running 7 tests" in text and "running 14 tests" in text
    assert "test result: ok" in text

rust_summary = json.loads((ROOT / "rust/results/summary.json").read_text())
assert len(rust_summary["kernels"]) == 12 and len(rust_summary["fft"]) == 5
assert all(row["relative_apply_error"] < 1e-6 for row in rust_summary["kernels"])
assert all(row["relative_apply_error"] < 2e-12 for row in rust_summary["fft"])
assert rust_summary["dft_butterfly"]["nnz"] == 22528
assert rust_summary["dft_butterfly"]["relative_error_vs_fft"] < 1e-12
core_summary = json.loads((ROOT / "rust/results/core-prototype-summary.json").read_text())
assert core_summary["exact_golden_keys"] == 72
assert core_summary["documented_tolerance_boundary_mismatches"] == 17
assert all(v["rust"] < 1e-8 for v in core_summary["achieved_errors"].values())
assert core_summary["max_lu_unit_residual"] < 1e-14

assert "dpdth = -n * (x * pn - pm)" not in (ROOT / "python/unified/explore/e10_sht_butterfly/sht.py").read_text()
for p in ROOT.rglob("__pycache__"):
    raise AssertionError(f"Generated cache folder must not ship: {p}")
for p in ROOT.rglob("target"):
    if p.is_dir():
        raise AssertionError(f"Generated target build folder must not ship: {p}")

pages = {}
for name in ("main.pdf", "slides.pdf"):
    info = subprocess.check_output(["pdfinfo", str(ROOT / name)], text=True)
    count = int(re.search(r"^Pages:\s+(\d+)", info, re.M).group(1))
    pages[name] = count
assert pages["main.pdf"] >= 16 and pages["slides.pdf"] >= 10
print(f"Package verified: required files and links OK; {pages['main.pdf']} report pages; "
      f"{pages['slides.pdf']} slide pages (10 core + backup).")
print("Recorded tests: 126 core, 24 showcase, 10 spherical-harmonic, 7 SVD comparisons, 21 Rust checks; "
      "72 exact + 17 documented reference-core differences.")
