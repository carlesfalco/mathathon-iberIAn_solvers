#!/usr/bin/env python3
"""Validate saved Rust outputs and rebuild the compact experiment summary."""
import json
import platform
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
records = []
for kind in ("laplace", "gaussian", "screened"):
    for n in (256, 512, 1024, 2048):
        path = RESULTS / f"{kind}-{n}.json"
        row = json.loads(path.read_text())[0]
        assert row["n"] == n, path
        assert row["relative_apply_error"] < 1e-6, (path, row["relative_apply_error"])
        assert 0 < row["compressed_entries"] < row["dense_entries"], path
        assert row["construction_note"].startswith("dense-to-compressed oracle")
        row["breakeven_applies_ignoring_dense_build"] = (
            row["compression_seconds"] * 1000
            / (row["dense_apply_ms_median"] - row["compressed_apply_ms_median"])
        )
        records.append(row)
fft = []
for n in (512, 1024, 2048, 4096, 8192):
    path = RESULTS / f"fft-{n}.json"
    row = json.loads(path.read_text())
    assert row["n"] == n
    assert row["relative_apply_error"] < 2e-12, (path, row["relative_apply_error"])
    fft.append(row)
dft_text = (RESULTS / "dft-as-butterfly.txt").read_text()
nnz_match = re.search(r"^nnz\s*: (\d+)$", dft_text, re.MULTILINE)
err_match = re.search(r"relative error vs FFT: ([0-9.eE+-]+)", dft_text)
assert nnz_match and nnz_match.group(1) == "22528", "exact DFT butterfly storage changed"
assert err_match and float(err_match.group(1)) < 1e-12
assert "OK: the DFT's complementary tiles are rank one" in dft_text
summary = {
    "protocol": {
        "build": "cargo test/run --release --all-targets",
        "timing": "one warm-up, raw samples, min/median, peak RSS; process creation excluded",
        "threads": "single-thread process; VM shared with other jobs",
        "compressed_construction": "dense-to-compressed column-pivoted-QR oracle",
        "dense_fft_baseline": "naive O(N^2) Rust loop with direct sin/cos; not NumPy FFT",
    },
    "environment": {
        "python": platform.python_version(),
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "kernel": platform.platform(),
    },
    "kernels": records,
    "fft": fft,
    "dft_butterfly": {
        "n": 1024,
        "nnz": int(nnz_match.group(1)),
        "relative_error_vs_fft": float(err_match.group(1)),
        "natural_and_mixed_order_falsifiers": "tested",
        "dense_matrix_reconstruction": "tested through N=32; full original audit through N=256",
    },
}
(RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(f"Validated {len(records)} compressed, {len(fft)} FFT, and 1 exact DFT-butterfly records.")
print(json.dumps(summary["environment"], indent=2))
for row in records:
    if row["n"] == 2048:
        print(f"{row['problem']:26s} storage={row['storage_ratio']:.4f} "
              f"error={row['relative_apply_error']:.2e} speedup={row['apply_speedup']:.2f}x "
              f"breakeven~{row['breakeven_applies_ignoring_dense_build']:.0f} applies")
