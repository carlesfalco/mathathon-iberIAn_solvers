# Numerical PDE demonstrations and figures

Supplementary code, numerical experiments, figures, data, and verification results for Team iberIAn_solvers' work on **Problem 29: Simplifying PDE Algorithms**.

LaTeX sources are intentionally **not tracked** in this repository. The official report is maintained separately in the **Final Report** chat. Compiled PDFs already in the repository are retained as reference documents.

## Main demonstrations

| Path | Contents |
|---|---|
| [`butterfly-demo/`](butterfly-demo/) | Minimal Python butterfly implementation for DFT and non-DFT oscillatory kernels, tests, accuracy/timing data, and comparison figures |
| [`consolidated/examples/`](consolidated/examples/) | Teaching implementations for the periodic FFT solver and phase-based butterfly construction |
| [`consolidated/python/`](consolidated/python/) | Larger Python experiment package |
| [`consolidated/rust/`](consolidated/rust/) | Rust apply benchmarks, DFT invariants, and reference-core prototype |
| [`section3/python/`](section3/python/) | Python implementations and experiments for the common-block-strategy package |
| [`section3/rust/`](section3/rust/) | Rust implementations and numerical results for that package |

## Figures and results

- `butterfly-demo/results/`: accuracy and scaling CSV files, comparison plots, and a measured-results guide.
- `consolidated/figures/` and `consolidated/results/`: consolidated figures and local/package verification receipts.
- `section3/figures/` and `section3/results/`: generated figures, raw numerical evidence, and audit notes.
- `section3/python/unified/explore/`: problem-specific experiments and saved outputs.

## Run the small butterfly checks

```bash
cd butterfly-demo
python3 test_butterfly.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 compare.py
```

The comparison script uses an FFT reference for the DFT test and explicit dense kernel evaluation for the non-DFT test. Absolute timings depend on the machine; they are teaching measurements rather than production benchmarks.

## Verify the larger packages

```bash
# Check paths, included evidence, and recorded results for section3
python3 section3/scripts/verify_package.py

# From consolidated/, reproduce the core numerical checks
cd consolidated
scripts/check_local.sh
```

The consolidated wrapper runs the small FFT and butterfly checks, Python core/showcase tests, and Rust unit/integration checks. It does not rerun the full twelve longer research suites. See the package READMEs for details and for which measurements are historical rather than regenerated.

Session records, attachments, LaTeX intermediates, caches, build directories, and local archives are excluded by `.gitignore`.
