# Target-only butterfly construction

Controlled implementations accompanying Section 5 of the consolidated report, adapted from `martin-stuff/additionalContext/`.

## What is compared

- **Original / mixed application:** compressed source coordinates, a middle coupling, then compressed target contributions.
- **Target-only application:** the partial-product invariant $P_{ij}^\ell w_{ij}^\ell\approx K_{ij}^\ell g_j^\ell$ at every level.
- **SVD version:** downward pivoted-QR construction from a common literal-SVD middle seed.
- **IDBF version:** original-column interpolative bases from a common two-sided ID middle seed.

Both versions use exact-rank oracle matrices so approximation differences do not confound the comparison. The IDBF driver independently checks random outputs, every coordinate vector, source states, and the target invariant at every slot and level.

## Run

From this directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 compare_butterfly.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 compare_idbf.py
```

The IDBF driver writes `results/idbf_benchmark_results.json`. Consolidated console receipts are in `../../results/local/target-only/`.

## Scope

The matrices are $K=AB^*/\sqrt{Nr}$ with exact global rank $r=4$, tested at $N=16,64,256$. Reported moments are fresh local medians on this VM with one BLAS/OpenMP thread; structural entry counts do not depend on wall time.

The constructors are visible-entry mathematical oracles. They are **not** implementations or benchmarks of IDBF's optimized sampled-entry construction. Exact-rank correctness does not establish error propagation for nontrivial truncation, production sparse-kernel speedups, or a new asymptotic application rate.
