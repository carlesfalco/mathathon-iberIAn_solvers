# Benchmarking the Rust port fairly

## What the independent measurements established

The efficiency audit ran the actual Python drivers read-only, with one configuration per process, BLAS pinned to one thread, warm-up, repeated samples, GC control, min/median statistics, and peak RSS. It did not change `updated/`.

Three results matter more than generic claims about language speed:

1. **The dominant cost depends on the regime.** At $N=4096$, the tested area problem's weak recursive factorization spends about 54% of its time in interpolative decomposition; the strong case spends about 86% in dense-block matrix multiplication and gather. An optimization aimed at only one of those can miss the real bottleneck in the other.
2. **The tempting overhead is in small operations.** One generic butterfly build makes 53,248 pivoted-QR calls; a one-candidate-column call costs about 53 µs. Small block LU solves cost microseconds despite little arithmetic. Sparse LIL-to-CSR assembly and fancy-index gather are also large measured terms. These are opportunities for specialized kernels and better layouts, not proof that “Rust is faster.”
3. **A ratio needs a size.** The generic B1 butterfly apply was about 0.75× dense-GEMV speed at $N=1024$ and 3.18× at $N=4096$: slower before the crossover, faster after it. Build time, construction memory and apply time answer different questions.

The audit also found a 30–50 µs SciPy call/validation floor on tiny IDs and a 5.7 µs small-block `lu_solve` call. These particular overheads can be reduced substantially. The same audit models possible gains from measured peak rates, but those multipliers are illustrative planning scenarios, not bounds or predictions.

## Current Rust protocol

`rust-unified/scripts/run.sh` uses a fresh example process for every case and records:

- problem, $N$, $\varepsilon$, leaf size and compressor convention;
- dense and compressed stored entries;
- independent dense-reference apply error;
- one warm-up, raw timed samples, min and median for dense and compressed apply;
- dense assembly and compression times separately;
- peak process RSS;
- deterministic test input and exact DFT invariants.

The saved file `results/summary.json` contains every sample. This is intentionally stronger evidence than a single heroic timing.

The current comparisons are **inside Rust**, not Rust-versus-Python races:

- dense matrix apply versus compressed matrix apply using the same Rust process and matrix;
- naive $\mathrm O(N^2)$ DFT versus a reusable RustFFT plan, for correctness and scaling;
- exact DFT-butterfly algebra versus a separate radix-two FFT and a dense reconstructed DFT matrix.

No production NumPy/BLAS or LAPACK parity is implied by these rows.

## What not to compare

| Invalid comparison | Why it misleads | Required alternative |
|---|---|---|
| Optimized NumPy/BLAS versus a naive Rust loop | Measures backend quality, not language speed | Same algorithm and equivalent optimized backend on both sides |
| Dense-to-compressed construction versus a genuine hierarchical builder | Different algorithms and memory requirements | Call the former an oracle; report it separately |
| A single small-$N$ apply ratio | Can sit before the dense-parallel crossover | Size sweep through the crossover, same error target |
| Factorization alone as “the solver’s cost” | Assembly and solve may dominate the application | Assembly, factor, solve, and end-to-end rows separately |
| Elimination flop count versus wall clock | Omits QR, sampling, gathers, allocations and memory traffic | Count each measured phase and report GB/s and call histogram |
| Stored factor entries versus process memory | Working matrices, temporaries and runtimes can dominate RSS | Report both stored entries and peak RSS |
| Two different tolerance/rank conventions | A different method can win by doing less work | Same $\varepsilon$, trees, permutations and acceptance error |
| Current Rust timings versus original arm64 timings | Different hardware, processes and algorithms | No direct ratio; rerun matched cases on one host |

## Promotion gate before any Rust-versus-Python headline

1. Bind a numerical backend deliberately and pin its version and LAPACK behavior.
2. Match Python's ordering, trees, permutations and compression choices.
3. Compare residuals first, not only ranks or storage. Skeleton counts can vary at the tolerance boundary.
4. Retain all raw timings, min/median, repeated cases and high-water RSS.
5. Use at least the existing P1/P2/B1 fixtures, then add physics only after its special-function/reference layer is verified.
6. Keep the dense-work-array limitation in the result caption. If both sides retain $O(N^2)$ memory, the measurement is about constants and implementation overhead, not a new asymptotic solver.

The full accepted measurement reports live at `.ultramath/smwha0t61/m2/report.md`, `validity-table.md`, and `protocol.md`, with scripts in `bench/` and raw JSON in `out/` under the same directory.
