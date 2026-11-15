# Rust port of updated/unified/{core,drivers} — evidence bundle

This directory is preserved as a **separate feasibility prototype**, not merged into the root crate's API. Its hand-written QR/LU routines make rank decisions differently at some tolerance boundaries; that is intentional evidence, not hidden failure. Use `../scripts/verify_reference_core.py` from the project for an acceptance check that expects the same 17 documented differences, requires all 11 recorded achieved errors to be below $10^{-8}$, and checks LU residuals separately.

Read **port-map.md** first. Everything else is either data it cites or the runnable prototype.

| path | what |
|---|---|
| `port-map.md` | the report: port surface, module map, 13 traps (file:line), measured equivalence, staged effort estimate |
| `evidence.txt` | captured output of every run quoted in the report (golden diff, LU check, timings, two-QR comparison) |
| `evidence-inventory.txt` | per-file / per-function LOC and the numpy+scipy primitive inventory of the Python port surface |
| `golden.txt` | 89 deterministic golden numbers produced by the Python drivers (`tools/golden.py`) |
| `rust/` | dependency-free prototype (`src/main.rs` 1214 lines, `src/cases.rs` 513): arena tree, weak/strong admissibility, pivoted-QR column ID, exact+proxy compressors, rskel factor/solve/stats, 1-D ID butterfly |
| `tools/` | `golden.py` (Python side of the harness), `inventory.py`, `dump_steps.py`, `mini.py`, `dump_block.py`, `bench_py.py` |
| `dumps/` | per-box decision dumps used to localize the Python/Rust divergences, and `block_M.txt` (the 304x55 ID block behind trap 5) |

## Run

From the delivered `rust/` directory, build the runnable first, then run the harness from this directory because it reads `golden.txt` by relative path:

```bash
cargo build --release --workspace
cd core-prototype
../target/release/proto check      # 72/89 keys exact, 17 rank-boundary differences -> exit 1 by design
VERBOSE=1 ../target/release/proto check | grep relerr
../target/release/proto bench | grep BENCH
../target/release/proto lutest
```

With a custom `CARGO_TARGET_DIR`, use that directory's `release/proto` instead. `../scripts/verify_reference_core.py` locates it correctly and turns the expected golden diff into an acceptance check. To regenerate the golden values through the real Python drivers, run `PYTHONPATH=../../python python3 tools/golden.py > golden.txt` here.

`proto check` exits non-zero while the 17 rank-decision keys differ; that is the finding, not a build failure.
