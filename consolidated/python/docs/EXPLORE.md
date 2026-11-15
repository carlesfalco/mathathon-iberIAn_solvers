**Historical design/specification, retained for reference.** This is not the current run guide or submission status page. Use `TESTING.md`, `STATUS.md` and `../README.md`; some commands and paths below refer to the original, unbundled project environment.

# Swap experiments: what happens when we exchange parts between algorithms?

Context: `unified/` implements three fast algorithms as one program with swappable parts (see SPEC.md and
README.md): tree, admissibility (Weak / Strong / Complementary), compression (one ID fed by Exact or Proxy
compressors; the SVD butterfly in `code/butterfly.py`), algebra (factor via `drivers/rskel.py`, apply via
`drivers/butterfly_id.py`). The harness view: a rank model decides admissibility, a compression engine
builds nested bases, a cost model picks representations, the algebra is fixed.

Each experiment swaps or replaces one part, states a hypothesis BEFORE running, searches over inputs for
regimes where the swap performs well and where it fails, and reports honestly (negative results are results).

## Rules
- Work only in your own directory `unified/explore/<your-id>/` (create it). Import `unified.core`,
  `unified.drivers`, `unified.bench.problems` and `code/butterfly.py` read-only. Never edit files outside
  your directory and never commit. If a driver lacks a hook you need, write a small wrapper in your directory.
- Keep it runnable: one script `run.py` that reproduces every number and figure (results in
  `results/*.json` + `*.png`) in a few minutes on a laptop (N ≤ 4096 unless cheap). Add `test_*.py` with at
  least a few correctness checks, and make sure `.venv/bin/python -m pytest -q unified/tests` still passes.
- Measure fairly: same accuracy target ε for every variant; report accuracy achieved, storage (numbers
  stored), and the relevant rank/iteration counts; wall time only as secondary evidence.
- Write `findings.md`: hypotheses, what you ran, a results table, the inputs where the swap
  wins/fails, and a claim table labelling each claim computed / cited / conjecture. Cite literature you
  rely on (see research/ and research/sota/ notes; e.g. H. Yang 2019 NUFFT-vs-BF, Engquist–Zhao 2018,
  Xing–Chow 2020 proxy analysis, Liu et al. 2021 butterfly in H-matrices, Halko–Martinsson–Tropp 2011).
- Other agents run in parallel on the same machine: keep runs moderate.
