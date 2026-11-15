# Coordinator verification artifacts

These are the lightweight checks rerun while judging the E6–E12 audit reports. They supplement, rather than replace, the detailed member evidence under `.ultramath/smwh5ojey/`.

## DFT, E1, and E12 rank/storage cross-checks

`crosschecks.py` is a preserved copy of the synthesis audit's read-only script. Run from the project root:

```bash
REPO="$PWD" PYTHONPATH="$PWD/python" \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 results/verification/crosschecks.py
```

The completed output is [crosschecks-output.txt](crosschecks-output.txt). It confirms:

- exact generic-ID DFT storage $2N\log_2N+2N$ and rank-one tiles at $N=256,512,1024,2048$;
- the E12 relative-rank and stacked-QR tables at $\lambda=0,5,80,320$;
- the recorded E1 storage table and DFT strong/complementary and weak/complementary ratios $2.45$ and $4.59$.

## Independent E12 deeper-tree cost check

Run from the project root:

```bash
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 results/verification/check_e12_n4096.py
```

This builds the scalar Oseen area problem at $N=4096$, $\lambda=320$, computes a dense reference solution, and compares strong admissibility with `dir:3,3,1` using exact ID at $\varepsilon=10^{-8}$ and the harness defaults. It writes [e12-n4096.json](e12-n4096.json). The saved run gave:

| Rule | Skeletons | Root | Stored entries | Elimination flops | Forward error | Residual-based backward error |
|---|---:|---:|---:|---:|---:|---:|
| strong | 7007 | 485 | 3,471,408 | $1.6978967507\times10^9$ | $1.45\times10^{-13}$ | $7.26\times10^{-14}$ |
| `dir:3,3,1` | 5490 | 163 | 4,463,489 | $2.8406305947\times10^9$ | $7.75\times10^{-14}$ | $3.87\times10^{-14}$ |

Thus skeletons fall $21.65\%$, while storage rises $28.58\%$ and elimination flops rise $67.30\%$. The root falls to $33.61\%$ of its strong-admissibility size. These are harness cost-model outputs, not wall-clock claims; construction/ID work is not included in the elimination-flop total.

## Standalone algebra checks

From the project root, `python3 results/examples.py` checks the separable-log tail, exact DFT residue-class ranks, a Cooley–Tukey recombination, the scalar transpose/adjoint counterexample, and 2,000 random complex congruence instances. Its passing output is `results/example-output.txt`.
