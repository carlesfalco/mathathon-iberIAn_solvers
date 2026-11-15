"""Independent cross-checks for the m8 synthesis audit (read-only w.r.t. updated/).

Run:  REPO=/path/to/repo PYTHONPATH=/path/to/repo/updated python3 crosschecks.py

A. DFT-as-butterfly entry count.  Claim (showcase dft-as-butterfly invariant; deck slide 24): the
   ID butterfly reproduces the radix-2 FFT entry count 2 N log2 N + 2 N exactly, every tile rank 1.
   This replicates the showcase construction: bit-reversal is realised as a *permuted kernel* on
   identity grids, not as a bit-reversed tree.
B. One-interpolation-matrix penalty for nonsymmetric kernels.  E12 R1: with the driver's stacked
   ID [K(F,B); K(B,F)^T] the rank is about 2x the single-block rank for lambda >= 1, while the
   *relative* eps-rank of one block is nearly direction-free.  This re-runs E12's own functions
   (unified.explore.e12_oseen.directional) on a grid of directions and lambda.
C. E1 storage ratios at N = 4096 recomputed from results/sweep.json (deck slide 23).
"""
from __future__ import annotations

import json
import os

import numpy as np

REPO = os.environ.get("REPO", os.getcwd())
UPD = os.environ.get("UNIFIED_ROOT", os.path.join(REPO, "python"))

from unified.core import kernels                                        # noqa: E402
from unified.bench import problems as P                                 # noqa: E402
from unified.drivers.butterfly_id import butterfly_id                    # noqa: E402
from unified.explore.e2_fft_replacements import fftswap                  # noqa: E402
from unified.explore.e12_oseen.oseen import scalar_oseen                 # noqa: E402
from unified.explore.e12_oseen.directional import block_rank, stacked_id_rank  # noqa: E402


def check_a() -> None:
    print("== A. DFT as butterfly (bit-reversed via permuted kernel), eps = 1e-10")
    rng = np.random.default_rng(0)
    for n in (256, 512, 1024, 2048):
        x, xi = P.b1_grids(n)
        x = np.asarray(x, float); xi = np.asarray(xi, float)
        perm = fftswap.bitrev(n)
        u = np.arange(n, dtype=float)
        g = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        bf = butterfly_id(fftswap.permuted_kernel(kernels.fio1d(0.0), x, xi, perm, perm), u, u, 1e-10)
        f = np.empty(n, complex)
        f[perm] = bf.apply(g[perm])
        exact = (-1.0) ** np.arange(n) * n * np.fft.ifft(g)
        err = np.linalg.norm(f - exact) / np.linalg.norm(exact)
        want = 2 * n * int(np.log2(n)) + 2 * n
        print(f"   n={n:5d} nnz={bf.nnz:7d} 2nlog2n+2n={want:7d} equal={bf.nnz == want} "
              f"max_rank={max(bf.ranks)} err_vs_fft={err:.1e}")


def check_b() -> None:
    print("== B. E12 R1 replication: w = 1/8, D = 2w, eps = 1e-8, 300 points per box")
    for lam in (0.0, 5.0, 80.0, 320.0):
        K = scalar_oseen(lam)
        rels, sts = [], []
        for th in (0.0, 0.5 * np.pi, np.pi):
            rel, ab, nrm = block_rank(K, 1 / 8, 2 / 8, float(th), 1e-8)
            st = stacked_id_rank(K, 1 / 8, 2 / 8, float(th), 1e-8)
            rels.append(rel); sts.append(st)
        print(f"   lam={lam:6g} rel(0/90/180)={rels} stacked(0/90/180)={sts} "
              f"stacked/max(rel)={max(sts) / max(rels):.2f}")


def check_c() -> None:
    print("== C. E1 storage ratios at N=4096, eps=1e-6 (deck slide 23)")
    f = os.path.join(UPD, "unified/explore/e1_which_structure/results/sweep.json")
    d = json.load(open(f))
    for row in d["rows"]:
        if row["N"] != 4096:
            continue
        r = row["results"]
        weak, strong, comp = r["weak"]["storage"], r["strong"]["storage"], r["complementary"]["storage"]
        print(f"   {row['kernel']:26s} weak/N={weak / 4096:7.0f} strong/N={strong / 4096:7.0f} "
              f"comp/N={comp / 4096:7.0f} strong/comp={strong / comp:5.2f} weak/comp={weak / comp:5.2f}")


if __name__ == "__main__":
    check_a()
    check_b()
    check_c()
