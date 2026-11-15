"""Dump one real ID block from the Python driver: the level-1 box of p1 (N=256, depth 2, weak/exact),
the ID's k and the achieved relative ID error.  Used to test a hand-written QR against LAPACK dgeqp3."""
import sys
import numpy as np
sys.path.insert(0, "../../../updated")
from unified.core import Tree, Weak, ExactCompressor, kernels
import unified.drivers.rskel as R

N, eps = 256, 1e-8
th = 2 * np.pi * np.arange(N) / N
r = 1 + 0.3 * np.cos(5 * th)
X = np.c_[r * np.cos(th), r * np.sin(th)]
w = np.sqrt(r ** 2 + (-1.5 * np.sin(5 * th)) ** 2) * 2 * np.pi / N
A = np.eye(N) + kernels.laplace2d(X, X) * w[None, :]
orig = R._skeletonize
saved = {}


def wrap(compressor, W, box, far, node, q_extra):
    res = orig(compressor, W, box, far, node, q_extra)
    if node.level == 1 and "done" not in saved:
        saved["done"] = True
        Wc = np.array(W, copy=True)
        M = np.vstack([Wc[np.ix_(far, box)], Wc[np.ix_(box, far)].T])
        sk, rd, T = res
        err = np.linalg.norm(M[:, rd] - M[:, sk] @ T) / np.linalg.norm(M)
        np.savetxt("block_M.txt", M, fmt="%.17g")
        print(f"python: shape {M.shape} k={len(sk)} id_relerr={err:.3e} cond={np.linalg.cond(M):.3e}")
        print("pivots", sk.tolist())
    return res


R._skeletonize = wrap
R.rskel_factor(A, Tree(X, depth=2), Weak(), ExactCompressor(A, eps))
