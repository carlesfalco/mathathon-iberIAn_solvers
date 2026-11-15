"""Dump the per-box skeletonization decisions of the Python driver, for diffing against the Rust port."""
import sys
import numpy as np
sys.path.insert(0, "../../../updated")
from unified.core import Tree, Weak, Strong, ExactCompressor, ProxyCompressor, kernels
import unified.drivers.rskel as R

adm = {"weak": Weak(), "strong": Strong()}[sys.argv[1]]
N, ml, eps = 256, 8, 1e-8
th = 2 * np.pi * np.arange(N) / N
r = 1 + 0.3 * np.cos(5 * th)
X = np.c_[r * np.cos(th), r * np.sin(th)]
w = np.sqrt(r ** 2 + (-1.5 * np.sin(5 * th)) ** 2) * 2 * np.pi / N
A = np.eye(N) + kernels.laplace2d(X, X) * w[None, :]
orig = R._skeletonize


def wrap(compressor, W, box, far, node, q_extra):
    res = orig(compressor, W, box, far, node, q_extra)
    print(f"step l={node.level} c={'_'.join(map(str, node.coord))} box={len(box)} near_box=? "
          f"far={len(far)} q={len(q_extra)} k={len(res.skel)} red={len(res.red)}")
    return res


R._skeletonize = wrap
F = R.rskel_factor(A, Tree(X, max_leaf=ml), adm, ExactCompressor(A, eps))
print("root", F.root.size, "nsteps", len(F.steps), "stats", F.stats["levels"])
