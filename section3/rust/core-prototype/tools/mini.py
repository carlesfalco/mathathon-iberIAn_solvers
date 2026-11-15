"""Smallest 2-D recursive-skeletonization case: p1, depth 1, weak/exact, N=64 -> residual + steps."""
import sys
import numpy as np
sys.path.insert(0, "../../../updated")
from unified.core import Tree, Weak, Strong, ExactCompressor, kernels
import unified.drivers.rskel as R

N = int(sys.argv[2]) if len(sys.argv) > 2 else 64
th = 2 * np.pi * np.arange(N) / N
r = 1 + 0.3 * np.cos(5 * th)
X = np.c_[r * np.cos(th), r * np.sin(th)]
w = np.sqrt(r ** 2 + (-1.5 * np.sin(5 * th)) ** 2) * 2 * np.pi / N
A = np.eye(N) + kernels.laplace2d(X, X) * w[None, :]
orig = R._skeletonize


def wrap(compressor, W, box, far, node, q_extra):
    res = orig(compressor, W, box, far, node, q_extra)
    print(f"step l={node.level} c={'_'.join(map(str, node.coord))} box={len(box)} far={len(far)} "
          f"q={len(q_extra)} k={len(res.skel)} red={len(res.red)}")
    return res


R._skeletonize = wrap
adm = Weak() if sys.argv[1] == "weak" else Strong()
F = R.rskel_factor(A, Tree(X, depth=int(sys.argv[3]) if len(sys.argv) > 3 else 1), adm, ExactCompressor(A, 1e-8))
b = np.sin(3.0 * np.arange(N)) + 0.5
print("root", F.root.size, "nsteps", len(F.steps))
print("relres", np.linalg.norm(A @ F.solve(b) - b) / np.linalg.norm(b))
