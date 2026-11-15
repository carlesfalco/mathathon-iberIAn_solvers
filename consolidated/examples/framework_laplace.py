#!/usr/bin/env python3
"""Minimal Section 3 example: map framework levers to the shipped Python API."""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from unified.core import ProxyCompressor, Tree, Weak
from unified.bench import problems as P
from unified.drivers.rskel import rskel_factor
import numpy as np


def main() -> None:
    # Problem-specific physics and kernel: supplied by the application.
    problem = P.p1_curve(512)

    # Lever 1: spatial partition; lever 2: weak off-diagonal interactions.
    tree = Tree(problem.X, max_leaf=32)
    admissibility = Weak()

    # Lever 3: interpolative representation; lever 4: proxy sampling.
    compressor = ProxyCompressor(problem.kernel, problem.X, 1e-8, A=problem.A)

    # Lever 5: recursive sparsification/elimination, independent of the API above.
    factor = rskel_factor(problem.A, tree, admissibility, compressor)
    b = np.ones(512)
    x = factor.solve(b)

    # Verify with the original kernel, not only with the approximate factor.
    residual = P.relres_kernel(problem, x, b)
    print(f"N=512, weak proxy recursive solve; independent kernel residual={residual:.6e}")
    assert residual < 1e-7


if __name__ == "__main__":
    main()
