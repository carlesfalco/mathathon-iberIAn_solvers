"""Wall-clock of the Python drivers on the same cases the Rust prototype benchmarks (this VM, one run)."""
import sys, time
import numpy as np
sys.path.insert(0, "../../../updated")
from unified.core import Tree, Strong, Weak, ExactCompressor, kernels
from unified.drivers import rskel, butterfly_id as bid

print("python", sys.version.split()[0], "numpy", np.__version__)
for N in (512, 1024):
    th = 2 * np.pi * np.arange(N) / N
    r = 1 + 0.3 * np.cos(5 * th)
    X = np.c_[r * np.cos(th), r * np.sin(th)]
    w = np.sqrt(r ** 2 + (-1.5 * np.sin(5 * th)) ** 2) * 2 * np.pi / N
    A = np.eye(N) + kernels.laplace2d(X, X) * w[None, :]
    t0 = time.perf_counter()
    F = rskel.rskel_factor(A, Tree(X, max_leaf=32), Strong(), ExactCompressor(A, 1e-8))
    t_fac = time.perf_counter() - t0
    b = np.sin(3.0 * np.arange(N)) + 0.5
    t0 = time.perf_counter(); x = F.solve(b); t_sol = time.perf_counter() - t0
    res = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
    print(f"python p1 strong exact N={N}: factor {t_fac:.3f} s, solve {t_sol:.4f} s, "
          f"root {F.root.size}, relres {res:.2e}")
for N in (512,):
    kernel, x, xi = bid.b1_problem(N, 1.0)
    t0 = time.perf_counter()
    bf = bid.butterfly_id(kernel, x, xi, 1e-8)
    print(f"python butterfly build N={N}: {time.perf_counter() - t0:.3f} s, nnz {bf.nnz}, ranks {bf.ranks}")
