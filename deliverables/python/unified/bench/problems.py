"""The standard test problems of unified/SPEC.md, built from the core API only.

Shared by the black-box tests (unified/tests/test_spec_*.py) and the benchmarks (unified/bench/run_*.py),
so every number in the README comes from exactly the matrices the tests check.

    P1 curve      starfish r = 1 + 0.3 cos 5θ, A = I + K * w[None, :],  K = laplace2d
    P2 area       N uniform random points in [0,1]^2 (seed 0), A = I + K / N
    P3 Helmholtz  P1 with helmholtz2d(k)  (complex matrix)
    B1 butterfly  fio1d(t) on x = arange(N)/N, xi = arange(N) - N/2

Each 2D problem returns a `Problem` with the dense matrix A, the points X, and `kernel`: a function of
coordinates whose off-diagonal values equal A's off-diagonal entries. That is what a ProxyCompressor needs.
For P1/P3 the column weights w_j are folded into the kernel by recovering θ = atan2(y, x) (the starfish
is star-shaped, so this is exact on the curve); at proxy points it just rescales rows, which does not
change the column space an interpolative decomposition (ID) sees.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import NamedTuple, cast
import numpy as np
from numpy.typing import DTypeLike

from unified.core import kernels
from unified.core.types import (ComplexArray, FrequencyGrid, Kernel, KernelMatrix, Matrix, Points, PositionGrid, RealArray,
                                Vector)

type QuadratureWeights = RealArray   # (N,) w_j: arc-length weights of the trapezoid rule on a curve
type Angles = RealArray              # θ: curve parameter values


@dataclass
class Problem:
    name: str
    X: Points                # (N, 2) points
    A: Matrix                # (N, N) dense system matrix
    kernel: Kernel           # kernel(X, Y) reproducing A's off-diagonal entries from coordinates


class CurveDiscretization(NamedTuple):
    """Nodes and weights of a closed-curve quadrature (unpacks as `X, w = starfish(n)`)."""
    X: Points                # (n, 2) nodes on the curve
    w: QuadratureWeights     # (n,) weights |γ'(θ_j)| 2π/n


class B1Grids(NamedTuple):
    """The B1 grids (unpacks as `x, xi = b1_grids(n)`)."""
    x: PositionGrid          # arange(N)/N
    xi: FrequencyGrid        # arange(N) - N/2 (integer-valued)


# ---- geometry ----------------------------------------------------------------------------------------
def starfish(n: int) -> CurveDiscretization:
    """Points and quadrature weights on the starfish curve r(θ) = 1 + 0.3 cos 5θ (trapezoid rule)."""
    th = 2 * np.pi * np.arange(n) / n
    X: Points = np.c_[_r(th) * np.cos(th), _r(th) * np.sin(th)]
    return CurveDiscretization(X, _speed(th) * 2 * np.pi / n)


def _r(th: Angles) -> RealArray:
    return 1 + 0.3 * np.cos(5 * th)


def _speed(th: Angles) -> RealArray:
    """|γ'(θ)| for γ(θ) = r(θ)(cos θ, sin θ):  sqrt(r^2 + r'^2)."""
    return np.sqrt(_r(th) ** 2 + (1.5 * np.sin(5 * th)) ** 2)


def _weighted(base: Kernel, n: int) -> Kernel:
    """kernel(X, Y) = base(X, Y) * w(Y) with w(y) = |γ'(θ(y))| 2π/n, θ(y) = atan2(y2, y1)."""
    def K(X: Points, Y: Points) -> KernelMatrix:
        Y = np.atleast_2d(Y)
        return base(X, Y) * (_speed(np.arctan2(Y[:, 1], Y[:, 0])) * 2 * np.pi / n)[None, :]
    return K


# ---- P1 / P2 / P3 ----------------------------------------------------------------------------------
def p1_curve(n: int) -> Problem:
    """P1: second-kind-like Laplace system on the starfish curve, A = I + K w."""
    X, w = starfish(n)
    K = _weighted(kernels.laplace2d, n)
    return Problem(f"P1(N={n})", X, dense_identity_plus(K, X), K)


def p2_area(n: int, seed: int = 0) -> Problem:
    """P2: N uniform random points in the unit square, A = I + K / N (volume problem)."""
    X = np.random.default_rng(seed).random((n, 2))
    K: Kernel = lambda P, Q: kernels.laplace2d(P, Q) / n
    return Problem(f"P2(N={n})", X, dense_identity_plus(K, X), K)


def p3_helmholtz(n: int, k: float) -> Problem:
    """P3: P1 with the Helmholtz kernel (i/4) H0(k r): complex, nonsymmetric after weighting."""
    X, _ = starfish(n)
    K = _weighted(kernels.helmholtz2d(k), n)
    return Problem(f"P3(N={n},k={k})", X, dense_identity_plus(K, X, complex), K)


def clustered(n: int, seed: int = 0) -> Problem:
    """Edge case: points in three tight Gaussian clusters plus a few outliers -> many empty boxes and
    leaves of very different occupancy. A = I + K / N, K = laplace2d."""
    rng = np.random.default_rng(seed)
    centers = np.array([[0.1, 0.1], [0.8, 0.3], [0.4, 0.9]])
    m = n - 8
    X: Points = np.vstack([centers[i % 3] + 0.01 * rng.standard_normal(2) for i in range(m)] + [rng.random((8, 2))])
    A: Matrix = np.eye(n) + kernels.laplace2d(X, X) / n
    return Problem(f"clustered(N={n})", X, A, lambda P, Q: kernels.laplace2d(P, Q) / n)


# ---- B1 ----------------------------------------------------------------------------------------------
def b1_grids(n: int) -> B1Grids:
    """B1 grids: positions x = arange(N)/N and integer frequencies xi = arange(N) - N/2."""
    xi = cast(FrequencyGrid, np.arange(n) - n // 2)   # integer-valued; consumers convert with asarray(xi, float)
    return B1Grids(np.arange(n) / n, xi)


def b1_matrix(n: int, t: float) -> ComplexArray:
    """Dense B1 matrix K[x, xi] = exp(2πi (x xi + t c(x) |xi|)). t = 0 is a shifted inverse DFT."""
    x, xi = b1_grids(n)
    return kernels.fio1d(t)(x, xi)


def dense_identity_plus(kernel: Kernel, X: Points, dtype: DTypeLike = float, chunk: int = 1024) -> Matrix:
    """A = I + kernel(X, X), built in row blocks so that N = 16384 needs only the N x N result
    (~2 GB real) and not the (N, N, 2) distance temporaries. The kernels have a zero diagonal."""
    n = len(X)
    A: Matrix = np.empty((n, n), dtype)
    for i in range(0, n, chunk):
        A[i:i + chunk] = kernel(X[i:i + chunk], X)
    A[np.diag_indices(n)] += 1
    return A


def relres_kernel(prob: Problem, x: Vector, b: Vector, chunk: int = 1024) -> np.floating:
    """||A x - b|| / ||b|| for A = I + kernel(X, X), recomputing A row block by row block.
    Used after factoring with overwrite_a=True, when the dense A has been overwritten."""
    r: Vector = x - b
    for i in range(0, len(b), chunk):
        rows = slice(i, i + chunk)
        r[rows] = r[rows] + prob.kernel(prob.X[rows], prob.X) @ x
    res: np.floating = np.linalg.norm(r) / np.linalg.norm(b)
    return res


def relres(A: Matrix, x: Vector, b: Vector) -> np.floating:
    """Relative residual ||A x - b|| / ||b||."""
    res: np.floating = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
    return res
