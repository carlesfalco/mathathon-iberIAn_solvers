"""E3 test inputs beyond SPEC P1-P3. Every problem is a unified.bench.problems.Problem (A, X, kernel) with
A = I + (off-diagonal kernel values), so ProxyCompressor can evaluate the same kernel at proxy points.

PDE kernels (Green's representation holds outside a box):
    p1, p2, p3                SPEC problems (Laplace curve / area, Helmholtz H0 curve)
    helm_area(k, which)       area points, kernel c * G(k r) with G in {H0 (i/4 H0^(1)), Y0 (-Y0/4), J0 (J0/4)}
                              H0 and Y0 are fundamental solutions; J0 is a smooth Helmholtz solution.
    helm_curve(k, which)      the same on the starfish curve (weights folded in, like P3)
    advdiff(b)                Green's function of -Lap u + beta . grad u, beta = (b, 0): NONSYMMETRIC
    double_layer              Laplace double layer on the starfish: NONSYMMETRIC (harmonic in both variables)
Non-PDE kernels (no Green's representation; proxy has no reason to work):
    rbf(sigma)                Gaussian exp(-r^2 / (2 sigma^2))
    cauchy(c)                 1 / (1 + (r/c)^2)
    skew_cauchy(c)            cauchy * (1 + 0.5 (x1 - y1)): NONSYMMETRIC, non-PDE
"""
from __future__ import annotations
from collections.abc import Callable
from typing import Literal

import numpy as np
from scipy.special import hankel1, y0, j0, k0
from . import _deps as P
from unified.core import kernels
from unified.core.types import Kernel, KernelMatrix, Points, RealArray

type Wavenumber = float
type HelmholtzFamily = Literal["H0", "Y0", "J0"]
type Distances = RealArray                                         # (m, n) matrix of |x_i - y_j|
type KernelBody = Callable[[Points, Points, Distances], KernelMatrix]  # f(X, Y, r) before zeroing the diagonal


def _d(X: Points, Y: Points) -> Distances:
    return kernels._dist(X, Y)


def _nodiag(f: KernelBody) -> Kernel:
    """Kernel from a body f(X, Y, r), with entries at r = 0 set to 0 (drivers add the diagonal)."""
    def K(X: Points, Y: Points) -> KernelMatrix:
        r = _d(X, Y)
        out = f(np.atleast_2d(X), np.atleast_2d(Y), r)
        out[r == 0] = 0
        return out
    return K


def helm_base(k: Wavenumber, which: HelmholtzFamily) -> Kernel:
    if which == "H0":
        return _nodiag(lambda X, Y, r: 0.25j * hankel1(0, k * np.where(r > 0, r, 1)))
    if which == "Y0":
        return _nodiag(lambda X, Y, r: -0.25 * y0(k * np.where(r > 0, r, 1)))
    if which == "J0":
        return _nodiag(lambda X, Y, r: 0.25 * j0(k * r))
    raise ValueError(which)


def _area(name: str, n: int, base: Kernel, dtype: type = float, seed: int = 0) -> P.Problem:
    X = np.random.default_rng(seed).random((n, 2))
    def K(Pp: Points, Q: Points) -> KernelMatrix:
        return base(Pp, Q) / n
    return P.make_problem(name, X, P.dense_identity_plus(K, X, dtype), K)


def _curve(name: str, n: int, base: Kernel, dtype: type = float) -> P.Problem:
    X, _ = P.starfish(n)
    K = P.weighted(base, n)
    return P.make_problem(name, X, P.dense_identity_plus(K, X, dtype), K)


def helm_area(n: int, k: Wavenumber, which: HelmholtzFamily = "H0") -> P.Problem:
    return _area(f"helm_area[{which}](N={n},k={k:.4g})", n, helm_base(k, which), complex if which == "H0" else float)


def helm_curve(n: int, k: Wavenumber, which: HelmholtzFamily = "H0") -> P.Problem:
    return _curve(f"helm_curve[{which}](N={n},k={k:.4g})", n, helm_base(k, which), complex if which == "H0" else float)


def advdiff(n: int, b: float = 20.0) -> P.Problem:
    """G(x, y) = exp(beta.(x-y)/2) K0(|beta| r / 2) / (2 pi): fundamental solution of -Lap + beta.grad."""
    def base(X: Points, Y: Points, r: Distances) -> KernelMatrix:
        dx = X[:, None, 0] - Y[None, :, 0]
        return np.exp(b * dx / 2) * k0(abs(b) * np.where(r > 0, r, 1) / 2) / (2 * np.pi)
    return _area(f"advdiff(N={n},b={b:g})", n, _nodiag(base))


def double_layer(n: int) -> P.Problem:
    """Laplace double layer D(x, y) = n_y . (x - y) / (2 pi |x - y|^2) on the starfish, A = I + D w.
    The normal n_y is recovered from y via theta = atan2 (exact on the curve, like P1's weights)."""
    def normal(Y: Points) -> Points:
        th = np.arctan2(Y[:, 1], Y[:, 0])
        r, dr = 1 + 0.3 * np.cos(5 * th), -1.5 * np.sin(5 * th)
        t = np.c_[dr * np.cos(th) - r * np.sin(th), dr * np.sin(th) + r * np.cos(th)]
        t /= np.linalg.norm(t, axis=1)[:, None]
        nrm: Points = np.c_[t[:, 1], -t[:, 0]]
        return nrm

    def base(X: Points, Y: Points, r: Distances) -> KernelMatrix:
        nY = normal(Y)
        diff = X[:, None, :] - Y[None, :, :]
        D: KernelMatrix = np.einsum("ijk,jk->ij", diff, nY) / (2 * np.pi * np.where(r > 0, r, 1) ** 2)
        return D
    return _curve(f"double_layer(N={n})", n, _nodiag(base))


def rbf(n: int, sigma: float = 0.2) -> P.Problem:
    return _area(f"rbf(N={n},s={sigma:g})", n, _nodiag(lambda X, Y, r: np.exp(-r ** 2 / (2 * sigma ** 2))))


def cauchy(n: int, c: float = 0.2) -> P.Problem:
    return _area(f"cauchy(N={n},c={c:g})", n, _nodiag(lambda X, Y, r: 1 / (1 + (r / c) ** 2)))


def skew_cauchy(n: int, c: float = 0.2) -> P.Problem:
    def base(X: Points, Y: Points, r: Distances) -> KernelMatrix:
        return (1 + 0.5 * (X[:, None, 0] - Y[None, :, 0])) / (1 + (r / c) ** 2)
    return _area(f"skew_cauchy(N={n},c={c:g})", n, _nodiag(base))
