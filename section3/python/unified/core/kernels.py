"""Kernels K(X, Y) -> matrix with entries K(x_i, y_j). Diagonal (x = y) entries are set to 0;
drivers add their own diagonal / quadrature terms."""
from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import hankel1

from .types import ComplexArray, FrequencyGrid, Kernel, PositionGrid, RealArray

type SpeedProfile = Callable[[RealArray], RealArray]                 # c(x): local wave speed
type ButterflyKernel = Callable[[PositionGrid, FrequencyGrid], ComplexArray]


def _dist(X: ArrayLike, Y: ArrayLike) -> RealArray:
    """Pairwise Euclidean distances |x_i - y_j|."""
    A = np.asarray(X, float); B = np.asarray(Y, float)
    A = A[:, None] if A.ndim == 1 else A
    B = B[:, None] if B.ndim == 1 else B
    r: RealArray = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
    return r


def laplace2d(X: ArrayLike, Y: ArrayLike) -> RealArray:
    """-1/(2 pi) log|x - y|  (2D Laplace Green's function)."""
    r = _dist(X, Y)
    with np.errstate(divide="ignore"):
        K = -np.log(r) / (2 * np.pi)
    K[r == 0] = 0.0
    return K


def helmholtz2d(k: float) -> Kernel:
    """(i/4) H0^(1)(k |x - y|)  (2D Helmholtz Green's function), returned as a kernel function."""
    def K(X: ArrayLike, Y: ArrayLike) -> ComplexArray:
        r = _dist(X, Y)
        out = np.zeros(r.shape, complex)
        nz = r > 0
        out[nz] = 0.25j * hankel1(0, k * r[nz])
        return out
    return K


def _sine_speed(x: RealArray) -> RealArray:
    return (2 + np.sin(2 * np.pi * x)) / 16


def fio1d(t: float = 1.0, c: SpeedProfile = _sine_speed) -> ButterflyKernel:
    """Butterfly test kernel: K(x, xi) = exp(2 pi i (x xi + t c(x) |xi|)), x in [0,1), xi integer frequencies."""
    def K(x: PositionGrid, xi: FrequencyGrid) -> ComplexArray:
        xr = np.asarray(x, float).ravel(); xir = np.asarray(xi, float).ravel()
        return np.exp(2j * np.pi * (np.outer(xr, xir) + t * c(xr)[:, None] * np.abs(xir)[None, :]))
    return K
