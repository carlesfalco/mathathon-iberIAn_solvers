"""E8 physics: the fractional Laplacian and the Riesz potential as dense point-set matrices.

Two operators, both built from ONE power-law kernel  k_p(x, y) = |x - y|^{-p}:

  hypersingular  p = d + 2s   the fractional Laplacian (-Lap)^s u(x) = C_{d,s} PV int (u(x) - u(y)) / |x-y|^{d+2s} dy
  Riesz          p = d - 2s   its inverse on R^d (Riesz potential I_{2s}), weakly singular, 0 < 2s < d

Fractional Poisson problem (the physics test):  (-Lap)^s u = 1 in the unit ball B of R^d, u = 0 outside B.
Exact solution (Getoor 1961; Dyda 2012):  u(x) = Gamma(d/2) / (4^s Gamma(1+s) Gamma(d/2+s)) * (1 - |x|^2)_+^s.

Discretisation (Riemann-sum "lattice quadrature", the simplest consistent scheme): lattice points x_i = h z_i,
z_i in Z^d, inside B. Because u = 0 on every lattice point outside B, the full-lattice sum gives
    (A u)_i = C_{d,s} h^d [ Z_d(s) h^{-(d+2s)} u_i - sum_{j != i, x_j in B} |x_i - x_j|^{-(d+2s)} u_j ],
    Z_d(s) = sum_{z in Z^d, z != 0} |z|^{-(d+2s)}   (Z_1 = 2 zeta(1+2s),  Z_2 = 4 zeta(1+s) beta(1+s)).
We divide by C_{d,s} h^{-2s}, so the matrix is dimensionless and O(1):
    A = Z_d(s) I - [ (h / |x_i - x_j|)^{d+2s} ]_{i != j},      right-hand side  b = h^{2s} f / C_{d,s}.
A is symmetric, strictly diagonally dominant (row sums of the off-diagonal part < Z_d), hence SPD.
Its off-diagonal entries ARE kernel values, so a proxy compressor may evaluate the kernel at proxy points.

Riesz second-kind test operator (a compression probe, NOT a physical model): A = I + h^d |x_i - x_j|^{-(d-2s)}
(zero diagonal), the analogue of SPEC P2 (P2 is its d = 2, s -> 1 logarithmic limit). The continuous Riesz
kernel is positive definite for 0 < 2s < d, but the zero-diagonal quadrature is NOT guaranteed to be: at
d = 2, s = 0.1 (p = 1.8), N = 2053 the matrix is indefinite (lambda_min = -0.083, cond 2.1e5). It is SPD for
s >= 0.25 in our runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import gamma, zeta

from unified.core.types import Kernel, KernelMatrix, Matrix, Points, RealArray, Vector

type Dim = Literal[1, 2]
type Order = float               # s in (0, 1): order of the fractional Laplacian
type Exponent = float            # p in k_p(r) = r^{-p}
type Family = Literal["hyper", "riesz"]


# ---- constants ---------------------------------------------------------------------------------------------
def c_ds(d: int, s: Order) -> float:
    """C_{d,s} = 4^s Gamma(d/2 + s) / (pi^{d/2} |Gamma(-s)|): normalises (-Lap)^s to symbol |xi|^{2s}."""
    return float(4 ** s * gamma(d / 2 + s) / (np.pi ** (d / 2) * abs(gamma(-s))))


def lattice_zeta(d: int, s: Order) -> float:
    """Z_d(s) = sum over nonzero z in Z^d of |z|^{-(d+2s)}: the diagonal of the lattice-quadrature matrix."""
    if d == 1:
        return float(2 * zeta(1 + 2 * s))
    if d == 2:   # sum_n r_2(n) n^{-sigma} = 4 zeta(sigma) beta(sigma), beta(sigma) = 4^-sigma (zeta(sigma,1/4) - zeta(sigma,3/4))
        sig = 1 + s
        beta = 4.0 ** (-sig) * (zeta(sig, 0.25) - zeta(sig, 0.75))
        return float(4 * zeta(sig) * beta)
    raise ValueError(d)


def dirichlet_beta(x: float, n: int = 40) -> float:
    """beta(x) = sum_k (-1)^k (2k+1)^{-x}, valid for all real x (Cohen-Villegas-Zagier acceleration)."""
    dd = (3 + np.sqrt(8)) ** n
    dd = (dd + 1 / dd) / 2
    b, c, acc = -1.0, -dd, 0.0
    for k in range(n):
        c = b - c
        acc += c * (2 * k + 1.0) ** (-x)
        b = (k + n) * (k - n) * b / ((k + 0.5) * (k + 1))
    return float(acc / dd)


def taylor_coef(d: int, s: Order) -> float:
    """kappa_d(s): (lattice sum - integral) of the quadratic Taylor term, in units of h^{2-2s} (-Lap u).
    Zeta-regularised lattice sums: kappa_1 = zeta(2s-1), kappa_2 = zeta(s) beta(s). Both are < 0 on (0, 1)."""
    if d == 1:
        return float(zeta(2 * s - 1))
    if d == 2:
        return float(zeta(s)) * dirichlet_beta(s)
    raise ValueError(d)


def neighbour_laplacian(X: Points, h: float) -> Matrix:
    """h^2 (-Lap_h) on the lattice with zero exterior values: 2d on the diagonal, -1 for each lattice neighbour."""
    r = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    L: Matrix = 2 * X.shape[1] * np.eye(len(X)) - (np.abs(r - h) < 1e-9 * h)
    return L


def exact_u(X: Points, s: Order) -> Vector:
    """u(x) = Gamma(d/2) / (4^s Gamma(1+s) Gamma(d/2+s)) (1 - |x|^2)_+^s, the solution for f = 1 on the unit ball."""
    d = X.shape[1]
    c = gamma(d / 2) / (4 ** s * gamma(1 + s) * gamma(d / 2 + s))
    u: Vector = c * np.maximum(1 - np.sum(X ** 2, axis=1), 0.0) ** s
    return u


# ---- the one kernel (dimension-agnostic, so it also accepts LIFTED proxy points) -------------------------
def power_kernel(p: Exponent, scale: float) -> Kernel:
    """K(X, Y) = scale * |x - y|^{-p}, zero where x = y. X and Y may have different numbers of columns:
    the shorter one is padded with zeros, i.e. points of R^d are embedded in R^{d+1} at t = 0.
    That embedding is all a lifted proxy needs (the kernel restricted to t = 0 is unchanged)."""
    def K(X: Points, Y: Points) -> KernelMatrix:
        A = np.atleast_2d(np.asarray(X, float)); B = np.atleast_2d(np.asarray(Y, float))
        D = max(A.shape[1], B.shape[1])
        A = np.pad(A, ((0, 0), (0, D - A.shape[1]))); B = np.pad(B, ((0, 0), (0, D - B.shape[1])))
        r = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
        out: KernelMatrix = scale * np.where(r > 0, r, 1.0) ** (-p)
        out[r == 0] = 0.0
        return out
    return K


# ---- point sets -------------------------------------------------------------------------------------------
def ball_lattice(d: Dim, n: int) -> tuple[Points, float]:
    """Lattice points h Z^d strictly inside the unit ball, about n of them. Returns (X, h)."""
    if d == 1:
        h = 2.0 / (n + 1)
        x: Points = (-1 + h * np.arange(1, n + 1, dtype=np.float64))[:, None]
        return x, h
    h = float(np.sqrt(np.pi / n))
    m = int(np.ceil(1 / h))
    g = h * np.arange(-m, m + 1)
    G = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    X: Points = G[np.sum(G ** 2, axis=1) < 1 - 1e-12]
    return X, h


@dataclass
class FracProblem:
    """Dense A, its points and kernel (off-diagonal entries of A = kernel(X, X)), right-hand side and exact u."""
    name: str
    family: Family
    d: int
    s: Order
    p: Exponent
    h: float
    X: Points
    A: Matrix
    kernel: Kernel
    b: Vector
    u_exact: Vector | None      # continuous exact solution at the points (fractional Poisson only)


def frac_poisson(d: Dim, s: Order, n: int, corrected: bool = True) -> FracProblem:
    """(-Lap)^s u = 1 on the unit ball, lattice quadrature, scaled as in the module docstring.
    corrected: add -kappa_d(s) * h^2(-Lap_h), the Taylor correction of the singular cell (raises the
    consistency order from h^{2-2s} to O(h)-ish in practice). It changes ONLY nearest-neighbour entries, so
    off-diagonal entries between points more than h apart are still kernel values."""
    X, h = ball_lattice(d, n)
    p = d + 2 * s
    K = power_kernel(p, -h ** p)
    A: Matrix = K(X, X) + lattice_zeta(d, s) * np.eye(len(X))
    if corrected:
        A = A - taylor_coef(d, s) * neighbour_laplacian(X, h)
    b: Vector = np.full(len(X), h ** (2 * s) / c_ds(d, s))
    return FracProblem(f"fracpoisson(d={d},s={s:g},N={len(X)})", "hyper", d, s, p, h, X, A, K, b, exact_u(X, s))


def riesz_second_kind(d: Dim, s: Order, n: int) -> FracProblem:
    """A = I + h^d |x - y|^{-(d-2s)} on the same lattice (needs 2s < d), b = 1. Not always SPD (see module doc)."""
    if not 0 < 2 * s < d:
        raise ValueError("Riesz potential kernel needs 0 < 2s < d")
    X, h = ball_lattice(d, n)
    p = d - 2 * s
    K = power_kernel(p, h ** d)
    A: Matrix = K(X, X) + np.eye(len(X))
    return FracProblem(f"riesz2k(d={d},s={s:g},N={len(X)})", "riesz", d, s, p, h, X, A, K, np.ones(len(X)), None)


def log_kernel(scale: float) -> Kernel:
    """K = -scale * log|x - y| (the p -> 0 member of the power family), dimension-agnostic like power_kernel."""
    def K(X: Points, Y: Points) -> KernelMatrix:
        A = np.atleast_2d(np.asarray(X, float)); B = np.atleast_2d(np.asarray(Y, float))
        D = max(A.shape[1], B.shape[1])
        A = np.pad(A, ((0, 0), (0, D - A.shape[1]))); B = np.pad(B, ((0, 0), (0, D - B.shape[1])))
        r = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
        out: KernelMatrix = -scale * np.log(np.where(r > 0, r, 1.0))
        return out
    return K


def power_second_kind(d: Dim, p: Exponent, n: int) -> FracProblem:
    """Break-test operator A = I + h^d k_p (zero diagonal) for ANY exponent p: p < 0 gives growing kernels
    |x - y|^{|p|} (no decay at infinity, so the lifted-proxy maximum-principle argument does not apply),
    p = 0 the logarithm. s is reported as (d - p) / 2 (the Riesz order that would give this p)."""
    X, h = ball_lattice(d, n)
    K = log_kernel(h ** d) if p == 0 else power_kernel(p, h ** d)
    A: Matrix = K(X, X) + np.eye(len(X))
    return FracProblem(f"power2k(d={d},p={p:g},N={len(X)})", "riesz", d, (d - p) / 2, p, h, X, A, K,
                       np.ones(len(X)), None)


def make(family: Family, d: Dim, s: Order, n: int) -> FracProblem:
    return frac_poisson(d, s, n) if family == "hyper" else riesz_second_kind(d, s, n)


def rel_err(x: ArrayLike, ref: ArrayLike) -> float:
    a = np.asarray(x); r = np.asarray(ref)
    return float(np.linalg.norm(a - r) / np.linalg.norm(r))


def linf_err(x: RealArray, ref: RealArray) -> float:
    return float(np.max(np.abs(x - ref)) / np.max(np.abs(ref)))
