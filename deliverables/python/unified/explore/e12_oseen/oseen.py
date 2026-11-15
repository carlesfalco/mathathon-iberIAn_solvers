"""E12 physics set-up: scalar and tensor 2D Oseen kernels, an area test matrix, and the Oseen drag on a cylinder.

Scalar model (advection-diffusion with a uniform stream along +x, "Oseen parameter" lam = U/(2 D)):
    -Lap u + 2 lam d_x u = delta     =>   G(x) = e^{lam x_1} K_0(lam |x|) / (2 pi)          [proved]
  (substitute u = e^{lam x_1} v: then (lam^2 - Lap) v = delta, the Yukawa equation). So
    G(x - y) = e^{lam x_1} * K_0(lam |x - y|)/(2 pi) * e^{-lam y_1}:
  every block of the Oseen matrix is a DIAGONAL SCALING of the same Yukawa block (same exact rank) [proved].
  Evaluated as kve(0, z) e^{lam (x_1 - r)} to avoid overflow/underflow.

Tensor Oseen ("Oseenlet", viscosity mu, Oseen parameter k = U/(2 nu)): velocity of
    mu (-Lap + 2k d_1) u + grad p = f delta,  div u = 0.
  With E = e^{k x_1} K_0(k r)/(2 pi) (the scalar G at lam = k) and Phi = log r/(2 pi) (Delta Phi = delta):
    p = f . grad Phi,  u_i = S_ij f_j,
    S_11 = (E - d_1(Phi + E)/(2k))/mu,  S_12 = S_21 = -d_2(Phi + E)/(2k mu),  S_22 = d_1(Phi + E)/(2k mu).
  Derivation (ours, checked in the tests): write u = (delta_ij A - d_i d_j B) f_j / mu. Incompressibility needs
  Lap B = A; the momentum equation needs L A = delta and L B = Phi with L = -Lap + 2k d_1; then
  2k d_1 B = Phi + E. As k -> 0, S -> Stokeslet (-log r I + x x^T/r^2)/(4 pi mu) + constant [tested].
  Near r = 0 (derived by series, tested):  S = Stokeslet + C + O(r log r),  C = diag(-c0, -1 - c0)/(4 pi mu),
  c0 = log(k/2) + gamma.

Drag check: circular cylinder of radius a at rest in a stream U e_1. Disturbance u' = -U e_1 on the body,
single-layer representation u' = int S(x - y) f(y) ds_y (f = force density on the fluid), drag = -int f_1 ds.
Lamb (1911): drag per unit length = 4 pi mu U / (1/2 - gamma - log(k a / 2))   (leading order as ka -> 0).
Quadrature: trapezoid rule in arclength for j != i; the 2x2 diagonal block uses the limits above plus the exact
circle identity int log|x_i - y| ds_y = 2 pi a log a (punctured trapezoid with a constant-density correction).
Only DIAGONAL 2x2 blocks are corrected, so every off-diagonal entry is a kernel value (the proxy needs that).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import kve

from unified.core.types import Matrix, Points, RealArray, Vector

EULER_GAMMA = 0.5772156649015329


def _diff(X: ArrayLike, Y: ArrayLike) -> tuple[RealArray, RealArray, RealArray]:
    A, B = np.asarray(X, float), np.asarray(Y, float)
    dx = A[:, None, 0] - B[None, :, 0]
    dy = A[:, None, 1] - B[None, :, 1]
    return dx, dy, np.hypot(dx, dy)


def scalar_oseen(lam: float) -> "ScalarOseen":
    return ScalarOseen(lam)


@dataclass(frozen=True)
class ScalarOseen:
    """K(X, Y)_ij = G(x_i - y_j), G = e^{lam x_1} K_0(lam r)/(2 pi); lam = 0 gives the Laplace kernel -log r/(2 pi)
    (the Stokes-limit control). Diagonal (r = 0) set to 0."""
    lam: float

    def __call__(self, X: Points, Y: Points) -> Matrix:
        dx, _, r = _diff(X, Y)
        out = np.zeros_like(r)
        nz = r > 0
        if self.lam == 0.0:
            out[nz] = -np.log(r[nz]) / (2 * np.pi)
        else:
            z = self.lam * r[nz]
            out[nz] = kve(0, z) * np.exp(self.lam * dx[nz] - z) / (2 * np.pi)
        return out


def oseen_tensor_blocks(k: float, dx: RealArray, dy: RealArray, mu: float = 1.0) -> tuple[RealArray, RealArray, RealArray]:
    """(S_11, S_12, S_22) at displacements (dx, dy) != 0 (formulas in the module docstring)."""
    r = np.hypot(dx, dy)
    z = k * r
    ek = np.exp(k * dx - z)                         # e^{k x_1} e^{-k r}; times kve gives e^{k x_1} K_nu(k r)
    K0, K1 = kve(0, z) * ek, kve(1, z) * ek
    E = K0 / (2 * np.pi)
    dE1 = (k * K0 - k * K1 * dx / r) / (2 * np.pi)
    dE2 = (-k * K1 * dy / r) / (2 * np.pi)
    dP1, dP2 = dx / (2 * np.pi * r ** 2), dy / (2 * np.pi * r ** 2)
    S11 = (E - (dP1 + dE1) / (2 * k)) / mu
    S12 = -(dP2 + dE2) / (2 * k * mu)
    S22 = (dP1 + dE1) / (2 * k * mu)
    return S11, S12, S22


def stokeslet_blocks(dx: RealArray, dy: RealArray, mu: float = 1.0) -> tuple[RealArray, RealArray, RealArray]:
    r2 = dx ** 2 + dy ** 2
    lg = -0.5 * np.log(r2)
    c = 1 / (4 * np.pi * mu)
    return c * (lg + dx ** 2 / r2), c * dx * dy / r2, c * (lg + dy ** 2 / r2)


def interleave(S11: RealArray, S12: RealArray, S21: RealArray, S22: RealArray) -> Matrix:
    """2x2 blocks -> (2m, 2n) matrix with dof order (point 0 comp 0, point 0 comp 1, point 1 comp 0, ...)."""
    m, n = S11.shape
    out: Matrix = np.empty((2 * m, 2 * n))
    out[0::2, 0::2], out[0::2, 1::2], out[1::2, 0::2], out[1::2, 1::2] = S11, S12, S21, S22
    return out


def oseen_tensor(k: float, X: Points, Y: Points, mu: float = 1.0) -> Matrix:
    """(2|X|, 2|Y|) Oseenlet matrix S(x_i - y_j) (velocity at x_i from a point force at y_j). x = y gives 0."""
    dx, dy, r = _diff(X, Y)
    nz = r > 0
    S11, S12, S22 = (np.zeros_like(r) for _ in range(3))
    a, b, c = oseen_tensor_blocks(k, dx[nz], dy[nz], mu)
    S11[nz], S12[nz], S22[nz] = a, b, c
    return interleave(S11, S12, S12, S22)


def near_constant(k: float, mu: float = 1.0) -> RealArray:
    """C = lim_{r->0} (S - Stokeslet) = diag(-c0, -1 - c0)/(4 pi mu), c0 = log(k/2) + gamma."""
    c0 = math.log(k / 2) + EULER_GAMMA
    out: RealArray = np.diag([-c0, -1.0 - c0]) / (4 * np.pi * mu)
    return out


@dataclass
class Cylinder:
    """First-kind single-layer Oseen system on a circle of radius a: A f = rhs, A = S W (+ diagonal blocks)."""
    k: float
    a: float
    n: int
    X: Points          # (n, 2) nodes
    w: float           # trapezoid weight a 2 pi / n
    A: Matrix          # (2n, 2n)
    rhs: Vector        # -U e_1 at every node (interleaved)

    @property
    def dof_points(self) -> Points:
        out: Points = np.repeat(self.X, 2, axis=0)
        return out

    def drag(self, f: Vector) -> float:
        return float(-np.sum(f[0::2]) * self.w)


def cylinder(k: float, n: int, a: float = 1.0, U: float = 1.0, mu: float = 1.0) -> Cylinder:
    th = 2 * np.pi * np.arange(n) / n
    X: Points = a * np.c_[np.cos(th), np.sin(th)]
    w = 2 * np.pi * a / n
    A = oseen_tensor(k, X, X, mu) * w
    dx, dy, r = _diff(X, X)
    off = ~np.eye(n, dtype=bool)
    sum_log = np.where(off, np.log(np.where(off, r, 1.0)), 0.0).sum(1) * w      # sum_{j != i} w log r_ij
    log_corr = -(2 * np.pi * a * math.log(a) - sum_log) / (4 * np.pi * mu)     # -(1/4 pi mu) * missing log mass
    C = near_constant(k, mu) * w
    for i in range(n):
        tau = np.array([-np.sin(th[i]), np.cos(th[i])])
        D = C + w * np.outer(tau, tau) / (4 * np.pi * mu) + log_corr[i] * np.eye(2)
        A[2 * i:2 * i + 2, 2 * i:2 * i + 2] = D
    rhs: Vector = np.tile([-U, 0.0], n)
    return Cylinder(k, a, n, X, w, A, rhs)


def lamb_drag(k: float, a: float = 1.0, U: float = 1.0, mu: float = 1.0) -> float:
    """Lamb (1911) Oseen drag per unit length on a cylinder: 4 pi mu U / (1/2 - gamma - log(k a / 2))."""
    return 4 * math.pi * mu * U / (0.5 - EULER_GAMMA - math.log(k * a / 2))


# ---- scalar area test matrix (SPEC P2 analogue) -----------------------------------------------------------
@dataclass
class AreaProblem:
    lam: float
    X: Points
    A: Matrix
    b: Vector
    kernel: ScalarOseen


def area_problem(N: int, lam: float, seed: int = 0) -> AreaProblem:
    """A = I + G_lam / N on N uniform random points in the unit square (seed 0, as SPEC P2)."""
    rng = np.random.default_rng(seed)
    X: Points = rng.random((N, 2))
    ker = ScalarOseen(lam)
    A: Matrix = np.eye(N) + ker(X, X) / N
    b: Vector = np.random.default_rng(seed + 1).standard_normal(N)
    return AreaProblem(lam, X, A, b, ker)


def rel_err(a: Vector, b: Vector) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))
