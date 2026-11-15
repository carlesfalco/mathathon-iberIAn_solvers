"""E11 physics/statistics set-up: Matérn covariances, the "Matérn family" Green's functions, GP problems.

Matérn covariance (smoothness nu, length scale ell, kappa = sqrt(2 nu)/ell):
    C_nu(r) = 2^{1-nu}/Gamma(nu) (kappa r)^nu K_nu(kappa r),   C_nu(0) = 1.
Green's function of L^beta = (kappa^2 - Laplacian)^beta in R^D (Matérn 1960; Whittle 1954; Lindgren-Rue-Lindström
2011 for the SPDE view):
    G_{beta,D}(r) ∝ r^{beta - D/2} K_{beta - D/2}(kappa r),     L G_{beta,D} = G_{beta-1,D}   (up to constants).
So C_nu = G_{beta,D} restricted to R^2 for every integer beta >= nu + 1 with D = 2 (beta - nu) [proved, by the
formula above]. (kappa^2 - Delta_D)[r^mu K_mu] ∝ r^{mu-1} K_{mu-1} for r > 0 in EVERY D (only the constant depends
on D), so the same radial formulas serve every lift; what singles out D = 2 (beta - nu) is that the chain terminates
(the last order is the Green's function of kappa^2 - Delta_D). Only the proxy GEOMETRY knows D.

Convention used by the proxy compressors: data points have 2 coordinates; proxy points have 3 (x, y, t), with
t >= 0 the extension coordinate (t = 0 for proxies in the plane). A StackedKernel sees which argument has 3
columns and stacks one block of rows per kernel order along that (proxy) axis.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import gamma, kv

from unified.core.types import Matrix, Points, RealArray, Vector


def _pairdist(X: ArrayLike, Y: ArrayLike) -> RealArray:
    """|X_i - Y_j| after padding the shorter coordinate list with t = 0 (data points live at t = 0)."""
    A, B = np.asarray(X, float), np.asarray(Y, float)
    d = max(A.shape[1], B.shape[1])
    A = np.pad(A, ((0, 0), (0, d - A.shape[1])))
    B = np.pad(B, ((0, 0), (0, d - B.shape[1])))
    r: RealArray = np.sqrt(np.maximum(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1), 0.0))
    return r


def radial_rK(mu: float, kappa: float, r: RealArray) -> RealArray:
    """f(r) = (kappa r)^mu K_|mu|(kappa r) for r > 0 (any sign of mu; K_mu = K_{-mu}). r = 0 is not allowed."""
    z = kappa * r
    out: RealArray = z ** mu * kv(abs(mu), z)
    return out


def matern_cov(nu: float, ell: float, r: RealArray) -> RealArray:
    """Matérn covariance C_nu(r) with unit variance; nu = 0.5 is e^{-r/ell}. C(0) = 1."""
    kappa = math.sqrt(2 * nu) / ell
    out = np.ones_like(r)
    nz = r > 0
    out[nz] = 2 ** (1 - nu) / gamma(nu) * radial_rK(nu, kappa, r[nz])
    return out


def kappa_of(nu: float, ell: float) -> float:
    return math.sqrt(2 * nu) / ell if nu > 0 else 1.0 / ell


def lift_order(nu: float) -> tuple[int, float]:
    """Smallest integer beta >= nu + 1 and the matching dimension D = 2 (beta - nu) in which C_nu is the Green's
    function of (kappa^2 - Delta)^beta. nu = 0.5 -> (2, 3); 1 -> (2, 2); 0.7 -> (2, 2.6); 2.5 -> (4, 3)."""
    beta = math.ceil(nu + 1 - 1e-12)
    return beta, 2.0 * (beta - nu)


@dataclass
class StackedKernel:
    """Kernel callable K(X, Y) for the compressors.

    data x data (2 columns each): the main kernel (rarely used: compressors read explicit entries from A).
    proxy x data / data x proxy (one argument with 3 columns): one block per entry of `orders`, stacked along the
    proxy axis; block m is r^{mu_m} K_{mu_m}(kappa r) with mu_m = orders[m], rescaled so that its largest entry
    equals the largest entry of block 0 (row scaling of the ID input only).
    `main` is the radial function of the matrix kernel (used for orders[0] and data x data)."""
    nu: float
    ell: float
    orders: list[float] = field(default_factory=list)   # mu values; [] means [nu] (main kernel only)
    kind: str = "matern"                                # "matern" | "k0" | "rte"

    @property
    def kappa(self) -> float:
        return kappa_of(self.nu, self.ell)

    def main(self, r: RealArray) -> RealArray:
        if self.kind == "matern":
            return matern_cov(self.nu, self.ell, r)
        out = np.zeros_like(r)
        nz = r > 0
        if self.kind == "k0":                           # Yukawa Green's function of R^2 (nu = 0 control)
            out[nz] = kv(0, r[nz] / self.ell) / (2 * np.pi)
        else:                                           # radiative transfer: e^{-mu r}/(2 pi r), mu = 1/ell
            out[nz] = np.exp(-r[nz] / self.ell) / (2 * np.pi * r[nz])
        return out

    def block(self, mu: float, r: RealArray) -> RealArray:
        if self.kind == "rte":                          # G_{1,3}-family of the 3D Yukawa: e^{-kr}/r only
            return self.main(r)
        if self.kind == "k0" or mu == self.nu:
            return self.main(r)
        return radial_rK(mu, self.kappa, r)

    def __call__(self, X: Points, Y: Points) -> Matrix:
        Xa, Ya = np.asarray(X, float), np.asarray(Y, float)
        r = _pairdist(Xa, Ya)
        proxy_rows = Xa.shape[1] == 3
        if not (proxy_rows or Ya.shape[1] == 3):
            return self.main(r)
        orders = self.orders or [self.nu]
        blocks: list[RealArray] = []
        ref = 1.0
        for m, mu in enumerate(orders):
            B = self.block(mu, r)
            s = float(np.max(np.abs(B))) or 1.0
            if m == 0:
                ref = s
            blocks.append(B * (ref / s))
        out: Matrix = np.vstack(blocks) if proxy_rows else np.hstack(blocks)
        return out


def multi_orders(nu: float) -> list[float]:
    """mu_m = nu - m, m = 0 .. beta-1: the kernels L^m C_nu that the representation of an order-2 beta operator needs."""
    beta, _ = lift_order(nu)
    return [nu - m for m in range(beta)]


# ---- GP regression problem ---------------------------------------------------------------------------------
def test_function(X: Points) -> Vector:
    out: Vector = np.sin(6 * X[:, 0]) * np.cos(4 * X[:, 1]) + 0.5 * np.exp(-20 * ((X[:, 0] - 0.6) ** 2 + (X[:, 1] - 0.3) ** 2))
    return out


@dataclass
class GPProblem:
    """A = C_nu(X, X) + sigma2 I, data y = f(X) + noise, test points Xs for the posterior mean."""
    name: str
    X: Points
    A: Matrix
    y: Vector
    Xs: Points
    kernel: StackedKernel
    sigma2: float


def gp_problem(N: int, nu: float, ell: float, sigma2: float, seed: int = 0, n_test: int = 400,
               kind: str = "matern") -> GPProblem:
    """Random points in the unit square (seed), y = f + N(0, 0.1^2) noise. kind = "k0"/"rte" builds the
    second-kind control A = I + K/N (diagonal of K set to 0) instead of a covariance."""
    rng = np.random.default_rng(seed)
    X: Points = rng.random((N, 2))
    Xs: Points = rng.random((n_test, 2))
    ker = StackedKernel(nu, ell, kind=kind)
    r = _pairdist(X, X)
    if kind == "matern":
        A: Matrix = matern_cov(nu, ell, r)
        A[np.diag_indices(N)] += sigma2
    else:
        A = np.eye(N) + ker.main(r) / N
    y: Vector = test_function(X) + 0.1 * rng.standard_normal(N)
    return GPProblem(f"{kind}-nu{nu}-l{ell}-s{sigma2}-N{N}", X, A, y, Xs, ker, sigma2)


def posterior_mean(prob: GPProblem, alpha: Vector) -> Vector:
    """m(x*) = C(x*, X) alpha, with alpha = (C + sigma2 I)^{-1} y."""
    Ks = matern_cov(prob.kernel.nu, prob.kernel.ell, _pairdist(prob.Xs, prob.X))
    out: Vector = Ks @ alpha
    return out


def log_marginal_likelihood(y: Vector, alpha: Vector, logdet: float) -> float:
    """log p(y) = -1/2 y^T alpha - 1/2 log det A - N/2 log 2 pi (Rasmussen-Williams eq. 2.30)."""
    return float(-0.5 * y @ alpha - 0.5 * logdet - 0.5 * len(y) * math.log(2 * math.pi))


def rel_err(a: Vector, b: Vector) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))
