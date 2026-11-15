"""Swap (b): the butterfly APPLY used inside a SOLVER. Solve K x = f for the 1D FIO
    K[x, xi] = a(xi) exp(2 pi i (x xi + t c(x) |xi|)),   x = j/N,  xi = -N/2 .. N/2-1,
with GMRES, where every product with K and with K^* goes through the ID-butterfly factors
(K ~= U G_{L-1} ... G_0 W, so K^* ~= W^* G_0^* ... G_{L-1}^* U^*: the adjoint costs the same O(N log N)).

a(xi) = exp(-sigma |xi| / (N/2)) is an optional damping (sigma = 0: the B1 kernel of SPEC.md).

Preconditioners (all applied on the RIGHT, so the GMRES residual is the true residual ||K x - f||):
  none      M = I
  adjoint   M = K^* / N                         (exact inverse at t = 0, sigma = 0: K / sqrt(N) is unitary)
  jacobian  M = (1/N) [Pi_+ K^* D_+ + Pi_- K^* D_-],   D_+- = diag(1 +- t c'(x)),  Pi_+- = restriction to
            xi > 0 / xi < 0 (xi = 0 uses D = I). The adjoint with the Jacobian of the canonical map
            x -> x +- t c(x) folded in: sum_x Phi'(x) e^{2 pi i (eta - xi) Phi(x)} ~= N delta_{xi eta} when
            Phi(x) = x +- t c(x) is a monotone circle map, i.e. the standard FIO parametrix.  [derived here]
  parametrix  jacobian followed by diag(1 / a(xi)^2): also undoes the known damping amplitude.
Caustic: Phi_- ' = 1 - t c'(x) with c' = (2 pi / 16) cos 2 pi x vanishes iff t >= 16 / (2 pi) = 2.546.
"""
from __future__ import annotations
from typing import Any, Callable, Literal

import numpy as np
import scipy.sparse as sp

from unified.core.types import (ComplexArray, Field, FrequencyGrid, KernelMatrix, Matrix, PositionGrid,
                                RealArray, Spectrum, Vector)
from unified.drivers.butterfly_id import IDButterfly, butterfly_id

# ---- domain aliases --------------------------------------------------------------------------------
type Time = float                                         # propagation time t of the FIO
type Damping = float                                      # sigma in a(xi) = exp(-sigma |xi| / (N/2))
type PrecondKind = Literal["none", "adjoint", "jacobian", "parametrix"]
type FIOKernel = Callable[[PositionGrid, FrequencyGrid], KernelMatrix]
type LinOp = Callable[[Vector], Vector]                   # v -> A v (one matrix-free product)
type ResidualHistory = list[float]                        # ||b - A x_k|| / ||b||, k = 0..iters
type SolveReport = dict[str, Any]
type SpectrumReport = dict[str, Any]

# typed handle on the (possibly untyped) driver entry point
_butterfly_id: Callable[..., IDButterfly] = butterfly_id

T_CAUSTIC: Time = 16 / (2 * np.pi)


def c(x: PositionGrid) -> RealArray:
    """Wave speed c(x) of the B1 kernel."""
    return (2 + np.sin(2 * np.pi * x)) / 16


def dc(x: PositionGrid) -> RealArray:
    """c'(x): the Jacobian of the canonical map x -> x +- t c(x) is 1 +- t c'(x)."""
    return 2 * np.pi * np.cos(2 * np.pi * x) / 16


def grids(N: int) -> tuple[PositionGrid, FrequencyGrid]:
    x: PositionGrid = np.arange(N) / N
    xi: FrequencyGrid = (np.arange(N) - N // 2).astype(float)
    return x, xi


def fio_kernel(N: int, t: Time, sigma: Damping = 0.0) -> FIOKernel:
    """K(x, xi) as a kernel of coordinates (works on blocks, as butterfly_id needs)."""
    def K(x_: PositionGrid, xi_: FrequencyGrid) -> KernelMatrix:
        x: PositionGrid = np.asarray(x_, float).ravel()
        xi: FrequencyGrid = np.asarray(xi_, float).ravel()
        amp: RealArray = np.exp(-sigma * np.abs(xi) / (N / 2))          # damping a(xi)
        out: ComplexArray = np.exp(2j * np.pi * (np.outer(x, xi) + t * c(x)[:, None] * np.abs(xi)[None, :]))
        return out * amp[None, :]
    return K


class ButterflyOp:
    """K and K^* through the sparse butterfly factors."""

    def __init__(self, N: int, t: Time, sigma: Damping = 0.0, eps: float = 1e-10) -> None:
        self.N, self.t, self.sigma = N, t, sigma
        self.x, self.xi = grids(N)
        self.kernel: FIOKernel = fio_kernel(N, t, sigma)
        self.bf: IDButterfly = _butterfly_id(self.kernel, self.x, self.xi, eps=eps)
        b = self.bf
        # adjoint factors: K^* ~= W^* G_0^* ... G_{L-1}^* U^*
        self._WH: sp.csr_matrix = b.W.conj().T.tocsr()
        self._GH: list[sp.csr_matrix] = [G.conj().T.tocsr() for G in b.G]
        self._UH: sp.csr_matrix = b.U.conj().T.tocsr()

    def matvec(self, g: Spectrum) -> Field:
        """f = K g."""
        apply: LinOp = self.bf.apply                    # typed handle on the driver's (untyped) apply
        return apply(g)

    def rmatvec(self, f: Field) -> Spectrum:
        """K^* f = W^* G_0^* ... G_{L-1}^* U^* f."""
        c_: Vector = np.asarray(self._UH @ f)
        for GH in reversed(self._GH):
            c_ = np.asarray(GH @ c_)
        g: Spectrum = np.asarray(self._WH @ c_)
        return g

    def precond(self, kind: PrecondKind) -> LinOp:
        """Right preconditioner M (an approximate inverse K^{-1}) of the given kind."""
        N, x, xi, t = self.N, self.x, self.xi, self.t
        if kind == "none":
            return lambda f: f
        if kind == "adjoint":
            return lambda f: self.rmatvec(f) / N
        if kind in ("jacobian", "parametrix"):
            Dp: RealArray = 1 + t * dc(x)                 # Jacobian of x -> x + t c(x)  (xi > 0 branch)
            Dm: RealArray = 1 - t * dc(x)                 # Jacobian of x -> x - t c(x)  (xi < 0 branch)
            pos, neg = xi > 0, xi < 0
            inv_a2: RealArray | float = (np.exp(2 * self.sigma * np.abs(xi) / (N / 2))
                                         if kind == "parametrix" else 1.0)

            def M(f: Field) -> Spectrum:
                Y: Matrix = self.rmatvec(np.c_[Dp * f, Dm * f, f]) / N
                out: Spectrum = inv_a2 * np.where(pos, Y[:, 0], np.where(neg, Y[:, 1], Y[:, 2]))
                return out
            return M
        raise ValueError(kind)


def _identity(v: Vector) -> Vector:
    return v


def gmres(A: LinOp, b: Vector, M: LinOp = _identity, tol: float = 1e-8,
          maxiter: int = 300) -> tuple[Vector, ResidualHistory]:
    """Full (unrestarted) right-preconditioned GMRES, modified Gram-Schmidt Arnoldi on A M.
    Returns (x, residual history ||b - A x_k|| / ||b||, k = 0..iters)."""
    n = b.shape[0]
    beta = float(np.linalg.norm(b))
    V: ComplexArray = np.zeros((n, maxiter + 1), complex)       # Krylov basis
    H: ComplexArray = np.zeros((maxiter + 1, maxiter), complex)  # upper Hessenberg matrix
    V[:, 0] = b / beta
    hist: ResidualHistory = [1.0]
    y: ComplexArray = np.zeros(0, complex)
    for j in range(maxiter):
        w: Vector = A(M(V[:, j]))
        for i in range(j + 1):
            H[i, j] = np.vdot(V[:, i], w)
            w = w - H[i, j] * V[:, i]
        H[j + 1, j] = np.linalg.norm(w)
        if abs(H[j + 1, j]) > 1e-14 * beta:
            V[:, j + 1] = w / H[j + 1, j]
        e1: ComplexArray = np.zeros(j + 2, complex)
        e1[0] = beta
        y = np.linalg.lstsq(H[:j + 2, :j + 1], e1, rcond=None)[0]
        hist.append(float(np.linalg.norm(H[:j + 2, :j + 1] @ y - e1) / beta))
        if hist[-1] <= tol or abs(H[j + 1, j]) <= 1e-14 * beta:
            break
    x: Vector = M(V[:, :len(y)] @ y)
    return x, hist


def solve_case(op: ButterflyOp, kind: PrecondKind, tol: float = 1e-8, maxiter: int = 300, seed: int = 0,
               check_dense: bool = True) -> SolveReport:
    """f = K x_true for a random x_true; solve with GMRES + preconditioner `kind`. [computed]"""
    rng = np.random.default_rng(seed)
    N = op.N
    x_true: Spectrum = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    f: Field = op.matvec(x_true)
    x, hist = gmres(op.matvec, f, op.precond(kind), tol=tol, maxiter=maxiter)
    out: SolveReport = dict(N=N, t=op.t, sigma=op.sigma, precond=kind, iters=len(hist) - 1,
                            converged=hist[-1] <= tol, final_res=hist[-1], hist=hist,
                            sol_err=float(np.linalg.norm(x - x_true) / np.linalg.norm(x_true)))
    if check_dense:  # residual with the exact K (not the butterfly)
        Kx: Field = np.concatenate([op.kernel(op.x[i:i + 512], op.xi) @ x for i in range(0, N, 512)])
        f_ex: Field = np.concatenate([op.kernel(op.x[i:i + 512], op.xi) @ x_true for i in range(0, N, 512)])
        out["true_res"] = float(np.linalg.norm(Kx - f_ex) / np.linalg.norm(f_ex))
    return out


def spectrum_stats(N: int, t: Time, sigma: Damping, kind: PrecondKind) -> SpectrumReport:
    """Dense diagnostics (small N): condition number of K and of the preconditioned K M, and the
    spread |lambda(K M) - 1| of its eigenvalues. [computed]"""
    x, xi = grids(N)
    K: KernelMatrix = fio_kernel(N, t, sigma)(x, xi)
    KM: Matrix
    if kind == "none":
        KM = K / np.sqrt(N)
    elif kind == "adjoint":
        KM = K @ K.conj().T / N
    else:
        Dp, Dm = 1 + t * dc(x), 1 - t * dc(x)
        KH: Matrix = K.conj().T
        Mx: Matrix = np.where((xi > 0)[:, None], KH * Dp[None, :],
                              np.where((xi < 0)[:, None], KH * Dm[None, :], KH)) / N
        if kind == "parametrix":
            Mx = Mx * np.exp(2 * sigma * np.abs(xi) / (N / 2))[:, None]
        KM = K @ Mx
    s: RealArray = np.linalg.svd(KM, compute_uv=False)
    sK: RealArray = np.linalg.svd(K, compute_uv=False)
    lam: ComplexArray = np.linalg.eigvals(KM)
    return dict(N=N, t=t, sigma=sigma, precond=kind, cond_K=float(sK[0] / sK[-1]), cond=float(s[0] / s[-1]),
                n_outliers=int(np.sum(np.abs(lam - 1) > 1e-3)),
                max_dist_from_1=float(np.max(np.abs(lam - 1))), median_dist_from_1=float(np.median(np.abs(lam - 1))))
