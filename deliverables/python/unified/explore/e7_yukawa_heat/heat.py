"""E7: the heat equation u_t = Δu inside the starfish, u = g on Γ, u(·, 0) = 0, by Rothe's method
(backward Euler in time FIRST, then a boundary integral equation per step), with boundary unknowns only.

Backward Euler:  (I − dtΔ) u^n = u^{n−1},  i.e.  (−Δ + κ²) u^n = κ² u^{n−1},  κ² = 1/dt.
The volume source κ² u^{n−1} is NOT computed with a volume potential. Instead we write

    u^n = Σ_{j=1}^{n} L_{n−j}[σ_j],     L_p = (−1/dt)^p / p! · ∂^p_{κ²} D_κ      (L_0 = D_κ, the double layer).

Differentiating (Δ − κ²) D_κ = 0 p times in κ² gives (Δ − κ²) ∂^p D = p ∂^{p−1} D, hence (I − dtΔ) L_p = L_{p−1}
and (I − dtΔ) u^n = u^{n−1} automatically (proved in findings.md). The boundary condition at step n is

    (−½ I + D_κ w) σ_n = g_n − Σ_{j<n} (L_{n−j} w) σ_j          (one SAME matrix every step: factor once).

This is the Chapko–Kress (1997) idea written for an interior problem; the history kernels are smooth for p ≥ 1.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.special import exp1

from unified.bench.problems import dense_identity_plus, starfish
from unified.core.types import Matrix, Points, RealArray, Vector

from .yukawa import Screening, dlp_kernel, monopole_p

type Solver = Callable[[Vector], Vector]          # σ = A^{-1} g for the step matrix A = −½I + D_κ w
type BoundaryData = Callable[[Points, float], RealArray]  # g(x, t)


@dataclass
class HeatRun:
    t: RealArray                 # (n_steps,) times t_n = n dt
    u: RealArray                 # (n_steps, n_targets) Rothe solution at the targets
    sigma: RealArray             # (n_steps, N) densities


def history_matrices(kappa: Screening, n: int, n_steps: int, targets: Points) -> tuple[list[Matrix], list[Matrix]]:
    """Boundary->boundary L_p w for p = 1..n_steps−1 and boundary->target L_p w for p = 0..n_steps−1."""
    X, _ = starfish(n)
    bb: list[Matrix] = [dense_identity_plus(dlp_kernel(kappa, n, p), X) - np.eye(n) for p in range(1, n_steps)]
    bt: list[Matrix] = [dlp_kernel(kappa, n, p)(targets, X) for p in range(n_steps)]
    return bb, bt


def rothe_heat(kappa: Screening, n: int, n_steps: int, g: BoundaryData, targets: Points, solve: Solver) -> HeatRun:
    """March n_steps backward-Euler steps of size dt = 1/κ²; `solve` applies the (fast or dense) inverse."""
    dt = 1.0 / kappa ** 2
    X, _ = starfish(n)
    bb, bt = history_matrices(kappa, n, n_steps, targets)
    sig = np.zeros((n_steps, n))
    u = np.zeros((n_steps, len(targets)))
    for s in range(n_steps):                       # step s+1, time (s+1) dt
        rhs = g(X, (s + 1) * dt)
        for j in range(s):                         # history: densities of earlier steps
            rhs = rhs - bb[s - j - 1] @ sig[j]
        sig[s] = solve(rhs)
        u[s] = sum((bt[s - j] @ sig[j] for j in range(s + 1)), np.zeros(len(targets)))
    t: RealArray = dt * np.arange(1, n_steps + 1, dtype=float)
    return HeatRun(t, u, sig)


# ---- analytic solutions: a unit point heat source switched on at t = 0 at x0 OUTSIDE the domain -------------
def semidiscrete_exact(kappa: Screening, x0: RealArray) -> BoundaryData:
    """Exact backward-Euler solution of u_t = Δu + δ_{x0}, u(0) = 0 (q = 1):  u^n = Σ_{p=0}^{n−1} M_p(|x − x0|).
    Satisfies (I − dtΔ)u^n = u^{n−1} + dt δ_{x0} exactly, since (I − dtΔ)M_0 = dt δ_{x0} (M_p: yukawa.monopole_p)."""
    dt = 1.0 / kappa ** 2
    def u(Y: Points, t: float) -> RealArray:
        nst = int(round(t / dt))
        r = np.linalg.norm(np.atleast_2d(Y) - x0, axis=1)
        out: RealArray = sum((monopole_p(kappa, p, r) for p in range(nst)), np.zeros(len(r)))
        return out
    return u


def continuous_exact(x0: RealArray) -> BoundaryData:
    """u(x, t) = E1(|x − x0|² / 4t) / (4π): the continuous solution of u_t = Δu + δ_{x0}, u(0) = 0."""
    def u(Y: Points, t: float) -> RealArray:
        r2 = np.sum((np.atleast_2d(Y) - x0) ** 2, axis=1)
        out: RealArray = exp1(r2 / (4 * t)) / (4 * np.pi)
        return out
    return u
