"""E9: the 2D Lippmann-Schwinger (LS) volume integral equation as a harness problem.

Physics. A time-harmonic wave (frequency fixed, wavenumber k in the background) hits an inhomogeneous
object whose refractive index squared is 1 + q(x), with q = 0 outside a bounded region (the "contrast").
The total field u solves  Δu + k²(1 + q) u = 0  plus a radiation condition on u - u_inc. Moving the q
term to the right and inverting (Δ + k²) with the outgoing Green's function G_k(x) = (i/4) H0^(1)(k|x|)
(so that (Δ + k²) G_k = -δ) gives the Lippmann-Schwinger equation

        u(x) - k² ∫ G_k(x - y) q(y) u(y) dy = u_inc(x),            x in supp q.

Discretization (Richmond 1965, "pulse basis + point matching"). Grid nodes x_j with spacing h filling the
support, weight h² each. The self-cell integral of G_k is replaced by the integral over the equal-area disk
of radius ρ = h/√π:
        s = ∫_{|y|<ρ} G_k(y) dy = (iπρ / 2k) H1^(1)(kρ) - 1/k².
Matrix: A = I + K D with K_ij = G_k(x_i, x_j) (i ≠ j), K_ii = s / h², D = diag(d), d_j = -k² q_j h².
The contrast multiplies the kernel on ONE side (columns) only.

Reference solutions.
* constant contrast q0 on a disk of radius a: the Bessel series ("Mie" for a circular cylinder).
* smooth radial bump q(r) = q0 (1 - r²/a²)³: per Fourier mode, a radial ODE integrated to 1e-12.
Both are expressed through the scattering coefficients b_n: outside the support
        u - u_inc = Σ_n b_n H_n^(1)(k r) e^{i n θ},
and the discrete b_n follow from Graf's addition theorem applied to the discrete LS representation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.special import h1vp, hankel1, jv, jvp

from unified.core import kernels
from unified.core.types import ComplexArray, Kernel, KernelMatrix, Matrix, Points, RealArray, Vector

type Wavenumber = float
type ContrastKind = Literal["disk", "bump", "square"]
type ScatCoeffs = ComplexArray        # b_n for n = -nmax..nmax
type ProxyMode = Literal["two_sided", "naive"]


@dataclass
class LSProblem:
    """A discretized Lippmann-Schwinger problem  A u = u_inc,  A = I + K D."""
    kind: ContrastKind
    k: Wavenumber
    q0: float
    a: float                 # disk radius / square half-width
    h: float                 # grid spacing
    X: Points                # (N, 2) grid nodes inside the support
    q: RealArray             # (N,) contrast at the nodes
    d: ComplexArray          # (N,) column scaling -k² q h²
    A: Matrix                # (N, N) dense complex system matrix
    self_term: complex       # s = ∫ over the equal-area disk of G_k

    @property
    def n(self) -> int:
        return len(self.X)

    def u_inc(self, direction: float = 0.0) -> ComplexArray:
        """Incident plane wave exp(i k x·(cos α, sin α)) at the nodes."""
        e = np.array([np.cos(direction), np.sin(direction)])
        out: ComplexArray = np.exp(1j * self.k * (self.X @ e))
        return out


def contrast(kind: ContrastKind, q0: float, a: float, X: Points) -> RealArray:
    """q(x) on arbitrary points: constant disk, C² radial bump, or constant square [-a, a]²."""
    r = np.hypot(X[:, 0], X[:, 1])
    if kind == "disk":
        out: RealArray = np.where(r < a, q0, 0.0)
    elif kind == "bump":
        out = np.where(r < a, q0 * np.clip(1 - (r / a) ** 2, 0, None) ** 3, 0.0)
    else:
        out = np.where(np.max(np.abs(X), axis=1) < a, q0, 0.0)
    return out


def grid_nodes(kind: ContrastKind, a: float, m: int) -> tuple[Points, float]:
    """Cell centres of an m x m grid on [-a, a]² that lie inside the support (staircase for the disk)."""
    h = 2 * a / m
    t = -a + h * (np.arange(m) + 0.5)
    G: Points = np.stack(np.meshgrid(t, t, indexing="ij"), axis=-1).reshape(-1, 2)
    keep = np.max(np.abs(G), axis=1) < a if kind == "square" else np.hypot(G[:, 0], G[:, 1]) < a
    return G[keep], h


def m_for_n(kind: ContrastKind, n_target: int) -> int:
    """Grid size m giving about n_target nodes (disk: N ≈ π m² / 4; square: N = m²)."""
    return int(round(np.sqrt(n_target if kind == "square" else 4 * n_target / np.pi)))


def self_term(k: Wavenumber, h: float) -> complex:
    """Richmond's equal-area-disk integral of (i/4) H0(k r):  (iπρ/2k) H1(kρ) - 1/k²,  ρ = h/√π."""
    rho = h / np.sqrt(np.pi)
    return complex(1j * np.pi * rho / (2 * k) * hankel1(1, k * rho) - 1 / k ** 2)


def build(kind: ContrastKind, k: Wavenumber, q0: float, m: int, a: float = 1.0) -> LSProblem:
    """Assemble the dense LS matrix A = I + K D on the grid nodes inside the support."""
    X, h = grid_nodes(kind, a, m)
    q = contrast(kind, q0, a, X)
    d: ComplexArray = (-k ** 2 * q * h ** 2).astype(complex)
    s = self_term(k, h)
    # G_k on a uniform grid depends only on the integer offset |Δi|, |Δj|: tabulate once (m² Hankel calls
    # instead of N²), then gather. Same values as kernels.helmholtz2d(k)(X, X) (tested).
    ij = np.rint((X + a - h / 2) / h).astype(np.intp)
    off = np.arange(m) * h
    table: Matrix = kernels.helmholtz2d(k)(np.c_[np.hypot(off[:, None], off[None, :]).ravel(),
                                                       np.zeros(m * m)], np.zeros((1, 2)))[:, 0].reshape(m, m)
    table[0, 0] = s / h ** 2
    K: Matrix = table[np.abs(ij[:, None, 0] - ij[None, :, 0]), np.abs(ij[:, None, 1] - ij[None, :, 1])]
    A: Matrix = K * d[None, :]
    A[np.diag_indices(len(X))] += 1
    return LSProblem(kind, k, q0, a, h, X, q, d, A, s)


# ---- the only new piece the proxy compressor needs: the kernel ------------------------------------
def ls_kernel(prob: LSProblem, mode: ProxyMode = "two_sided") -> Kernel:
    """Kernel for ProxyCompressor reproducing A's off-diagonal entries:  kernel(x, y) = G_k(x, y) · c(y).

    two_sided: c(y) = d_j if y is grid node j, else 1. At proxy points (never grid nodes) the transpose
               block kernel(X_B, P)ᵀ = G_k(P, X_B) is then UNscaled, which is exactly the relation the row
               operation needs (rows of A_BFᵀ = D_F K_FB have the relations of K_FB).
    naive:     c(y) = -k² q(y) h² evaluated as a field (zero outside the support), i.e. what one would write
               by "multiplying the kernel by the contrast". Rows at proxy points outside supp q vanish.
    """
    k, h = prob.k, prob.h
    G = kernels.helmholtz2d(k)
    if mode == "naive":
        def Kn(X: Points, Y: Points) -> KernelMatrix:
            Y = np.atleast_2d(Y)
            cy = -k ** 2 * contrast(prob.kind, prob.q0, prob.a, Y) * h ** 2
            out: KernelMatrix = G(X, Y) * cy[None, :]
            return out
        return Kn
    lo = -prob.a + h / 2
    m = int(round(2 * prob.a / h))
    index = -np.ones((m, m), np.intp)
    ij = np.rint((prob.X - lo) / h).astype(np.intp)
    index[ij[:, 0], ij[:, 1]] = np.arange(prob.n)

    def Kt(X: Points, Y: Points) -> KernelMatrix:
        Y = np.atleast_2d(Y)
        cy: ComplexArray = np.ones(len(Y), complex)
        g = np.rint((Y - lo) / h).astype(np.intp)
        ok = np.all((g >= 0) & (g < m), axis=1)
        j = np.full(len(Y), -1, np.intp)
        j[ok] = index[g[ok, 0], g[ok, 1]]
        hit = j >= 0
        hit[hit] = np.linalg.norm(prob.X[j[hit]] - Y[hit], axis=1) < 1e-9 * h   # exact node match only
        cy[hit] = prob.d[j[hit]]
        out: KernelMatrix = G(X, Y) * cy[None, :]
        return out
    return Kt


# ---- reference solutions ----------------------------------------------------------------------------
def nmax_for(k: Wavenumber, q0: float, a: float) -> int:
    return int(k * np.sqrt(1 + max(q0, 0.0)) * a + 20)


def mie_coeffs(k: Wavenumber, q0: float, a: float, nmax: int) -> tuple[ScatCoeffs, ComplexArray]:
    """Constant contrast q0 on the disk r < a, incident exp(i k x1) = Σ iⁿ J_n(kr) e^{inθ}.
    Returns (b_n, c_n), n = -nmax..nmax:  inside u = Σ c_n J_n(k1 r) e^{inθ}, k1 = k √(1 + q0);
    outside u - u_inc = Σ b_n H_n(kr) e^{inθ}. From continuity of u and ∂_r u at r = a."""
    n = np.arange(-nmax, nmax + 1)
    k1 = k * np.sqrt(1 + q0 + 0j)
    ka, k1a = k * a, k1 * a
    Jn, dJn, Hn, dHn = jv(n, ka), jvp(n, ka), hankel1(n, ka), h1vp(n, ka)
    Jn1, dJn1 = jv(n, k1a), jvp(n, k1a)
    den = Jn1 * k * dHn - k1 * dJn1 * Hn
    c: ComplexArray = (1j ** n) * (Jn * k * dHn - k * dJn * Hn) / den
    b: ComplexArray = (c * Jn1 - (1j ** n) * Jn) / Hn
    return b, c


def mie_field(k: Wavenumber, q0: float, a: float, X: Points, nmax: int) -> ComplexArray:
    """Total field inside the disk from the Bessel series."""
    _, c = mie_coeffs(k, q0, a, nmax)
    n = np.arange(-nmax, nmax + 1)
    r, th = np.hypot(X[:, 0], X[:, 1]), np.arctan2(X[:, 1], X[:, 0])
    k1 = k * np.sqrt(1 + q0 + 0j)
    out: ComplexArray = (jv(n[None, :], k1 * r[:, None]) * np.exp(1j * n[None, :] * th[:, None])) @ c
    return out


def radial_coeffs(k: Wavenumber, q0: float, a: float, nmax: int, kind: ContrastKind = "bump") -> ScatCoeffs:
    """b_n for a radial contrast by integrating, per mode, v = r^|n| w with
    w'' + (2|n|+1)/r w' + k²(1 + q(r)) w = 0,  w(0) = 1, w'(0) = 0, to r = a (rtol 1e-12), then matching
    c v = iⁿ J_n(kr) + b H_n(kr) and its r-derivative at r = a."""
    def qr(r: float) -> float:
        return float(contrast(kind, q0, a, np.array([[r, 0.0]]))[0])
    out: ScatCoeffs = np.zeros(2 * nmax + 1, complex)
    for i, n in enumerate(range(-nmax, nmax + 1)):
        mm = abs(n)
        kap0 = k ** 2 * (1 + qr(0.0))
        r0 = 1e-6 * a

        def rhs(r: float, y: NDArray[np.float64], mm: int = mm) -> list[float]:
            return [y[1], -(2 * mm + 1) / r * y[1] - k ** 2 * (1 + qr(r * (1 - 1e-15))) * y[0]]
        y0 = [1 - kap0 * r0 ** 2 / (4 * (mm + 1)), -kap0 * r0 / (2 * (mm + 1))]
        sol = solve_ivp(rhs, (r0, a), y0, method="DOP853", rtol=1e-12, atol=1e-14)
        w, dw = sol.y[0, -1], sol.y[1, -1]
        L = mm / a + dw / w                              # v'(a) / v(a)
        ka = k * a
        out[i] = (1j ** n) * (k * jvp(n, ka) - L * jv(n, ka)) / (L * hankel1(n, ka) - k * h1vp(n, ka))
    return out


def discrete_coeffs(prob: LSProblem, u: Vector, nmax: int) -> ScatCoeffs:
    """b_n of the discrete solution: u - u_inc = k² Σ_j G(x - x_j) q_j h² u_j and Graf's theorem
    H0(k|x - y|) = Σ_n H_n(kr) e^{inθ} J_n(kρ) e^{-inφ} (r > ρ) give b_n = (i/4) k² Σ_j J_n(kρ_j) e^{-inφ_j} q_j h² u_j."""
    n = np.arange(-nmax, nmax + 1)
    rho, phi = np.hypot(prob.X[:, 0], prob.X[:, 1]), np.arctan2(prob.X[:, 1], prob.X[:, 0])
    B = jv(n[None, :], prob.k * rho[:, None]) * np.exp(-1j * n[None, :] * phi[:, None])
    out: ScatCoeffs = 0.25j * prob.k ** 2 * (B.T @ (prob.q * prob.h ** 2 * u))
    return out


def unitarity_defect(b: ScatCoeffs, nmax: int) -> float:
    """Energy conservation for a lossless (real q) scatterer: each partial-wave S-matrix entry
    s_n = 1 + 2 b_n i⁻ⁿ has |s_n| = 1. Returns max_n | |s_n| - 1 |."""
    n = np.arange(-nmax, nmax + 1)
    return float(np.max(np.abs(np.abs(1 + 2 * b * (1j ** (-n))) - 1)))


def coeff_error(b: ScatCoeffs, b_ref: ScatCoeffs) -> float:
    """Relative l2 error of scattering coefficients (∝ far-field L2 error, by Parseval)."""
    return float(np.linalg.norm(b - b_ref) / np.linalg.norm(b_ref))
