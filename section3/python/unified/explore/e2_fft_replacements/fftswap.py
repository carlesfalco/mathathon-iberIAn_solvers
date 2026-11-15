"""E2: replace the butterfly by FFT-based algorithms.

(a) Exact: the DFT in bit-reversed ordering. If both the rows (positions j) and the columns (frequencies k)
    are listed in bit-reversed order, the tile "row box at depth l x column box at depth L-l" is
        j = j0 + 2^l a,  k = k0 + 2^(L-l) m   =>   jk/N = (j0 k0 + 2^(L-l) j0 m + 2^l a k0)/N + a m,
    and a m is an integer, so the tile is u(a) v(m)^T: EXACTLY rank 1.                         [proved]
    The ID butterfly (unified/drivers/butterfly_id.py) builds its trees from coordinates, so we feed it
    *virtual* coordinates u = 0..N-1 and a wrapped kernel that maps u -> x[bitrev(u)] (`permuted_kernel`).
    With ranks 1 the butterfly has nnz = N (W) + 2N per stage x L stages + N (U) = 2N log2 N + 2N: the
    radix-2 FFT, with unimodular "twiddle" entries in G_l.                         [proved / computed]

(b) Low-rank phase + FFT (NUFFT style, H. Yang, arXiv:1803.04128, "NUFFT or butterfly?").  [cited]
        K(x, xi) = exp(2 pi i x xi) * M(x, xi),   M(x, xi) = exp(2 pi i t c(x) |xi|).
    If M ~= sum_{r<R} a_r(x) b_r(xi) (truncated SVD at relative eps), then
        K g = sum_r a_r(x) * DFT[b_r(xi) g(xi)](x)        -> R FFTs of length N.
    On the B1 grids (x = j/N, xi = k - N/2):  sum_k e^{2 pi i x_j xi_k} h_k = (-1)^j N ifft(h)_j.
"""
from __future__ import annotations
import time
from typing import Callable

import numpy as np
import scipy.linalg as sla

from unified.core import kernels
from unified.core.kernels import ButterflyKernel, SpeedProfile
from unified.core.types import (ComplexArray, FrequencyGrid, IndexArray, PositionGrid, RealArray,
                                Spectrum, Field)

# ---- domain aliases -------------------------------------------------------------------------------------
type Permutation = IndexArray     # (N,) p with virtual index u <-> physical index p[u]
type LowRank = int                # R: number of separated terms = number of FFTs per apply
type StorageCount = int           # numbers stored by a representation
type MultiplyAdds = int           # complex multiply-adds per apply (our cost unit; = nnz for sparse factors)
type Bandwidth = float            # B = t * dc * N / 2: wavelengths of phase perturbation across the domain
type VirtualCoords = RealArray    # (N,) 0..N-1 as floats: what the coordinate-based Tree sees

# ---- media c(x) on [0, 1) -----------------------------------------------------------------------------
MEDIA: dict[str, SpeedProfile] = {
    "constant": lambda x: np.full_like(np.asarray(x, float), 1 / 8),
    "weak":     lambda x: (20 + np.sin(2 * np.pi * x)) / 160,          # range 1/80: nearly constant
    "sine":     lambda x: (2 + np.sin(2 * np.pi * x)) / 16,            # SPEC B1 default, range 1/8
    "lens":     lambda x: (1 + 2 * np.exp(-((x - 0.5) / 0.05) ** 2)) / 16,  # same range 1/8, 5x steeper
}


def medium_range(name: str, n: int = 100001) -> float:
    """Delta c = max c - min c of a medium on [0, 1)."""
    c = MEDIA[name](np.arange(n) / n)
    return float(c.max() - c.min())


# ---- (a) bit-reversed orderings -----------------------------------------------------------------------
def bitrev(N: int) -> Permutation:
    """Bit-reversal permutation of 0..N-1 (N = 2^L)."""
    L = int(round(np.log2(N)))
    u = np.arange(N)
    out: Permutation = np.zeros(N, np.intp)
    for b in range(L):
        out |= ((u >> b) & 1) << (L - 1 - b)
    return out


def permuted_kernel(kernel: ButterflyKernel, x: PositionGrid, xi: FrequencyGrid, prow: Permutation,
                    pcol: Permutation) -> Callable[[VirtualCoords, VirtualCoords], ComplexArray]:
    """Wrapper so the coordinate-based Tree sees a permuted ordering: virtual row u means x[prow[u]],
    virtual column v means xi[pcol[v]]. Use with virtual coordinates u = v = arange(N) as the 'points'."""
    def K(u: VirtualCoords, v: VirtualCoords) -> ComplexArray:
        return kernel(x[prow[np.asarray(u, int)]], xi[pcol[np.asarray(v, int)]])
    return K


# ---- (b) low-rank phase + R FFTs ----------------------------------------------------------------------
class LowRankPhaseFFT:
    """K g = sum_r a_r(x) DFT[b_r g], with M = exp(2 pi i t c(x)|xi|) ~= A @ B (rank R, relative eps).

    M depends on xi only through |xi|, so the SVD is taken of the N x (N/2+1) matrix of distinct |xi|,
    with columns weighted by sqrt(multiplicity) so the singular values are exactly those of M."""

    def __init__(self, N: int, t: float, c: SpeedProfile, eps: float) -> None:
        t0 = time.perf_counter()
        self.N: int = N
        x: PositionGrid = np.arange(N) / N
        xi: FrequencyGrid = (np.arange(N) - N // 2).astype(float)
        ax = np.abs(xi)
        uniq, inv, mult = np.unique(ax, return_inverse=True, return_counts=True)
        w = np.sqrt(mult)
        Mt: ComplexArray = np.exp(2j * np.pi * t * np.outer(c(x), uniq)) * w[None, :]
        U, s, Vh = sla.svd(Mt, full_matrices=False)
        R = max(1, int(np.sum(s > eps * s[0])))
        self.sigma: RealArray = s                       # singular values of M (all of them)
        self.R: LowRank = R
        self.A: ComplexArray = U[:, :R] * s[:R]         # a_r(x): N x R
        self.B: ComplexArray = (Vh[:R] / w[None, :])[:, inv]   # b_r(xi): R x N
        self.sign: RealArray = (-1.0) ** np.arange(N, dtype=float)
        self.build_time = time.perf_counter() - t0

    def apply(self, g: Spectrum) -> Field:
        """R batched FFTs: f_j = (-1)^j sum_r a_r(j) N ifft(b_r g)_j."""
        H: ComplexArray = np.fft.ifft(self.B * g[None, :], axis=1) * self.N
        f: Field = self.sign * np.einsum("jr,rj->j", self.A, H)
        return f

    @property
    def storage(self) -> StorageCount:
        """Numbers stored: a_r on N points, b_r on the N/2+1 distinct |xi| (FFT twiddles O(N) ignored)."""
        return self.R * (self.N + self.N // 2 + 1)


def fft_cost(N: int) -> MultiplyAdds:
    """Complex multiply-adds of one length-N FFT, counted as nnz of the rank-1 (bit-reversed) butterfly
    of part (a): 2N log2 N + 2N. Same unit as a butterfly's nnz (one multiply-add per stored nonzero)."""
    return 2 * N * int(round(np.log2(N))) + 2 * N


def lowrank_cost(N: int, R: LowRank) -> MultiplyAdds:
    """R FFTs plus two diagonal scalings (b_r g and a_r * ...) per term."""
    return R * (fft_cost(N) + 2 * N)


def b1_kernel(t: float, medium: str) -> ButterflyKernel:
    return kernels.fio1d(t, MEDIA[medium])


def exact_apply(kernel: ButterflyKernel, x: PositionGrid, xi: FrequencyGrid, g: Spectrum,
                chunk: int = 512) -> Field:
    """Reference f = K g, evaluating K a block of rows at a time."""
    return np.concatenate([kernel(x[i:i + chunk], xi) @ g for i in range(0, len(x), chunk)])


def time_it(f: Callable[[], object], reps: int = 5) -> float:
    """Median wall time of f() over `reps` runs (after one warm-up)."""
    f()
    ts: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts))
