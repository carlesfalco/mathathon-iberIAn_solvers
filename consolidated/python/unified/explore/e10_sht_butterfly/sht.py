"""The associated Legendre transform as a butterfly kernel, and the small wrapper needed to feed it to
`unified.drivers.butterfly_id` unchanged.

Physics. A field on the sphere (temperature, geopotential, CMB anisotropy) is expanded in spherical
harmonics  f(θ, φ) = Σ_{l<n} Σ_{|m|<=l} a_lm  P̃_l^m(cos θ) e^{imφ}.  The φ direction is an FFT. For each
fixed order m what remains is the matrix
        A^(m)[j, l] = P̃_l^m(cos θ_j),     j = 0..n-1 (Gauss-Legendre latitudes),  l = m..n-1,
with P̃_l^m the associated Legendre function normalised so that ∫_{-1}^{1} P̃_l^m(x)^2 dx = 1.
Synthesis is  f_m = A a_m ;  analysis is  a_m = Aᵀ diag(w) f_m  (exact by Gauss quadrature, because
P̃_l^m P̃_l'^m is a polynomial of degree l + l' <= 2n - 2).                              [proved]

The kernel swap. The driver only ever asks for blocks  kernel(x[rows], xi[cols]).  We hand it
    x  = the tree coordinate of the latitudes (θ by default: the "clustering" dial),
    xi = the degree l (as a float),
and a kernel that looks the entries up in a table computed once by the three-term recurrence. Nothing in
the driver changes; the only extra code is (i) the table, (ii) zero-padding because A^(m) is n x (n-m)
(the driver sizes W by len(x)), (iii) applying the transposed factors for analysis.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Literal, cast

import numpy as np
from numpy.typing import NDArray
import scipy.linalg as sla
from scipy.special import roots_legendre

from unified.core import Tree
from unified.core.kernels import ButterflyKernel
from unified.core.types import IDResult, Matrix, ComplexArray, FrequencyGrid, IndexArray, PositionGrid, RealArray
from unified.drivers.butterfly_id import IDButterfly, butterfly_id

type Coord = Literal["theta", "x", "bitrev", "random"]     # dial 1: which coordinate the trees cut

# ---- associated Legendre functions -------------------------------------------------------------------
def alf_table(theta: RealArray, m: int, lmax: int) -> RealArray:
    """P̃_l^m(cos θ_j) for l = m..lmax, shape (len(theta), lmax - m + 1).

    Orthonormal on [-1, 1] (no Condon-Shortley sign). Three-term recurrence in l with a per-node
    exponent (an "X-number"-style rescaling, cf. Fukushima 2012), so that P̃_m^m = c_m sin^m θ, which
    underflows for large m near the poles, still recovers correctly past the turning point. [computed:
    agrees with scipy.special.lpmv to 1e-13 for l <= 60]"""
    th = np.asarray(theta, float)
    x, s = np.cos(th), np.sin(th)
    nl = lmax - m + 1
    out = np.zeros((len(th), nl))
    if nl <= 0:
        return out
    k = np.arange(1, m + 1)
    logc = 0.5 * np.log((2 * m + 1) / 2.0) + 0.5 * float(np.sum(np.log((2 * k - 1) / (2.0 * k))))
    with np.errstate(divide="ignore"):
        E = logc + m * np.log(s)                 # log |P̃_m^m|, may be -inf only if s == 0 (never at Gauss nodes)
    p_prev = np.zeros_like(x)                    # scaled P̃_{l-2}
    p = np.ones_like(x)                          # scaled P̃_{l-1}: true value = p * exp(E)
    out[:, 0] = np.exp(E)
    big = 1e150
    for i in range(1, nl):
        l = m + i
        a = np.sqrt((4.0 * l * l - 1.0) / (l * l - m * m))
        b = np.sqrt(((l - 1.0) ** 2 - m * m) / (4.0 * (l - 1.0) ** 2 - 1.0))
        p_new = a * (x * p - b * p_prev)
        p_prev, p = p, p_new
        scale = np.abs(p) > big
        if scale.any():                           # rescale: keep magnitudes O(1), move the size into E
            p[scale] /= big
            p_prev[scale] /= big
            E[scale] += np.log(big)
        with np.errstate(under="ignore", over="ignore"):
            out[:, i] = p * np.exp(np.minimum(E, 700.0))
    return out


def _legendre_pn(x: RealArray, n: int) -> tuple[RealArray, RealArray]:
    """P_n(x) and P_{n-1}(x) (classical normalisation) by the three-term recurrence."""
    p0, p1 = np.ones_like(x), x.copy()
    for k in range(2, n + 1):
        p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
    return p1, p0


def gauss_nodes(n: int) -> tuple[RealArray, RealArray]:
    """Gauss-Legendre latitudes as colatitudes θ_j (increasing) and weights w_j (sum 2).
    Newton polishing uses the positive theta derivative d P_n / dθ = n(x P_n − P_{n-1})/sin(theta).
    Orthonormality of the resulting node/weight set is verified independently. [computed]"""
    xg, _ = roots_legendre(n)
    th = np.sort(np.arccos(xg))
    for _ in range(2):
        x = np.cos(th)
        pn, pm = _legendre_pn(x, n)
        # (1-x^2) P_n'(x) = n(P_{n-1} - x P_n), with dx/dtheta = -sin(theta);
        # the two signs cancel, so d P_n/dtheta = n(x P_n - P_{n-1})/sin(theta).
        dpdth = n * (x * pn - pm) / np.sin(th)
        th = th - pn / dpdth
    x = np.cos(th)
    pn, pm = _legendre_pn(x, n)
    dpdth = n * (x * pn - pm) / np.sin(th)
    w = 2.0 / dpdth ** 2                                 # w_j = 2 / ((1 - x^2) P_n'(x)^2) = 2 / (dP_n/dθ)^2
    return th, np.asarray(w, float)


# ---- problem and kernel ------------------------------------------------------------------------------
def bitrev(n: int) -> IndexArray:
    """Bit-reversal permutation of 0..n-1 (n a power of 2)."""
    L = int(round(np.log2(n)))
    assert 2 ** L == n
    r = np.zeros(n, dtype=np.intp)
    for b in range(L):
        r |= ((np.arange(n) >> b) & 1) << (L - 1 - b)
    return r


@dataclass
class SHTProblem:
    """A^(m) on n Gauss latitudes, its tree coordinates, and a table kernel for the driver."""
    n: int
    m: int
    theta: RealArray        # (n,) colatitudes
    w: RealArray            # (n,) Gauss weights
    ls: RealArray           # (n - m,) degrees m..n-1
    A: RealArray            # (n, n - m) dense reference
    xr: RealArray           # (n,) row tree coordinate
    xc: RealArray           # (n - m,) column tree coordinate

    def kernel(self) -> ButterflyKernel:
        """kernel(xr[rows], xc[cols]) = A[rows, cols]: maps the coordinates back to indices."""
        ro, co = np.argsort(self.xr), np.argsort(self.xc)
        xs, cs = self.xr[ro], self.xc[co]
        A = self.A

        def K(xq: PositionGrid, cq: FrequencyGrid) -> ComplexArray:
            i = ro[np.searchsorted(xs, np.asarray(xq, float))]
            j = co[np.searchsorted(cs, np.asarray(cq, float))]
            blk: RealArray = A[np.ix_(i, j)]
            return cast(ComplexArray, blk)       # real entries; the driver's ID and factors accept them as-is
        return K


def sht_problem(n: int, m: int, coord: Coord = "theta", clip: float = 0.0, seed: int = 0) -> SHTProblem:
    """Build A^(m). `coord` picks the tree coordinate (dial 1). `clip` > 0 zeroes entries with
    |A_ij| < clip * max|A| (the evanescent region below the turning point; a kernel-only change)."""
    theta, w = gauss_nodes(n)
    A = alf_table(theta, m, n - 1)
    if clip > 0:
        A = np.where(np.abs(A) < clip * np.abs(A).max(), 0.0, A)
    ls = np.arange(m, n, dtype=float)
    nc = n - m
    if coord == "theta":
        xr, xc = theta.copy(), ls.copy()
    elif coord == "x":
        xr, xc = np.cos(theta), ls.copy()
    elif coord == "bitrev":                       # E2's trick for the DFT: positions = bit-reversed indices
        assert m == 0, "bit reversal needs n - m a power of 2"
        xr, xc = bitrev(n).astype(float), bitrev(nc).astype(float)
    else:
        rng = np.random.default_rng(seed)
        xr, xc = rng.permutation(n).astype(float), rng.permutation(nc).astype(float)
    return SHTProblem(n, m, theta, w, ls, A, xr, xc)


# ---- the butterfly (driver unchanged) ----------------------------------------------------------------
@dataclass
class SHTButterfly:
    """ID butterfly of A^(m), plus synthesis (A a) and analysis (Aᵀ diag(w) f) through its factors."""
    bf: IDButterfly
    n: int
    nc: int
    w: RealArray

    def synth(self, a: NDArray[np.inexact]) -> NDArray[np.complex128]:
        """f = A a.  The driver's W has len(x) = n columns; only the first n - m are used, so pad."""
        g = np.zeros((self.n,) + a.shape[1:], dtype=complex)
        g[: self.nc] = a
        return cast(NDArray[np.complex128], self.bf.apply(g))

    def adjoint(self, f: NDArray[np.inexact]) -> NDArray[np.complex128]:
        """Aᵀ f through the transposed factors: Wᵀ G_0ᵀ ... G_{L-1}ᵀ Uᵀ f  (A is real, so ᵀ not ᴴ)."""
        c = self.bf.U.T @ f
        for Gl in reversed(self.bf.G):
            c = Gl.T @ c
        out = cast(NDArray[np.complex128], self.bf.W.T @ c)
        return out[: self.nc]

    def analyze(self, f: NDArray[np.inexact]) -> NDArray[np.complex128]:
        """a = Aᵀ diag(w) f  (Gauss quadrature; exact for band-limited f)."""
        wf = self.w[:, None] * f if f.ndim == 2 else self.w * f
        return self.adjoint(wf)

    @property
    def nnz(self) -> int:
        return self.bf.nnz


def build(p: SHTProblem, eps: float) -> SHTButterfly:
    """Two trees of equal depth L = log2(n) over the chosen coordinates; then the unmodified driver."""
    L = int(np.ceil(np.log2(p.n)))
    rt, ct = Tree(p.xr, depth=L), Tree(p.xc, depth=L)
    bf = butterfly_id(p.kernel(), p.xr, p.xc, eps, row_tree=rt, col_tree=ct)
    return SHTButterfly(bf, p.n, p.n - p.m, p.w)


# ---- diagnostics -------------------------------------------------------------------------------------
def turning_class(theta: RealArray, ls: RealArray, m: int) -> NDArray[np.int8]:
    """+1 where P_l^m(cos θ) oscillates ((l + 1/2) sin θ > m), 0 where it is evanescent (WKB). [cited]"""
    osc: NDArray[np.int8] = ((ls[None, :] + 0.5) * np.sin(theta)[:, None] > m).astype(np.int8)
    return osc


def tile_ranks(p: SHTProblem, level: int, eps: float, absolute: bool = False) -> list[dict[str, float]]:
    """Numerical eps-rank of every complementary tile (row box at depth `level`, column box at depth
    L - level) of A^(m), relative to the tile's own norm (what the driver's ID uses) or, if `absolute`,
    to ||A||_2. Each record also says whether the tile is oscillatory, evanescent, or straddles the
    turning curve."""
    L = int(np.ceil(np.log2(p.n)))
    rt, ct = Tree(p.xr, depth=L), Tree(p.xc, depth=L)
    normA = float(np.linalg.norm(p.A, 2))
    osc = turning_class(p.theta, p.ls, p.m)
    out: list[dict[str, float]] = []
    for a in rt.levels[level]:
        for b in ct.levels[L - level]:
            blk = p.A[np.ix_(a.idx, b.idx)]
            s = np.linalg.svd(blk, compute_uv=False)
            ref = normA if absolute else (s[0] if s.size else 0.0)
            r = int(np.sum(s > eps * ref)) if ref > 0 else 0
            o = osc[np.ix_(a.idx, b.idx)]
            cls = 2.0 if o.all() else (0.0 if not o.any() else 1.0)   # 2 osc, 1 straddle, 0 evanescent
            out.append(dict(rank=r, cls=cls, theta=float(p.theta[a.idx].mean()), l=float(p.ls[b.idx].mean()),
                            size=float(blk.size), norm=float(s[0]) if s.size else 0.0))
    return out


def cheb_problem(n: int, coord: Coord = "theta") -> SHTProblem:
    """The cleanest case: the Chebyshev transform T_l(x_j) = cos(l θ_j) at the n Chebyshev points
    θ_j = π (j + ½) / n (a DCT-II). Stored as an SHTProblem with m = 0 and weights such that
    analysis is again  a = Aᵀ diag(w) f  with w_j = 2/n, after the l = 0 column is scaled by 1/√2."""
    theta = np.pi * (np.arange(n) + 0.5) / n
    ls = np.arange(n, dtype=float)
    A = np.cos(np.outer(theta, ls))
    A[:, 0] /= np.sqrt(2.0)                       # makes Aᵀ (2/n) A = I exactly   [proved: discrete orthogonality]
    w = np.full(n, 2.0 / n)
    if coord == "theta":
        xr, xc = theta.copy(), ls.copy()
    elif coord == "x":
        xr, xc = np.cos(theta), ls.copy()
    elif coord == "bitrev":
        xr, xc = bitrev(n).astype(float), bitrev(n).astype(float)
    else:
        rng = np.random.default_rng(0)
        xr, xc = rng.permutation(n).astype(float), rng.permutation(n).astype(float)
    return SHTProblem(n, 0, theta, w, ls, A, xr, xc)


# ---- dial-4 variant: absolute tolerance in the ID (wrapper; no file outside this directory changes) ----
@contextmanager
def absolute_tolerance(scale: float) -> Iterator[None]:
    """Inside this block the butterfly driver's ID keeps pivots with |R_kk| > eps * scale instead of
    eps * |R_00| (relative to the tile). `scale` = max|A|, so exponentially small tiles get rank 0.
    Implemented by rebinding the name `interp_decomp` inside unified.drivers.butterfly_id for the
    duration of the block (restored afterwards)."""
    import importlib
    drv: Any = importlib.import_module("unified.drivers.butterfly_id")   # Any: we rebind a module global
    orig = drv.interp_decomp

    def id_abs(M: Matrix, eps: float) -> IDResult:
        n = M.shape[1]
        if n == 0 or M.shape[0] == 0 or not np.any(M):
            return IDResult(np.zeros(0, np.intp), np.arange(n, dtype=np.intp), np.zeros((0, n)))
        _, R, piv = sla.qr(M, mode="economic", pivoting=True)
        k = int(np.sum(np.abs(np.diag(R)) > eps * scale))
        skel, red = piv[:k].astype(np.intp), piv[k:].astype(np.intp)
        T = sla.solve_triangular(R[:k, :k], R[:k, k:]) if k else np.zeros((0, n))
        return IDResult(skel, red, T)

    drv.interp_decomp = id_abs
    try:
        yield
    finally:
        drv.interp_decomp = orig
