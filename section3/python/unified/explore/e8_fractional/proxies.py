"""Dial 4 (sampling) for power-law kernels |x - y|^{-p}: three proxy point sets, one class each.

All three subclass the core ProxyCompressor and override ONLY `proxy_points`, so the driver's FMM-LU Q-set
path (rskel._skeletonize), which calls `comp.proxy_points(node)` and `comp.kernel(Pp, X[box])`, works as-is.

  plane     core ProxyCompressor: a circle in the plane (2D) / the two points c +- r (1D). Green's
            representation in R^d: valid for the 2D log / 1D |x| kernels only.
  lifted    NEW: proxy points on the upper half of the sphere |(z - c, t)| = r in R^{d+1} (a semicircle for
            d = 1, a hemisphere for d = 2). Requires the kernel to accept (d+1)-dim points
            (fractional.power_kernel pads with t = 0). Why it works: |X|^{-p} with X = (x, t') in
            R^d x R^m, m = p + 2 - d, is the Laplace Green's function of R^{p+2}; it depends on t' only via
            |t'| = t, so the sphere of R^{p+2} reduces to the (d+1)-dim hemisphere in (z, t). A combination of
            box sources that vanishes on that sphere vanishes outside it (exterior maximum principle, decay
            at infinity since p > 0), in particular on the far points (t = 0). Proved for integer m; for
            fractional m the same argument runs through the weighted operator div(t^{m-1} grad) of the
            Caffarelli-Silvestre extension (conjecture for m > 2, i.e. outside the A_2 weight range).
  annulus   kernel-agnostic volume proxy (Minden-Damle-Ho-Ying 2017 annulus; Xing-Chow 2020 sample the whole
            far domain): rings of radius r .. r_max in the plane, geometrically spaced; r_max must reach the
            farthest point of the domain.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from unified.core import ProxyCompressor
from unified.core.tree import Node
from unified.core.types import Kernel, Matrix, Points


class LiftedProxy(ProxyCompressor):
    """Hemisphere proxy in R^{d+1}: n_equator points on the equator (= the plane circle) and n_lat latitudes.
    n_lat = 0 gives back exactly the plane circle (2D). In 1D: n_equator + 1 points on the semicircle."""

    def __init__(self, kernel: Kernel, points: ArrayLike, eps: float, A: Matrix | None = None,
                 n_proxy: int = 64, radius_factor: float = 2.5, n_lat: int = 8) -> None:
        super().__init__(kernel, points, eps, A=A, n_proxy=n_proxy, radius_factor=radius_factor)
        self.n_lat = n_lat

    def proxy_points(self, node: Node) -> Points:
        r, c, d = self.radius_factor * node.half_width, node.center, self.points.shape[1]
        if d == 1:   # semicircle (z, t) = (c + r cos th, r sin th), th in [0, pi]
            th = np.pi * np.arange(self.n_proxy + 1) / self.n_proxy
            semi: Points = np.c_[c[0] + r * np.cos(th), r * np.sin(th)]
            return semi
        out: list[Points] = []
        for j in range(self.n_lat + 1):          # polar angle phi from the equator (j = 0) to the pole
            phi = 0.5 * np.pi * j / (self.n_lat + 1) if self.n_lat else 0.0
            m = max(4, int(round(self.n_proxy * np.cos(phi))))
            th = 2 * np.pi * (np.arange(m) + 0.5 * (j % 2)) / m
            rho, t = r * np.cos(phi), r * np.sin(phi)
            out.append(np.c_[c[0] + rho * np.cos(th), c[1] + rho * np.sin(th), np.full(m, t)])
        if self.n_lat:
            out.append(np.array([[c[0], c[1], r]]))   # the pole
        return np.vstack(out)


class AnnulusProxy(ProxyCompressor):
    """Volume proxy in the plane: n_rings rings with radii geometric from r to r_max (2D: n_proxy points each;
    1D: the two points c +- radius per ring)."""

    def __init__(self, kernel: Kernel, points: ArrayLike, eps: float, A: Matrix | None = None,
                 n_proxy: int = 64, radius_factor: float = 2.5, n_rings: int = 8,
                 r_max: float | None = None) -> None:
        super().__init__(kernel, points, eps, A=A, n_proxy=n_proxy, radius_factor=radius_factor)
        P = self.points
        self.n_rings = n_rings
        self.r_max = r_max if r_max is not None else float(np.linalg.norm(P.max(0) - P.min(0)))

    def proxy_points(self, node: Node) -> Points:
        r = self.radius_factor * node.half_width
        radii = np.geomspace(r, max(self.r_max, 1.01 * r), self.n_rings)
        c, d = node.center, self.points.shape[1]
        if d == 1:
            pts1: Points = np.concatenate([c[0] - radii, c[0] + radii])[:, None]
            return pts1
        out: list[Points] = []
        for j, rad in enumerate(radii):
            th = 2 * np.pi * (np.arange(self.n_proxy) + 0.5 * (j % 2)) / self.n_proxy
            out.append(c[None, :] + rad * np.c_[np.cos(th), np.sin(th)])
        return np.vstack(out)
