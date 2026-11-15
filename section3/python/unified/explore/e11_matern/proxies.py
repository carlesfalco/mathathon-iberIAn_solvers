"""E11 dial 4 (sampling): proxy geometries for Matérn kernels. One class, `MaternProxy`, subclassing the core
ProxyCompressor and overriding ONLY `proxy_points`, so the driver's FMM-LU Q-set path (rskel._skeletonize calls
comp.proxy_points(node) and comp.kernel(Pp, X[box])) works unchanged. Which kernel ORDERS are sampled is decided by
the StackedKernel (matern.py): one order = the stock "one kernel" proxy, beta orders = the multi-kernel proxy.

Geometries (all return 3 columns (x, y, t); t is the extension coordinate):
  plane     circle of n_proxy points, radius radius_factor * half-width, t = 0          (core proxy)
  lifted    hemisphere |(z - c, t)| = r, t >= 0, with n_lat latitudes (E8's LiftedProxy, re-implemented here
            because E8's directory is still being edited)
  annulus   n_rings circles, radii geometric from r to the domain diameter, t = 0  (Minden-Damle-Ho-Ying 2017)
  xc        Xing-Chow-style selected volume proxy: per box size, 3000 random candidates in the far annulus
            (radii log-uniform in [r, diameter]) reduced by a pivoted QR of K(candidates, box samples) at eps/10
            (Xing & Chow 2020, Alg. 5.1 without densification; same recipe as E8's XCProxy)
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.linalg as sla

from unified.core import ProxyCompressor
from unified.core.tree import Node
from unified.core.types import Points

from .matern import StackedKernel

type Geometry = Literal["plane", "lifted", "annulus", "xc"]


class MaternProxy(ProxyCompressor):
    """Proxy compressor whose proxy point set is one of the four geometries above."""

    def __init__(self, kernel: StackedKernel, points: Points, eps: float, geometry: Geometry,
                 A: np.ndarray | None = None, n_proxy: int = 64, radius_factor: float = 2.5, n_lat: int = 4,
                 n_rings: int = 8, n_cand: int = 3000, seed: int = 0) -> None:
        super().__init__(kernel, points, eps, A=A, n_proxy=n_proxy, radius_factor=radius_factor)
        self.geometry, self.n_lat, self.n_rings = geometry, n_lat, n_rings
        self.skernel = kernel
        P = self.points
        self.r_max = float(np.linalg.norm(P.max(0) - P.min(0)))
        self.n_cand, self.seed = n_cand, seed
        self.cache: dict[float, Points] = {}

    def _circle(self, c: np.ndarray, rad: float, m: int, shift: float, t: float = 0.0) -> Points:
        th = 2 * np.pi * (np.arange(m) + shift) / m
        out: Points = np.c_[c[0] + rad * np.cos(th), c[1] + rad * np.sin(th), np.full(m, t)]
        return out

    def _hemisphere(self, c: np.ndarray, r: float) -> Points:
        out: list[Points] = []
        for j in range(self.n_lat + 1):                  # polar angle phi from the equator (j = 0) to the pole
            phi = 0.5 * np.pi * j / (self.n_lat + 1)
            m = max(4, int(round(self.n_proxy * np.cos(phi))))
            out.append(self._circle(c, r * np.cos(phi), m, 0.5 * (j % 2), r * np.sin(phi)))
        if self.n_lat:
            out.append(np.array([[c[0], c[1], r]]))
        return np.vstack(out)

    def _select(self, hw: float) -> Points:
        """Xing-Chow-style selection, relative to the box centre, cached per box size."""
        rng = np.random.default_rng(self.seed)
        r1 = self.radius_factor * hw
        rad = np.exp(rng.uniform(np.log(r1), np.log(max(self.r_max, 1.01 * r1)), self.n_cand))
        th = rng.uniform(0, 2 * np.pi, self.n_cand)
        cand: Points = np.c_[rad * np.cos(th), rad * np.sin(th), np.zeros(self.n_cand)]
        g = np.linspace(-hw, hw, 20)
        box: Points = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
        K = self.skernel(cand, box)
        _, R, piv = sla.qr(K.T, mode="economic", pivoting=True)
        dg = np.abs(np.diag(R))
        k = int(np.sum(dg > 0.1 * self.eps * dg[0]))
        sel: Points = cand[piv[:k] % self.n_cand]
        return np.unique(sel, axis=0)

    def proxy_points(self, node: Node) -> Points:
        c, r = node.center, self.radius_factor * node.half_width
        if self.geometry == "plane":
            return self._circle(c, r, self.n_proxy, 0.0)
        if self.geometry == "lifted":
            return self._hemisphere(c, r)
        if self.geometry == "annulus":
            radii = np.geomspace(r, max(self.r_max, 1.01 * r), self.n_rings)
            return np.vstack([self._circle(c, rad, self.n_proxy, 0.5 * (j % 2)) for j, rad in enumerate(radii)])
        hw = round(node.half_width, 12)
        if hw not in self.cache:
            self.cache[hw] = self._select(node.half_width)
        out: Points = self.cache[hw] + np.r_[c, 0.0][None, :]
        return out
