"""E6 physics: rigid particles in 2D Stokes flow (viscosity mu = 1), as a dense second-kind Nystrom matrix.

Everything the harness needs from the physics lives here. The harness itself (tree, admissibility,
rskel_factor) is not touched: a vector field with 2 components per boundary point becomes 2 unknowns
("dofs") per point, INTERLEAVED (dof 2i = x-component at point i, dof 2i+1 = y-component), and the tree
is built over the dof coordinates (each point listed twice), so both components of a point always sit
in the same box.

Kernels (r = x - y, rho = |r|, interleaved (2m, 2n) blocks):
    Stokeslet       G(x, y)    = (1/4pi) (-log rho I + r r^T / rho^2)          velocity of a point force
    stresslet (DL)  D(x, y; n) = (1/pi) (r . n_y) r r^T / rho^4                 double layer, source normal n_y
    Laplace DL      d(x, y; n) = (r . n_y) / (2 pi rho^2)                       scalar comparison
Diagonal (self) limits on a smooth curve with curvature kappa and unit tangent t (derived in findings.md):
    D -> -(kappa / 2pi) t t^T,     d -> -kappa / 4pi.

Completed double layer (Power-Miranda 1987 in 3D; 2D: Rachh-Greengard 2016). For bodies B_1..B_K with
given forces F_k and torques T_k, write the exterior velocity as
    u = u_inf + D[mu] + sum_k ( G(., c_k) F_k + R(., c_k) T_k ),      R(x, c) T = T (x - c)^perp / (4 pi |x - c|^2).
The exterior limit of D is (1/2 I + K); its null space on each body is the 3 rigid motions. Adding the
rigid-motion projector P removes it and makes the boundary condition "u = rigid motion" read
    (1/2 I + K + P) mu = -u_inf - sum_k (G F_k + R T_k),        rigid motion of body k = -(P mu)|_k.
So the solved matrix is A = 1/2 I + K W + P: kernel entries plus a rank-3-per-body correction P.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from unified.core.types import IndexArray, Matrix, Points, RealArray, Vector

type Normals = RealArray          # (n, 2) outward unit normals
type QuadWeights = RealArray      # (n,) trapezoid weights (arc length per point)
type BodyId = NDArray[np.intp]    # (n,) which body each boundary point belongs to
type RigidBasis = RealArray       # (2n_b, 3) columns: x-translation, y-translation, rotation about c
type Mobility = RealArray         # (K, 3): (U_x, U_y, Omega) of each body

INV4PI = 1.0 / (4 * np.pi)


@dataclass
class Geometry:
    """Boundary points of all bodies, concatenated body by body (counter-clockwise, outward normals)."""
    X: Points
    normal: Normals
    tangent: Normals
    kappa: RealArray
    w: QuadWeights
    body: BodyId
    centers: Points                  # (K, 2) reference point c_k of each body (torques and rotations)
    pw: RealArray | None = None      # weights of the inner product defining P (default: w). Any positive weights
                                     # give a valid formulation; rough ones are used to stress the proxy (R7)

    def proj_w(self) -> RealArray:
        return self.w if self.pw is None else self.pw

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def K(self) -> int:
        return int(self.centers.shape[0])

    def dof_points(self) -> Points:
        """Coordinates of the 2n interleaved dofs (each point twice): what the Tree is built on."""
        P: Points = np.repeat(self.X, 2, axis=0)
        return P

    def dofs_of_body(self, k: int) -> IndexArray:
        pts = np.flatnonzero(self.body == k)
        out: IndexArray = np.stack([2 * pts, 2 * pts + 1], 1).ravel()
        return out


def ellipse(n: int, a: float, b: float, center: tuple[float, float] = (0.0, 0.0), angle: float = 0.0) -> Geometry:
    """Ellipse with semi-axes a (along `angle`) and b, n points equispaced in the parameter."""
    th = 2 * np.pi * np.arange(n) / n
    c, s = np.cos(angle), np.sin(angle)
    Rm = np.array([[c, -s], [s, c]])
    p = np.c_[a * np.cos(th), b * np.sin(th)] @ Rm.T
    dp = np.c_[-a * np.sin(th), b * np.cos(th)] @ Rm.T
    sp = np.linalg.norm(dp, axis=1)
    t = dp / sp[:, None]
    kap = a * b / sp ** 3
    return Geometry(p + np.asarray(center), np.c_[t[:, 1], -t[:, 0]], t, kap, sp * 2 * np.pi / n,
                    np.zeros(n, np.intp), np.asarray([center], float))


def starfish(n: int, amp: float = 0.3, arms: int = 5, scale: float = 1.0,
             center: tuple[float, float] = (0.0, 0.0), angle: float = 0.0) -> Geometry:
    """Starfish r = scale (1 + amp cos(arms th)) (the P1 curve of SPEC.md at scale 1)."""
    th = 2 * np.pi * np.arange(n) / n + angle
    r = scale * (1 + amp * np.cos(arms * (th - angle)))
    dr = -scale * amp * arms * np.sin(arms * (th - angle))
    d2r = -scale * amp * arms ** 2 * np.cos(arms * (th - angle))
    p = np.c_[r * np.cos(th), r * np.sin(th)]
    dp = np.c_[dr * np.cos(th) - r * np.sin(th), dr * np.sin(th) + r * np.cos(th)]
    d2p = np.c_[d2r * np.cos(th) - 2 * dr * np.sin(th) - r * np.cos(th),
                d2r * np.sin(th) + 2 * dr * np.cos(th) - r * np.sin(th)]
    sp = np.linalg.norm(dp, axis=1)
    t = dp / sp[:, None]
    kap = (dp[:, 0] * d2p[:, 1] - dp[:, 1] * d2p[:, 0]) / sp ** 3
    return Geometry(p + np.asarray(center), np.c_[t[:, 1], -t[:, 0]], t, kap, sp * 2 * np.pi / n,
                    np.zeros(n, np.intp), np.asarray([center], float))


def union(parts: list[Geometry]) -> Geometry:
    """Several bodies as one geometry (body ids renumbered 0..K-1)."""
    return Geometry(np.vstack([g.X for g in parts]), np.vstack([g.normal for g in parts]),
                    np.vstack([g.tangent for g in parts]), np.concatenate([g.kappa for g in parts]),
                    np.concatenate([g.w for g in parts]),
                    np.concatenate([np.full(g.n, k, np.intp) for k, g in enumerate(parts)]),
                    np.vstack([g.centers for g in parts]))


# ---- kernels ------------------------------------------------------------------------------------------
def _interleave(B: NDArray[np.float64]) -> RealArray:
    """(m, n, 2, 2) per-pair 2x2 blocks -> (2m, 2n) matrix with rows 2i+a, columns 2j+b."""
    m, n = B.shape[:2]
    out: RealArray = B.transpose(0, 2, 1, 3).reshape(2 * m, 2 * n)
    return out


def stokeslet(X: Points, Y: Points) -> RealArray:
    """G(x_i, y_j) as an interleaved (2m, 2n) matrix; pairs with x = y are set to 0."""
    r = X[:, None, :] - Y[None, :, :]
    r2 = np.einsum("ijk,ijk->ij", r, r)
    nz = r2 > 0
    safe = np.where(nz, r2, 1.0)
    B = (r[..., :, None] * r[..., None, :]) / safe[..., None, None]
    B -= (0.5 * np.log(safe))[..., None, None] * np.eye(2)
    B[~nz] = 0.0
    return _interleave(INV4PI * B)


def stresslet_dl(X: Points, Y: Points, NY: Normals) -> RealArray:
    """Double-layer kernel D(x_i, y_j; n_j) as an interleaved (2m, 2n) matrix; x = y pairs set to 0."""
    r = X[:, None, :] - Y[None, :, :]
    r2 = np.einsum("ijk,ijk->ij", r, r)
    nz = r2 > 0
    safe = np.where(nz, r2, 1.0)
    rn = np.einsum("ijk,jk->ij", r, NY)
    B = (rn / (np.pi * safe ** 2))[..., None, None] * (r[..., :, None] * r[..., None, :])
    B[~nz] = 0.0
    return _interleave(B)


def laplace_dl(X: Points, Y: Points, NY: Normals) -> RealArray:
    """Scalar Laplace double layer (r . n_y) / (2 pi |r|^2); x = y pairs set to 0."""
    r = X[:, None, :] - Y[None, :, :]
    r2 = np.einsum("ijk,ijk->ij", r, r)
    nz = r2 > 0
    out: RealArray = np.where(nz, np.einsum("ijk,jk->ij", r, NY) / (2 * np.pi * np.where(nz, r2, 1.0)), 0.0)
    return out


# ---- the Nystrom matrices -----------------------------------------------------------------------------
def rigid_basis(g: Geometry, k: int) -> RigidBasis:
    """Rigid motions of body k sampled at its dofs: (1,0), (0,1), (x - c)^perp = (-(y - c_y), x - c_x)."""
    pts = np.flatnonzero(g.body == k)
    d = g.X[pts] - g.centers[k]
    V = np.zeros((2 * pts.size, 3))
    V[0::2, 0] = 1.0
    V[1::2, 1] = 1.0
    V[0::2, 2], V[1::2, 2] = -d[:, 1], d[:, 0]
    return V


def projector_blocks(g: Geometry) -> list[tuple[IndexArray, Matrix, Matrix]]:
    """Per body k: (dofs, V, Z) with P restricted to the body = V Z, Z = (V^T W V)^{-1} V^T W (weighted L2 projector)."""
    out: list[tuple[IndexArray, Matrix, Matrix]] = []
    for k in range(g.K):
        dofs = g.dofs_of_body(k)
        V = rigid_basis(g, k)
        Wd = np.repeat(g.proj_w()[g.body == k], 2)
        Z: Matrix = np.linalg.solve(V.T @ (Wd[:, None] * V), (V * Wd[:, None]).T)
        out.append((dofs, V, Z))
    return out


def stokes_matrix(g: Geometry, with_projector: bool = True) -> Matrix:
    """A = 1/2 I + K W + P  (interleaved, size 2n). K: stresslet with the curvature diagonal limit."""
    K = stresslet_dl(g.X, g.X, g.normal)
    tt = g.tangent[:, :, None] * g.tangent[:, None, :]
    diag = -(g.kappa / (2 * np.pi))[:, None, None] * tt
    for a in range(2):
        for b in range(2):
            K[2 * np.arange(g.n) + a, 2 * np.arange(g.n) + b] = diag[:, a, b]
    A: Matrix = 0.5 * np.eye(2 * g.n) + K * np.repeat(g.w, 2)[None, :]
    if with_projector:
        for dofs, V, Z in projector_blocks(g):
            A[np.ix_(dofs, dofs)] += np.asarray(V @ Z, float)
    return A


def laplace_matrix(g: Geometry, with_projector: bool = True) -> Matrix:
    """Scalar analogue: A = 1/2 I + K_L W + P_L, P_L = projector onto constants per body (Laplace "elastance" form)."""
    K = laplace_dl(g.X, g.X, g.normal)
    K[np.arange(g.n), np.arange(g.n)] = -g.kappa / (4 * np.pi)
    A: Matrix = 0.5 * np.eye(g.n) + K * g.w[None, :]
    if with_projector:
        for k in range(g.K):
            idx = np.flatnonzero(g.body == k)
            pw = g.proj_w()[idx]
            A[np.ix_(idx, idx)] += np.ones((idx.size, 1)) @ (pw[None, :] / pw.sum())
    return A


def area_problem(n: int, scale: float = 1.0, physics: str = "stokes", seed: int = 0) -> tuple[Geometry, Matrix]:
    """P2 analogue: n random point forces in [0, scale]^2, A = I + G w with w = scale^2 / n (a volume sum, not a
    boundary integral). Stokes: G = Stokeslet (2n dofs); Laplace: -log r / 2 pi. Normals are unused."""
    X = scale * np.random.default_rng(seed).random((n, 2))
    w = np.full(n, scale ** 2 / n)
    z = np.zeros(n)
    g = Geometry(X, np.c_[z + 1, z], np.c_[z, z + 1], z, w, np.zeros(n, np.intp), np.zeros((1, 2)))
    if physics == "stokes":
        A: Matrix = np.eye(2 * n) + stokeslet(X, X) * np.repeat(w, 2)[None, :]
    else:
        r = np.linalg.norm(X[:, None] - X[None], axis=2)
        A = np.eye(n) - np.log(np.where(r > 0, r, 1.0)) / (2 * np.pi) * w[None, :]
    return g, A


# ---- right-hand sides and read-out ---------------------------------------------------------------------
def rhs_forces(g: Geometry, F: RealArray, T: RealArray) -> Vector:
    """-(sum_k G(x, c_k) F_k + R(x, c_k) T_k) at every boundary dof. F: (K, 2), T: (K,)."""
    u = np.zeros(2 * g.n)
    for k in range(g.K):
        u += stokeslet(g.X, g.centers[k:k + 1]) @ F[k]
        d = g.X - g.centers[k]
        rot = T[k] * INV4PI * np.c_[-d[:, 1], d[:, 0]] / np.einsum("ij,ij->i", d, d)[:, None]
        u += rot.ravel()
    out: Vector = -u
    return out


def rhs_shear(g: Geometry, gamma: float = 1.0) -> Vector:
    """-u_inf for simple shear u_inf = (gamma y, 0)."""
    u = np.zeros((g.n, 2))
    u[:, 0] = gamma * g.X[:, 1]
    out: Vector = -u.ravel()
    return out


def rigid_motions(g: Geometry, mu: Vector) -> Mobility:
    """(U_x, U_y, Omega) of every body = -(P mu) expressed in the rigid basis."""
    out = np.zeros((g.K, 3))
    for k, (dofs, _, Z) in enumerate(projector_blocks(g)):
        out[k] = -(Z @ mu[dofs])
    return out


def velocity_field(g: Geometry, mu: Vector, targets: Points) -> RealArray:
    """Double-layer part D[mu](x) at off-boundary targets (plain trapezoid rule; accurate away from the bodies)."""
    u = stresslet_dl(targets, g.X, g.normal) @ (mu * np.repeat(g.w, 2))
    out: RealArray = np.asarray(u, float).reshape(-1, 2)
    return out


# ---- analytic answers -----------------------------------------------------------------------------------
def circle_mobility(a: float, F: RealArray, T: float) -> RealArray:
    """Single circle of radius a in unbounded 2D Stokes flow, u = G F + O(1/r) at infinity (no added constant):
    U = (1/2 - log a) F / (4 pi),  Omega = T / (4 pi a^2)."""
    out: RealArray = np.r_[(0.5 - np.log(a)) * INV4PI * np.asarray(F, float), T * INV4PI / a ** 2]
    return out


def jeffery_omega(a: float, b: float, phi: float, gamma: float = 1.0) -> float:
    """Angular velocity of a force- and torque-free ellipse (semi-axes a, b, major axis at angle phi) in simple
    shear u_inf = (gamma y, 0): Omega = -gamma (a^2 sin^2 phi + b^2 cos^2 phi) / (a^2 + b^2)  (Jeffery 1922)."""
    return float(-gamma * (a ** 2 * np.sin(phi) ** 2 + b ** 2 * np.cos(phi) ** 2) / (a ** 2 + b ** 2))


def random_suspension(K: int, n_per: int, box: float, gap: float, seed: int = 0,
                      a_range: tuple[float, float] = (0.25, 0.45), aspect: tuple[float, float] = (0.4, 1.0),
                      max_tries: int = 20000) -> Geometry:
    """K random non-overlapping ellipses in [0, box]^2 whose boundary points stay >= gap apart."""
    rng = np.random.default_rng(seed)
    parts: list[Geometry] = []
    tries = 0
    while len(parts) < K:
        tries += 1
        if tries > max_tries:
            raise RuntimeError(f"could only place {len(parts)} of {K} bodies")
        a = rng.uniform(*a_range)
        b = a * rng.uniform(*aspect)
        c = (float(rng.uniform(a + gap, box - a - gap)), float(rng.uniform(a + gap, box - a - gap)))
        e = ellipse(n_per, a, b, c, float(rng.uniform(0, np.pi)))
        if all(np.min(np.linalg.norm(e.X[:, None] - q.X[None], axis=2)) >= gap
               and np.linalg.norm(np.subtract(c, q.centers[0])) > max(a, float(np.max(np.linalg.norm(q.X - q.centers[0], axis=1)))) for q in parts):
            parts.append(e)
    return union(parts)
