"""E12 dial 2 (admissibility): a stream-aware ("directional") rule, and a probe of far-block ranks vs direction.

Directional(a_down, a_up, a_cross): a same-level box Q is NEAR to box B iff
    -a_up <= (Q.coord_x - B.coord_x) <= a_down   and   |Q.coord_y - B.coord_y| <= a_cross.
Strong is Directional(1, 1, 1). Stretching along the stream axis keeps the closest wake blocks exact.
Because the driver compresses rows and columns with ONE ID of M = [A_FB; A_BF^T], the row side sees the
wake DOWNSTREAM of B and the column side the wake UPSTREAM of B (G(x_B - y_F) is large when F is upstream),
so the natural stretch is symmetric (a_down = a_up); the asymmetric rule is measured too.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from unified.core import interp_decomp
from unified.core.tree import Node, Tree
from unified.core.types import Kernel, Points


@dataclass
class Directional:
    """Anisotropic near field: stretched along the stream (x) axis. Implements the Admissibility protocol."""
    a_down: int = 2
    a_up: int = 2
    a_cross: int = 1
    name: str = field(init=False, default="")

    def __post_init__(self) -> None:
        self.name = f"dir({self.a_down},{self.a_up},{self.a_cross})"

    def near_nodes(self, tree: Tree, node: Node) -> list[Node]:
        out: list[Node] = []
        cx, cy = node.coord[0], node.coord[1]
        for dx in range(-self.a_up, self.a_down + 1):
            for dy in range(-self.a_cross, self.a_cross + 1):
                if dx == 0 and dy == 0:
                    continue
                nb = tree.node(node.level, (cx + dx, cy + dy))
                if nb is not None:
                    out.append(nb)
        return out


def block_rank(kernel: Kernel, w: float, D: float, theta: float, eps: float, n: int = 300, seed: int = 0,
               abs_scale: float | None = None) -> tuple[int, int, float]:
    """Source box B = [-w/2, w/2]^2 (n random points), target box F of the same size centred at D (cos t, sin t).
    theta = 0 is DOWNSTREAM of B (stream along +x). Returns (relative eps-rank of K(F, B),
    absolute eps-rank with threshold eps * abs_scale (abs_scale defaults to the block norm), block 2-norm)."""
    rng = np.random.default_rng(seed)
    Y: Points = (rng.random((n, 2)) - 0.5) * w
    X: Points = (rng.random((n, 2)) - 0.5) * w + D * np.array([np.cos(theta), np.sin(theta)])
    s = np.linalg.svd(kernel(X, Y), compute_uv=False)
    rel = int(np.sum(s > eps * s[0])) if s[0] > 0 else 0
    scale = abs_scale if abs_scale is not None else float(s[0])
    ab = int(np.sum(s > eps * scale))
    return rel, ab, float(s[0])


def stacked_id_rank(kernel: Kernel, w: float, D: float, theta: float, eps: float, n: int = 300) -> int:
    """Rank of the driver's stacked ID input [K(F, B); K(B, F)^T] for one far box (rows + columns together)."""
    rng = np.random.default_rng(0)
    Y: Points = (rng.random((n, 2)) - 0.5) * w
    X: Points = (rng.random((n, 2)) - 0.5) * w + D * np.array([np.cos(theta), np.sin(theta)])
    M = np.vstack([kernel(X, Y), kernel(Y, X).T])
    return len(interp_decomp(M, eps).skel)
