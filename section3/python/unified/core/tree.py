"""One tree for every driver: a uniform-depth 2^d-ary tree (binary in 1D, quadtree in 2D) over points.

Boxes live on a regular grid at each level: box (level l, coord c) covers
    [lo + c * w_l, lo + (c + 1) * w_l),   w_l = size / 2**l   (per dimension).
Only non-empty boxes are stored. Because the tree is uniform, two boxes at the same level are
neighbours iff their integer coords differ by at most 1 in every dimension. That makes the near
field, the far field and the FMM interaction list exact integer tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .types import GridCoord, IndexArray, Points, RealArray


@dataclass(eq=False)
class Node:
    """A box of the tree. `idx` lists ALL point indices inside the box (not just a skeleton)."""
    level: int
    coord: GridCoord                   # integer grid coordinates at this level
    center: RealArray                  # (d,)
    half_width: float                  # boxes are cubes
    idx: IndexArray                    # sorted indices of the points inside this box
    parent: Node | None = None
    children: list[Node] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def __repr__(self) -> str:
        return f"Node(l={self.level}, c={self.coord}, n={len(self.idx)})"


class Tree:
    """Uniform tree of depth `depth` (root = level 0). Pick depth directly or via max_leaf."""

    points: Points
    n: int
    d: int
    size: float
    lo: RealArray
    depth: int
    levels: list[list[Node]]
    root: Node

    def __init__(self, points: ArrayLike, depth: int | None = None, max_leaf: int = 32,
                 bbox: tuple[ArrayLike, ArrayLike] | None = None) -> None:
        P: Points = np.asarray(points, dtype=float)
        if P.ndim == 1:
            P = P[:, None]
        self.points, (self.n, self.d) = P, P.shape
        if bbox is None:  # data box, padded slightly so the largest point falls inside the last box
            lo, hi = P.min(0), P.max(0)
            self.size = float(np.max(hi - lo)) * (1 + 1e-9) or 1.0
            self.lo = lo - 1e-12
        else:             # user box [lo, hi): used exactly, so points on dyadic boundaries land correctly
            lo, hi = np.asarray(bbox[0], float), np.asarray(bbox[1], float)
            self.size, self.lo = float(np.max(hi - lo)), lo
        if depth is None:  # smallest depth with average occupied-leaf size <= max_leaf
            depth = 0
            while depth < 30:
                occupied = len({tuple(c) for c in self._coords(depth)})
                if self.n / occupied <= max_leaf:
                    break
                depth += 1
        self.depth = depth
        self.levels = []
        self._by_key: dict[tuple[int, GridCoord], Node] = {}
        for l in range(depth + 1):
            groups: dict[GridCoord, list[int]] = {}
            for i, c in enumerate(map(tuple, self._coords(l))):
                groups.setdefault(c, []).append(i)
            nodes: list[Node] = []
            w = self.size / 2 ** l
            for c in sorted(groups):
                node = Node(l, c, self.lo + (np.array(c) + 0.5) * w, w / 2, np.array(groups[c], dtype=np.intp))
                if l > 0:
                    par = self._by_key[(l - 1, tuple(x // 2 for x in c))]
                    node.parent = par
                    par.children.append(node)
                nodes.append(node)
                self._by_key[(l, c)] = node
            self.levels.append(nodes)
        self.root = self.levels[0][0]

    def _coords(self, l: int) -> NDArray[np.int64]:
        """Integer grid coordinates of every point at level l, shape (n, d)."""
        w = self.size / 2 ** l
        coords: NDArray[np.int64] = np.minimum(np.floor((self.points - self.lo) / w).astype(np.int64), 2 ** l - 1)
        return coords

    # ---- queries -------------------------------------------------------------------------
    def node(self, level: int, coord: Sequence[int]) -> Node | None:
        return self._by_key.get((level, tuple(coord)))

    def leaves(self) -> list[Node]:
        return self.levels[-1]

    def nodes_postorder(self) -> list[Node]:
        """Leaves first, root last (the order used by factorizations)."""
        return [n for lvl in reversed(self.levels) for n in lvl]

    def neighbors(self, node: Node) -> list[Node]:
        """Same-level non-empty boxes touching `node` (excluding itself)."""
        out: list[Node] = []
        for off in np.ndindex(*(3,) * self.d):
            c = tuple(node.coord[k] + off[k] - 1 for k in range(self.d))
            if c != node.coord:
                nb = self._by_key.get((node.level, c))
                if nb is not None:
                    out.append(nb)
        return out

    def interaction_list(self, node: Node) -> list[Node]:
        """FMM interaction list: children of the parent's neighbours (and parent) that are not adjacent to node."""
        if node.parent is None:
            return []
        candidates = [ch for p in [node.parent] + self.neighbors(node.parent) for ch in p.children]
        return [c for c in candidates
                if c is not node and max(abs(c.coord[k] - node.coord[k]) for k in range(self.d)) > 1]
