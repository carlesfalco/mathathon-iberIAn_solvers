"""Shared type vocabulary: names from the papers, so signatures read like the math.

These are aliases (documentation for readers and for mypy), not runtime-checked types: a `FrequencyGrid`
and a `PositionGrid` are both float arrays, and nothing stops you from swapping them. Shapes are given in
comments because NumPy's type system does not track them.
"""
from __future__ import annotations

from typing import Any, Callable, NamedTuple, Protocol, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # avoid an import cycle at runtime
    from .tree import Node, Tree

# ---- scalars and plain arrays ---------------------------------------------------------------------
type Real = np.float64
type RealArray = NDArray[np.float64]
type ComplexArray = NDArray[np.complex128]
type Array = NDArray[np.inexact[Any, Any]]                    # real or complex data
type IndexArray = NDArray[np.intp]                            # integer indices into a point set / matrix

# ---- geometry ----------------------------------------------------------------------------------------
type Points = RealArray            # (n, d) point coordinates, d = 1 or 2
type PositionGrid = RealArray      # (N,) positions x in [0, 1)   (rows of a butterfly matrix)
type FrequencyGrid = RealArray | NDArray[np.int64]   # (N,) frequencies xi, often integers (columns of a butterfly matrix)
type GridCoord = tuple[int, ...]   # integer box coordinates at one tree level

# ---- linear algebra ------------------------------------------------------------------------------------
type Matrix = Array                # dense 2D array
type KernelMatrix = Matrix         # entries K(x_i, y_j)
type Vector = Array                # (n,) right-hand side, density, coefficients
type Spectrum = Vector             # g(xi): input of a butterfly apply
type Field = Vector                # f(x) = K g: output of a butterfly apply

# ---- compression vocabulary ----------------------------------------------------------------------
type SkeletonIdx = IndexArray      # unknowns that "speak for" a box
type RedundantIdx = IndexArray     # unknowns eliminated because the skeletons reproduce their far field
type InterpMatrix = Matrix         # T with  M[:, red] ~= M[:, skel] @ T
type TilePair = tuple[Node, Node]  # butterfly tile: (row node at depth l, column node at depth L - l), one phase-space cell


type LUFactors = tuple[Matrix, NDArray[np.int32]]   # (LU, pivots) as returned by scipy.linalg.lu_factor
type SparseFactor = Any                              # one scipy.sparse factor of a butterfly (CSR); scipy-stubs typing is coarse


class IDResult(NamedTuple):
    """Interpolative decomposition M[:, red] ~= M[:, skel] @ T (a tuple, so `skel, red, T = ...` works)."""
    skel: SkeletonIdx
    red: RedundantIdx
    T: InterpMatrix


type Kernel = Callable[[Points, Points], KernelMatrix]        # K(X, Y) -> matrix of K(x_i, y_j)


class Compressor(Protocol):
    """Anything that skeletonizes a box against the far field selected by an admissibility rule."""
    def __call__(self, box_idx: IndexArray, far_idx: IndexArray, node: Node) -> IDResult: ...


class Admissibility(Protocol):
    """Rule deciding which interactions of a box are kept exact (near) instead of compressed (far)."""
    name: str
    def near_nodes(self, tree: Tree, node: Node) -> list[Node]: ...

