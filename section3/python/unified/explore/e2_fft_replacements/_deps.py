"""Typed shim over the two butterfly implementations this experiment compares against.

`unified/drivers/butterfly_id.py` and `code/butterfly.py` are owned by other workers and are not (yet)
annotated, so importing them statically would make `mypy --strict` on this directory fail inside files we
may not edit. We load them at runtime and describe the parts we use with Protocols. No behaviour change.
"""
from __future__ import annotations

import importlib
import importlib.util
import pathlib
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from unified.core.types import ComplexArray, FrequencyGrid, PositionGrid

type SparseFactor = Any                 # scipy.sparse CSR matrix (scipy ships no stubs here)
type StageRanks = list[int]             # max skeleton size at each butterfly stage l = 0..L
type ButterflyKernel = Any              # Callable[[PositionGrid, FrequencyGrid], ComplexArray]; Any so the
                                        # driver may annotate its own kernel type however it likes


class ButterflyFactorization(Protocol):
    """K ~= U G_{L-1} ... G_0 W (the ID butterfly of unified/drivers/butterfly_id.py)."""
    G: list[SparseFactor]
    ranks: StageRanks
    build_time: float

    @property
    def nnz(self) -> int: ...
    def apply(self, g: NDArray[Any]) -> ComplexArray: ...


class SVDButterflyModule(Protocol):
    """The functions of code/butterfly.py (SVD reference butterfly) that we call."""
    def butterfly_factorize(self, K: ComplexArray, eps: float = ...
                            ) -> tuple[SparseFactor, list[SparseFactor], SparseFactor, StageRanks]: ...
    def butterfly_apply(self, W: SparseFactor, G: list[SparseFactor], Uf: SparseFactor,
                        g: NDArray[Any]) -> ComplexArray: ...


_drv: Any = importlib.import_module("unified.drivers.butterfly_id")


def butterfly_id(kernel: ButterflyKernel, x: PositionGrid | FrequencyGrid, xi: PositionGrid | FrequencyGrid,
                 eps: float = 1e-8) -> ButterflyFactorization:
    """ID butterfly of K[i, j] = kernel(x_i, xi_j) (delegates to unified.drivers.butterfly_id)."""
    return cast(ButterflyFactorization, _drv.butterfly_id(kernel, np.asarray(x, float), np.asarray(xi, float), eps))


def load_svd_reference() -> SVDButterflyModule:
    """Import the packaged read-only SVD butterfly reference."""
    return cast(SVDButterflyModule, importlib.import_module("unified.reference.butterfly"))
