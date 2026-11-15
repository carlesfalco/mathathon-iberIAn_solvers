"""Typed shim over unified/bench/problems.py (not annotated, so a static import would break `mypy --strict`
on this directory). Loaded at runtime; the parts we use are described by a Protocol. No behaviour change."""
from __future__ import annotations

import importlib
from typing import Any, Protocol, cast

from unified.core.types import Kernel, Matrix, Points, RealArray

_P: Any = importlib.import_module("unified.bench.problems")

type QuadWeights = RealArray     # (n,) trapezoid weights on the starfish


class Problem(Protocol):
    """unified.bench.problems.Problem: dense A = I + kernel(X, X), points X, and the kernel itself."""
    name: str
    X: Points
    A: Matrix
    kernel: Kernel


def make_problem(name: str, X: Points, A: Matrix, kernel: Kernel) -> Problem:
    return cast(Problem, _P.Problem(name, X, A, kernel))


def p3_helmholtz(n: int, k: float) -> Problem:
    return cast(Problem, _P.p3_helmholtz(n, k))


def starfish(n: int) -> tuple[Points, QuadWeights]:
    return cast(tuple[Points, QuadWeights], _P.starfish(n))


def dense_identity_plus(kernel: Kernel, X: Points, dtype: type = float) -> Matrix:
    return cast(Matrix, _P.dense_identity_plus(kernel, X, dtype))
