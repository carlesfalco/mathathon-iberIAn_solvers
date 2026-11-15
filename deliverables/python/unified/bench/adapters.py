"""Thin adapters from the SPEC wording to the drivers' concrete names.

SPEC.md fixes `rskel_factor(A, tree, admissibility, compressor) -> Factorization` with `.solve(b)` and
`.stats`, but not the stats keys, and for the butterfly it only says "explicit sparse factors
(scipy.sparse) and apply(g)". These helpers read those loosely specified parts in one place, so the
black-box tests and the benchmarks never depend on more than the SPEC promises. If a lookup fails, the
error message says which SPEC item could not be found.
"""
from __future__ import annotations
import inspect
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence, cast

import numpy as np
import scipy.sparse as sp

from unified.core.kernels import ButterflyKernel
from unified.core.types import Field, FrequencyGrid, KernelMatrix, PositionGrid, Spectrum

# ---- vocabulary --------------------------------------------------------------------------------------
# The SPEC fixes no schema for stats or for the butterfly object, so these adapters are duck-typed on
# purpose; `Any` below marks exactly the values whose type is up to the driver.
type StatsLike = Mapping[str, Any] | object        # a stats dict, or an object whose vars() hold the stats
type StatValue = Any                               # whatever the driver stored under a key
type PerLevel = Mapping[int, Any] | Sequence[Any]  # per-level stats: {level: value} or [value, ...]
type ButterflyLike = Any                           # the driver's butterfly object (IDButterfly here)
type SparseFactor = Any                            # a scipy.sparse matrix (sp.issparse(F) is True)


# ---- rskel (A1) ----------------------------------------------------------------------------------
def rskel() -> ModuleType:
    from unified.drivers import rskel as mod
    return mod


def _stat(stats: StatsLike, *needles: str, required: bool = True) -> StatValue:
    """First stats entry whose key contains all `needles` (stats may be a dict or an object).
    Per-level stats stored as {"levels": {l: {key: value}}} are returned as {l: value}."""
    d: Mapping[str, Any] = stats if isinstance(stats, dict) else vars(stats)
    for key, val in d.items():
        if all(n in key.lower() for n in needles):
            return val
    lv = d.get("levels")
    if isinstance(lv, dict) and lv and all(isinstance(v, dict) for v in lv.values()):
        for key in next(iter(lv.values())):
            if all(n in key.lower() for n in needles):
                return {l: v[key] for l, v in lv.items()}
    if required:
        raise KeyError(f"stats has no entry matching {needles}; keys = {sorted(d)}")
    return None


def root_size(stats: StatsLike) -> int:
    """SPEC: stats show the root size (number of unknowns in the final dense LU)."""
    return int(_stat(stats, "root"))


def factor_time(stats: StatsLike) -> float:
    """SPEC: stats include wall time."""
    return float(_stat(stats, "time"))


def skeleton_sizes(stats: StatsLike) -> PerLevel:
    """SPEC: skeleton sizes per level (returned as whatever the driver stores: list or dict)."""
    sizes: PerLevel = _stat(stats, "skel", "max", required=False) or _stat(stats, "skel")
    return sizes


def eliminated_per_level(stats: StatsLike) -> PerLevel:
    """SPEC: number of eliminated (redundant) unknowns per level."""
    counts: PerLevel = _stat(stats, "elim", required=False) or _stat(stats, "red")
    return counts


def _as_counts(v: PerLevel) -> list[float]:
    """Per-level stats as a flat list of numbers, whether stored as list, dict or list of lists."""
    vals = list(v.values()) if isinstance(v, dict) else list(v)
    return [float(np.sum(x)) if np.ndim(x) else float(x) for x in vals]


def total_eliminated(stats: StatsLike) -> int:
    return int(round(sum(_as_counts(eliminated_per_level(stats)))))


def max_skeleton(stats: StatsLike) -> int:
    """Largest skeleton count recorded at any level (max over per-level entries)."""
    v = skeleton_sizes(stats)
    vals = list(v.values()) if isinstance(v, dict) else list(v)
    return int(max(np.max(x) if np.ndim(x) else x for x in vals if np.size(x)))


# ---- butterfly (A2) ------------------------------------------------------------------------------
_BUILD_NAMES = ("butterfly_id_factor", "butterfly_factor", "butterfly_factorize", "factor", "build",
                "butterfly_id")


def butterfly_build(K: KernelMatrix, x: PositionGrid, xi: FrequencyGrid, eps: float,
                    kernel: ButterflyKernel | None = None) -> ButterflyLike:
    """Build the ID butterfly of K[i, j] = kernel(x_i, xi_j) over position grid x and frequency grid xi.
    The dense K is passed if the driver wants a matrix, `kernel` if it wants entry access.
    Returns the driver's object; use `butterfly_apply` / `butterfly_factors` on it."""
    from unified.drivers import butterfly_id as mod
    f: Callable[..., ButterflyLike] | None = None
    for name in _BUILD_NAMES:
        f = getattr(mod, name, None)
        if callable(f) and not inspect.isclass(f):
            break
    else:
        raise AttributeError(f"butterfly_id has none of {_BUILD_NAMES}; public names: "
                             f"{[n for n in dir(mod) if not n.startswith('_')]}")
    assert f is not None
    params = list(inspect.signature(f).parameters)
    kw: dict[str, object] = {}
    for p in params:
        pl = p.lower()
        if pl == "kernel" and kernel is not None:
            kw[p] = kernel
        elif pl in ("k", "a", "matrix", "kmat"):
            kw[p] = K
        elif pl in ("x", "xs", "points", "rows", "row_points"):
            kw[p] = x
        elif pl in ("xi", "freq", "freqs", "cols", "col_points"):
            kw[p] = xi
        elif pl in ("eps", "tol"):
            kw[p] = eps
    return f(**kw)


def butterfly_apply(bf: ButterflyLike, g: Spectrum) -> Field:
    """SPEC: `apply(g)` computes K @ g through the sparse factors."""
    from unified.drivers import butterfly_id as mod
    if hasattr(bf, "apply"):
        return cast(Field, bf.apply(g))
    if hasattr(mod, "apply"):
        return cast(Field, mod.apply(bf, g))
    if hasattr(mod, "butterfly_apply"):
        return cast(Field, mod.butterfly_apply(bf, g))
    raise AttributeError("no apply(g) found on the butterfly object or module")


def butterfly_factors(bf: ButterflyLike) -> list[SparseFactor]:
    """SPEC: explicit scipy.sparse factors. Returns the list of sparse matrices found on the object."""
    for name in ("factors", "mats", "matrices"):
        v = getattr(bf, name, None)
        if v is not None:
            return list(v)
    out: list[SparseFactor]
    if isinstance(bf, (tuple, list)):
        out = []
        for v in bf:
            if sp.issparse(v):
                out.append(v)
            elif isinstance(v, (list, tuple)) and v and all(sp.issparse(u) for u in v):
                out.extend(v)
        return out
    out = []
    for v in vars(bf).values():
        if sp.issparse(v):
            out.append(v)
        elif isinstance(v, (list, tuple)) and v and all(sp.issparse(u) for u in v):
            out.extend(v)
    return out


def butterfly_nnz(bf: ButterflyLike) -> int:
    """Storage = total nonzeros over all sparse factors."""
    if hasattr(bf, "nnz"):
        n = bf.nnz
        return int(n() if callable(n) else n)
    return int(sum(F.nnz for F in butterfly_factors(bf)))
