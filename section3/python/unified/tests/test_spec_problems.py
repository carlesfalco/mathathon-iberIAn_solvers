"""Checks that the shared SPEC problems (unified/bench/problems.py) are what SPEC.md says they are,
so failures in the driver tests cannot come from a wrong test matrix."""
from typing import Callable, cast

import numpy as np
import pytest
from unified.bench import problems as P


@pytest.mark.parametrize("make", [lambda: P.p1_curve(300), lambda: P.p2_area(300),
                                  lambda: P.p3_helmholtz(300, 20), lambda: P.clustered(300)])
def test_kernel_reproduces_offdiagonal_entries(make: Callable[[], P.Problem]) -> None:
    """The proxy compressor needs kernel(X, Y) == A off the diagonal."""
    prob = make()
    K = prob.kernel(prob.X, prob.X)
    off = ~np.eye(len(prob.X), dtype=bool)
    assert np.allclose(K[off], prob.A[off], rtol=1e-12, atol=1e-14)


def test_p1_weights_integrate_perimeter() -> None:
    """sum w_j = arc length of the starfish (trapezoid rule is spectrally accurate here)."""
    from scipy.integrate import quad
    _, w = P.starfish(400)
    speed = cast(Callable[[float], float], P._speed)   # vectorized helper; also accepts a scalar θ
    exact = quad(speed, 0, 2 * np.pi, limit=200)[0]
    assert abs(w.sum() - exact) < 1e-10 * exact


def test_p1_well_conditioned_p3_complex() -> None:
    assert np.linalg.cond(P.p1_curve(400).A) < 100
    assert np.iscomplexobj(P.p3_helmholtz(100, 5).A)


def test_b1_shapes() -> None:
    x, xi = P.b1_grids(64)
    assert x[0] == 0 and xi[0] == -32 and xi[-1] == 31
    assert P.b1_matrix(64, 1.0).shape == (64, 64)
