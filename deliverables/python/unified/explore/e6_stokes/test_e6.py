"""Correctness checks for E6 (Stokes). Small N; < 1 min."""
from __future__ import annotations

import dataclasses

import numpy as np

from . import stokes as S
from .harness import case


def test_rigid_motions_span_exterior_null_space() -> None:
    """(1/2 I + K) v = 0 for the 3 rigid motions (spectral convergence on the starfish)."""
    g = S.starfish(256)
    A0 = S.stokes_matrix(g, with_projector=False)
    assert np.abs(A0 @ S.rigid_basis(g, 0)).max() < 1e-12


def test_circle_mobility_matches_closed_form() -> None:
    g = S.ellipse(64, 0.7, 0.7)
    F, T = np.array([0.3, -1.0]), 0.7
    mu = np.linalg.solve(S.stokes_matrix(g), S.rhs_forces(g, F[None], np.array([T])))
    assert np.allclose(S.rigid_motions(g, mu)[0], S.circle_mobility(0.7, F, T), atol=1e-13)


def test_jeffery_rate() -> None:
    g = S.ellipse(128, 1.0, 0.4, angle=0.3)
    mu = np.linalg.solve(S.stokes_matrix(g), S.rhs_shear(g))
    assert abs(S.rigid_motions(g, mu)[0, 2] - S.jeffery_omega(1.0, 0.4, 0.3)) < 1e-12


def test_unchanged_driver_solves_stokes_with_exact_and_proxy() -> None:
    """The kernel swap alone (exact sampling) and the block proxy, weak and strong, reach ~eps."""
    g = S.starfish(256)
    for rule in ("weak", "strong"):
        for kind in ("exact", "proxy-sl", "proxy-both"):
            assert case(g, "stokes", rule, kind, eps=1e-10)["err"] < 1e-8, (rule, kind)


def test_projector_is_seen_by_proxy_even_without_rigid_rows() -> None:
    """Refuted hypothesis (r5): P's far blocks are carried by the explicit rows of the same body near the box,
    so dropping the extra rigid rows does not hurt, even with rough projector weights."""
    g = S.starfish(256)
    g = dataclasses.replace(g, pw=g.w * np.random.default_rng(0).uniform(0.2, 1.8, g.n))
    for rule in ("weak", "strong"):
        assert case(g, "stokes", rule, "proxy-norigid", eps=1e-10)["err"] < 1e-8


def test_degenerate_scale_single_layer_vanishes_inside() -> None:
    """At R = sqrt(e) a constant-density Stokeslet layer on a circle gives zero velocity inside."""
    R, m = float(np.sqrt(np.e)), 256
    th = 2 * np.pi * np.arange(m) / m
    P = R * np.c_[np.cos(th), np.sin(th)]
    x = np.array([[0.1, 0.2], [-0.3, 0.1]])
    u = S.stokeslet(x, P) @ np.tile([1.0, 0.0], m) * (2 * np.pi * R / m)
    assert np.abs(u).max() < 1e-14
