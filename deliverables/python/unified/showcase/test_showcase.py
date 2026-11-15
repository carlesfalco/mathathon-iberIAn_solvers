"""Smoke tests: every model passes its own check in quick mode, and the line counter counts what it says."""
from __future__ import annotations

import inspect

import pytest

from unified.drivers import butterfly_id, rskel

from . import reproduce
from .models import MODELS, Model


@pytest.mark.parametrize("model", MODELS, ids=[m.name for m in MODELS])
def test_model_passes_quick(model: Model[object]) -> None:
    row = reproduce.run_model(model, "quick")
    assert row["status"] == "PASS", {k: row.get(k) for k in ("status", "error", "metric", "ref_fast", "ref_tol",
                                                               "fast_vs_dense", "dense_tol", "invariants")}


@pytest.mark.parametrize("model", MODELS, ids=[m.name for m in MODELS])
def test_catalogue_entry_is_complete(model: Model[object]) -> None:
    assert model.physics.endswith(".") and model.references and model.dial_code and model.physics_code
    assert set(model.params) == {"quick", "full"}
    assert {id(o) for o in model.dial_code}.isdisjoint(id(o) for o in model.physics_code)
    assert reproduce.count(model.dial_code) > 0 and reproduce.count(model.physics_code) > 0


def _sample(x: int) -> int:
    """Docstring lines are not counted.

    Neither is this one."""
    # a comment line is not counted

    y = x + 1   # a trailing comment does not hide the code
    return (y
            * 2)


def test_code_lines_skips_blank_comment_and_docstring() -> None:
    assert reproduce.code_lines(_sample) == 4          # def, y = ..., and the two lines of the return


def test_count_dedupes_objects() -> None:
    assert reproduce.count((_sample, _sample)) == reproduce.code_lines(_sample)


def test_drivers_are_the_unchanged_shared_ones() -> None:
    """The showcase calls the shared drivers; it defines no driver of its own."""
    from . import models
    src = inspect.getsource(models)
    assert "rskel_factor(" in src and "butterfly_id(" in src
    assert vars(models)["rskel_factor"] is rskel.rskel_factor and vars(models)["bfly"] is butterfly_id


def test_failed_check_is_reported_not_hidden() -> None:
    """A model whose check raises shows up as ERROR in the table instead of disappearing."""
    m = MODELS[0]
    broken = Model(**{**m.__dict__, "name": "broken", "check": lambda _: 1 / 0})
    row = reproduce.run_model(broken, "quick")
    assert row["status"] == "ERROR" and "ZeroDivisionError" in row["error"]
    assert "ERROR" in reproduce.plain_table([row])
