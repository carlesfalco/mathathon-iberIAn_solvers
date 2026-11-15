"""Run every model in the showcase catalogue and print one honest table.

    .venv/bin/python -m unified.showcase.reproduce --quick          # every model, small sizes (< ~3 min)
    .venv/bin/python -m unified.showcase.reproduce                  # full sizes (< ~20 min)
    .venv/bin/python -m unified.showcase.reproduce --only stokes-particles,dft-as-butterfly

Writes unified/showcase/results.json and unified/showcase/RESULTS.md.

Line counts are computed here, never typed in: for each model, the source of every object listed in
`dial_code` (kernel, sampler / proxy override, wrappers, forked driver) and in `physics_code` (formulation,
quadrature, geometry, analytic reference) is read with inspect.getsource, and the lines that carry code are
counted: blank lines, comment lines and docstrings are excluded.
"""
from __future__ import annotations

import argparse
import ast
import inspect
import io
import json
import os
import platform
import textwrap
import time
import tokenize
import traceback
from typing import Any

import numpy as np

from .models import MODELS, CheckResult, Mode, Model

HERE = os.path.dirname(os.path.abspath(__file__))
_SKIP = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER}

MATCHED_FOOTNOTE = (
    '"Matched" means one of three things, named per model: **exact solution** = a closed-form solution of the '
    "continuous problem, so agreement is limited by the discretization and the tolerance is the documented "
    "discretization level, not ε; **invariant** = an identity that must hold (FFT identity and entry count, "
    "Parseval, Rayleigh expansion); **published behaviour** = a qualitative or scaling result (bounded tile rank). "
    "No row reproduces a table from a paper number for number.")
COLUMNS_FOOTNOTE = (
    "*dial lines* = code that plugs the physics into the harness (kernel, sampler / proxy override, wrappers, "
    "forked driver); *physics lines* = formulation, quadrature, geometry and analytic reference. Both are counted "
    "automatically with inspect.getsource (non-blank, non-comment, docstrings excluded). `unified/core` and "
    "`unified/drivers` are imported unchanged by every model. *error vs analytic* is the fast solution against "
    "the reference, with the dense solution's error in brackets: the gap between the two is what the fast solver "
    "adds. *fast vs dense* is ‖x_fast − x_dense‖/‖x_dense‖ (solves) or the same for outputs (transforms). "
    "*time* is fast / dense wall seconds on one laptop (secondary evidence; drivers keep dense work arrays).")


# ---- automatic line counting ---------------------------------------------------------------------------
def code_lines(obj: Any) -> int:
    """Lines of `obj`'s source that hold code: no blank lines, no comment-only lines, no docstrings."""
    src = textwrap.dedent(inspect.getsource(obj))
    doc: set[int] = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str) and first.end_lineno is not None):
                doc.update(range(first.lineno, first.end_lineno + 1))
    code: set[int] = set()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type not in _SKIP:
            code.update(range(tok.start[0], tok.end[0] + 1))
    return len(code - doc)


def count(objs: tuple[Any, ...]) -> int:
    """Sum of code_lines over distinct objects (an object listed twice is counted once)."""
    seen: dict[int, Any] = {id(o): o for o in objs}
    return sum(code_lines(o) for o in seen.values())


def _qualname(o: Any) -> str:
    return f"{getattr(o, '__module__', '?')}.{getattr(o, '__qualname__', repr(o))}"


# ---- running ---------------------------------------------------------------------------------------------
def run_model(model: Model[Any], mode: Mode) -> dict[str, Any]:
    """Build and check one model. Exceptions are recorded as a failed row, never hidden."""
    overlap = {id(o) for o in model.dial_code} & {id(o) for o in model.physics_code}
    assert not overlap, f"{model.name}: an object is listed as both dial and physics code"
    row: dict[str, Any] = dict(
        name=model.name, physics=model.physics, matched=model.matched, matched_text=model.matched_text,
        references=[r.__dict__ for r in model.references], dials=model.dials.__dict__,
        params=dict(model.params[mode]), source=model.source, note=model.note,
        dial_lines=count(model.dial_code), physics_lines=count(model.physics_code),
        dial_objects=[_qualname(o) for o in model.dial_code],
        physics_objects=[_qualname(o) for o in model.physics_code])
    t0 = time.perf_counter()
    try:
        with np.errstate(all="ignore"):
            res: CheckResult = model.check(model.build(model.params[mode]))
        row.update(status="PASS" if res.passed else "FAIL", metric=res.metric, ref_fast=res.ref_fast,
                   ref_dense=res.ref_dense, ref_tol=res.ref_tol, fast_vs_dense=res.fast_vs_dense,
                   dense_tol=res.dense_tol, t_fast=res.t_fast, t_dense=res.t_dense,
                   invariants=res.invariants, extra=res.extra, display=res.display)
    except Exception as e:   # a crash is a result: report it in the table
        row.update(status="ERROR", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
    row["t_total"] = time.perf_counter() - t0
    return row


# ---- formatting ------------------------------------------------------------------------------------------
def _e(v: float | None) -> str:
    return "—" if v is None else f"{v:.1e}"


def table_rows(rows: list[dict[str, Any]]) -> list[list[str]]:
    out = []
    for r in rows:
        refs = "; ".join(f"{x['authors']} {x['year']}" for x in r["references"][:2])
        matched = f"{r['matched_text']} [{r['matched']}] ({refs})"
        if r["status"] == "ERROR":
            err, fvd, tm = r["error"][:60], "—", f"{r['t_total']:.1f}"
        else:
            shown = r["display"] or _e(r["ref_fast"])
            err = f"{shown} (dense {_e(r['ref_dense'])}, tol {r['ref_tol']:.0e})"
            fvd = f"{_e(r['fast_vs_dense'])} (tol {r['dense_tol']:.0e})"
            tm = f"{r['t_fast']:.1f} / {r['t_dense']:.1f}"
        out.append([r["name"], matched, str(r["dial_lines"]), str(r["physics_lines"]), err, fvd, tm, r["status"]])
    return out


HEADER = ["model", "matched result (reference)", "solver-dial lines", "physics-setup lines",
          "error vs analytic / invariant", "fast vs dense", "time fast / dense (s)", "status"]


def plain_table(rows: list[dict[str, Any]]) -> str:
    body = table_rows(rows)
    short = [[c if i != 1 else c.split(" [")[0] for i, c in enumerate(r)] for r in body]   # terminal: shorter
    w = [max(len(h), *(len(r[i]) for r in short)) for i, h in enumerate(HEADER)]
    line = " | ".join(h.ljust(w[i]) for i, h in enumerate(HEADER))
    sep = "-+-".join("-" * x for x in w)
    return "\n".join([line, sep] + [" | ".join(c.ljust(w[i]) for i, c in enumerate(r)) for r in short])


def markdown(rows: list[dict[str, Any]], mode: Mode, wall: float) -> str:
    md = ["# Showcase results", "",
          f"Mode `{mode}`, total wall time {wall:.0f} s, Python {platform.python_version()} on {platform.machine()}. "
          "Generated by `unified/showcase/reproduce.py`; do not edit by hand.", "",
          "| " + " | ".join(HEADER) + " |", "|" + "---|" * len(HEADER)]
    md += ["| " + " | ".join(c.replace("|", "\\|") for c in r) + " |" for r in table_rows(rows)]
    failed = [r for r in rows if r["status"] != "PASS"]
    md += ["", "**Failed checks:** " + (", ".join(f"`{r['name']}` ({r['status']})" for r in failed)
                                        if failed else "none in this run."), "",
           "## What the columns mean", "", MATCHED_FOOTNOTE, "", COLUMNS_FOOTNOTE, "", "## Per model", ""]
    for r in rows:
        md += [f"### {r['name']}", "", r["physics"], ""]
        md += [f"- **Matched ({r['matched']}):** {r['matched_text']}."]
        md += [f"- **Reference:** {x['authors']} ({x['year']}), {x['where']}: {x['result']}." for x in r["references"]]
        d = r["dials"]
        md += [f"- **Dials:** tree = {d['tree']}; admissibility = {d['admissibility']}; representation = "
               f"{d['representation']}; sampling = {d['sampling']}; algebra = {d['algebra']}.",
               f"- **Parameters:** {r['params']}. **Physics from:** {r['source']}."]
        if r["status"] == "ERROR":
            md += [f"- **ERROR:** `{r['error']}`"]
        else:
            md += [f"- **Check:** {r['metric']}: fast {_e(r['ref_fast'])}, dense {_e(r['ref_dense'])}, "
                   f"tolerance {r['ref_tol']:.0e}; fast vs dense {_e(r['fast_vs_dense'])} (tolerance {r['dense_tol']:.0e})."]
            for k, v in r["invariants"].items():
                md += [f"- **Invariant** {k}: {'holds' if v else 'FAILS'}."]
        if r["note"]:
            md += [f"- **Note:** {r['note']}"]
        md += [f"- **Dial code counted ({r['dial_lines']} lines):** " + ", ".join(f"`{o}`" for o in r["dial_objects"]),
               f"- **Physics code counted ({r['physics_lines']} lines):** "
               + ", ".join(f"`{o}`" for o in r["physics_objects"]), ""]
    return "\n".join(md) + "\n"


def main(argv: list[str] | None = None) -> list[dict[str, Any]]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="small sizes, all models in under ~3 minutes")
    ap.add_argument("--only", default="", help="comma-separated model names")
    ap.add_argument("--no-write", action="store_true", help="do not write results.json / RESULTS.md")
    args = ap.parse_args(argv)
    mode: Mode = "quick" if args.quick else "full"
    names = [s for s in args.only.split(",") if s]
    models = [m for m in MODELS if not names or m.name in names]
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for m in models:
        print(f"[{mode}] {m.name} ...", flush=True)
        rows.append(run_model(m, mode))
        r = rows[-1]
        print(f"    {r['status']}  ({r['t_total']:.1f} s)" + (f"  {r.get('error', '')}" if r["status"] == "ERROR" else ""),
              flush=True)
    wall = time.perf_counter() - t0
    print()
    print(plain_table(rows))
    print()
    print(MATCHED_FOOTNOTE.replace("**", ""))
    failed = [r["name"] for r in rows if r["status"] != "PASS"]
    print(f"\nFAILED: {', '.join(failed)}" if failed else "\nAll checks passed.")
    print(f"Total wall time: {wall:.1f} s ({mode})")
    if not args.no_write:
        clean = [{k: v for k, v in r.items() if k != "traceback"} | ({"traceback": r["traceback"]} if "traceback" in r else {})
                 for r in rows]
        with open(os.path.join(HERE, "results.json"), "w") as f:
            json.dump(dict(mode=mode, wall_s=wall, models=clean), f, indent=1, default=float)
        with open(os.path.join(HERE, "RESULTS.md"), "w") as f:
            f.write(markdown(rows, mode, wall))
    return rows


def console_cli(argv: list[str] | None = None) -> None:
    """Console-script wrapper: keep main's list-returning API and return process success."""
    main(argv)


if __name__ == "__main__":
    main()
