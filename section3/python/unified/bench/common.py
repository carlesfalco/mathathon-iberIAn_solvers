"""Shared plotting and output helpers for the benchmarks: one style, one results folder.

Colors follow a fixed categorical order (slot 1 blue, slot 2 orange, slot 3 aqua), assigned to an
entity and never to a rank: weak = blue, strong = orange everywhere; in butterfly plots ID = blue,
SVD reference = orange. Every figure has a legend plus direct end-of-line labels.
"""
from __future__ import annotations
import json
import pathlib
import platform
import time
from typing import Any, Sequence, cast

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray

type SeriesData = Sequence[float] | NDArray[Any]   # one plotted coordinate list (list or 1D array)
type Slot = int                          # index into SERIES / MARKERS: a fixed colour per entity
type JSONPayload = dict[str, Any]        # anything json.dumps accepts after `_jsonable`

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
MARKERS = ["o", "s", "^", "D"]


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2, "lines.markersize": 6, "font.size": 10, "legend.frameon": False,
    })


def line(ax: Axes, x: SeriesData, y: SeriesData, slot: Slot, label: str, direct: bool = True) -> None:
    """One series: 2px line, markers, colour from its fixed slot, optional label at the line's end."""
    ax.plot(x, y, color=SERIES[slot], marker=MARKERS[slot], label=label)
    if direct and len(x):
        ax.annotate(label, (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                    color=INK2, fontsize=8, va="center")


def save(fig: Figure, name: str) -> pathlib.Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def dump(name: str, payload: JSONPayload) -> pathlib.Path:
    """Write results/<name>.json with a small provenance header (all numbers are COMPUTED here)."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {"label": "computed", "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
               "machine": f"{platform.machine()} / {platform.python_version()} / numpy {np.__version__}",
               **payload}
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=_jsonable))
    return path


def _jsonable(o: object) -> int | float | list[Any] | str:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return cast(list[Any], o.tolist())
    return str(o)
