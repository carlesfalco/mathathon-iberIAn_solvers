#!/usr/bin/env python3
"""Generate plots for the report from data included in this deliverable."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "latex" / "figures"
OUT.mkdir(exist_ok=True)
NAVY, TEAL, ORANGE, GREY = "#183247", "#168C88", "#CF694E", "#657886"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white"})

def copy(name):
    import shutil
    shutil.copyfile(ROOT / "figures" / name, OUT / name)

# The two-trees comparison, not a universal rank theorem.
d = json.loads((ROOT / "results" / "butterfly-discovery-ranks.json").read_text())
t = d["tiling_comparison"]
ns = [r["N"] for r in t]
fig, ax = plt.subplots(figsize=(7.4, 4.0))
ax.plot(ns, [r["fixed_4_by_4_tiling_max_rank"] for r in t], "o-", color=ORANGE,
        lw=2, label="fixed 4x4 tiling")
ax.plot(ns, [r["middle_max_rank"] for r in t], "s-", color=TEAL,
        lw=2, label="middle complementary tiling")
ax.set_xscale("log", base=2)
ax.set_xticks(ns, [str(x) for x in ns])
ax.set_ylim(0, max(r["fixed_4_by_4_tiling_max_rank"] for r in t) + 6)
ax.set_xlabel("matrix size N")
ax.set_ylabel("largest tile rank (10^-6 tolerance)")
ax.set_title("Tile geometry determines the amount of local information")
ax.grid(alpha=0.2)
ax.legend(frameon=False, loc="upper left")
fig.savefig(OUT / "tile-ranks.png", dpi=160, bbox_inches="tight")
plt.close(fig)

for name in ("dft-storage.png", "rust-scaling.png", "showcase-speed.png", "proxy-circle.png"):
    copy(name)


# A structural figure, not a fitted measurement: two trees and complementary levels.
from matplotlib.patches import ConnectionPatch

levels = 3
leaves = 2**levels
fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.9), width_ratios=[1.05, 1.05, 1.55])


def draw_interval_tree(ax, variable, selected_depth, selected_index, selected_name,
                       depth_label_side):
    """Draw tree nodes as the actual contiguous index intervals at each depth."""
    centers = {}
    for depth in range(levels + 1):
        size = leaves // 2**depth
        y = levels - depth
        centers[depth] = []
        for index, start in enumerate(range(0, leaves, size)):
            stop = start + size
            xmid = (start + stop) / 2
            centers[depth].append((xmid, y, start, stop))
            selected = depth == selected_depth and index == selected_index
            rect = plt.Rectangle((start + 0.06, y - 0.15), size - 0.12, 0.30,
                                 ec=TEAL if selected else NAVY,
                                 fc='#DDF1EF' if selected else 'white',
                                 lw=2.0 if selected else 1.0, zorder=2)
            ax.add_patch(rect)
            if depth < levels:
                for child_start in (start, start + size // 2):
                    child_mid = child_start + size / 4
                    ax.plot([xmid, child_mid], [y - 0.15, y - 0.85 + 0.15],
                            color=GREY, lw=0.8, zorder=0)
            if depth == levels:
                ax.text(xmid, y, str(index), ha='center', va='center', fontsize=8,
                        color=NAVY, zorder=3)
        if depth_label_side == 'left':
            ax.text(-1.05, y, f'depth {depth}', fontsize=9, color=GREY,
                    ha='right', va='center')
        else:
            ax.text(leaves + 0.85, y, f'depth {depth}', fontsize=9, color=GREY,
                    ha='left', va='center')

    x, y, start, stop = centers[selected_depth][selected_index]
    if stop - start >= 2:
        ax.text(x, y, selected_name, color=TEAL, fontsize=9, weight='bold',
                ha='center', va='center', zorder=4)
    ax.text(leaves / 2, levels + 0.92, variable, fontsize=13, color=NAVY,
            weight='bold', ha='center')
    ax.text(leaves / 2, levels + 0.46,
            'row boxes split as $l$ grows' if depth_label_side == 'left'
            else 'partners use depth $L-l$',
            fontsize=9.5, color=GREY, ha='center')
    ax.text(leaves / 2, -0.67, 'leaf indices', fontsize=8.5, color=GREY, ha='center')
    ax.set(xlim=(-1.7, leaves + 1.65), ylim=(-0.9, levels + 1.35),
           aspect='equal', adjustable='box')
    ax.axis('off')
    return centers


centers_A = draw_interval_tree(axes[0], 'Position tree $X$', 2, 2, '$A$', 'left')
centers_B = draw_interval_tree(axes[1], 'Frequency tree $\\xi$', 1, 0, '$B$', 'right')
axes[0].text(4, -1.17, '$A$ contains positions $4,5$; $|A|=2$',
             fontsize=9, color=TEAL, ha='center')
axes[1].text(4, -1.17, '$B$ contains frequencies $0,1,2,3$; $|B|=4$',
             fontsize=9, color=TEAL, ha='center')

selected_A = centers_A[2][2]
selected_B = centers_B[1][0]
pair_arrow = ConnectionPatch(
    xyA=(selected_A[3] + 0.10, selected_A[1]),
    xyB=(selected_B[2] - 0.10, selected_B[1]),
    coordsA='data', coordsB='data', axesA=axes[0], axesB=axes[1],
    arrowstyle='-|>', color=TEAL, lw=2.0,
    connectionstyle='arc3,rad=-0.12', shrinkA=0, shrinkB=0)
fig.add_artist(pair_arrow)

# At stage l the row interval has 2^(L-l) entries and its partner has 2^l.
ax = axes[2]
ax.axis('off')
ax.text(3.9, 4.20, 'Complementary tiles: eight entries each',
        fontsize=13, color=NAVY, weight='bold', ha='center')
ax.text(3.9, 3.76, 'row intervals split; column intervals merge',
        fontsize=9.5, color=GREY, ha='center')
cell = 0.21
for l in range(levels + 1):
    rows = leaves // 2**l       # |A|
    cols = 2**l                 # |B|
    xmid = 0.95 + 2.02*l
    ymid = 1.92
    width, height = rows*cell, cols*cell
    selected = l == 2
    rect = plt.Rectangle((xmid - width/2, ymid - height/2), width, height,
                         ec=TEAL if selected else NAVY,
                         fc='#DDF1EF' if selected else 'white',
                         lw=2.0 if selected else 1.2)
    ax.add_patch(rect)
    # Unit cells make the constant entry area visible, rather than just asserted.
    for i in range(1, rows):
        xx = xmid - width/2 + i*cell
        ax.plot([xx, xx], [ymid-height/2, ymid+height/2], color=NAVY, lw=0.35, alpha=0.5)
    for j in range(1, cols):
        yy = ymid - height/2 + j*cell
        ax.plot([xmid-width/2, xmid+width/2], [yy, yy], color=NAVY, lw=0.35, alpha=0.5)
    ax.text(xmid, 3.20, f'$l={l}$', fontsize=10, color=TEAL if selected else NAVY,
            weight='bold' if selected else 'normal', ha='center')
    ax.text(xmid, 0.48, f'$|A|={rows},\\;|B|={cols}$', fontsize=9.5,
            color=NAVY, ha='center')
    ax.text(xmid, 0.10, 'tile area = 8', fontsize=8.5,
            color=GREY, ha='center')
ax.annotate('', xy=(6.85, -0.32), xytext=(0.75, -0.32),
            arrowprops={'arrowstyle': '-|>', 'color': GREY, 'lw': 1.2})
ax.text(3.8, -0.62, 'increasing $l$: thinner in rows, wider in columns',
        fontsize=9, color=GREY, ha='center')
ax.set(xlim=(-0.1, 7.65), ylim=(-0.85, 4.55), aspect='equal', adjustable='box')

fig.tight_layout(w_pad=2.2)
fig.savefig(OUT / 'two-trees.png', dpi=180, bbox_inches='tight')
plt.close(fig)

print("Report figures regenerated from included data.")
