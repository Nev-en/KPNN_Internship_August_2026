"""Why a seed alone does not reproduce weights -- the figure for the finding.

A seed does not label weights. It sets the start of one shared stream of random
numbers, and every draw takes the next values from wherever the stream stands.
Change what runs before a layer and the layer gets different numbers, even with
the same seed and the same init code.

This bit us in kpnn2_port.py: it copied kpnn v1's initialisation bound exactly,
but overwrote the weights *after* the constructor had already consumed 7841
numbers. Same distribution, correlation 0.005 to v1's actual values, and the
carefully matched init rule did nothing at all.

Three panels:
    A  the mechanism, drawn
    B  what it cost -- correlation of the initial weights to v1
    C  what it cost downstream -- TF-level agreement with Karla's run

The two port versions use figure_style's STAGE_COLORS ("own_init" for no
reseed, "infused" for reseeded), the same colours every other figure uses for
those stages. Numbers are the measured ones from this project (INS, seed 1, for
panel B; 20 seeds per condition for panel C). Run from the project root::

    python src/comp/plot_seed_stream.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import figure_style as fs

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "comp" / "t2" / "plots" / "plots_seed_stream"

NO_RESEED = fs.STAGE_COLORS["own_init"]
RESEEDED = fs.STAGE_COLORS["infused"]
STREAM = "0.9"

CONDITIONS = fs.CONDITIONS
# TF-layer correlation per seed against Karla's kpnn v1 run, before and after
# apply_v1_init learned to reseed.
BEFORE = [0.902, 0.842, 0.939, 0.814, 0.867]
AFTER = [0.960, 0.976, 0.991, 0.966, 0.942]

# Correlation of the initial sparse-layer weights to v1's, INS seed 1.
INIT_R = {"port, no reseed": 0.005, "port2 / port, reseeded": 1.000}


def panel_mechanism(ax):
    """The stream as a row of numbers, and who eats which part of it."""
    ax.set_title("A   One seed, one stream — position decides", loc="left")
    ax.axis("off")

    values = ["12", "867", "2", "34374", "456", "91", "7", "…"]
    for i, value in enumerate(values):
        ax.add_patch(plt.Rectangle((i, 2.3), 0.9, 0.6, facecolor=STREAM,
                                   edgecolor="none"))
        ax.text(i + 0.45, 2.6, value, ha="center", va="center")
    ax.text(-0.3, 2.6, "stream\nafter seed", ha="right", va="center")

    # v1 takes the first three; the port's constructor eats them first instead.
    ax.add_patch(plt.Rectangle((0, 1.35), 2.9, 0.55, facecolor="none",
                               edgecolor=fs.REFERENCE_COLOR, lw=0.8))
    ax.text(1.45, 1.62, "v1: sparse layer", ha="center", va="center")
    ax.text(-0.3, 1.62, "v1", ha="right", va="center")

    ax.add_patch(plt.Rectangle((0, 0.5), 2.9, 0.55, facecolor=STREAM,
                               edgecolor="0.5", lw=0.5))
    ax.text(1.45, 0.77, "constructor (dense draw)", ha="center", va="center")
    ax.add_patch(plt.Rectangle((3, 0.5), 2.9, 0.55, facecolor=NO_RESEED,
                               edgecolor="none"))
    ax.text(4.45, 0.77, "sparse layer", ha="center", va="center")
    ax.text(-0.3, 0.77, "port, no reseed:\nshifted", ha="right", va="center")

    ax.text(0, -0.35, "Same seed, same init rule — the layer simply starts "
                      "further along the stream.")
    ax.set_xlim(-2.6, 8.3)
    ax.set_ylim(-0.7, 3.2)


def panel_init(ax):
    """What the shift did to the initial weights."""
    ax.set_title("B   Initial weights vs kpnn v1 (INS, seed 1)", loc="left")
    labels = list(INIT_R)
    values = [INIT_R[k] for k in labels]
    bars = ax.barh(labels, values, color=[NO_RESEED, RESEEDED], height=0.4)
    for bar, value in zip(bars, values):
        ax.text(value + 0.03, bar.get_y() + bar.get_height() / 2,
                f"r = {value:.3f}", va="center")
    ax.set_xlim(0, 1.30)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("correlation to v1's weights")
    ax.axvline(1.0, ls=":", lw=0.5, color=fs.GUIDE_COLOR)
    fs.tidy(ax)
    ax.text(0.0, -0.5, "Both use v1's bound √(6 / average fan-in):\n"
                       "the spread matches, the values do not.",
            transform=ax.get_yaxis_transform(), clip_on=False)


def panel_downstream(ax):
    """What it cost in the actual comparison against Karla's run."""
    ax.set_title("C   TF-level agreement with Karla's kpnn v1 run\n"
                 "(per seed, 20 seeds per condition)", loc="left")
    x = np.arange(len(CONDITIONS))
    width = 0.38
    ax.bar(x - width / 2, BEFORE, width, label="port, no reseed", color=NO_RESEED)
    ax.bar(x + width / 2, AFTER, width, label="port, reseeded", color=RESEEDED)

    for i, (before, after) in enumerate(zip(BEFORE, AFTER)):
        ax.text(i + width / 2, after + 0.004, f"{after:.3f}", ha="center")
        ax.text(i - width / 2, before + 0.004, f"{before:.3f}", ha="center")

    ax.set_xticks(x)
    ax.set_xticklabels(CONDITIONS)
    ax.set_ylim(0.75, 1.06)
    ax.set_ylabel("Pearson r")
    ax.legend(loc="upper left", ncol=2)
    ax.grid(axis="y", ls=":", lw=0.4, color=fs.ZERO_COLOR)
    ax.set_axisbelow(True)
    fs.tidy(ax)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, _ = fs.subplots(160, 100, 1, 1)
    fig.clear()
    grid = fig.add_gridspec(2, 2, height_ratios=[0.8, 1.2])

    panel_mechanism(fig.add_subplot(grid[0, :]))
    panel_init(fig.add_subplot(grid[1, 0]))
    panel_downstream(fig.add_subplot(grid[1, 1]))

    fig.suptitle("A seed does not label weights — it sets a position in one shared stream")
    out = fs.save(fig, OUT_DIR / "seed_stream_finding")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
