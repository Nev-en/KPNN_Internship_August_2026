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

Numbers are the measured ones from this project (INS, seed 1, for panel B;
20 seeds per condition for panel C). Run from the project root::

    python src/comp/plot_seed_stream.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "comp" / "t2" / "plots" / "plots_seed_stream"

WRONG = "#E45756"   # drawing at the wrong stream position
RIGHT = "#54A24B"   # reseeded, drawing at v1's position
GREY = "#BAB0AC"

CONDITIONS = ["IFNB", "IFNG", "INS", "TGFB1", "TNFA"]
# TF-layer correlation per seed against Karla's kpnn v1 run, before and after
# apply_v1_init learned to reseed.
BEFORE = [0.902, 0.842, 0.939, 0.814, 0.867]
AFTER = [0.960, 0.976, 0.991, 0.966, 0.942]

# Correlation of the initial sparse-layer weights to v1's, INS seed 1.
INIT_R = {"port, no reseed": 0.005, "port2 / port, reseeded": 1.000}


def panel_mechanism(ax):
    """The stream as a row of numbers, and who eats which part of it."""
    ax.set_title("A   One seed, one stream — position decides", loc="left",
                 fontsize=11, fontweight="bold")
    ax.axis("off")

    values = ["12", "867", "2", "34374", "456", "91", "7", "…"]
    for i, value in enumerate(values):
        ax.add_patch(plt.Rectangle((i, 2.3), 0.9, 0.6, facecolor=GREY,
                                   edgecolor="white", alpha=0.6))
        ax.text(i + 0.45, 2.6, value, ha="center", va="center", fontsize=8.5)
    ax.text(-0.3, 2.6, "stream\nafter seed", ha="right", va="center", fontsize=8.5)

    # v1 takes the first three; the port's constructor eats them first instead.
    ax.add_patch(plt.Rectangle((0, 1.35), 2.9, 0.55, facecolor=RIGHT, alpha=0.30,
                               edgecolor=RIGHT, lw=1.4))
    ax.text(1.45, 1.62, "v1: sparse layer", ha="center", va="center", fontsize=9)
    ax.text(-0.3, 1.62, "correct", ha="right", va="center", fontsize=8.5, color=RIGHT)

    ax.add_patch(plt.Rectangle((0, 0.5), 2.9, 0.55, facecolor=GREY, alpha=0.45,
                               edgecolor="0.5", lw=1.0))
    ax.text(1.45, 0.77, "constructor (dense draw)", ha="center", va="center", fontsize=9)
    ax.add_patch(plt.Rectangle((3, 0.5), 2.9, 0.55, facecolor=WRONG, alpha=0.30,
                               edgecolor=WRONG, lw=1.4))
    ax.text(4.45, 0.77, "sparse layer", ha="center", va="center", fontsize=9)
    ax.text(-0.3, 0.77, "shifted", ha="right", va="center", fontsize=8.5, color=WRONG)

    ax.text(0, -0.35, "Same seed, same init rule — the layer simply starts "
                      "further along the stream.",
            fontsize=9, color="0.25")
    ax.set_xlim(-2.6, 8.3)
    ax.set_ylim(-0.7, 3.2)


def panel_init(ax):
    """What the shift did to the initial weights."""
    ax.set_title("B   Initial weights vs kpnn v1\n(INS, seed 1)", loc="left",
                 fontsize=11, fontweight="bold")
    labels = list(INIT_R)
    values = [INIT_R[k] for k in labels]
    bars = ax.barh(labels, values, color=[WRONG, RIGHT], height=0.32, alpha=0.85)
    for bar, value in zip(bars, values):
        ax.text(value + 0.03, bar.get_y() + bar.get_height() / 2,
                f"r = {value:.3f}", va="center", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1.30)
    # Keep the two bars close together rather than spread over the whole panel.
    ax.set_ylim(-0.55, 1.55)
    ax.set_xlabel("correlation to v1's weights")
    ax.axvline(1.0, ls=":", lw=1, color="0.6")
    ax.tick_params(labelsize=9.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(0.0, -0.44, "Both use v1's bound √(6 / average fan-in):\n"
                        "the spread matches, the values do not.",
            fontsize=9, color="0.25", transform=ax.get_yaxis_transform(),
            clip_on=False)


def panel_downstream(ax):
    """What it cost in the actual comparison against Karla's run."""
    ax.set_title("C   TF-level agreement with Karla's kpnn v1 run\n"
                 "(per seed, 20 seeds per condition)", loc="left",
                 fontsize=11, fontweight="bold")
    x = np.arange(len(CONDITIONS))
    width = 0.38
    ax.bar(x - width / 2, BEFORE, width, label="before: no reseed",
           color=WRONG, alpha=0.85)
    ax.bar(x + width / 2, AFTER, width, label="after: reseeded",
           color=RIGHT, alpha=0.85)

    for i, (before, after) in enumerate(zip(BEFORE, AFTER)):
        ax.text(i + width / 2, after + 0.006, f"{after:.3f}", ha="center",
                fontsize=8, fontweight="bold")
        ax.text(i - width / 2, before + 0.006, f"{before:.3f}", ha="center",
                fontsize=8, color="0.35")

    ax.set_xticks(x)
    ax.set_xticklabels(CONDITIONS)
    ax.set_ylim(0.75, 1.02)
    ax.set_ylabel("Pearson r")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(axis="y", ls=":", lw=0.6, color="0.85")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13.5, 8.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[0.85, 1.15])

    panel_mechanism(fig.add_subplot(grid[0, :]))
    panel_init(fig.add_subplot(grid[1, 0]))
    panel_downstream(fig.add_subplot(grid[1, 1]))

    fig.suptitle(
        "A seed does not label weights — it sets a position in one shared stream",
        fontsize=14, fontweight="bold",
    )
    out = OUT_DIR / "seed_stream_finding.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
