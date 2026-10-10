"""How SparseLinear and MaskedLinear eat the random number stream (INS).

Panel A  how many numbers each layer takes before training starts
Panel B  which number ends up on which connection

Counts are measured, not estimated: seed the stream, build the layer, then look
up the next value's position in the same stream (see docstring of count_draws
in the module that produced them). For INS, 783 genes -> 10 TFs, 845 edges:

    v1  SparseLinear        845   one per edge, in edgelist order
        nn.Linear(10, 1)     11   10 weights + 1 bias
        total                856

    kpnn2  MaskedLinear    7840   10 x 783 full matrix + 10 row biases
           nn.Linear(10,1)   11
           total           7851

Run from the project root::

    python src/comp/plot_rng_consumption.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import figure_style as fs

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "comp" / "t2" / "plots" / "plots_rng_consumption"

V1 = fs.REFERENCE_COLOR
KPNN2 = fs.STAGE_COLORS["own_init"]
STREAM = "0.9"

# Measured draw counts for INS (783 genes, 10 TFs, 845 edges).
SEGMENTS = {
    "v1 — SparseLinear": [
        ("845 edge weights\n+ 11 for the output layer", 845, V1),
        ("", 11, "0.45"),
    ],
    "kpnn2 — MaskedLinear": [
        ("7840 = 10 x 783 matrix cells + 10 row biases\n+ 11 for the output layer", 7840, KPNN2),
        ("", 11, "0.45"),
    ],
}
TOTAL = 7851

# A toy network for panel B: 4 TFs x 8 genes with 6 edges, drawn as a grid.
N_TF, N_GENE = 4, 8
EDGES = [(0, 1), (0, 5), (1, 2), (2, 0), (2, 6), (3, 3)]   # (TF row, gene column)


def panel_stream(ax):
    """Two bars on one axis: how far into the stream each layer reaches."""
    ax.set_title("A   How many random numbers the layer takes before training\n"
                 "Same seed, same stream — but kpnn2 is 9x further along when training "
                 "starts, so every later draw (shuffling, dropout) differs.", loc="left")
    for y, (name, parts) in enumerate(SEGMENTS.items()):
        start = 0
        for label, width, color in parts:
            ax.add_patch(plt.Rectangle((start, y - 0.22), width, 0.44, facecolor=color,
                                       edgecolor="white", lw=0.4))
            start += width
        ax.text(start + 90, y, f"{start} numbers", va="center")
        ax.text(0, y - 0.33, parts[0][0].replace("\n", "   "), va="bottom")
    ax.set_xlim(0, TOTAL * 1.16)
    ax.set_ylim(1.6, -0.75)
    ax.set_yticks([0, 1], labels=list(SEGMENTS))
    ax.set_xlabel("position in the random number stream")
    ax.spines[["left", "top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)


def grid(ax, numbers, title, color, note):
    """One toy weight matrix: live edges carry the number they were given."""
    ax.set_title(title, loc="left")
    ax.axis("off")
    for tf in range(N_TF):
        for gene in range(N_GENE):
            live = (tf, gene) in EDGES
            ax.add_patch(plt.Rectangle((gene, N_TF - 1 - tf), 0.92, 0.92,
                                       facecolor=color if live else STREAM,
                                       edgecolor="white", lw=0.5))
            if live:
                ax.text(gene + 0.46, N_TF - 1 - tf + 0.46, f"#{numbers[(tf, gene)]}",
                        ha="center", va="center", color="white")
    ax.text(-0.3, N_TF / 2, "TFs", ha="right", va="center", rotation=90)
    ax.text(N_GENE / 2, -0.5, "genes", ha="center", va="top")
    ax.text(0, -1.25, note, va="top")
    ax.set_xlim(-1.4, N_GENE + 0.3)
    ax.set_ylim(-2.2, N_TF + 0.3)
    ax.set_aspect("equal")


def panel_mapping(axes):
    """Panel B: same six edges, different numbers on them."""
    v1_numbers = {edge: k + 1 for k, edge in enumerate(EDGES)}        # edgelist order
    # kpnn2 walks the whole matrix row by row, blanks included.
    dense = {(tf, gene): tf * N_GENE + gene + 1
             for tf in range(N_TF) for gene in range(N_GENE)}
    kpnn2_numbers = {edge: dense[edge] for edge in EDGES}
    grid(axes[0], v1_numbers, "B   v1: one draw per edge, in edgelist order",
         V1, "6 edges -> numbers #1-#6.\nGrey cells are never drawn.")
    grid(axes[1], kpnn2_numbers, "      kpnn2: one draw per matrix cell, row by row",
         KPNN2, "32 cells -> numbers #1-#32.\nThe same 6 edges get #2, #6, #11, #17, #23, #28.")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, _ = fs.subplots(160, 95, 1, 1)
    fig.clear()
    outer = fig.add_gridspec(2, 1, height_ratios=[0.8, 1.2], hspace=0.35)
    panel_stream(fig.add_subplot(outer[0]))
    inner = outer[1].subgridspec(1, 2, wspace=0.1)
    panel_mapping([fig.add_subplot(inner[0]), fig.add_subplot(inner[1])])

    fig.suptitle("The same seed gives the two layers different weights — "
                 "they take different numbers, in a different order")
    out = fs.save(fig, OUT_DIR / "rng_consumption")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
