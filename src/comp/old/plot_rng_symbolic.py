"""The random number stream, symbolically: who takes which draw.

A schematic, not a data plot. One toy layer -- 2 TFs x 4 genes with three real
connections e1, e2, e3 -- drawn three times against the same stream z1, z2, ...:

    v1 (SparseLinear)     one draw per connection, in edgelist order
    kpnn2 (MaskedLinear)  one draw per matrix cell, row by row, blanks included
    after the fix         reseed, then draw flat in edgelist order again

The same connection therefore gets a different number in each of the first two
rows, and training starts at a different position in the stream.

Real counts for INS (783 genes, 10 TFs, 845 edges) are printed underneath; they
were measured by seeding the stream, building the layer, and looking up where
the next value sits in that same stream.

Run from the project root::

    python src/comp/plot_rng_symbolic.py
"""
from pathlib import Path

import matplotlib.pyplot as plt

import figure_style as fs

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "comp" / "t2" / "plots" / "plots_rng_symbolic"

V1 = fs.REFERENCE_COLOR
KPNN2 = fs.STAGE_COLORS["own_init"]
BLANK = "0.9"
N = 14                      # how many draws the schematic shows

# What each row does with draw z1 ... zN. "" = the draw is not taken by this row.
V1_ROW = ["e1", "e2", "e3", "w1", "w2", "b", "", "", "", "", "", "", "", ""]
KPNN2_ROW = ["M", "e1", "M", "e2", "b", "e3", "M", "M", "M", "b", "w1", "w2", "b", ""]
FIX_ROW = V1_ROW

ROWS = [
    ("v1 — SparseLinear\none draw per connection", V1_ROW, V1, 7),
    ("kpnn2 — MaskedLinear\none draw per matrix cell", KPNN2_ROW, KPNN2, 14),
    ("after the fix: reseed,\nthen draw like v1", FIX_ROW, V1, 7),
]

CELL, GAP = 0.86, 1.0


def draw_row(ax, y, labels, color, start_training):
    """One row of cells under the stream, with its training-start marker."""
    for i, label in enumerate(labels):
        if not label:
            continue
        masked = label == "M"
        ax.add_patch(plt.Rectangle((i * GAP, y), CELL, 0.62, lw=0.4,
                                   facecolor=BLANK if masked else color,
                                   edgecolor="white"))
        ax.text(i * GAP + CELL / 2, y + 0.31, label, ha="center", va="center",
                color="black" if masked or color is KPNN2 else "white")
    x = (start_training - 1) * GAP
    ax.annotate("", xy=(x + CELL, y + 0.31), xytext=(x + CELL + 1.5, y + 0.31),
                arrowprops=dict(arrowstyle="<-", lw=0.5, color="black"))
    ax.text(x + CELL + 1.7, y + 0.31, f"training starts at z{start_training}", va="center")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = fs.subplots(160, 56)
    ax.axis("off")

    # The stream itself.
    for i in range(N):
        ax.add_patch(plt.Rectangle((i * GAP, 0), CELL, 0.62, facecolor=BLANK,
                                   edgecolor="white", lw=0.4))
        ax.text(i * GAP + CELL / 2, 0.31, f"z{i + 1}", ha="center", va="center")
    ax.text(-0.4, 0.31, "stream after the seed", ha="right", va="center")
    ax.text(N * GAP + 0.1, 0.31, "…", va="center")

    for n, (name, labels, color, start) in enumerate(ROWS):
        y = -1.3 - n * 1.3
        draw_row(ax, y, labels, color, start)
        ax.text(-0.4, y + 0.31, name, ha="right", va="center")

    ax.text(0, -5.0,
            "e1, e2, e3 = the three real connections · M = masked cell (no connection) · "
            "b = bias · w = output weight\n"
            "Same seed, same stream — but e1 gets z1 in v1 and z2 in kpnn2, and training "
            "starts 7 draws later.\n"
            "For INS (783 genes, 10 TFs, 845 edges): v1 takes 845 + 11 = 856 draws, "
            "kpnn2 takes 7830 + 10 + 11 = 7851.", va="top")

    ax.set_xlim(-6.2, N * GAP + 5.2)
    ax.set_ylim(-6.3, 0.9)
    out = fs.save(fig, OUT_DIR / "rng_symbolic")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
