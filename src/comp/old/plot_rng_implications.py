"""What the stream means for the two scripts (slide 13).

Two consequences, one panel each:

    1  the layer itself: SparseLinear draws one number per connection,
       MaskedLinear one per matrix cell -- 856 against 7851 for INS.
    2  the rest of the loop draws too, and not in the same order: v1 shuffles
       with numpy and runs a validation pass, the port shuffled with torch and
       had none, and dropout masks land on genes in a different order.

That is why identical starting weights alone (r = 0.94-0.99) were not enough,
and only matching every draw gave r = 1.000.

Run from the project root::

    python src/comp/plot_rng_implications.py
"""
from pathlib import Path

import matplotlib.pyplot as plt

import figure_style as fs

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "comp" / "t2" / "plots" / "plots_rng_implications"

V1 = fs.REFERENCE_COLOR
PORT = fs.STAGE_COLORS["own_init"]
SOFT = "0.9"

# Panel A: measured draw counts for INS (783 genes, 10 TFs, 845 connections).
INIT = [("v1 — SparseLinear", 845, 856, V1, "one draw per connection"),
        ("kpnn2 — MaskedLinear", 7840, 7851, PORT, "one draw per matrix cell")]
TOTAL = 7851

# Panel B: what one epoch draws, in order.
EPOCH = {
    "v1": [("torch", "int64 for the\ntrain loader"), ("numpy", "shuffle the\ntrain cells"),
           ("torch", "dropout mask\nper batch"), ("torch", "int64 for the\nval loader"),
           ("numpy", "shuffle the\nval cells")],
    "the port, before the fix": [("torch", "int64 for the\ntrain loader"),
                                 ("torch", "shuffle the\ntrain cells"),
                                 ("torch", "dropout mask\nper batch")],
}
BOX_W, BOX_GAP = 3.1, 3.4


def panel_init(ax):
    ax.set_title("1   The layer: how many numbers it takes before training starts",
                 loc="left")
    for y, (name, first, total, color, note) in enumerate(INIT):
        ax.add_patch(plt.Rectangle((0, -y - 0.22), first, 0.44, facecolor=color,
                                   edgecolor="white", lw=0.4))
        ax.add_patch(plt.Rectangle((first, -y - 0.22), total - first, 0.44,
                                   facecolor="0.45", edgecolor="white", lw=0.4))
        ax.text(total + 90, -y, f"{total} numbers", va="center")
        ax.text(-150, -y, name, ha="right", va="center")
        ax.text(0, -y - 0.33, note, va="top")
    ax.set_xlim(-3400, TOTAL * 1.18)
    ax.set_ylim(-1.75, 0.6)
    ax.set_yticks([])
    ax.set_xticks(range(0, 8001, 2000))   # the negative range only holds labels
    ax.set_xlabel("position in the torch stream")
    ax.spines[["left", "top", "right"]].set_visible(False)


def panel_loop(ax):
    ax.set_title("2   The rest of the loop draws as well — and not in the same order",
                 loc="left")
    ax.axis("off")
    for y, (name, steps) in enumerate(EPOCH.items()):
        color = V1 if name == "v1" else PORT
        for i, (stream, text) in enumerate(steps):
            x = i * BOX_GAP
            ax.add_patch(plt.Rectangle((x, -y * 1.75 - 0.75), BOX_W, 0.75,
                                       facecolor=color if stream == "torch" else SOFT,
                                       edgecolor="white", lw=0.5))
            ax.text(x + BOX_W / 2, -y * 1.75 - 0.38, text, ha="center", va="center",
                    color="white" if stream == "torch" and color is V1 else "black")
            ax.text(x + BOX_W / 2, -y * 1.75 - 0.85, stream, ha="center", va="top")
        ax.text(-0.3, -y * 1.75 - 0.38, name, ha="right", va="center")
    ax.plot([2 * BOX_GAP - 0.15, 2 * BOX_GAP - 0.15], [0.1, -2.6], lw=0.5, ls=":",
            color="0.5")
    ax.text(2 * BOX_GAP - 0.05, 0.15,
            "from here the two runs see different batches and different dropout masks",
            va="bottom")
    ax.text(0, -3.6,
            "Dropout masks are drawn by position, and v1 orders the genes by the edgelist "
            "while kpnn2 sorts them alphabetically —\nso even identical numbers would drop "
            "different genes.", va="top")
    ax.set_xlim(-6.0, 5 * BOX_GAP + 0.5)
    ax.set_ylim(-4.6, 0.8)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, _ = fs.subplots(160, 96, 1, 1)
    fig.clear()
    grid = fig.add_gridspec(2, 1, height_ratios=[0.75, 1.25], hspace=0.3)
    panel_init(fig.add_subplot(grid[0]))
    panel_loop(fig.add_subplot(grid[1]))
    fig.suptitle("Same seed, same sequence — but the two scripts take different numbers "
                 "from it, in a different order")
    out = fs.save(fig, OUT_DIR / "rng_implications")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
