"""The random number stream itself — nothing about kpnn (slide 12).

Four rules, one row each, almost no text:

    1  a seed picks one fixed sequence
    2  every draw takes the next numbers, however it is split up
    3  one extra draw in front shifts everything after it
    4  torch and numpy are separate sequences

The values are the real ones for seed 42 (torch.rand / np.random.rand).

Run from the project root::

    python src/comp/plot_stream_basics.py
"""
from pathlib import Path

import matplotlib.pyplot as plt

import figure_style as fs

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "comp" / "t2" / "plots" / "plots_stream_basics"

TORCH = ["0.8823", "0.9150", "0.3829", "0.9593", "0.3904", "0.6009"]
NUMPY = ["0.3745", "0.9507", "0.7320"]

INK = fs.REFERENCE_COLOR
SOFT = "0.88"
TAKEN = fs.STAGE_COLORS["own_init"]
CELL, GAP = 1.5, 1.62


def cells(ax, x0, y, values, facecolor=SOFT, textcolor="black"):
    for i, value in enumerate(values):
        ax.add_patch(plt.Rectangle((x0 + i * GAP, y), CELL, 0.62, facecolor=facecolor,
                                   edgecolor="white", lw=0.5))
        ax.text(x0 + i * GAP + CELL / 2, y + 0.31, value, ha="center", va="center",
                color=textcolor)


def bracket(ax, x0, n, y, label):
    """A span under n cells, naming the call that took them."""
    width = (n - 1) * GAP + CELL
    ax.plot([x0, x0 + width], [y, y], lw=0.6, color=INK)
    ax.text(x0 + width / 2, y - 0.12, label, ha="center", va="top")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = fs.subplots(160, 82)
    ax.axis("off")
    label_x = -0.4

    # 1 — the sequence
    ax.text(label_x, 0.31, "seed 42", ha="right", va="center")
    cells(ax, 0, 0, TORCH)
    ax.text(6 * GAP + 0.2, 0.31, "…", va="center")
    ax.text(0, 1.0, "A seed does not pick numbers for one purpose — it picks one fixed "
                    "sequence.", va="bottom")

    # 2 — consumption in order
    y = -2.2
    cells(ax, 0, y, TORCH, TAKEN)
    bracket(ax, 0, 2, y - 0.18, "rand(2)")
    bracket(ax, 2 * GAP, 4, y - 0.18, "rand(4)")
    ax.text(label_x, y + 0.31, "two calls", ha="right", va="center")
    ax.text(0, y + 1.0, "Every draw takes the next numbers. How the calls are split "
                        "does not matter.", va="bottom")

    # 3 — a shift
    y = -5.2
    ax.add_patch(plt.Rectangle((0, y), CELL, 0.62, facecolor=INK, edgecolor="white", lw=0.5))
    ax.text(CELL / 2, y + 0.31, TORCH[0], ha="center", va="center", color="white")
    cells(ax, GAP, y, TORCH[1:], TAKEN)
    bracket(ax, 0, 1, y - 0.18, "one extra draw")
    bracket(ax, GAP, 5, y - 0.18, "rand(5) now starts one position later")
    ax.text(label_x, y + 0.31, "shifted", ha="right", va="center")
    ax.text(0, y + 1.0, "One draw in front of it shifts everything that follows.",
            va="bottom")

    # 4 — separate streams
    y = -8.2
    cells(ax, 0, y, TORCH[:3], TAKEN)
    numpy_x = 3.6 * GAP
    cells(ax, numpy_x, y, NUMPY, TAKEN)
    ax.text(numpy_x, y + 0.72, "numpy", va="bottom")
    ax.text(0, y + 0.72, "torch", va="bottom")
    ax.text(0, y + 1.35, "torch, numpy and python's random each have their own sequence.",
            va="bottom")

    ax.set_xlim(-4.6, 7 * GAP + 0.6)
    ax.set_ylim(-9.6, 1.9)
    out = fs.save(fig, OUT_DIR / "stream_basics")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
