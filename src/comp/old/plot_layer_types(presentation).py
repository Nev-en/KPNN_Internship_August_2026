"""What SparseLinear and MaskedLinear store, as a network and as a matrix.

A schematic, not a data plot. One toy layer: 8 genes -> 4 TFs with six real
connections, shown twice:

    v1 SparseLinear     the network has only the six connections; stored is an
                        index list plus one weight each. The matrix exists only
                        inside the forward pass (torch.sparse_coo_tensor ->
                        torch.sparse.mm).
    kpnn2 MaskedLinear  the network is fully connected, but the 26 connections
                        that should not exist are shaded out by a 0/1 mask.
                        Stored is the full 4 x 8 matrix; F.linear multiplies
                        with weights x mask, which equals v1's matrix.

Run from the project root::

    python src/comp/plot_layer_types.py
"""
from pathlib import Path

import matplotlib.pyplot as plt

import figure_style as fs

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "comp" / "t2" / "plots" / "plots_layer_types"

V1 = fs.REFERENCE_COLOR
KPNN2 = fs.STAGE_COLORS["own_init"]
STORED = "0.78"          # stored, but masked out
EMPTY = "0.94"           # not part of the matrix at all
SHADED = "0.91"          # a connection the mask switches off

N_TF, N_GENE = 4, 8
EDGES = [(0, 1), (0, 5), (1, 2), (2, 0), (2, 6), (3, 3)]   # (TF row, gene column)
CELL = 0.9
WIDTH = N_GENE * CELL

NET_W, NET_H = 2.6, 4.0      # the network sketch: width between the two columns
NODE = 0.16


def network(ax, x0, y_top, color, all_connections):
    """Genes (left column) -> TFs (right column); dead edges shaded or absent."""
    gene_y = [y_top - 0.25 - i * (NET_H / (N_GENE - 1)) for i in range(N_GENE)]
    tf_y = [y_top - 0.9 - i * ((NET_H - 1.3) / (N_TF - 1)) for i in range(N_TF)]

    if all_connections:      # draw the shaded ones first, so live edges sit on top
        for tf in range(N_TF):
            for gene in range(N_GENE):
                if (tf, gene) not in EDGES:
                    ax.plot([x0, x0 + NET_W], [gene_y[gene], tf_y[tf]],
                            lw=0.3, color=SHADED, zorder=1)
    for tf, gene in EDGES:
        ax.plot([x0, x0 + NET_W], [gene_y[gene], tf_y[tf]], lw=0.9, color=color, zorder=2)

    for y in gene_y:
        ax.add_patch(plt.Circle((x0, y), NODE, facecolor="white", edgecolor="black",
                                lw=0.4, zorder=3))
    for y in tf_y:
        ax.add_patch(plt.Circle((x0 + NET_W, y), NODE, facecolor=color,
                                edgecolor="black", lw=0.4, zorder=3))
    ax.text(x0, y_top + 0.1, "genes", ha="center", va="bottom")
    ax.text(x0 + NET_W, y_top + 0.1, "TFs", ha="center", va="bottom")


def matrix(ax, x0, y_top, live_color, dead_color, dead_edge=None, labels=None, title=None,
           axis_labels=True):
    """The toy weight matrix with its top-left corner at (x0, y_top)."""
    for tf in range(N_TF):
        for gene in range(N_GENE):
            live = (tf, gene) in EDGES
            ax.add_patch(plt.Rectangle(
                (x0 + gene * CELL, y_top - (tf + 1) * CELL), CELL * 0.92, CELL * 0.92,
                facecolor=live_color if live else dead_color,
                edgecolor="white" if live or dead_edge is None else dead_edge, lw=0.4))
            text = labels.get((tf, gene), "") if labels else ""
            if text:
                ax.text(x0 + gene * CELL + CELL * 0.46,
                        y_top - (tf + 1) * CELL + CELL * 0.46, text,
                        ha="center", va="center",
                        color="white" if live and live_color is V1 else "black")
    if title:
        ax.text(x0, y_top + 0.15, title, va="bottom")
    if axis_labels:
        ax.text(x0 - 0.25, y_top - N_TF * CELL / 2, "TFs", ha="right", va="center",
                rotation=90)
    ax.text(x0 + WIDTH / 2, y_top - N_TF * CELL - 0.2, "genes", ha="center", va="top")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = fs.subplots(160, 120)
    ax.axis("off")
    mat_x = NET_W + 2.2                     # matrices start right of the network

    # --- v1 -------------------------------------------------------------------
    ax.text(0, 0, "v1 — SparseLinear: only the real connections exist",
            va="bottom", fontweight="bold")
    network(ax, 0, -1.4, V1, all_connections=False)
    ax.text(NET_W / 2, -6.1, "6 connections", ha="center", va="top")

    for k, (tf, gene) in enumerate(EDGES):
        ax.add_patch(plt.Rectangle((mat_x + k * 1.45, -1.9), 1.0, 0.6, facecolor=V1,
                                   edgecolor="white", lw=0.4))
        ax.text(mat_x + k * 1.45 + 0.5, -1.6, f"w{k + 1}", ha="center", va="center",
                color="white")
        ax.text(mat_x + k * 1.45 + 0.5, -2.1, f"g{gene}→T{tf}", ha="center", va="top")
    ax.text(mat_x, -1.25, "stored: index list + 6 weights", va="bottom")

    matrix(ax, mat_x, -3.3, V1, "none", "0.8",
           {edge: f"w{k + 1}" for k, edge in enumerate(EDGES)})
    ax.text(mat_x + WIDTH + 0.6, -3.3 - N_TF * CELL / 2,
            "torch.sparse.mm builds this matrix\nin the forward pass.\n"
            "The empty cells are never stored\nand never drawn from the stream.",
            va="center")

    # --- kpnn2 ----------------------------------------------------------------
    top = -9.0
    ax.text(0, top, "kpnn2 — MaskedLinear: every connection exists, the mask shades "
                    "26 of them out", va="bottom", fontweight="bold")
    network(ax, 0, top - 1.4, KPNN2, all_connections=True)
    ax.text(NET_W / 2, top - 6.1, "32 connections,\n6 of them active", ha="center", va="top")

    grid_top = top - 1.9
    middle = grid_top - N_TF * CELL / 2
    ax.text(mat_x, grid_top + 0.75, "stored: all 32 weights, plus the mask", va="bottom")
    matrix(ax, mat_x, grid_top, KPNN2, STORED, None, None, "weights")

    mask_x = mat_x + WIDTH + 1.3
    ax.text(mask_x - 0.65, middle, "×", ha="center", va="center")
    matrix(ax, mask_x, grid_top, KPNN2, EMPTY, None,
           {(tf, gene): ("1" if (tf, gene) in EDGES else "0")
            for tf in range(N_TF) for gene in range(N_GENE)}, "mask", axis_labels=False)
    ax.text(mask_x + WIDTH + 0.6, middle,
            "= what F.linear multiplies with,\nthe same matrix as v1's.\n"
            "The 26 shaded cells are stored and\nwere drawn — they are multiplied by 0.",
            va="center")

    # --- the consequence ------------------------------------------------------
    ax.text(0, -17.2,
            "Both compute the same thing: six weights act, the rest is zero.\n"
            "But for INS (783 genes, 10 TFs, 845 connections) v1 draws 845 numbers and "
            "kpnn2 draws 7830 —\nso with the same seed the same connection ends up with a "
            "different number.", va="top")

    ax.set_xlim(-1.4, mask_x + WIDTH + 7.0)
    ax.set_ylim(-19.4, 0.9)
    out = fs.save(fig, OUT_DIR / "layer_types")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
