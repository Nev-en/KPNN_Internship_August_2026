"""Two comparison tables in one scatter: a grey run behind a coloured one.

Same grid as plot_karla_benjamin_scatter.py (layers as rows, conditions as
columns), but each panel carries two runs against the same reference:

    background   light grey, drawn first -- the run you are improving on
    foreground   condition colour -- the run you are showing

Both r values sit above the panel, the grey one in grey. Axis limits cover both
runs, so the two clouds are directly comparable.

Environment variables
---------------------
COMPARISON_IN    foreground table folder (default karla_new_vs_kpnn2_port)
COMPARISON_BG    background table folder (default karla_new_vs_kpnn2_port_alt)
PLOT_OUT         figure folder (default plots/plots_overlay)
COMP_SUBDIR      t1 or t2 (default t2)
X_LABEL, Y_LABEL, BG_LABEL   override the names drawn in the figure

Both tables must share the reference column (x). Run from the project root::

    python src/comp/plot_overlay_scatter.py
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

import figure_style as fs

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN_NAME = os.environ.get("COMPARISON_IN", "karla_new_vs_kpnn2_port")
BG_NAME = os.environ.get("COMPARISON_BG", "karla_new_vs_kpnn2_port_alt")
OUT_NAME = os.environ.get("PLOT_OUT", "plots/plots_overlay")

COMP_SUBDIR = os.environ.get("COMP_SUBDIR", "t2")
COMP_ROOT = PROJECT_ROOT / "comp" / COMP_SUBDIR if COMP_SUBDIR else PROJECT_ROOT / "comp"
OUT_DIR = COMP_ROOT / OUT_NAME

COMBINED_NAME = "all_conditions_comparison.csv"
SEP, DECIMAL = ";", ","
META_COLS = {"Condition", "Seed", "Class", "Layer", "Node", "diff"}

BG_COLOR = "0.72"            # the light grey of the background run
BG_ALPHA = {0: 0.30, 1: 0.55}
FG_ALPHA = {0: 0.30, 1: 0.70}


def load(name: str):
    path = COMP_ROOT / name / COMBINED_NAME
    if not path.exists():
        raise SystemExit(f"missing {path}")
    df = pd.read_csv(path, sep=SEP, decimal=DECIMAL)
    cols = [c for c in df.columns if c not in META_COLS]
    if len(cols) != 2:
        raise SystemExit(f"{path}: expected 2 value columns, found {cols}")
    print(f"{name}: {len(df):,} rows, '{cols[0]}' vs '{cols[1]}'")
    return df, cols[0], cols[1]


def panel(ax, fg, bg, cond, layer, x_col, fg_col, bg_col):
    """Background run first, then the foreground run on top."""
    marker, area = fs.LAYER_MARKERS[layer], fs.LAYER_POINT_AREA[layer]
    lo = min(fg[x_col].min(), fg[fg_col].min(), bg[bg_col].min())
    hi = max(fg[x_col].max(), fg[fg_col].max(), bg[bg_col].max())
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    lo, hi = lo - pad, hi + pad

    ax.axhline(0, lw=0.4, color=fs.ZERO_COLOR, zorder=0)
    ax.axvline(0, lw=0.4, color=fs.ZERO_COLOR, zorder=0)
    ax.plot([lo, hi], [lo, hi], ls="--", lw=0.5, color=fs.GUIDE_COLOR, zorder=3)

    ax.scatter(bg[x_col], bg[bg_col], s=area, marker=marker, alpha=BG_ALPHA[layer],
               color=BG_COLOR, linewidths=0, zorder=1, rasterized=True)
    ax.scatter(fg[x_col], fg[fg_col], s=area, marker=marker, alpha=FG_ALPHA[layer],
               color=fs.CONDITION_COLORS.get(cond, "0.4"), linewidths=0, zorder=2,
               rasterized=True)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    fs.tidy(ax)

    r_fg = np.corrcoef(fg[x_col], fg[fg_col])[0, 1]
    r_bg = np.corrcoef(bg[x_col], bg[bg_col])[0, 1]
    ax.text(0.0, 1.04, f"r={r_fg:.4f}", transform=ax.transAxes, va="bottom")
    ax.text(1.0, 1.04, f"(r={r_bg:.4f})", transform=ax.transAxes, va="bottom",
            ha="right", color=BG_COLOR)
    return r_fg, r_bg


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output {OUT_DIR}\n")
    fg_df, x_col, fg_col = load(IN_NAME)
    bg_df, bg_x_col, bg_col = load(BG_NAME)
    if bg_x_col != x_col:
        raise SystemExit(f"the two tables use different references: {x_col} vs {bg_x_col}")
    fs.label_overrides(x_col, fg_col)
    if os.environ.get("BG_LABEL"):
        fs.DISPLAY_NAMES[bg_col] = os.environ["BG_LABEL"]

    conds = [c for c in fs.CONDITIONS if (fg_df["Condition"] == c).any()]
    fig, axes = fs.subplots(160, 88, 2, len(conds))
    axes = np.asarray(axes).reshape(2, len(conds))

    for col, cond in enumerate(conds):
        for row, layer in enumerate((0, 1)):
            pick = lambda d: d[(d["Condition"] == cond) & (d["Layer"] == layer)]
            fg, bg = pick(fg_df), pick(bg_df)
            if fg.empty or bg.empty:
                axes[row, col].axis("off")
                continue
            r_fg, r_bg = panel(axes[row, col], fg, bg, cond, layer, x_col, fg_col, bg_col)
            if row == 0:
                axes[row, col].set_title(cond, pad=12)
            print(f"[{cond}] L{layer}: r={r_fg:.4f}  background r={r_bg:.4f}")

    for row, layer in enumerate((0, 1)):
        axes[row, 0].set_ylabel(f"{fs.LAYER_NAMES[layer]}\n{fs.display(fg_col)}")
    axes[1, len(conds) // 2].set_xlabel(fs.display(x_col))
    fig.legend(handles=[
        Line2D([], [], ls="none", marker="s", ms=3, color="0.45",
               label=f"{fs.display(fg_col)} (coloured, in front)"),
        Line2D([], [], ls="none", marker="s", ms=3, color=BG_COLOR,
               label=f"{fs.display(bg_col)} (grey, behind)"),
    ], loc="outside lower center", ncol=2)
    fig.suptitle(f"{fs.display(x_col)} vs {fs.display(fg_col)}, "
                 f"with {fs.display(bg_col)} behind it")
    out = fs.save(fig, OUT_DIR / "all_conditions_overlay")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
