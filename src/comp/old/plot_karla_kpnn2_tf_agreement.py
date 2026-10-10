"""TF-level agreement between Karla's kpnn v1 run and a kpnn2 run.

Why this exists next to plot_karla_benjamin_scatter.py
------------------------------------------------------
That script plots one point per (seed, node) and draws y=x, which is the right
picture when two runs are supposed to be identical. kpnn2 runs that draw their
random numbers differently from v1 (native, minimal, the port with
rng="torch") do not reproduce seed N, and judged that way even a faithful
implementation looks like a scatter cloud.

So this figure asks whether the run recovers the same biology. For each TF it
plots the mean score across the 20 seeds, with whiskers for +-1 SD across seeds.
If the run is faithful, points sit near y=x and, more importantly, each run's
mean falls inside the other's seed spread.

Figures
-------
`tf_agreement.svg`     one panel per condition, class 0 filled and class 1 open,
                       Karla mean (x) vs kpnn2 mean (y), whiskers = +-1 SD.
`tf_rank_agreement.svg` per condition, TFs ordered by Karla's mean score, with
                       both runs' means -- shows whether the ranking survives.
                       Karla black, the compared run in the condition colour.

Styling comes from figure_style.py. Reads the table written by
compare_karla_kpnn2.py.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy import stats

import figure_style as fs

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN_NAME = os.environ.get("COMPARISON_IN", "karla_new_vs_kpnn2_native")
OUT_NAME = os.environ.get("PLOT_OUT", "plots/plots_karla_new_vs_kpnn2_native")

# comp/ is split by experiment generation: t1 holds the kpnn v1 era material,
# t2 the kpnn2 one. Set COMP_SUBDIR="" to read/write flat in comp/ as before.
COMP_SUBDIR = os.environ.get("COMP_SUBDIR", "t2")
COMP_ROOT = PROJECT_ROOT / "comp" / COMP_SUBDIR if COMP_SUBDIR else PROJECT_ROOT / "comp"

IN_DIR = COMP_ROOT / IN_NAME
OUT_DIR = COMP_ROOT / OUT_NAME

COMBINED_NAME = "all_conditions_comparison.csv"
SEP = ";"
DECIMAL = ","

CONDITIONS = fs.CONDITIONS
# Standard numbering: layer 1 is the first hidden layer, i.e. the TF nodes.
TF_LAYER = 1
TF_MARKER = fs.LAYER_MARKERS[TF_LAYER]

META_COLS = {"Condition", "Seed", "Class", "Layer", "Node", "diff"}


def load() -> tuple[pd.DataFrame, str, str]:
    path = IN_DIR / COMBINED_NAME
    if not path.exists():
        raise SystemExit(f"missing {path} — run compare_karla_kpnn2.py first")
    df = pd.read_csv(path, sep=SEP, decimal=DECIMAL)
    value_cols = [c for c in df.columns if c not in META_COLS]
    if len(value_cols) != 2:
        raise SystemExit(f"expected exactly 2 value columns, found {value_cols}")
    return df, value_cols[0], value_cols[1]


def summarise(df, x_col, y_col):
    """Mean and SD across seeds, per condition/class/node."""
    grouped = df.groupby(["Condition", "Class", "Node"], as_index=False).agg(
        x_mean=(x_col, "mean"), x_sd=(x_col, "std"),
        y_mean=(y_col, "mean"), y_sd=(y_col, "std"),
    )
    # A run's mean is "inside the other's spread" when it sits within +-1 SD.
    grouped["within_spread"] = (
        (grouped["y_mean"] - grouped["x_mean"]).abs() <= grouped["x_sd"].fillna(0)
    )
    return grouped


def plot_agreement(summary, x_col, y_col, conds):
    fig, axes = fs.subplots(160, 54, 1, len(conds))
    axes = np.atleast_1d(axes)

    for ax, cond in zip(axes, conds):
        sub = summary[summary["Condition"] == cond]
        color = fs.CONDITION_COLORS.get(cond, "0.4")
        lo = float(min(sub.x_mean.min(), sub.y_mean.min()))
        hi = float(max(sub.x_mean.max(), sub.y_mean.max()))
        pad = 0.08 * (hi - lo) if hi > lo else 1.0
        lo, hi = lo - pad, hi + pad

        ax.axhline(0, lw=0.4, color=fs.ZERO_COLOR, zorder=0)
        ax.axvline(0, lw=0.4, color=fs.ZERO_COLOR, zorder=0)
        ax.plot([lo, hi], [lo, hi], ls="--", lw=0.5, color=fs.GUIDE_COLOR, zorder=3)

        for cls in (0, 1):
            part = sub[sub["Class"] == cls]
            if part.empty:
                continue
            ax.errorbar(
                part.x_mean, part.y_mean,
                xerr=part.x_sd.fillna(0), yerr=part.y_sd.fillna(0),
                fmt=TF_MARKER, ms=2.5, mew=0.6, lw=0.5, capsize=1,
                color=color, mfc=fs.face(color, cls), mec=color, zorder=2,
            )

        r = sub.x_mean.corr(sub.y_mean)
        rho = stats.spearmanr(sub.x_mean, sub.y_mean).statistic
        inside = sub.within_spread.mean()
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.set_title(
            f"{cond}  (n={len(sub)} TF·class)\n"
            f"r={r:+.3f}  ρ={rho:+.3f}\nwithin ±1 SD: {inside:.0%}"
        )
        fs.tidy(ax)
    axes[0].set_ylabel(f"{fs.display(y_col)} — mean over seeds")
    fig.supxlabel(f"{fs.display(x_col)} — mean over seeds")
    fig.legend(handles=fs.class_legend_handles(TF_MARKER), loc="outside lower right", ncol=2)
    fig.suptitle(
        f"TF-level agreement — {fs.display(x_col)} vs {fs.display(y_col)}\n"
        "point = one TF, whiskers = ±1 SD across the 20 seeds"
    )
    out = fs.save(fig, OUT_DIR / "tf_agreement")
    print(f"-> {out}")


def plot_ranks(summary, x_col, y_col, conds):
    fig, axes = fs.subplots(160, 160, len(conds), 1)
    axes = np.atleast_1d(axes)

    for ax, cond in zip(axes, conds):
        sub = summary[(summary["Condition"] == cond) & (summary["Class"] == 0)]
        sub = sub.sort_values("x_mean")
        pos = np.arange(len(sub))
        color = fs.CONDITION_COLORS.get(cond, "0.4")
        # Offset the two runs a little so their whiskers do not overlap.
        ax.errorbar(pos - 0.12, sub.x_mean, yerr=sub.x_sd.fillna(0), fmt=TF_MARKER,
                    ms=2.5, color=fs.REFERENCE_COLOR, capsize=1, lw=0.5)
        ax.errorbar(pos + 0.12, sub.y_mean, yerr=sub.y_sd.fillna(0), fmt=TF_MARKER,
                    ms=2.5, color=color, capsize=1, lw=0.5)
        ax.axhline(0, lw=0.4, color=fs.ZERO_COLOR)
        ax.set_xticks(pos)
        ax.set_xticklabels(sub.Node, rotation=45, ha="right")
        rho = stats.spearmanr(sub.x_mean, sub.y_mean).statistic
        ax.set_title(f"{cond} — class 0, TFs ordered by {fs.display(x_col)}   ρ={rho:+.3f}")
        ax.set_ylabel("score")
        ax.grid(axis="y", ls=":", lw=0.4, color=fs.ZERO_COLOR)
        ax.set_axisbelow(True)
        fs.tidy(ax)
    handles = [
        Line2D([], [], ls="none", marker=TF_MARKER, ms=3, color=fs.REFERENCE_COLOR,
               label=fs.display(x_col)),
        Line2D([], [], ls="none", marker=TF_MARKER, ms=3, color="0.6",
               label=f"{fs.display(y_col)} (condition colour)"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=2)
    fig.suptitle("Does the TF ranking survive?")
    out = fs.save(fig, OUT_DIR / "tf_rank_agreement")
    print(f"-> {out}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, x_col, y_col = load()
    fs.label_overrides(x_col, y_col)
    df = df[df["Layer"] == TF_LAYER]
    if df.empty:
        raise SystemExit("no TF-layer rows in the comparison table")

    conds = [c for c in CONDITIONS if (df["Condition"] == c).any()]
    summary = summarise(df, x_col, y_col)

    print(f"Input  {IN_DIR}\nOutput {OUT_DIR}\n")
    for cond in conds:
        sub = summary[summary["Condition"] == cond]
        rho = stats.spearmanr(sub.x_mean, sub.y_mean).statistic
        print(f"[{cond}] {len(sub)} TF·class  r={sub.x_mean.corr(sub.y_mean):+.3f}  "
              f"rho={rho:+.3f}  within ±1 SD: {sub.within_spread.mean():.0%}")

    plot_agreement(summary, x_col, y_col, conds)
    plot_ranks(summary, x_col, y_col, conds)


if __name__ == "__main__":
    main()
