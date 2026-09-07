"""TF-level agreement between Karla's kpnn v1 run and the kpnn2 port.

Why this exists next to plot_karla_benjamin_scatter.py
------------------------------------------------------
That script plots one point per (seed, node) and draws y=x, which is the right
picture when two runs are supposed to be identical -- as when comparing two kpnn
v1 runs against each other. The kpnn2 port is not bit-identical by construction:
it re-implements the training loop, so seed N does not reproduce Karla's seed N.
Judged that way even a perfectly faithful port looks like a scatter cloud.

So this figure asks the question that actually matters: does the port recover
the same biology? For each TF it plots the mean score across the 20 seeds, with
whiskers for +-1 SD across seeds. If the port is faithful, points sit near y=x
and, more importantly, each run's mean falls inside the other's seed spread.

Figures
-------
`tf_agreement.png`     one panel per condition (class 0 and class 1 overlaid),
                       Karla mean (x) vs kpnn2 mean (y), whiskers = +-1 SD.
`tf_rank_agreement.png` per condition, TFs ordered by Karla's mean score, with
                       both runs' means -- shows whether the ranking survives.

Reads the table written by compare_karla_kpnn2.py.
"""
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

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

CONDITIONS = ["IFNB", "IFNG", "INS", "TGFB1", "TNFA"]
# Standard numbering: layer 1 is the first hidden layer, i.e. the TF nodes.
TF_LAYER = 1
CLASS_STYLE = {
    0: {"color": "#4C78A8", "label": "class 0"},
    1: {"color": "#E45756", "label": "class 1"},
}

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
    # Height leaves room for the two-line suptitle and the two-line panel
    # titles; equal aspect makes the axes taller than the data box alone.
    fig, axes = plt.subplots(
        1, len(conds), figsize=(4.2 * len(conds), 6.0), constrained_layout=True
    )
    axes = np.atleast_1d(axes)

    for ax, cond in zip(axes, conds):
        sub = summary[summary["Condition"] == cond]
        lo = float(min(sub.x_mean.min(), sub.y_mean.min()))
        hi = float(max(sub.x_mean.max(), sub.y_mean.max()))
        pad = 0.08 * (hi - lo) if hi > lo else 1.0
        lo, hi = lo - pad, hi + pad

        ax.axhline(0, lw=0.5, color="0.88", zorder=0)
        ax.axvline(0, lw=0.5, color="0.88", zorder=0)
        ax.plot([lo, hi], [lo, hi], ls="--", lw=1.0, color="0.4", zorder=3)

        for cls, style in CLASS_STYLE.items():
            part = sub[sub["Class"] == cls]
            if part.empty:
                continue
            ax.errorbar(
                part.x_mean, part.y_mean,
                xerr=part.x_sd.fillna(0), yerr=part.y_sd.fillna(0),
                fmt="o", ms=5, lw=0.9, alpha=0.75, capsize=2,
                color=style["color"], label=style["label"], zorder=2,
            )

        r = sub.x_mean.corr(sub.y_mean)
        rho = stats.spearmanr(sub.x_mean, sub.y_mean).statistic
        inside = sub.within_spread.mean()
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel(f"{x_col} — mean over seeds")
        ax.set_ylabel(f"{y_col} — mean over seeds")
        ax.set_title(
            f"{cond}  (n={len(sub)} TF·class)\n"
            f"r={r:+.3f}  ρ={rho:+.3f}  within ±1 SD: {inside:.0%}",
            fontsize=9.5,
        )
    axes[0].legend(fontsize=8, loc="upper left", frameon=False)
    fig.suptitle(
        f"TF-level agreement — {x_col} vs {y_col}\n"
        "point = one TF, whiskers = ±1 SD across the 20 seeds",
        fontsize=13,
    )
    out = OUT_DIR / "tf_agreement.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"-> {out}")


def plot_ranks(summary, x_col, y_col, conds):
    fig, axes = plt.subplots(
        len(conds), 1, figsize=(9.5, 2.6 * len(conds)), constrained_layout=True
    )
    axes = np.atleast_1d(axes)

    for ax, cond in zip(axes, conds):
        sub = summary[(summary["Condition"] == cond) & (summary["Class"] == 0)]
        sub = sub.sort_values("x_mean")
        pos = np.arange(len(sub))
        ax.errorbar(pos, sub.x_mean, yerr=sub.x_sd.fillna(0), fmt="o", ms=5,
                    color="#54A24B", capsize=2, lw=0.9, label=x_col)
        ax.errorbar(pos, sub.y_mean, yerr=sub.y_sd.fillna(0), fmt="s", ms=4.5,
                    color="#B279A2", capsize=2, lw=0.9, alpha=0.85, label=y_col)
        ax.axhline(0, lw=0.5, color="0.85")
        ax.set_xticks(pos)
        ax.set_xticklabels(sub.Node, rotation=45, ha="right", fontsize=8)
        rho = stats.spearmanr(sub.x_mean, sub.y_mean).statistic
        ax.set_title(f"{cond} — class 0, TFs ordered by {x_col}   ρ={rho:+.3f}",
                     fontsize=10)
        ax.set_ylabel("score")
        ax.grid(axis="y", ls=":", lw=0.6, color="0.85")
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=8, frameon=False, ncol=2)
    fig.suptitle("Does the TF ranking survive the port?", fontsize=13)
    out = OUT_DIR / "tf_rank_agreement.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"-> {out}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, x_col, y_col = load()
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
