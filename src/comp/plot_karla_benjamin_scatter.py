"""Plots comparing Karla's and Benjamin's KPNN attributions.

Input
-----
The combined table written by compare_karla_benjamin.py
(`all_conditions_comparison.csv`, ';'-separated, ',' decimal, with a Condition
column). If that file is absent the older per-condition `<COND>_comparison.csv`
files (plain comma CSVs) are read instead, so older result folders still work.

Figures
-------
`<COND>_scatter.png`           Karla (x) vs Benjamin (y), one panel per layer.
`all_conditions_scatter.png`   the same, layers as rows, conditions as columns.
`diff_boxplots.png`            |Karla - Benjamin| per condition, box = IQR,
                               whiskers = 1.5x IQR, points = values outside them.

Scatter panels are split by layer because the two differ by orders of magnitude:
layer 1 (hidden / TF nodes) is a few dozen nodes with attributions up to ~20,
layer 0 (input / gene features) is thousands of nodes packed near zero.
Overlaying them hides layer 0 completely.

Paths
-----
Everything lives under comp/<COMP_SUBDIR>/, where COMP_SUBDIR splits the work by
experiment generation: t1 is the kpnn v1 era, t2 (the default) the kpnn2 one.
COMPARISON_IN names the table folder inside it and PLOT_OUT the figure folder --
change the two together, or the output folder name will not match the data it
holds. Both accept nested names, e.g. PLOT_OUT=plots/plots_foo. Set
COMP_SUBDIR="" to read and write flat in comp/ as before.
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN_NAME = os.environ.get("COMPARISON_IN", "karla_new_vs_kpnn2_native")
OUT_NAME = os.environ.get("PLOT_OUT", "plots/plots_karla_new_vs_kpnn2_native")

# comp/ is split by experiment generation: t1 holds the kpnn v1 era material,
# t2 the kpnn2 one. Set COMP_SUBDIR="" to read/write flat in comp/ as before.
COMP_SUBDIR = os.environ.get("COMP_SUBDIR", "t2")
COMP_ROOT = PROJECT_ROOT / "comp" / COMP_SUBDIR if COMP_SUBDIR else PROJECT_ROOT / "comp"

IN_DIR = COMP_ROOT / IN_NAME
OUT_DIR = COMP_ROOT / OUT_NAME

# must match compare_karla_benjamin.py
COMBINED_NAME = "all_conditions_comparison.csv"
SEP = ";"
DECIMAL = ","

CONDITIONS = ["IFNB", "IFNG", "INS", "TGFB1", "TNFA"]

# Everything else in the table is metadata; the two remaining columns are the
# pair being compared. Reading them from the file lets the same script plot
# Karla-vs-run tables and run-vs-run tables (compare_runs.py).
META_COLS = {"Condition", "Seed", "Class", "Layer", "Node", "diff"}

# Standard numbering: layer 0 is the input layer, layer 1 the first hidden
# layer. Comparison tables written before this change use the opposite
# convention and would be mislabelled here -- regenerate them first.
LAYERS = {
    0: {"name": "layer 0 — input / gene features", "color": "#4C78A8", "s": 3, "alpha": 0.20},
    1: {"name": "layer 1 — hidden / TF nodes", "color": "#E45756", "s": 16, "alpha": 0.55},
}
BOX_COLORS = {"IFNB": "#4C78A8", "IFNG": "#F58518", "INS": "#54A24B",
              "TGFB1": "#B279A2", "TNFA": "#E45756"}

# |diff| spans ~1e-14 to ~25 and contains exact zeros, so a plain log axis
# cannot draw all of it. symlog is linear below LINTHRESH (where the zeros sit)
# and logarithmic above it.
LINTHRESH = 1e-12


def load_comparison() -> pd.DataFrame:
    """Read the combined table, falling back to per-condition files."""
    combined = IN_DIR / COMBINED_NAME
    if combined.exists():
        df = pd.read_csv(combined, sep=SEP, decimal=DECIMAL)
        if "Condition" not in df.columns:
            raise SystemExit(f"{combined} has no 'Condition' column")
        print(f"Reading {combined.name} ({len(df):,} rows)")
        return df

    frames = []
    for cond in CONDITIONS:
        path = IN_DIR / f"{cond}_comparison.csv"
        if not path.exists():
            print(f"[{cond}] missing {path.name} — skipping")
            continue
        part = pd.read_csv(path)
        part["Condition"] = cond
        frames.append(part)
    if not frames:
        raise SystemExit(f"no {COMBINED_NAME} and no *_comparison.csv in {IN_DIR}")
    print(f"Reading per-condition files ({len(frames)} conditions)")
    return pd.concat(frames, ignore_index=True)


def value_columns(df):
    """The two compared columns, in file order."""
    cols = [c for c in df.columns if c not in META_COLS]
    if len(cols) != 2:
        raise SystemExit(f"expected exactly 2 value columns, found {cols}")
    return cols


def draw_scatter(ax, df, layer, title, x_col, y_col):
    """One scatter panel: first value column on x, second on y, y=x reference."""
    st = LAYERS[layer]
    x, y = df[x_col].to_numpy(), df[y_col].to_numpy()

    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    lo, hi = lo - pad, hi + pad

    ax.axhline(0, lw=0.5, color="0.88", zorder=0)
    ax.axvline(0, lw=0.5, color="0.88", zorder=0)
    # y = x means the two runs agree exactly
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.0, color="0.4", zorder=3)

    ax.scatter(x, y, s=st["s"], alpha=st["alpha"], color=st["color"],
               linewidths=0, zorder=2, rasterized=True)

    r = np.corrcoef(x, y)[0, 1]
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{title}\nr={r:.4f}  max|diff|={df['diff'].max():.3g}", fontsize=9.5)
    return r


def make_boxplots(df, conds, x_col, y_col):
    """|a - b| per condition: IQR box, 1.5x IQR whiskers, outlier points."""
    data = [df.loc[df["Condition"] == c, "diff"].to_numpy() for c in conds]
    all_zero = all(d.max() == 0 for d in data)

    fig, ax = plt.subplots(figsize=(1.6 * len(conds) + 4.5, 7.0), constrained_layout=True)
    bp = ax.boxplot(
        data, tick_labels=conds, whis=1.5, showfliers=True, patch_artist=True,
        widths=0.55,
        medianprops=dict(color="black", lw=1.8),
        boxprops=dict(lw=1.0, edgecolor="black"),
        whiskerprops=dict(lw=1.0, color="black"),
        capprops=dict(lw=1.0, color="black"),
        flierprops=dict(marker=".", markersize=1.6, markerfacecolor="black",
                        markeredgecolor="none", alpha=0.18),
    )
    for patch, cond in zip(bp["boxes"], conds):
        patch.set_facecolor(BOX_COLORS.get(cond, "#888888"))
        patch.set_alpha(0.55)
    for fl in bp["fliers"]:          # ~44k points across all boxes
        fl.set_rasterized(True)

    ax.set_yscale("symlog", linthresh=LINTHRESH)
    if all_zero:                       # identical runs: symlog can't autoscale a flat 0
        ax.set_ylim(-LINTHRESH, 1.0)
    ax.set_ylabel(f"|{x_col} − {y_col}|")
    ax.set_xlabel("Condition")
    subtitle = ("every value identical — the two runs agree exactly"
                if all_zero else
                "box = interquartile range · whiskers = 1.5 × IQR · "
                "points = values beyond the whiskers")
    ax.set_title(f"Disagreement between {x_col} and {y_col}\n{subtitle}", fontsize=11)
    ax.grid(axis="y", ls=":", lw=0.6, color="0.8")
    ax.set_axisbelow(True)

    out = OUT_DIR / "diff_boxplots.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)

    print(f"\n-> {out}")
    for c, d in zip(conds, data):
        q1, med, q3 = np.percentile(d, [25, 50, 75])
        iqr = q3 - q1
        n_out = int(((d > q3 + 1.5 * iqr) | (d < max(q1 - 1.5 * iqr, 0))).sum())
        print(f"   {c:6s} median={med:.3g}  IQR=[{q1:.3g}, {q3:.3g}]  "
              f"beyond whiskers={n_out:,} ({n_out / len(d):.1%})  max={d.max():.3g}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Input  {IN_DIR}\nOutput {OUT_DIR}\n")

    df = load_comparison()
    x_col, y_col = value_columns(df)
    print(f"comparing '{x_col}' (x) vs '{y_col}' (y)")
    conds = [c for c in CONDITIONS if (df["Condition"] == c).any()]
    frames = {c: df[df["Condition"] == c] for c in conds}

    # --- one scatter figure per condition: the two layers side by side ---
    for cond, sub in frames.items():
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.0), constrained_layout=True)
        stats = []
        for ax, layer in zip(axes, (0, 1)):
            part = sub[sub["Layer"] == layer]
            if part.empty:
                ax.axis("off")
                continue
            stats.append((layer, draw_scatter(ax, part, layer, LAYERS[layer]["name"], x_col, y_col), len(part)))
        fig.suptitle(f"{cond} — {x_col} vs {y_col}", fontsize=13)
        out = OUT_DIR / f"{cond}_scatter.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        desc = "  ".join(f"L{l}: n={n:,} r={r:.4f}" for l, r, n in stats)
        print(f"[{cond}] {desc}  -> {out.name}")

    # --- overview: layers as rows, conditions as columns ---
    fig, axes = plt.subplots(2, len(conds), figsize=(4.0 * len(conds), 8.6),
                             constrained_layout=True)
    # subplots() drops dimensions when a grid is 1 wide, so force the 2 x ncond shape
    axes = np.asarray(axes).reshape(2, len(conds))
    for col, (cond, sub) in enumerate(frames.items()):
        for row, layer in enumerate((0, 1)):
            ax = axes[row, col]
            part = sub[sub["Layer"] == layer]
            if part.empty:
                ax.axis("off")
                continue
            draw_scatter(ax, part, layer, cond if row == 0 else "", x_col, y_col)
            if row == 1:
                r = np.corrcoef(part[x_col], part[y_col])[0, 1]
                ax.set_title(f"r={r:.4f}", fontsize=9.5)
    for row, layer in enumerate((0, 1)):
        axes[row, 0].set_ylabel(f"{LAYERS[layer]['name']}\n{y_col}", fontsize=10)
    fig.suptitle(f"{x_col} vs {y_col} — {IN_NAME}", fontsize=15)
    out = OUT_DIR / "all_conditions_scatter.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"\n-> {out}")

    # --- boxplots of the disagreement, one box per condition ---
    make_boxplots(df, conds, x_col, y_col)


if __name__ == "__main__":
    main()
