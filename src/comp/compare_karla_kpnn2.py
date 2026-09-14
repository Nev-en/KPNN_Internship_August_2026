"""Compare Karla's kpnn v1 attributions against a kpnn2 run.

Same job as compare_karla_benjamin.py, but for the kpnn2 port and with the
reference export read from its real location. Output is deliberately written in
the format plot_karla_benjamin_scatter.py already expects -- Condition, Seed,
Class, Layer, Node, two value columns, diff -- so the existing plotting script
runs over it unchanged:

    COMPARISON_IN=karla_new_vs_kpnn2 PLOT_OUT=plots/plots_karla_new_vs_kpnn2 \
        python plot_karla_benjamin_scatter.py

Both land under comp/<COMP_SUBDIR>/, which defaults to t2 -- comp/ is split by
experiment generation (t1 = kpnn v1 era, t2 = kpnn2). Set COMP_SUBDIR="" to
write flat into comp/.

What "the same seed" means here
-------------------------------
Rows are matched on seed. For kpnn2_port.py with rng="v1" (the default) seed N
really is Karla's seed N: the port consumes random numbers exactly as kpnn v1
does and matches her export to r = 1.000 per seed, max|diff| around 3e-5.
Every other kpnn2 script (native, minimal, port2, or the port with
rng="torch") draws differently, so there seed N is just another random run and
the scatter should be read as a distribution comparison, not as a diff.
"""
import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Karla's kpnn v1 reference run. Defaults to the in-repo karla_new run; set
# KARLA_ROOT to an absolute path to compare against a different export, e.g. the
# KPNN_Importances_Karla delivery. Either layout is found by karla_scores_dir.
KARLA_ROOT = Path(
    os.environ.get("KARLA_ROOT", PROJECT_ROOT / "results_kpnn/KPNN_results_karla_new")
)

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results_kpnn2/results_kpnn2_native")
OUT_NAME = os.environ.get("COMPARISON_OUT", "karla_new_vs_kpnn2")

# comp/ is split by experiment generation: t1 holds the kpnn v1 era material,
# t2 the kpnn2 one. COMPARISON_OUT names the folder inside that; set
# COMP_SUBDIR="" to write flat into comp/ as before.
COMP_SUBDIR = os.environ.get("COMP_SUBDIR", "t2")
COMP_ROOT = PROJECT_ROOT / "comp" / COMP_SUBDIR if COMP_SUBDIR else PROJECT_ROOT / "comp"

KPNN2_ROOT = PROJECT_ROOT / RESULTS_DIR

CONDITIONS = ["IFNB", "IFNG", "INS", "TGFB1", "TNFA"]
# Standard numbering: layer 0 is the input layer, layer 1 the first hidden
# layer. The older comparison tables in comp/ use the opposite convention, so a
# table written before this change cannot be plotted with the current scripts.
LAYER_FILES = {0: "input_layer_feature_attributions.csv", 1: "hidden_layer_1_attributions.csv"}

# CSV formatting for Excel, matching compare_karla_benjamin.py: a German or
# Austrian Excel expects ';' as the column separator and ',' as the decimal mark.
SEP = ";"
DECIMAL = ","

KARLA_COL = "Karla KPNN"
# Names the second column, and with it the plot axes. Set COMPARISON_LABEL to
# tell two kpnn2 variants apart in the figures.
KPNN2_COL = os.environ.get("COMPARISON_LABEL", "kpnn2")


def kpnn2_scores_dir(cond):
    """Newest aggregated_scores folder for a condition, whichever layout wrote it.

    The two kpnn2 entry points nest their output differently:
      kpnn2_port.py  <root>/kpnn_output/sergio/<COND>_o_A_vs_B/alter_nothing/
                         kpnn_output_<ts>/importance_scores/aggregated_scores
      kpnn2_native.py    <root>/<COND>/run_<ts>/aggregated_scores

    Both put timestamps in the folder name, so a lexical sort is chronological.
    """
    bases = [
        KPNN2_ROOT / "kpnn_output/sergio" / f"{cond}_o_A_vs_B",
        KPNN2_ROOT / f"{cond}_o_A_vs_B",
        KPNN2_ROOT / cond,
    ]
    probe = Path("class_0") / LAYER_FILES[0]
    runs = [
        path
        for base in bases
        if base.is_dir()
        for path in base.glob("**/aggregated_scores")
        if (path / probe).exists()
    ]
    return sorted(runs)[-1] if runs else None


def karla_scores_dir(cond):
    """Locate Karla's aggregated scores, whichever layout this export uses.

    Seen so far:
      <COND>_A_vs_B/class_0/...
      <COND>_A_vs_B/aggregated_scores/class_0/...
      <COND>_A_vs_B/alter_nothing/<run>/importance_scores/aggregated_scores/class_0/...
    """
    base = KARLA_ROOT / f"{cond}_A_vs_B"
    if not base.is_dir():
        return None
    probe = Path("class_0") / LAYER_FILES[0]
    for cand in (base, base / "aggregated_scores"):
        if (cand / probe).exists():
            return cand
    nested = sorted(p for p in base.glob("**/aggregated_scores") if (p / probe).exists())
    return nested[-1] if nested else None


def load_long(path, layer):
    df = pd.read_csv(path)
    df = df.assign(seed=df["model_id"].str.split("_").str[0].astype(int)).drop(columns="model_id")
    long = df.melt(id_vars="seed", var_name="node", value_name="value")
    long["layer"] = layer
    return long


def main():
    out_dir = COMP_ROOT / OUT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Karla  {KARLA_ROOT}")
    print(f"kpnn2  {KPNN2_ROOT}")
    print(f"Output {out_dir}\n")

    all_conditions = []

    for cond in CONDITIONS:
        kpnn2_cond_dir = kpnn2_scores_dir(cond)
        if kpnn2_cond_dir is None:
            print(f"[{cond}] no aggregated_scores found under {KPNN2_ROOT} — skipping")
            continue
        karla_cond_dir = karla_scores_dir(cond)
        if karla_cond_dir is None:
            print(f"[{cond}] no Karla scores under {KARLA_ROOT / f'{cond}_A_vs_B'} — skipping")
            continue

        rows = []
        for cls in (0, 1):
            for layer, fname in LAYER_FILES.items():
                k_path = karla_cond_dir / f"class_{cls}" / fname
                n_path = kpnn2_cond_dir / f"class_{cls}" / fname
                if not k_path.exists() or not n_path.exists():
                    print(f"SKIP missing: {k_path if not k_path.exists() else n_path}")
                    continue
                k = load_long(k_path, layer)
                n = load_long(n_path, layer)
                merged = k.merge(
                    n, on=["seed", "node", "layer"],
                    suffixes=("_karla", "_kpnn2"), validate="1:1",
                )
                if len(merged) != len(k) or len(merged) != len(n):
                    raise ValueError(
                        f"{cond} class_{cls} {fname}: merge dropped rows "
                        f"(karla={len(k)}, kpnn2={len(n)}, merged={len(merged)}) — "
                        f"seed/node sets differ between the two runs"
                    )
                merged["class"] = cls
                rows.append(merged)

        if not rows:
            print(f"[{cond}] no comparable files found — skipping (checked {karla_cond_dir})")
            continue

        out = pd.concat(rows, ignore_index=True)
        out["diff"] = (out["value_karla"] - out["value_kpnn2"]).abs()
        out["condition"] = cond
        out = out.rename(columns={
            "condition": "Condition",
            "seed": "Seed", "class": "Class", "layer": "Layer", "node": "Node",
            "value_karla": KARLA_COL, "value_kpnn2": KPNN2_COL,
        })
        out = out[["Condition", "Seed", "Class", "Layer", "Node", KARLA_COL, KPNN2_COL, "diff"]]
        out = out.sort_values(["Seed", "Class", "Layer", "Node"]).reset_index(drop=True)
        all_conditions.append(out)

        # Correlation per layer says more than the raw diff here, because the two
        # runs are not expected to agree value-by-value.
        stats = []
        for layer in (0, 1):
            part = out[out["Layer"] == layer]
            if part.empty:
                continue
            stats.append(f"L{layer} r={part[KARLA_COL].corr(part[KPNN2_COL]):+.3f}")
        print(f"[{cond}] {len(out):,} rows  {'  '.join(stats)}  "
              f"mean|diff|={out['diff'].mean():.4g}")

    if not all_conditions:
        raise SystemExit(f"nothing comparable found under {KPNN2_ROOT}")

    combined = pd.concat(all_conditions, ignore_index=True)
    combined = combined.sort_values(
        ["Condition", "Seed", "Class", "Layer", "Node"]).reset_index(drop=True)

    out_path = out_dir / "all_conditions_comparison.csv"
    combined.to_csv(out_path, index=False, sep=SEP, decimal=DECIMAL)

    print(f"\nWrote {len(combined):,} rows to {out_path}")
    print(f"  separator='{SEP}'  decimal='{DECIMAL}'  conditions: "
          f"{', '.join(combined['Condition'].unique())}")
    print(f"\nNext: COMPARISON_IN={OUT_NAME} PLOT_OUT=plots/plots_{OUT_NAME} "
          f"python src/comp_t1/plot_karla_benjamin_scatter.py")


if __name__ == "__main__":
    main()
