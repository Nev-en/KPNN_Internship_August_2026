"""Compare two of Benjamin's own KPNN runs against each other.

Same table layout as compare_karla_benjamin.py, but both sides are results_*
folders instead of Karla's export. Used to isolate the effect of an environment
change: results_t1_2 was produced on .venv310 with torch 2.4.1, results_t1_3 on
the same venv after aligning every package to Karla's pins (torch 2.4.0 etc.).

Writes one combined CSV with a Condition column, ';'-separated with ',' as the
decimal mark so Excel opens it directly.
"""
import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RUN_A = os.environ.get("RUN_A", "results_kpnn2/results_kpnn2_minimal")
RUN_B = os.environ.get("RUN_B", "results_kpnn2/results_kpnn2_minimal_b")
LABEL_A = os.environ.get("LABEL_A", "t1_2 KPNN")
LABEL_B = os.environ.get("LABEL_B", "t1_3 KPNN")
def _short(run):
    """Last path segment, without the results_ prefix — RUN_A may be nested."""
    return Path(run).name.replace("results_", "")


OUT_NAME = os.environ.get("COMPARISON_OUT", f"{_short(RUN_A)}_vs_{_short(RUN_B)}")

# comp/ is split by experiment generation: t1 holds the kpnn v1 era material,
# t2 the kpnn2 one. Set COMP_SUBDIR="" to write flat into comp/ as before.
COMP_SUBDIR = os.environ.get("COMP_SUBDIR", "t2")
COMP_ROOT = PROJECT_ROOT / "comp" / COMP_SUBDIR if COMP_SUBDIR else PROJECT_ROOT / "comp"

CONDITIONS = ["IFNB", "IFNG", "INS", "TGFB1", "TNFA"]
# Standard numbering: layer 0 is the input layer, layer 1 the first hidden
# layer. Tables written before this change use the opposite convention.
LAYER_FILES = {0: "input_layer_feature_attributions.csv", 1: "hidden_layer_1_attributions.csv"}

# must match compare_karla_benjamin.py / plot_karla_benjamin_scatter.py
SEP = ";"
DECIMAL = ","


def load_long(path, layer):
    df = pd.read_csv(path)
    df = df.assign(seed=df["model_id"].str.split("_").str[0].astype(int)).drop(columns="model_id")
    long = df.melt(id_vars="seed", var_name="node", value_name="value")
    long["layer"] = layer
    return long


def scores_dir(results_dir, cond):
    """Newest aggregated_scores folder for a condition, whichever layout wrote it.

    The kpnn entry points nest their output differently:
      kpnn v1 / kpnn2_port.py
          <root>/kpnn_output/sergio/<COND>_o_A_vs_B/alter_nothing/
          kpnn_output_<ts>/importance_scores/aggregated_scores
      kpnn2_native.py
          <root>/<COND>/run_<ts>/aggregated_scores

    Folder names carry timestamps, so a lexical sort is chronological.
    """
    root = PROJECT_ROOT / results_dir
    bases = [
        root / "kpnn_output/sergio" / f"{cond}_o_A_vs_B",
        root / f"{cond}_o_A_vs_B",
        root / cond,
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


def main():
    out_dir = COMP_ROOT / OUT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Comparing {RUN_A} vs {RUN_B} -> {out_dir}\n")

    all_conditions = []
    for cond in CONDITIONS:
        a_dir, b_dir = scores_dir(RUN_A, cond), scores_dir(RUN_B, cond)
        if a_dir is None or b_dir is None:
            missing = RUN_A if a_dir is None else RUN_B
            print(f"[{cond}] no run found under {missing} — skipping")
            continue

        rows = []
        for cls in (0, 1):
            for layer, fname in LAYER_FILES.items():
                pa, pb = a_dir / f"class_{cls}" / fname, b_dir / f"class_{cls}" / fname
                if not pa.exists() or not pb.exists():
                    print(f"SKIP missing: {pa if not pa.exists() else pb}")
                    continue
                a, b = load_long(pa, layer), load_long(pb, layer)
                merged = a.merge(b, on=["seed", "node", "layer"],
                                 suffixes=("_a", "_b"), validate="1:1")
                if len(merged) != len(a) or len(merged) != len(b):
                    raise ValueError(
                        f"{cond} class_{cls} {fname}: merge dropped rows "
                        f"(a={len(a)}, b={len(b)}, merged={len(merged)}) — "
                        f"seed/node sets differ between the two runs")
                merged["class"] = cls
                rows.append(merged)

        if not rows:
            print(f"[{cond}] no comparable files found — skipping")
            continue

        out = pd.concat(rows, ignore_index=True)
        out["diff"] = (out["value_a"] - out["value_b"]).abs()
        out["condition"] = cond
        out = out.rename(columns={
            "condition": "Condition", "seed": "Seed", "class": "Class",
            "layer": "Layer", "node": "Node",
            "value_a": LABEL_A, "value_b": LABEL_B,
        })
        out = out[["Condition", "Seed", "Class", "Layer", "Node", LABEL_A, LABEL_B, "diff"]]
        out = out.sort_values(["Seed", "Class", "Layer", "Node"]).reset_index(drop=True)
        all_conditions.append(out)

        identical = int((out["diff"] == 0).sum())
        print(f"[{cond}] {len(out):,} rows  Max diff: {out['diff'].max():.6g}  "
              f"Mean diff: {out['diff'].mean():.6g}  "
              f"Exactly identical: {identical:,} ({identical / len(out):.1%})")

    if not all_conditions:
        raise SystemExit("nothing comparable found")

    combined = pd.concat(all_conditions, ignore_index=True)
    combined = combined.sort_values(
        ["Condition", "Seed", "Class", "Layer", "Node"]).reset_index(drop=True)

    out_path = out_dir / "all_conditions_comparison.csv"
    combined.to_csv(out_path, index=False, sep=SEP, decimal=DECIMAL)

    n_same = int((combined["diff"] == 0).sum())
    print(f"\nWrote {len(combined):,} rows to {out_path}")
    print(f"  separator='{SEP}'  decimal='{DECIMAL}'")
    print(f"  overall max |diff| = {combined['diff'].max():.6g}")
    print(f"  rows with diff exactly 0: {n_same:,} / {len(combined):,} "
          f"({n_same / len(combined):.2%})")


if __name__ == "__main__":
    main()
