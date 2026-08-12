import os
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KARLA_ROOT = PROJECT_ROOT / "comp/KPNN_Importances_Karla/KPNN_Importances_Karla"

# Which run to compare against, and where to write the comparison CSVs.
# Both can be overridden via environment variables.
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results_t1_2")
OUT_NAME = os.environ.get("COMPARISON_OUT", "results_new_venv")

BEN_ROOT = PROJECT_ROOT / RESULTS_DIR / "kpnn_output/sergio"

CONDITIONS = ["IFNB", "IFNG", "INS", "TGFB1", "TNFA"]
LAYER_FILES = {0: "hidden_layer_1_attributions.csv", 1: "input_layer_feature_attributions.csv"}

# CSV formatting for Excel. A German/Austrian Excel expects ';' as the column
# separator and ',' as the decimal mark; an English-locale Excel expects ',' and
# '.'. Mixing them (';' with '.') splits the columns correctly but makes Excel
# read the numbers as text, so these two belong together.
SEP = ";"
DECIMAL = ","


def latest_run(cond):
    """Newest kpnn_output_* folder for a condition (timestamped names sort chronologically)."""
    base = BEN_ROOT / f"{cond}_o_A_vs_B" / "alter_nothing"
    runs = sorted(p for p in base.glob("kpnn_output_*") if p.is_dir())
    return runs[-1].name if runs else None


def load_long(path, layer):
    df = pd.read_csv(path)
    df = df.assign(seed=df["model_id"].str.split("_").str[0].astype(int)).drop(columns="model_id")
    long = df.melt(id_vars="seed", var_name="node", value_name="value")
    long["layer"] = layer
    return long


def main():
    out_dir = PROJECT_ROOT / "comp" / OUT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Comparing Karla vs {RESULTS_DIR} -> {out_dir}\n")

    all_conditions = []

    for cond in CONDITIONS:
        run = latest_run(cond)
        if run is None:
            print(f"[{cond}] no kpnn_output_* run found under {BEN_ROOT} — skipping")
            continue
        karla_cond_dir = KARLA_ROOT / f"{cond}_A_vs_B"
        if not (karla_cond_dir / "class_0").exists():
            karla_cond_dir = karla_cond_dir / "aggregated_scores"
        ben_cond_dir = BEN_ROOT / f"{cond}_o_A_vs_B" / "alter_nothing" / run / "importance_scores" / "aggregated_scores"

        rows = []
        for cls in (0, 1):
            for layer, fname in LAYER_FILES.items():
                k_path = karla_cond_dir / f"class_{cls}" / fname
                b_path = ben_cond_dir / f"class_{cls}" / fname
                if not k_path.exists() or not b_path.exists():
                    print(f"SKIP missing: {k_path if not k_path.exists() else b_path}")
                    continue
                k = load_long(k_path, layer)
                b = load_long(b_path, layer)
                merged = k.merge(b, on=["seed", "node", "layer"],
                                 suffixes=("_karla", "_benjamin"), validate="1:1")
                if len(merged) != len(k) or len(merged) != len(b):
                    raise ValueError(
                        f"{cond} class_{cls} {fname}: merge dropped rows "
                        f"(karla={len(k)}, benjamin={len(b)}, merged={len(merged)}) — "
                        f"seed/node sets differ between the two runs"
                    )
                merged["class"] = cls
                rows.append(merged)

        if not rows:
            print(f"[{cond}] no comparable files found — skipping (checked {karla_cond_dir})")
            continue

        out = pd.concat(rows, ignore_index=True)
        out["diff"] = (out["value_karla"] - out["value_benjamin"]).abs()
        out["condition"] = cond
        out = out.rename(columns={
            "condition": "Condition",
            "seed": "Seed", "class": "Class", "layer": "Layer", "node": "Node",
            "value_karla": "Karla KPNN", "value_benjamin": "Benjamin KPNN",
        })
        out = out[["Condition", "Seed", "Class", "Layer", "Node",
                   "Karla KPNN", "Benjamin KPNN", "diff"]]
        out = out.sort_values(["Seed", "Class", "Layer", "Node"]).reset_index(drop=True)
        all_conditions.append(out)

        print(f"[{cond}] {len(out):,} rows  "
              f"Max diff: {out['diff'].max():.6g}  Mean diff: {out['diff'].mean():.6g}  "
              f"Rows with diff > 1e-4: {(out['diff'] > 1e-4).sum():,}")

    if not all_conditions:
        raise SystemExit(f"nothing comparable found under {BEN_ROOT}")

    # --- single combined table, Condition column demarks the source ---
    combined = pd.concat(all_conditions, ignore_index=True)
    combined = combined.sort_values(
        ["Condition", "Seed", "Class", "Layer", "Node"]).reset_index(drop=True)

    out_path = out_dir / "all_conditions_comparison.csv"
    combined.to_csv(out_path, index=False, sep=SEP, decimal=DECIMAL)

    print(f"\nWrote {len(combined):,} rows to {out_path}")
    print(f"  separator='{SEP}'  decimal='{DECIMAL}'  conditions: "
          f"{', '.join(combined['Condition'].unique())}")


if __name__ == "__main__":
    main()
