"""
SERGIO ground-truth benchmark.

This script does NOT generate SERGIO KO datasets.
It assumes the following files already exist per pathway:
  - X_nt.csv
  - y_nt.csv
  - X_ko.csv
  - y_ko.csv
  - basals.csv
  - activity_delta_vs_abs_deltabasal.csv

Main purpose:
  Compare SERGIO perturbation effect scores against:
  1. KPNN hidden node importance
  2. Decoupler activity
  3. baseline feature AUC
  4. random shuffled baseline
"""

#!/usr/bin/env python3
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon, ttest_rel
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


# ====================== ROOTS ===============================
PROJECT_ROOT = Path(__file__).resolve().parents[3]

SERGIO_ROOT = PROJECT_ROOT / "results_t1" / "KO_simulation" / "sergio"
KPNN_ROOT   = PROJECT_ROOT / "results_t1" / "kpnn_output" / "sergio"

# Pathway TF-only edgelists (flat layout: EDGELIST_DIR/{pathway}_tf_only_edgelist_filtered.tsv)
EDGELIST_DIR = PROJECT_ROOT / "src" / "Karla_kpnn_csv"

# KPNN run/output settings
RUN_TAG = "alter_nothing"
HIDDEN_LAYER_FILE = "hidden_layer_1_attributions.csv"

# Pathways and random-baseline settings
PATHWAYS = ["IFNB", "IFNG", "INS", "TGFB1", "TNFA"]
N_RANDOM_SHUFFLES = 1000
RANDOM_SEED = 42

# Optional manual class-name overrides for plot labels.
# Use this if y_nt.csv only contains A/B or 0/1 and not the real cell-line names.
# Format:
#   "PATHWAY": {"A": "actual cell line for SERGIO y=0", "B": "actual cell line for SERGIO y=1"}
# Example:
#   "TGFB1": {"A": "CVCL206", "B": "CVCL0023"}
CLASS_NAME_OVERRIDES = {
    # "IFNB":  {"A": "CVCL...", "B": "CVCL..."},
    # "IFNG":  {"A": "CVCL...", "B": "CVCL..."},
    # "INS":   {"A": "CVCL...", "B": "CVCL..."},
    # "TGFB1": {"A": "CVCL206", "B": "CVCL..."},
    # "TNFA":  {"A": "CVCL...", "B": "CVCL..."},
}

# =============================================================================
# General helpers
# =============================================================================
def find_latest_kpnn_out(run_dir: Path) -> Path:
    outs = sorted(run_dir.glob("kpnn_output_*"))
    if not outs:
        raise FileNotFoundError(f"No kpnn_output_* under: {run_dir}")
    return outs[-1]


def sep_centroid(A: np.ndarray, B: np.ndarray) -> float:
    if A.size == 0 or B.size == 0:
        return np.nan
    return float(np.linalg.norm(A.mean(axis=0) - B.mean(axis=0)))


def get_sergio_class_names(run_dir: Path, pathway: str | None = None) -> tuple[str, str]:
    """
    Reads class names from y_nt.csv as reliably as possible.

    Returns:
      class_A_name for SERGIO y=0
      class_B_name for SERGIO y=1

    Because the KPNN labels were swapped during training:
      SERGIO A / y=0 = KPNN class 1
      SERGIO B / y=1 = KPNN class 0
    """
    if pathway is not None and pathway in CLASS_NAME_OVERRIDES:
        override = CLASS_NAME_OVERRIDES[pathway]
        class_A_name = override.get("A", "SERGIO A / y=0")
        class_B_name = override.get("B", "SERGIO B / y=1")
        return class_A_name, class_B_name

    y_nt_path = run_dir / "y_nt.csv"
    y_nt = pd.read_csv(y_nt_path)

    candidate_cols = [
        "cell_type",
        "cell_line",
        "class_name",
        "cell_type_name",
        "celltype",
        "celltype_name",
    ]

    name_col = None
    for c in candidate_cols:
        if c in y_nt.columns:
            name_col = c
            break

    if name_col is None:
        return "SERGIO A / y=0", "SERGIO B / y=1"

    class_A = y_nt.loc[y_nt["y"] == 0, name_col].dropna().astype(str).unique()
    class_B = y_nt.loc[y_nt["y"] == 1, name_col].dropna().astype(str).unique()

    class_A_name = class_A[0] if len(class_A) > 0 else "SERGIO A / y=0"
    class_B_name = class_B[0] if len(class_B) > 0 else "SERGIO B / y=1"

    return class_A_name, class_B_name


def compute_effect_scores(run_dir: Path) -> pd.DataFrame:
    """
    EFFECT1.0:
        distance(A_U, B_U) - distance(A_P, B_P)
        positive = KO makes A and B more similar
        negative = KO makes A and B more different

    EFFECT2.0:
        Magnitude-based perturbation response difference:
        mean_g( |A_P_g - A_U_g| - |B_P_g - B_U_g| )
        positive = A/KPNN-class-1 responds more strongly
        negative = B/KPNN-class-0 responds more strongly

    EFFECT2.1:
        Mean absolute response difference:
        mean_g(|A_P_g - A_U_g|) - mean_g(|B_P_g - B_U_g|)
        positive = A/KPNN-class-1 responds more strongly
        negative = B/KPNN-class-0 responds more strongly

    EFFECT3.0:
        Signed directional movement dominance:
        (A movement toward B baseline) - (B movement toward A baseline)
        positive = A/KPNN-class-1 movement dominates
        negative = B/KPNN-class-0 movement dominates

    EFFECT3.1:
        Direction-agnostic class-specific movement score:
        abs(A movement relative to B baseline) - abs(B movement relative to A baseline)
        positive = A/KPNN-class-1 is more affected by KO
        negative = B/KPNN-class-0 is more affected by KO
    """
    X_nt = pd.read_csv(run_dir / "X_nt.csv")
    y_nt = pd.read_csv(run_dir / "y_nt.csv")
    X_ko = pd.read_csv(run_dir / "X_ko.csv")
    y_ko = pd.read_csv(run_dir / "y_ko.csv")

    X_nt.columns = X_nt.columns.str.replace("_leaf$", "", regex=True)
    X_ko.columns = X_ko.columns.str.replace("_leaf$", "", regex=True)

    common_genes = [g for g in X_nt.columns if g in X_ko.columns]
    X_nt, X_ko = X_nt[common_genes], X_ko[common_genes]

    maskA_nt = y_nt["y"].to_numpy() == 0
    maskB_nt = y_nt["y"].to_numpy() == 1

    A_U = X_nt.loc[maskA_nt]
    B_U = X_nt.loc[maskB_nt]

    d_base = sep_centroid(A_U.values, B_U.values)

    tf_col = "tf" if "tf" in y_ko.columns else "gene"
    tf_list = sorted(set(y_ko[tf_col].astype(str).str.strip()))

    rows = []

    for tf in tf_list:
        subset = y_ko[tf_col].astype(str).str.strip() == tf
        yk = y_ko.loc[subset, "y"].to_numpy()

        A_P = X_ko.loc[subset].loc[yk == 0]
        B_P = X_ko.loc[subset].loc[yk == 1]

        if A_P.empty or B_P.empty:
            warnings.warn(f"Missing class for KO {tf}, skipping.")
            continue

        # ---------- EFFECT1.0 ----------
        d_ko = sep_centroid(A_P.values, B_P.values)
        effect1_0 = d_base - d_ko

        effect1_0_direction = (
            "more_similar" if effect1_0 > 0
            else "more_different" if effect1_0 < 0
            else "no_change"
        )

        # ---------- EFFECT2.0 ----------
        A_response_abs = (A_P.mean(axis=0) - A_U.mean(axis=0)).abs()
        B_response_abs = (B_P.mean(axis=0) - B_U.mean(axis=0)).abs()
        effect2_0 = (A_response_abs - B_response_abs).mean()

        # ---------- EFFECT2.1 ----------
        A_mean_distance = (A_P.mean(axis=0) - A_U.mean(axis=0)).abs().mean()
        B_mean_distance = (B_P.mean(axis=0) - B_U.mean(axis=0)).abs().mean()
        effect2_1 = A_mean_distance - B_mean_distance

        # ---------- EFFECT3.0 ----------
        effect_A_toward_B = d_base - sep_centroid(A_P.values, B_U.values)
        effect_B_toward_A = d_base - sep_centroid(B_P.values, A_U.values)
        effect3_0 = effect_A_toward_B - effect_B_toward_A

        # ---------- EFFECT3.1 ----------
        effect3_1_classA = abs(d_base - sep_centroid(A_P.values, B_U.values))
        effect3_1_classB = abs(d_base - sep_centroid(B_P.values, A_U.values))
        effect3_1 = effect3_1_classA - effect3_1_classB

        # ---------- EFFECT3.2 ----------
        shift_A = sep_centroid(A_P.values, A_U.values)
        shift_B = sep_centroid(B_P.values, B_U.values)

        effect3_2_classA = effect3_1_classA + shift_A
        effect3_2_classB = effect3_1_classB + shift_B
        effect3_2 = effect3_2_classA - effect3_2_classB

        # ---------- EFFECT3.3 ----------
        effect3_3 = effect3_2_classA + effect3_2_classB

        rows.append({
            "tf": tf,

            # raw EFFECT1.0 distances for checking and interpretation
            "d_base": d_base,
            "d_ko": d_ko,
            "effect1_0_direction": effect1_0_direction,

            "effect1_0": effect1_0,
            "effect2_0": effect2_0,
            "effect2_1": effect2_1,
            "effect3_0": effect3_0,
            "effect3_1": effect3_1,
            "effect3_2": effect3_2,
            "effect3_3": effect3_3,

            "delta_sep_euclidean": effect1_0,
        })

    return pd.DataFrame(rows)

def load_average_importance_scores(class1_csv: Path, class0_csv: Path) -> pd.DataFrame:
    """
    Seedwise abs(class1) - abs(class0), then mean across seeds.

    Because labels were swapped during KPNN training:
      KPNN class_1 = SERGIO A / y=0
      KPNN class_0 = SERGIO B / y=1

    Therefore:
      positive HNI_avg = more important for SERGIO A / KPNN class 1
      negative HNI_avg = more important for SERGIO B / KPNN class 0
    """
    df1 = pd.read_csv(class1_csv).drop(columns="model_id", errors="ignore")
    df0 = pd.read_csv(class0_csv).drop(columns="model_id", errors="ignore")

    df1 = df1.apply(pd.to_numeric, errors="coerce")
    df0 = df0.apply(pd.to_numeric, errors="coerce")

    common_cols = sorted(set(df1.columns) & set(df0.columns))
    if not common_cols:
        raise ValueError("No overlapping TF columns between class_1 and class_0.")

    df1 = df1[common_cols]
    df0 = df0[common_cols]

    df1_abs = df1.abs()
    df0_abs = df0.abs()

    n_seeds = min(len(df1_abs), len(df0_abs))
    if n_seeds == 0:
        raise ValueError("No seed rows found in one or both DataFrames.")

    df1_abs = df1_abs.iloc[:n_seeds].reset_index(drop=True)
    df0_abs = df0_abs.iloc[:n_seeds].reset_index(drop=True)

    s = (df1_abs - df0_abs).mean(axis=0)

    out = (
        s.reset_index()
         .rename(columns={"index": "tf_real", 0: "HNI_avg"})
         .sort_values("tf_real")
         .reset_index(drop=True)
    )
    return out


def find_edgelist_for_pathway(pathway: str) -> Path:
    candidates = [
        EDGELIST_DIR / f"{pathway}_tf_only_edgelist_filtered.tsv",
        EDGELIST_DIR / f"{pathway}_tf_only_edgelist.tsv",
        EDGELIST_DIR / pathway / f"{pathway}_tf_only_edgelist_filtered.tsv",
        EDGELIST_DIR / pathway / f"{pathway}_tf_only_edgelist.tsv",
    ]

    for c in candidates:
        if c.exists():
            return c

    hits = sorted(EDGELIST_DIR.glob(f"{pathway}*tf_only*filtered*.tsv"))
    if hits:
        return hits[0]

    folder = EDGELIST_DIR / pathway
    if folder.exists():
        hits = sorted(folder.glob("*tf_only*filtered*.tsv"))
        if hits:
            return hits[0]

    raise FileNotFoundError(
        f"Could not find edgelist for pathway={pathway} under {EDGELIST_DIR}. "
        f"Tried: {[str(x) for x in candidates]}"
    )


def build_real_to_TFk_mapping(edgelist_tsv: Path, basals_csv: Path) -> dict:
    """
    Uses edgelist parent order -> TF1..TFN, where N comes from basals.csv.
    """
    edges = pd.read_csv(edgelist_tsv, sep="\t", dtype=str)
    if "parent" not in edges.columns:
        raise ValueError(f"Edgelist missing 'parent' column: {edgelist_tsv} (cols={edges.columns.tolist()})")

    parents = sorted(edges["parent"].astype(str).str.strip().unique())

    basals = pd.read_csv(basals_csv)
    if "tf" not in basals.columns:
        raise ValueError(f"basals.csv missing 'tf' column: {basals_csv} (cols={basals.columns.tolist()})")

    N = len(basals["tf"])
    return {old: f"TF{i+1}" for i, old in enumerate(parents[:N])}


def signed_unsigned_transform(df: pd.DataFrame, x: str, y: str, mode: str) -> pd.DataFrame:
    """
    mode="basal": keep sign for both x and y
    mode="effect": currently also keeps sign for directional interpretation
    """
    out = df.copy()
    if mode == "basal":
        return out
    if mode == "effect":
        # Keep signed values. Do not abs-transform unless you intentionally want magnitude-only comparisons.
        # out[x] = out[x].abs()
        # out[y] = out[y].abs()
        return out
    raise ValueError(f"Unknown mode: {mode}")


# =============================================================================
# Data loaders and mapping helpers
# =============================================================================
def load_sergio_metrics(pathway: str) -> pd.DataFrame:
    run_dir = SERGIO_ROOT / f"sergio_{pathway}"  # run_dir = SERGIO_ROOT / f"sergio_{pathway}_loop"

    basals_path = run_dir / "basals.csv"
    if not basals_path.exists():
        raise FileNotFoundError(f"Missing basals.csv for {pathway}: {basals_path}")

    class_A_name, class_B_name = get_sergio_class_names(run_dir, pathway=pathway)

    basals = pd.read_csv(basals_path)[["tf", "delta_basal"]]
    effect_df = compute_effect_scores(run_dir)[
        [
            "tf",
            "d_base",
            "d_ko",
            "effect1_0_direction",
            "effect1_0",
            "effect2_0",
            "effect2_1",
            "effect3_0",
            "effect3_1",
            "effect3_2",
            "effect3_3",
            "delta_sep_euclidean",
        ]
    ]

    out = basals.merge(effect_df, on="tf", how="inner")
    out["pathway"] = pathway
    out["class_A_name"] = class_A_name
    out["class_B_name"] = class_B_name
    return out


def load_decoupler_activity(pathway: str) -> pd.DataFrame:
    run_dir = SERGIO_ROOT / f"sergio_{pathway}"  # run_dir = SERGIO_ROOT / f"sergio_{pathway}_loop"
    act_path = run_dir / "activity_delta_vs_abs_deltabasal.csv"
    if not act_path.exists():
        raise FileNotFoundError(f"Missing activity file for {pathway}: {act_path}")
    return pd.read_csv(act_path)[["tf", "delta_activity"]]


def compute_feature_auc_scores(run_dir: Path) -> pd.DataFrame:
    """
    Baseline feature score from non-targeting data:
    AUC per feature for distinguishing class 0 vs class 1 in X_nt / y_nt.

    Returns columns:
      tf, baseline_score

    baseline_score is signed as (0.5 - AUC), so:
      > 0  => higher in class 0
      < 0  => higher in class 1
    """
    X_nt = pd.read_csv(run_dir / "X_nt.csv")
    y_nt = pd.read_csv(run_dir / "y_nt.csv")

    X_nt.columns = X_nt.columns.str.replace("_leaf$", "", regex=True)

    if "y" not in y_nt.columns:
        raise ValueError(f"y_nt.csv missing 'y' column: {run_dir / 'y_nt.csv'}")

    y = y_nt["y"].to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError(f"Only one class present in y_nt for {run_dir}")

    rows = []
    for col in X_nt.columns:
        vals = pd.to_numeric(X_nt[col], errors="coerce").to_numpy()

        if np.all(~np.isfinite(vals)):
            continue
        if np.nanstd(vals) == 0:
            continue

        mask = np.isfinite(vals)
        if mask.sum() < 2:
            continue
        if len(np.unique(y[mask])) < 2:
            continue

        try:
            auc = roc_auc_score(y[mask], vals[mask])
            rows.append({
                "tf_real": col,
                "baseline_auc": float(auc),
                "baseline_score": float(0.5 - auc),
            })
        except Exception:
            continue

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(f"No usable baseline AUC scores computed for {run_dir}")

    return out


def load_baseline_scores_mapped_to_TFk(pathway: str) -> pd.DataFrame:
    """
    Load baseline feature AUC scores and map them to TF1..TFN if needed.

    Returns columns:
      tf, baseline_score, baseline_auc
    """
    run_dir = SERGIO_ROOT / f"sergio_{pathway}"  # run_dir = SERGIO_ROOT / f"sergio_{pathway}_loop"

    base = compute_feature_auc_scores(run_dir)  # tf_real, baseline_auc, baseline_score
    basals_csv = run_dir / "basals.csv"

    basals = pd.read_csv(basals_csv)
    if "tf" not in basals.columns:
        raise ValueError(f"basals.csv missing 'tf' column: {basals_csv}")

    sergio_tfs = set(basals["tf"].astype(str))

    # Case 1: features already match TF1..TFN
    if set(base["tf_real"].astype(str)) & sergio_tfs:
        base = base.rename(columns={"tf_real": "tf"})
        return base[["tf", "baseline_score", "baseline_auc"]].copy()

    # Case 2: map real TF names -> TF1..TFN using edgelist order
    edgelist_tsv = find_edgelist_for_pathway(pathway)
    mapping = build_real_to_TFk_mapping(edgelist_tsv, basals_csv)

    base["tf"] = base["tf_real"].map(mapping)
    base = base.dropna(subset=["tf"]).copy()

    if base.empty:
        print(f"[WARN] {pathway}: baseline mapping produced 0 TFs.")
        print(f"       Edgelist used: {edgelist_tsv}")

    return base[["tf", "baseline_score", "baseline_auc"]].copy()


def load_random_baseline_scores_mapped_to_TFk(pathway: str, seed: int = 42) -> pd.DataFrame:
    """
    Negative control: take the mapped baseline scores and randomly shuffle
    the baseline_score values across TFs within a pathway.
    """
    base = load_baseline_scores_mapped_to_TFk(pathway).copy()

    if base.empty:
        raise ValueError(f"No baseline scores available to shuffle for pathway {pathway}")

    rng = np.random.default_rng(seed)
    shuffled_scores = base["baseline_score"].to_numpy(copy=True)
    rng.shuffle(shuffled_scores)

    out = base[["tf"]].copy()
    out["random_baseline_score"] = shuffled_scores
    return out


def compute_random_baseline_spearman_summary(
    ser_df: pd.DataFrame,
    base_df: pd.DataFrame,
    x: str,
    mode: str,
    n_shuffles: int = 1000,
    seed: int = 42,
) -> tuple[float, float, np.ndarray]:
    """
    For one pathway:
      - merge SERGIO metrics with mapped baseline scores
      - shuffle baseline_score across TFs n_shuffles times
      - compute Spearman each time against x
      - return mean, std, and the full null distribution
    """
    merged = ser_df.merge(base_df[["tf", "baseline_score"]], on="tf", how="inner").copy()
    merged = merged[["tf", x, "baseline_score"]].dropna()

    if len(merged) < 2:
        return np.nan, np.nan, np.array([])

    rng = np.random.default_rng(seed)
    xvals = merged[x].to_numpy(dtype=float)
    yvals = merged["baseline_score"].to_numpy(dtype=float)

    # Keep signed values here to match signed effect-score comparisons.
    # If you intentionally want magnitude-only nulls, uncomment these lines.
    # if mode == "effect":
    #     xvals = np.abs(xvals)
    #     yvals = np.abs(yvals)

    corrs = []
    for _ in range(n_shuffles):
        shuffled = yvals.copy()
        rng.shuffle(shuffled)
        rho, _ = spearmanr(xvals, shuffled)
        corrs.append(rho)

    corrs = np.array(corrs, dtype=float)
    return float(np.nanmean(corrs)), float(np.nanstd(corrs, ddof=1)), corrs


def load_kpnn_hni_mapped_to_TFk(pathway: str) -> pd.DataFrame:
    """
    Loads KPNN HNI, then maps tf_real -> TF1..TFN so it can merge with SERGIO basals/sep.
    Returns: tf, HNI_avg
    """
    run_dir = KPNN_ROOT / f"{pathway}_A_vs_B" / RUN_TAG
    kpnn_out = find_latest_kpnn_out(run_dir)

    class0 = kpnn_out / "importance_scores/aggregated_scores/class_0" / HIDDEN_LAYER_FILE
    class1 = kpnn_out / "importance_scores/aggregated_scores/class_1" / HIDDEN_LAYER_FILE

    if not class0.exists():
        raise FileNotFoundError(f"Missing class_0 HNI file for {pathway}: {class0}")
    if not class1.exists():
        raise FileNotFoundError(f"Missing class_1 HNI file for {pathway}: {class1}")

    hni_real = load_average_importance_scores(class1, class0)

    sergio_dir = SERGIO_ROOT / f"sergio_{pathway}"  # sergio_dir = SERGIO_ROOT / f"sergio_{pathway}_loop"
    basals_csv = sergio_dir / "basals.csv"
    edgelist_tsv = find_edgelist_for_pathway(pathway)
    mapping = build_real_to_TFk_mapping(edgelist_tsv, basals_csv)

    hni_real["tf"] = hni_real["tf_real"].map(mapping)
    hni = hni_real.dropna(subset=["tf"])[["tf", "HNI_avg"]].copy()

    if hni.empty:
        print(f"[WARN] {pathway}: HNI mapping produced 0 TFs.")
        print(f"        Example HNI tf_real: {hni_real['tf_real'].head(5).tolist()}")
        print(f"        Edgelist used: {edgelist_tsv}")

    return hni


# =============================================================================
# Statistics and plotting helpers
# =============================================================================
def compute_spearman_table(all_df: pd.DataFrame, x: str, y: str, mode: str) -> pd.DataFrame:
    """
    Returns per-pathway Spearman.
    Columns: pathway, spearman, p, n
    """
    rows = []
    if all_df is None or all_df.empty:
        return pd.DataFrame(columns=["pathway", "spearman", "p", "n"])

    for p, sub in all_df.groupby("pathway"):
        sub = sub[[x, y]].dropna()
        sub = signed_unsigned_transform(sub, x=x, y=y, mode=mode)
        n = len(sub)
        if n < 2:
            rho, pval = np.nan, np.nan
        else:
            rho, pval = spearmanr(sub[x], sub[y])
        rows.append({"pathway": p, "spearman": float(rho) if rho is not None else np.nan, "p": float(pval) if pval is not None else np.nan, "n": int(n)})

    out = pd.DataFrame(rows).sort_values("spearman", ascending=False, na_position="last").reset_index(drop=True)
    return out


def compute_random_baseline_summary_table(
    sergio_by_pathway: dict,
    baseline_by_pathway: dict,
    x: str,
    mode: str,
    n_shuffles: int = 1000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      summary_df: one row per pathway with mean/std of shuffled Spearman
      full_df: full null distribution with one row per shuffle per pathway
    """
    summary_rows = []
    full_rows = []

    for i, pathway in enumerate(PATHWAYS):
        if pathway not in sergio_by_pathway or pathway not in baseline_by_pathway:
            summary_rows.append({
                "pathway": pathway,
                "random_mean": np.nan,
                "random_std": np.nan,
                "n_shuffles": 0,
            })
            continue

        mean_r, std_r, corrs = compute_random_baseline_spearman_summary(
            ser_df=sergio_by_pathway[pathway],
            base_df=baseline_by_pathway[pathway],
            x=x,
            mode=mode,
            n_shuffles=n_shuffles,
            seed=seed + i,
        )

        summary_rows.append({
            "pathway": pathway,
            "random_mean": mean_r,
            "random_std": std_r,
            "n_shuffles": int(len(corrs)),
        })

        for j, c in enumerate(corrs):
            full_rows.append({
                "pathway": pathway,
                "shuffle_idx": j,
                "spearman_random": float(c),
            })

    summary_df = pd.DataFrame(summary_rows)
    full_df = pd.DataFrame(full_rows)
    return summary_df, full_df


def paired_pvalue(corr_k: pd.DataFrame, corr_d: pd.DataFrame) -> tuple[float, int, str]:
    """
    Paired comparison of per-pathway Spearman(KPNN) vs Spearman(Decoupler).
    """
    m = corr_k.merge(corr_d, on="pathway", suffixes=("_kpnn", "_dec"), how="inner")
    m = m[np.isfinite(m["spearman_kpnn"]) & np.isfinite(m["spearman_dec"])]

    n = len(m)
    if n < 2:
        return np.nan, n, "NA"

    diffs = m["spearman_kpnn"].to_numpy() - m["spearman_dec"].to_numpy()

    if n >= 3 and np.any(np.abs(diffs) > 0):
        try:
            stat, p = wilcoxon(diffs, alternative="two-sided", zero_method="wilcox")
            return float(p), n, "Wilcoxon"
        except Exception:
            pass

    stat, p = ttest_rel(
        m["spearman_kpnn"].to_numpy(),
        m["spearman_dec"].to_numpy(),
        nan_policy="omit"
    )
    return float(p), n, "Paired t-test"


def plot_summary(all_df: pd.DataFrame, x: str, y: str, mode: str, out_png: Path, title: str):
    if all_df is None or all_df.empty:
        print(f"[SUMMARY] all_df is empty for plot '{title}'. Skipping.")
        return

    dfp = all_df[[x, y, "pathway"]].dropna().copy()
    dfp = signed_unsigned_transform(dfp, x=x, y=y, mode=mode)

    corr = compute_spearman_table(all_df, x=x, y=y, mode=mode)
    avg_rho = float(np.nanmean(corr["spearman"].to_numpy())) if not corr.empty else np.nan

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 6),
        gridspec_kw={"width_ratios": [1, 1.6]}
    )

    ax1.bar(corr["pathway"], corr["spearman"])
    ax1.axhline(0, linewidth=1)
    ax1.set_ylabel("Spearman ρ")
    ax1.set_title(f"{title}\nAvg ρ={avg_rho:.2f} | Spearman per pathway")
    ax1.tick_params(axis="x", rotation=45)

    for p, sub in dfp.groupby("pathway"):
        ax2.scatter(sub[x], sub[y], s=70, alpha=0.85, label=p)

        if len(sub) >= 2:
            xx = sub[x].to_numpy()
            yy = sub[y].to_numpy()
            m, b = np.polyfit(xx, yy, 1)
            xs = np.linspace(xx.min(), xx.max(), 50)
            ax2.plot(xs, m * xs + b, linewidth=2)

    if x == "delta_sep_euclidean":
        ax2.set_xlabel("EFFECT1.0 / ΔSEP\n+ = KO makes A/B more similar\n- = KO makes A/B more different")
    else:
        ax2.set_xlabel(x)

    ax2.set_ylabel(y)
    ax2.set_title(f"{title}\nAll TFs pooled (colored by pathway)")
    ax2.legend(fontsize=9, frameon=False)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out_png}")

    print("[SUMMARY] Spearman per pathway:")
    for _, r in corr.iterrows():
        print(f"  - {r['pathway']}: spearman={r['spearman']:.3f} (p={r['p']:.2e}) n={int(r['n'])}")


def plot_plain_effect_scatter(
    df: pd.DataFrame,
    effect_col: str,
    effect_name: str,
    out_dir: Path,
):
    """
    Creates one plain scatterplot per pathway and effect score.

    y-axis:
      positive HNI = important for cell type A / KPNN class 1
      negative HNI = important for cell type B / KPNN class 0

    x-axis:
      EFFECT1.0:
        positive = KO makes A/B more similar
        negative = KO makes A/B more different
      Other effect scores:
        positive = stronger A-oriented effect
        negative = stronger B-oriented effect
    """
    required = ["pathway", "tf", effect_col, "HNI_avg", "class_A_name", "class_B_name"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[WARN] Missing columns for {effect_name}: {missing}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    for pathway, sub in df.groupby("pathway"):
        sub = sub[required].dropna().copy()
        if sub.empty:
            continue

        class_A = str(sub["class_A_name"].iloc[0])
        class_B = str(sub["class_B_name"].iloc[0])

        rho, pval = (np.nan, np.nan)
        if len(sub) >= 2:
            rho, pval = spearmanr(sub[effect_col], sub["HNI_avg"])

        fig, ax = plt.subplots(figsize=(8.5, 7.2))
        ax.axhline(0, linewidth=1)
        ax.axvline(0, linewidth=1)

        ax.scatter(
            sub[effect_col],
            sub["HNI_avg"],
            s=85,
            alpha=0.85,
        )

        for _, row in sub.iterrows():
            ax.text(
                row[effect_col],
                row["HNI_avg"],
                str(row["tf"]),
                fontsize=8,
                ha="left",
                va="bottom",
            )

            y_label = (
            f"KPNN HNI\n\n"
            f"+ = important for\n"
            f"    {class_A} (KPNN class 1)\n\n"
            f"- = important for\n"
            f"    {class_B} (KPNN class 0)"
        )

        if effect_col == "effect1_0" or effect_col == "delta_sep_euclidean":

            x_label = (
                f"{effect_name}\n\n"
                f"+ = KO makes classes more similar\n"
                f"    A = {class_A}\n"
                f"    B = {class_B}\n\n"
                f"- = KO makes classes more different\n"
                f"    A = {class_A}\n"
                f"    B = {class_B}"
            )

        else:

            x_label = (
                f"{effect_name}\n\n"
                f"+ = stronger effect in\n"
                f"    {class_A}\n\n"
                f"- = stronger effect in\n"
                f"    {class_B}"
            )

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        ax.set_title(
            f"{pathway}: {effect_name} vs KPNN HNI\n"
            f"Spearman ρ={rho:.3f}, p={pval:.2e}, n={len(sub)}"
        )

        plt.tight_layout()
        out_png = out_dir / f"{pathway}_{effect_name}_plain_scatter.svg"
        plt.savefig(out_png, dpi=250, bbox_inches="tight")
        plt.close()

        print(f"✅ Saved plain scatter: {out_png}")


def plot_method_comparison_bars(
    corr_k: pd.DataFrame,
    corr_d: pd.DataFrame,
    corr_b: pd.DataFrame,
    corr_r_summary: pd.DataFrame,
    out_png: Path,
    title: str,
):
    """
    Grouped bars per pathway:
      KPNN vs Decoupler vs Baseline vs Random baseline mean ± std
    """
    b = corr_b.rename(columns={
        "spearman": "spearman_baseline",
        "p": "p_baseline",
        "n": "n_baseline"
    })

    r = corr_r_summary.rename(columns={
        "random_mean": "spearman_random",
        "random_std": "spearman_random_std"
    })

    m = (
        corr_k.merge(corr_d, on="pathway", suffixes=("_kpnn", "_dec"), how="outer")
              .merge(b, on="pathway", how="outer")
              .merge(r, on="pathway", how="outer")
    )

    m = m.set_index("pathway").reindex(PATHWAYS).reset_index()

    out_csv = out_png.with_suffix(".csv")
    m.to_csv(out_csv, index=False)
    print(f"📄 Saved data: {out_csv}")

    pval, n_pairs, method = paired_pvalue(corr_k, corr_d)

    x = np.arange(len(m))
    w = 0.20

    kp = m["spearman_kpnn"].to_numpy(dtype=float)
    dc = m["spearman_dec"].to_numpy(dtype=float)
    bl = m["spearman_baseline"].to_numpy(dtype=float)
    rb = m["spearman_random"].to_numpy(dtype=float)
    rb_std = m["spearman_random_std"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(13, 5.8))
    ax.bar(x - 1.5*w, kp, width=w, label="KPNN", alpha=0.95)
    ax.bar(x - 0.5*w, dc, width=w, label="Decoupler", alpha=0.95)
    ax.bar(x + 0.5*w, bl, width=w, label="Baseline", alpha=0.95)
    ax.bar(
        x + 1.5*w,
        rb,
        width=w,
        yerr=rb_std,
        capsize=4,
        label="Random baseline (mean ± SD)",
        alpha=0.95
    )
    ax.axhline(0, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(m["pathway"].tolist(), rotation=0)
    ax.set_ylabel("Spearman ρ")

    pv_txt = "p=NA" if not np.isfinite(pval) else f"p={pval:.2e}"
    ax.set_title(f"{title}\nPaired {method} KPNN vs Decoupler ({n_pairs} pathways): {pv_txt}")

    ax.legend(frameon=False)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out_png}")


def restrict_other_methods_to_kpnn_tfs(all_kpnn, all_dec, all_base, all_rand):
    """
    Keep only pathway/TF pairs that are present in the KPNN comparison.
    This makes Decoupler, Baseline, and Random baseline use exactly the same
    SERGIO effect values as KPNN.
    """
    if all_kpnn.empty:
        print("[WARN] all_kpnn is empty; cannot restrict other methods to KPNN TFs.")
        return all_kpnn, all_dec, all_base, all_rand

    kpnn_keys = all_kpnn[["pathway", "tf"]].drop_duplicates()

    print("\n================ FILTER TO KPNN TFs ================")
    print(f"KPNN pathway/TF pairs used as reference: {len(kpnn_keys)}")

    for p in PATHWAYS:
        k_tfs = set(all_kpnn.loc[all_kpnn["pathway"] == p, "tf"])
        d_tfs = set(all_dec.loc[all_dec["pathway"] == p, "tf"])
        b_tfs = set(all_base.loc[all_base["pathway"] == p, "tf"])
        r_tfs = set(all_rand.loc[all_rand["pathway"] == p, "tf"])

        print(f"\n[PATHWAY] {p}")
        print(f"  KPNN TFs:      {len(k_tfs)} {sorted(k_tfs)}")
        print(f"  Decoupler TFs: {len(d_tfs)} {sorted(d_tfs)}")
        print(f"  Baseline TFs:  {len(b_tfs)} {sorted(b_tfs)}")
        print(f"  Random TFs:    {len(r_tfs)} {sorted(r_tfs)}")

        print(f"  Removed from Decoupler: {sorted(d_tfs - k_tfs)}")
        print(f"  Removed from Baseline:  {sorted(b_tfs - k_tfs)}")
        print(f"  Removed from Random:    {sorted(r_tfs - k_tfs)}")

        if not all_dec.empty:
            extra_dec = all_dec[
                (all_dec["pathway"] == p) &
                (~all_dec["tf"].isin(k_tfs))
            ].copy()

            if not extra_dec.empty:
                extra_dec["abs_effect"] = extra_dec["delta_sep_euclidean"].abs()
                print("  Extra Decoupler rows that will be removed:")
                print(
                    extra_dec[
                        ["pathway", "tf", "delta_basal", "delta_sep_euclidean", "abs_effect"]
                    ].sort_values("abs_effect", ascending=False)
                )

    before = {
        "KPNN": len(all_kpnn),
        "Decoupler": len(all_dec),
        "Baseline": len(all_base),
        "Random": len(all_rand),
    }

    all_dec = all_dec.merge(kpnn_keys, on=["pathway", "tf"], how="inner")
    all_base = all_base.merge(kpnn_keys, on=["pathway", "tf"], how="inner")
    all_rand = all_rand.merge(kpnn_keys, on=["pathway", "tf"], how="inner")

    after = {
        "KPNN": len(all_kpnn),
        "Decoupler": len(all_dec),
        "Baseline": len(all_base),
        "Random": len(all_rand),
    }

    print("\n[ROW COUNTS BEFORE FILTER]")
    print(before)
    print("[ROW COUNTS AFTER FILTER]")
    print(after)

    print("\n[ABS EFFECT1.0 / ΔSEP RANGES AFTER FILTER]")
    for name, df in {
        "KPNN": all_kpnn,
        "Decoupler": all_dec,
        "Baseline": all_base,
        "Random": all_rand,
    }.items():
        if df.empty:
            print(f"{name}: EMPTY")
        else:
            print(
                f"{name}: min={df['delta_sep_euclidean'].abs().min():.4f}, "
                f"max={df['delta_sep_euclidean'].abs().max():.4f}"
            )

    print("====================================================\n")

    return all_kpnn, all_dec, all_base, all_rand

def make_sergio_boxplot_values(all_kpnn, all_dec, all_base, all_rand, out_csv: Path) -> pd.DataFrame:
    
    effect_cols = {
        "DELTA_BASAL": "delta_basal",
        "EFFECT1_0": "effect1_0",
        "EFFECT2_0": "effect2_0",
        "EFFECT2_1": "effect2_1",
        "EFFECT3_0": "effect3_0",
        "EFFECT3_1": "effect3_1",
        "EFFECT3_2": "effect3_2",
        "EFFECT3_3": "effect3_3",
    }

    method_specs = [
        ("KPNN", all_kpnn, "HNI_avg"),
        ("DECOUPLER", all_dec, "delta_activity"),
        ("BASELINE", all_base, "baseline_score"),
        ("RANDOM_BASELINE", all_rand, "random_baseline_score"),
    ]

    rows = []

    for effect_name, effect_col in effect_cols.items():

        if effect_name == "EFFECT3_3":
            effect_modes = ["magnitude"]
        else:
            effect_modes = ["signed", "magnitude"]

        for effect_mode in effect_modes:
            for method_mode in ["signed", "magnitude"]:
                for method_name, df, method_col in method_specs:
                    if df is None or df.empty:
                        continue
                    if effect_col not in df.columns or method_col not in df.columns:
                        continue

                    for pathway, sub in df.groupby("pathway"):
                        tmp = sub[[effect_col, method_col]].dropna().copy()

                        if effect_mode == "magnitude":
                            tmp[effect_col] = tmp[effect_col].abs()

                        if method_mode == "magnitude":
                            tmp[method_col] = tmp[method_col].abs()

                        if len(tmp) < 2:
                            rho, pval = np.nan, np.nan
                        elif tmp[effect_col].nunique() < 2 or tmp[method_col].nunique() < 2:
                            rho, pval = np.nan, np.nan
                        else:
                            rho, pval = spearmanr(tmp[effect_col], tmp[method_col])

                        rows.append({
                            "pathway": pathway,
                            "method": method_name,
                            "effect_score_type": effect_name,
                            "effect_mode": effect_mode,
                            "method_mode": method_mode,
                            "spearman_r": rho,
                            "spearman_p": pval,
                            "n": len(tmp),
                        })

    box_df = pd.DataFrame(rows)
    box_df = box_df[box_df["spearman_r"].notna()].copy()
    box_df.to_csv(out_csv, index=False)
    print(f"📄 SERGIO boxplot CSV saved: {out_csv}")
    return box_df


def plot_sergio_overview_boxplot_paper_style(box_df: pd.DataFrame, out_dir: Path):
    if box_df.empty:
        print("[WARN] SERGIO boxplot dataframe is empty.")
        return

    df = box_df.copy()

    df["effect_score_type"] = pd.Categorical(
        df["effect_score_type"],
        categories=[
            "DELTA_BASAL",
            "EFFECT1_0",
            "EFFECT2_0",
            "EFFECT2_1",
            "EFFECT3_0",
            "EFFECT3_1",
            "EFFECT3_2",
            "EFFECT3_3",
        ],
        ordered=True,
    )

    df["effect_mode"] = pd.Categorical(
        df["effect_mode"],
        categories=["signed", "magnitude"],
        ordered=True,
    )

    pretty_effect = {
        "DELTA_BASAL": "Δbasal",
        "EFFECT1_0": "EFFECT1.0",
        "EFFECT2_0": "EFFECT2.0",
        "EFFECT2_1": "EFFECT2.1",
        "EFFECT3_0": "EFFECT3.0",
        "EFFECT3_1": "EFFECT3.1",
        "EFFECT3_2": "EFFECT3.2",
        "EFFECT3_3": "EFFECT3.3",
    }

    df["x_group"] = (
        df["effect_score_type"].astype(str).map(pretty_effect)
        + "\n"
        + df["effect_mode"].map({
            "signed": "signed",
            "magnitude": "unsigned",
        }).astype(str)
    )

    x_order = []
    for effect in [
        "DELTA_BASAL",
        "EFFECT1_0",
        "EFFECT2_0",
        "EFFECT2_1",
        "EFFECT3_0",
        "EFFECT3_1",
        "EFFECT3_2",
        "EFFECT3_3",
    ]:
        modes = ["magnitude"] if effect == "EFFECT3_3" else ["signed", "magnitude"]

        for mode in modes:
            x_order.append(
                pretty_effect[effect] + "\n" +
                ("signed" if mode == "signed" else "unsigned")
            )

    method_order = ["KPNN", "DECOUPLER", "BASELINE", "RANDOM_BASELINE"]

    palettes = {
        "signed": {
            "KPNN": "#E69F00",
            "DECOUPLER": "#0072B2",
            "BASELINE": "#666666",
            "RANDOM_BASELINE": "#009E73",
        },
        "magnitude": {
            "KPNN": "#F6C267",
            "DECOUPLER": "#8EC5E8",
            "BASELINE": "#BDBDBD",
            "RANDOM_BASELINE": "#8FDCC2",
        },
    }

    import seaborn as sns

    sns.set_theme(style="white", context="paper")

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(18, 9),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.18},
    )

    panel_titles = {
        "signed": "Signed method scores",
        "magnitude": "Unsigned method scores",
    }

    for ax, method_mode in zip(axes, ["signed", "magnitude"]):
        sub = df[df["method_mode"] == method_mode].copy()

        sns.boxplot(
            data=sub,
            x="x_group",
            y="spearman_r",
            hue="method",
            order=x_order,
            hue_order=method_order,
            palette=palettes[method_mode],
            width=0.72,
            linewidth=1.2,
            fliersize=0,
            ax=ax,
        )

        sns.stripplot(
            data=sub,
            x="x_group",
            y="spearman_r",
            hue="method",
            order=x_order,
            hue_order=method_order,
            palette=palettes[method_mode],
            dodge=True,
            size=3,
            alpha=0.45,
            linewidth=0,
            ax=ax,
        )

        ax.axhline(0, color="black", linewidth=0.9, alpha=0.7)
        ax.set_title(panel_titles[method_mode], fontsize=16, weight="bold", pad=10)
        ax.set_xlabel("")
        ax.set_ylabel("Spearman ρ", fontsize=13)
        ax.tick_params(axis="y", labelsize=11)
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
        ax.set_axisbelow(True)

        sns.despine(ax=ax, top=True, right=True)

        handles, labels = ax.get_legend_handles_labels()
        if ax.legend_ is not None:
            ax.legend_.remove()

    handles, labels = axes[0].get_legend_handles_labels()
    handles = handles[:len(method_order)]
    labels = labels[:len(method_order)]

    fig.legend(
        handles,
        labels,
        title="Method",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=False,
        fontsize=12,
        title_fontsize=12,
    )

    axes[-1].tick_params(axis="x", labelsize=10, rotation=0)

    fig.suptitle(
        "Method–SERGIO score agreement across simulated effect definitions",
        fontsize=18,
        weight="bold",
        y=1.08,
    )

    fig.supxlabel("SERGIO score definition", fontsize=13, y=0.04)

    plt.tight_layout()

    out_png = out_dir / "SERGIO_overview_boxplot_two_panel_paper_style.png"
    out_pdf = out_dir / "SERGIO_overview_boxplot_two_panel_paper_style.pdf"

    plt.savefig(out_png, dpi=400, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

    print(f"📦 SERGIO overview boxplot saved: {out_png}")

# =============================================================================
# Main workflow
# =============================================================================
def main():
    kpnn_rows = []
    dec_rows = []
    baseline_rows = []
    random_baseline_rows = []
    sergio_by_pathway = {}
    baseline_by_pathway = {}

    # -------------------------------------------------------------------------
    # 1) Load and merge each pathway
    # -------------------------------------------------------------------------
    for p in PATHWAYS:
        print(f"\n[LOAD] Pathway={p}")

        try:
            ser = load_sergio_metrics(p)
            sergio_by_pathway[p] = ser.copy()
        except Exception as e:
            print(f"[WARN] Skipping {p}: failed to load SERGIO metrics: {e}")
            continue

        # ---- KPNN merge (with mapping) ----
        try:
            hni = load_kpnn_hni_mapped_to_TFk(p)
            kpnn = ser.merge(hni, on="tf", how="inner")
            if kpnn.empty:
                print(f"[WARN] {p}: SERGIO×KPNN merge empty (after mapping).")
            else:
                kpnn_rows.append(kpnn)
        except Exception as e:
            print(f"[WARN] {p}: failed to load/merge KPNN HNI: {e}")

        # ---- Decoupler merge ----
        try:
            act = load_decoupler_activity(p)
            dec = ser.merge(act, on="tf", how="inner")
            if dec.empty:
                print(f"[WARN] {p}: SERGIO×Decoupler merge empty.")
            else:
                dec_rows.append(dec)
        except Exception as e:
            print(f"[WARN] {p}: failed to load/merge Decoupler activity: {e}")

        # ---- Baseline merge ----
        try:
            base = load_baseline_scores_mapped_to_TFk(p)
            baseline = ser.merge(base, on="tf", how="inner")
            baseline_by_pathway[p] = base.copy()
            if baseline.empty:
                print(f"[WARN] {p}: SERGIO×Baseline merge empty.")
            else:
                baseline_rows.append(baseline)
        except Exception as e:
            print(f"[WARN] {p}: failed to load/merge baseline scores: {e}")

        # ---- Random baseline merge (negative control) ----
        try:
            rand_base = load_random_baseline_scores_mapped_to_TFk(p, seed=42)
            rand = ser.merge(rand_base, on="tf", how="inner")
            if rand.empty:
                print(f"[WARN] {p}: SERGIO×Random-baseline merge empty.")
            else:
                random_baseline_rows.append(rand)
        except Exception as e:
            print(f"[WARN] {p}: failed to load/merge random baseline scores: {e}")

    # -------------------------------------------------------------------------
    # 2) Combine pathway-specific results
    # -------------------------------------------------------------------------
    all_kpnn = pd.concat(kpnn_rows, ignore_index=True) if kpnn_rows else pd.DataFrame()
    all_dec  = pd.concat(dec_rows, ignore_index=True) if dec_rows else pd.DataFrame()
    all_base = pd.concat(baseline_rows, ignore_index=True) if baseline_rows else pd.DataFrame()
    all_rand = pd.concat(random_baseline_rows, ignore_index=True) if random_baseline_rows else pd.DataFrame()

    # -------------------------------------------------------------------------
    # 3) Diagnostic printouts for checking intermediate values
    # -------------------------------------------------------------------------
    print("\n[CHECK EFFECT1.0 / ΔSEP values]")
    if not all_kpnn.empty:
        print(all_kpnn["delta_sep_euclidean"].describe())
        print("Negative values:", (all_kpnn["delta_sep_euclidean"] < 0).sum())
        print("Positive values:", (all_kpnn["delta_sep_euclidean"] > 0).sum())
        print("Zero values:", (all_kpnn["delta_sep_euclidean"] == 0).sum())

        print("\n[SMALLEST EFFECT1.0 VALUES: more different after KO if negative]")
        print(
            all_kpnn[["pathway", "tf", "d_base", "d_ko", "effect1_0", "effect1_0_direction"]]
            .sort_values("effect1_0")
            .head(20)
        )

        print("\n[LARGEST EFFECT1.0 VALUES: more similar after KO if positive]")
        print(
            all_kpnn[["pathway", "tf", "d_base", "d_ko", "effect1_0", "effect1_0_direction"]]
            .sort_values("effect1_0", ascending=False)
            .head(20)
        )

        print("\n[CHECK effect1_0 equals delta_sep_euclidean]")
        print(np.allclose(all_kpnn["effect1_0"], all_kpnn["delta_sep_euclidean"], equal_nan=True))
    else:
        print("[WARN] all_kpnn empty; cannot print EFFECT1.0 diagnostics.")

    print("\n[BEFORE FILTER] TGFB1 baseline preview:")
    if not all_base.empty:
        print(
            all_base[all_base["pathway"] == "TGFB1"][
                ["tf", "delta_basal", "d_base", "d_ko", "effect1_0", "effect2_0", "effect2_1", "baseline_score", "baseline_auc"]
            ].head(20)
        )

    # -------------------------------------------------------------------------
    # 4) Restrict all methods to the TFs available in the KPNN comparison
    # -------------------------------------------------------------------------
    all_kpnn, all_dec, all_base, all_rand = restrict_other_methods_to_kpnn_tfs(
        all_kpnn, all_dec, all_base, all_rand
    )

    # Rebuild these dictionaries AFTER filtering, so random-baseline summaries
    # use the same KPNN TF subset as all other methods.
    baseline_by_pathway = {
        p: all_base.loc[
            all_base["pathway"] == p,
            ["tf", "baseline_score", "baseline_auc"]
        ].copy()
        for p in PATHWAYS
    }

    sergio_by_pathway = {
        p: all_kpnn.loc[
            all_kpnn["pathway"] == p,
            [
                "tf",
                "delta_basal",
                "d_base",
                "d_ko",
                "effect1_0_direction",
                "effect1_0",
                "effect2_0",
                "effect2_1",
                "effect3_0",
                "effect3_1",
                "effect3_2",
                "effect3_3",
                "delta_sep_euclidean",
                "pathway",
                "class_A_name",
                "class_B_name",
            ]
        ].copy()
        for p in PATHWAYS
    }

    print("\n[AFTER FILTER] TGFB1 baseline preview:")
    if not all_base.empty:
        print(
            all_base[all_base["pathway"] == "TGFB1"][
                ["tf", "delta_basal", "d_base", "d_ko", "effect1_0", "effect2_0", "effect2_1", "baseline_score", "baseline_auc"]
            ].head(20)
        )

    print("\n[DEBUG] all_kpnn shape:", all_kpnn.shape)
    if not all_kpnn.empty:
        print("[DEBUG] all_kpnn columns:", all_kpnn.columns.tolist())
        print("[DEBUG] pathways in all_kpnn:", sorted(all_kpnn["pathway"].unique().tolist()))
    else:
        print("[DEBUG] all_kpnn is empty.")

    print("[DEBUG] all_dec shape:", all_dec.shape)
    if not all_dec.empty:
        print("[DEBUG] all_dec columns:", all_dec.columns.tolist())
        print("[DEBUG] pathways in all_dec:", sorted(all_dec["pathway"].unique().tolist()))
    else:
        print("[DEBUG] all_dec is empty.")

    print("[DEBUG] all_base shape:", all_base.shape)
    if not all_base.empty:
        print("[DEBUG] all_base columns:", all_base.columns.tolist())
        print("[DEBUG] pathways in all_base:", sorted(all_base["pathway"].unique().tolist()))
    else:
        print("[DEBUG] all_base is empty.")

    print("[DEBUG] all_rand shape:", all_rand.shape)
    if not all_rand.empty:
        print("[DEBUG] all_rand columns:", all_rand.columns.tolist())
        print("[DEBUG] pathways in all_rand:", sorted(all_rand["pathway"].unique().tolist()))
    else:
        print("[DEBUG] all_rand is empty.")

    # -------------------------------------------------------------------------
    # 5) Save merged result tables
    # -------------------------------------------------------------------------
    out_dir = KPNN_ROOT / "ALL_PATHWAYS_SUMMARY"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not all_kpnn.empty:
        all_kpnn.to_csv(out_dir / "all_pathways_kpnn_vs_sergio.csv", index=False)
    else:
        print("[WARN] all_kpnn empty; not writing all_pathways_kpnn_vs_sergio.csv")

    if not all_dec.empty:
        all_dec.to_csv(out_dir / "all_pathways_decoupler_vs_sergio.csv", index=False)

    if not all_base.empty:
        all_base.to_csv(out_dir / "all_pathways_baseline_vs_sergio.csv", index=False)

    if not all_rand.empty:
        all_rand.to_csv(out_dir / "all_pathways_random_baseline_vs_sergio.csv", index=False)

    # -------------------------------------------------------------------------
    # 6) Create overview boxplot values and plot
    # -------------------------------------------------------------------------
    sergio_box_csv = out_dir / "SERGIO_ALL_EFFECTS_boxplot_values.csv"

    sergio_box_df = make_sergio_boxplot_values(
        all_kpnn=all_kpnn,
        all_dec=all_dec,
        all_base=all_base,
        all_rand=all_rand,
        out_csv=sergio_box_csv,
    )

    plot_sergio_overview_boxplot_paper_style(
        box_df=sergio_box_df,
        out_dir=out_dir,
    )

    # -------------------------------------------------------------------------
    # 7) Plain effect-score scatterplots
    # -------------------------------------------------------------------------
    scatter_dir = out_dir / "effect_score_plain_scatterplots"
    scatter_dir.mkdir(parents=True, exist_ok=True)

    effect_score_columns = {
        "EFFECT1_0": "effect1_0",
        "EFFECT2_0": "effect2_0",
        "EFFECT2_1": "effect2_1",
        "EFFECT3_0": "effect3_0",
        "EFFECT3_1": "effect3_1",
        "EFFECT3_2": "effect3_2",
        "EFFECT3_3": "effect3_3",
    }

    for effect_name, effect_col in effect_score_columns.items():
        if effect_col not in all_kpnn.columns:
            print(f"[WARN] Skipping {effect_name}: {effect_col} not in all_kpnn.")
            continue

        plot_plain_effect_scatter(
            df=all_kpnn,
            effect_col=effect_col,
            effect_name=effect_name,
            out_dir=scatter_dir,
        )

    # -------------------------------------------------------------------------
    # 8) Existing summary plots
    # -------------------------------------------------------------------------
    plot_summary(
        all_kpnn,
        x="delta_basal",
        y="HNI_avg",
        mode="basal",
        out_png=out_dir / "1_KPNN_HNI_vs_delta_basal_ALL.svg",
        title="SERGIO: KPNN HNI vs Δbasal (signed)"
    )

    plot_summary(
        all_dec,
        x="delta_basal",
        y="delta_activity",
        mode="basal",
        out_png=out_dir / "3_DECOUPLER_activity_vs_delta_basal_ALL.svg",
        title="SERGIO: Decoupler Δactivity vs Δbasal (signed)"
    )

    plot_summary(
        all_rand,
        x="delta_basal",
        y="random_baseline_score",
        mode="basal",
        out_png=out_dir / "5_RANDOM_BASELINE_vs_delta_basal_ALL.svg",
        title="SERGIO: Random baseline vs Δbasal (signed)"
    )

    plot_summary(
        all_kpnn,
        x="delta_sep_euclidean",
        y="HNI_avg",
        mode="effect",
        out_png=out_dir / "2_KPNN_HNI_vs_EFFECT_ALL.svg",
        title="SERGIO: KPNN HNI vs EFFECT1.0 / ΔSEP (signed)"
    )

    plot_summary(
        all_dec,
        x="delta_sep_euclidean",
        y="delta_activity",
        mode="effect",
        out_png=out_dir / "4_DECOUPLER_activity_vs_EFFECT_ALL.svg",
        title="SERGIO: Decoupler Δactivity vs EFFECT1.0 / ΔSEP (signed)"
    )

    plot_summary(
        all_rand,
        x="delta_sep_euclidean",
        y="random_baseline_score",
        mode="effect",
        out_png=out_dir / "6_RANDOM_BASELINE_vs_EFFECT_ALL.svg",
        title="SERGIO: Random baseline vs EFFECT1.0 / ΔSEP (signed)"
    )

    for effect_name, effect_col in {
        "EFFECT2_0": "effect2_0",
        "EFFECT2_1": "effect2_1",
        "EFFECT3_0": "effect3_0",
        "EFFECT3_1": "effect3_1",
        "EFFECT3_2": "effect3_2",
        "EFFECT3_3": "effect3_3",
    }.items():

        plot_summary(
            all_kpnn,
            x=effect_col,
            y="HNI_avg",
            mode="effect",
            out_png=out_dir / f"KPNN_HNI_vs_{effect_name}_ALL.svg",
            title=f"SERGIO: KPNN HNI vs {effect_name}"
        )

        plot_summary(
            all_dec,
            x=effect_col,
            y="delta_activity",
            mode="effect",
            out_png=out_dir / f"DECOUPLER_activity_vs_{effect_name}_ALL.svg",
            title=f"SERGIO: Decoupler Δactivity vs {effect_name}"
        )

        plot_summary(
            all_rand,
            x=effect_col,
            y="random_baseline_score",
            mode="effect",
            out_png=out_dir / f"RANDOM_BASELINE_vs_{effect_name}_ALL.svg",
            title=f"SERGIO: Random baseline vs {effect_name}"
        )

    # -------------------------------------------------------------------------
    # 9) Bar plots comparing methods per pathway
    # -------------------------------------------------------------------------
    corr_k_basal = compute_spearman_table(all_kpnn, x="delta_basal", y="HNI_avg", mode="basal")
    corr_d_basal = compute_spearman_table(all_dec,  x="delta_basal", y="delta_activity", mode="basal")
    corr_k_basal.to_csv(out_dir / "KPNN_delta_basal_spearman.csv", index=False)
    corr_d_basal.to_csv(out_dir / "Decoupler_delta_basal_spearman.csv", index=False)

    corr_b_basal = compute_spearman_table(all_base, x="delta_basal", y="baseline_score", mode="basal")
    corr_b_basal.to_csv(out_dir / "Baseline_delta_basal_spearman.csv", index=False)

    corr_r_basal_summary, corr_r_basal_full = compute_random_baseline_summary_table(
        sergio_by_pathway=sergio_by_pathway,
        baseline_by_pathway=baseline_by_pathway,
        x="delta_basal",
        mode="basal",
        n_shuffles=N_RANDOM_SHUFFLES,
        seed=RANDOM_SEED,
    )
    corr_r_basal_summary.to_csv(out_dir / "RandomBaseline_delta_basal_spearman_summary.csv", index=False)
    corr_r_basal_full.to_csv(out_dir / "RandomBaseline_delta_basal_spearman_null.csv", index=False)

    plot_method_comparison_bars(
        corr_k_basal,
        corr_d_basal,
        corr_b_basal,
        corr_r_basal_summary,
        out_png=out_dir / "7_COMPARE_KPNN_vs_DECOUPLER_vs_BASELINE_vs_RANDOM_delta_basal.svg",
        title="COMPARE: KPNN vs Decoupler vs Baseline vs Random baseline (Δbasal task)"
    )

    corr_k_eff = compute_spearman_table(all_kpnn, x="delta_sep_euclidean", y="HNI_avg", mode="effect")
    corr_d_eff = compute_spearman_table(all_dec,  x="delta_sep_euclidean", y="delta_activity", mode="effect")
    corr_k_eff.to_csv(out_dir / "KPNN_effect_spearman.csv", index=False)
    corr_d_eff.to_csv(out_dir / "Decoupler_effect_spearman.csv", index=False)

    corr_b_eff = compute_spearman_table(all_base, x="delta_sep_euclidean", y="baseline_score", mode="effect")
    corr_b_eff.to_csv(out_dir / "Baseline_effect_spearman.csv", index=False)

    corr_r_eff_summary, corr_r_eff_full = compute_random_baseline_summary_table(
        sergio_by_pathway=sergio_by_pathway,
        baseline_by_pathway=baseline_by_pathway,
        x="delta_sep_euclidean",
        mode="effect",
        n_shuffles=N_RANDOM_SHUFFLES,
        seed=RANDOM_SEED,
    )
    corr_r_eff_summary.to_csv(out_dir / "RandomBaseline_effect_spearman_summary.csv", index=False)
    corr_r_eff_full.to_csv(out_dir / "RandomBaseline_effect_spearman_null.csv", index=False)

    plot_method_comparison_bars(
        corr_k_eff,
        corr_d_eff,
        corr_b_eff,
        corr_r_eff_summary,
        out_png=out_dir / "8_COMPARE_KPNN_vs_DECOUPLER_vs_BASELINE_vs_RANDOM_EFFECT.svg",
        title="COMPARE: KPNN vs Decoupler vs Baseline vs Random baseline (EFFECT1.0 / ΔSEP task)"
    )

    for effect_name, effect_col in {
        "EFFECT2_0": "effect2_0",
        "EFFECT2_1": "effect2_1",
        "EFFECT3_0": "effect3_0",
        "EFFECT3_1": "effect3_1",
        "EFFECT3_2": "effect3_2",
        "EFFECT3_3": "effect3_3",
    }.items():

        corr_k_extra = compute_spearman_table(all_kpnn, x=effect_col, y="HNI_avg", mode="effect")
        corr_d_extra = compute_spearman_table(all_dec,  x=effect_col, y="delta_activity", mode="effect")
        corr_b_extra = compute_spearman_table(all_base, x=effect_col, y="baseline_score", mode="effect")

        corr_k_extra.to_csv(out_dir / f"KPNN_{effect_name}_spearman.csv", index=False)
        corr_d_extra.to_csv(out_dir / f"Decoupler_{effect_name}_spearman.csv", index=False)
        corr_b_extra.to_csv(out_dir / f"Baseline_{effect_name}_spearman.csv", index=False)

        corr_r_extra_summary, corr_r_extra_full = compute_random_baseline_summary_table(
            sergio_by_pathway=sergio_by_pathway,
            baseline_by_pathway=baseline_by_pathway,
            x=effect_col,
            mode="effect",
            n_shuffles=N_RANDOM_SHUFFLES,
            seed=RANDOM_SEED,
        )

        corr_r_extra_summary.to_csv(out_dir / f"RandomBaseline_{effect_name}_spearman_summary.csv", index=False)
        corr_r_extra_full.to_csv(out_dir / f"RandomBaseline_{effect_name}_spearman_null.csv", index=False)

        plot_method_comparison_bars(
            corr_k_extra,
            corr_d_extra,
            corr_b_extra,
            corr_r_extra_summary,
            out_png=out_dir / f"COMPARE_KPNN_vs_DECOUPLER_vs_BASELINE_vs_RANDOM_{effect_name}.svg",
            title=f"COMPARE: KPNN vs Decoupler vs Baseline vs Random baseline ({effect_name})"
        )
        
    # -------------------------------------------------------------------------
    # 10) Final paired statistics and output summary
    # -------------------------------------------------------------------------
    p_basal, n_basal, m_basal = paired_pvalue(corr_k_basal, corr_d_basal)
    p_eff,   n_eff,   m_eff   = paired_pvalue(corr_k_eff, corr_d_eff)

    print("\n[STATS] Paired KPNN vs Decoupler per-pathway Spearman:")
    print(f"  - Δbasal task:  method={m_basal}, n={n_basal}, p={'NA' if not np.isfinite(p_basal) else f'{p_basal:.2e}'}")
    print(f"  - EFFECT1.0 task:  method={m_eff},   n={n_eff},   p={'NA' if not np.isfinite(p_eff) else f'{p_eff:.2e}'}")

    print("\n✅ Done.")
    print(f"  Output folder: {out_dir}")
    print(f"  Scatter folder: {scatter_dir}")
    if not all_kpnn.empty:
        print(f"  - {out_dir}/all_pathways_kpnn_vs_sergio.csv")
    if not all_dec.empty:
        print(f"  - {out_dir}/all_pathways_decoupler_vs_sergio.csv")


if __name__ == "__main__":
    main()
