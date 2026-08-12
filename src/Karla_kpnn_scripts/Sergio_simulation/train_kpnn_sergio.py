# train_kpnn_on_sergio.py
import os
import anndata as ad
import pandas as pd
from pathlib import Path
import yaml
import numpy as np
import kpnn

# ===== paths =====
# COND / RESULTS_DIR can be overridden via environment variables.
COND = os.environ.get("SERGIO_COND", "INS")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results_t1_2")

project_root = Path(__file__).resolve().parents[3]
edgelist_path = project_root / f"src/Karla_kpnn_csv/{COND}_tf_only_edgelist_filtered.tsv"
config_yaml = Path(__file__).resolve().parent / "kpnn_config.yaml"
output_path = project_root / f"{RESULTS_DIR}/kpnn_output/sergio/{COND}_o_A_vs_B/alter_nothing"

sergio_run_dir = project_root / f"{RESULTS_DIR}/KO_simulation/sergio/sergio_{COND}"


def _find_latest_split_files(run_dir: Path) -> tuple[str, str, str]:
    """Locate the most recent preprocessed_* folder written by
    preprocess_sergio_for_kpnn.py (kpnn.preprocess_data timestamps each run)."""
    candidates = sorted(run_dir.glob("preprocessed_*"))
    if not candidates:
        raise FileNotFoundError(
            f"No preprocessed_* folders found under {run_dir}. "
            "Run preprocess_sergio_for_kpnn.py first."
        )
    latest = candidates[-1]
    train = sorted(latest.glob("*.h5ad_train"))
    val = sorted(latest.glob("*.h5ad_val"))
    test = sorted(latest.glob("*.h5ad_test"))
    if not (train and val and test):
        raise FileNotFoundError(f"Missing train/val/test files under {latest}")
    return str(train[0]), str(val[0]), str(test[0])


# explicit split files (already decompressed/renamed to .h5ad)
train_fp, val_fp, test_fp = _find_latest_split_files(sergio_run_dir)

# ===== load config (use exactly what's in YAML) =====
with open(config_yaml, "r") as f:
    cfg = yaml.safe_load(f)

additional_settings = cfg.get("additional_settings", {})
nn_hyperparameters  = cfg.get("nn_hyperparameters", {})

label_col = additional_settings.get("data_label_column", "cell_type_class_int")

# ===== load data =====
train_data = ad.read_h5ad(train_fp)
val_data   = ad.read_h5ad(val_fp)
test_data  = ad.read_h5ad(test_fp)
datasets = {"train": train_data, "val": val_data, "test": test_data}

# ensure integer label column exists (0/1)
def ensure_binary_label(adata: ad.AnnData, label_col: str):
    if label_col in adata.obs.columns:
        return
    if "y" in adata.obs.columns and set(np.unique(adata.obs["y"])) <= {0, 1}:
        adata.obs[label_col] = adata.obs["y"].astype("Int64")
        return
    if "cell_type" in adata.obs.columns and set(adata.obs["cell_type"].astype(str).unique()) <= {"A", "B"}:
        adata.obs[label_col] = adata.obs["cell_type"].map({"A": 0, "B": 1}).astype("Int64")
        return
    raise KeyError(f"Missing '{label_col}' and could not derive it from 'y' or 'cell_type'.")

for split in datasets.values():
    ensure_binary_label(split, label_col)

# ===== edgelist =====
edgelist = pd.read_csv(edgelist_path, sep="\t")

# ===== train =====
output_path.mkdir(parents=True, exist_ok=True)
kpnn.train_kpnns(
    edgelist=edgelist,
    datasets=datasets,
    nn_hyperparameters=nn_hyperparameters,   # <-- pass YAML directly
    additional_settings=additional_settings, # <-- pass YAML directly
    remove_layers=None,
    random_seeds=list(range(1, 21)),
    network_bias_method=None,
    set_feature_values=None,
    save_models=False,
    compute_attr=True,
    captum_n_steps=50,
    get_all_attributions=False,
    get_activations=False,
    compute_infidelity=False,     # set True if you want; YAML has the perturbation counts
    compute_max_sensitivity=False,
    filter_classes={0: 1, 1: 0},          # dataset already binary 0/1
    check_edgelist=False,
    output_path=output_path,
)

print("✅ Training done. Output at:", output_path)
