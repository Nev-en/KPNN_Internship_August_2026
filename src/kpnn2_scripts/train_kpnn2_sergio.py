"""kpnn2 port of train_kpnn_sergio.py.

Same as train_kpnn2_sergio_rseed.py, except the splits are located the way
Karla's non-rseed script does it: under the newest preprocessed_* folder that
kpnn.preprocess_data timestamped into the SERGIO run directory. That
preprocessing step still belongs to kpnn v1; only training moves to kpnn2.

Environment variables
---------------------
SERGIO_COND     condition to run (default INS)
RESULTS_DIR     output root, also where the SERGIO run dir is read from
KPNN2_SEEDS     number of random seeds (default 20)
"""
import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kpnn2_port

# ===== paths =====
COND = os.environ.get("SERGIO_COND", "INS")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results_kpnn2/results_kpnn2_v1")

project_root = Path(__file__).resolve().parents[2]
edgelist_path = project_root / f"src/Karla_kpnn_csv/{COND}_tf_only_edgelist_filtered.tsv"
config_yaml = (
    project_root / "src/Karla_kpnn_scripts/Sergio_simulation/kpnn_config.yaml"
)
output_path = (
    project_root / f"{RESULTS_DIR}/kpnn_output/sergio/{COND}_o_A_vs_B/alter_nothing"
)

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


train_fp, val_fp, test_fp = _find_latest_split_files(sergio_run_dir)

# ===== load config (use exactly what's in YAML) =====
with open(config_yaml, "r") as f:
    cfg = yaml.safe_load(f)

additional_settings = cfg.get("additional_settings", {})
nn_hyperparameters = cfg.get("nn_hyperparameters", {})

label_col = additional_settings.get("data_label_column", "cell_type_class_int")

# ===== load data =====
train_data = ad.read_h5ad(train_fp)
val_data = ad.read_h5ad(val_fp)
test_data = ad.read_h5ad(test_fp)
datasets = {"train": train_data, "val": val_data, "test": test_data}


# ensure integer label column exists (0/1)
def ensure_binary_label(adata: ad.AnnData, label_col: str):
    if label_col in adata.obs.columns:
        return
    if "y" in adata.obs.columns and set(np.unique(adata.obs["y"])) <= {0, 1}:
        adata.obs[label_col] = adata.obs["y"].astype("Int64")
        return
    if "cell_type" in adata.obs.columns and set(
        adata.obs["cell_type"].astype(str).unique()
    ) <= {"A", "B"}:
        adata.obs[label_col] = adata.obs["cell_type"].map({"A": 0, "B": 1}).astype("Int64")
        return
    raise KeyError(
        f"Missing '{label_col}' and could not derive it from 'y' or 'cell_type'."
    )


for split in datasets.values():
    ensure_binary_label(split, label_col)

# ===== edgelist =====
edgelist = pd.read_csv(edgelist_path, sep="\t")

# ===== train =====
output_path.mkdir(parents=True, exist_ok=True)
n_seeds = int(os.environ.get("KPNN2_SEEDS", "20"))
kpnn2_port.run(
    edgelist=edgelist,
    datasets=datasets,
    nn_hyperparameters=nn_hyperparameters,
    additional_settings=additional_settings,
    output_path=output_path,
    random_seeds=list(range(1, n_seeds + 1)),
    captum_n_steps=50,
    filter_classes={0: 1, 1: 0},  # matches Karla's kpnn v1 call (swaps labels)
    condition=COND,
)

print("Training done. Output at:", output_path)
