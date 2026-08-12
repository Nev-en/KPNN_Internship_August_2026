# train_kpnn_on_sergio.py

import os

# ============================================================
# Reproducibility settings that must be configured BEFORE
# importing numpy / torch / pandas / anndata
# ============================================================

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


# ===== imports =====

import random
import numpy as np
import torch
import anndata as ad
import pandas as pd
from pathlib import Path
import yaml
import kpnn


# ============================================================
# Reproducibility
# ============================================================

SEED = 1

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Force single-threaded CPU execution
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Require deterministic PyTorch algorithms
torch.use_deterministic_algorithms(True)

# ===== paths =====
project_root = Path("/home/people/kmangano/masterthesis")
edgelist_path = project_root / "data/edgelist/TGFB1/TGFB1_tf_only_edgelist_filtered.tsv"
config_yaml = project_root / "src/masterscript/kpnn_config.yaml"
output_path = project_root / "results/kpnn_output/sergio/TGFB1_repro1_A_vs_B/alter_nothing"

# explicit split files (already decompressed/renamed to .h5ad)
train_fp = "/vscratch/kmangano/KO_simulation/sergio/sergio_TGFB1/preprocessed_sergio_TGFB1_nt_only.h5ad_20251105_110510/sergio_TGFB1_nt_only.h5ad_train.h5ad"
val_fp   = "/vscratch/kmangano/KO_simulation/sergio/sergio_TGFB1/preprocessed_sergio_TGFB1_nt_only.h5ad_20251105_110510/sergio_TGFB1_nt_only.h5ad_val.h5ad"
test_fp  = "/vscratch/kmangano/KO_simulation/sergio/sergio_TGFB1/preprocessed_sergio_TGFB1_nt_only.h5ad_20251105_110510/sergio_TGFB1_nt_only.h5ad_test.h5ad"

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
