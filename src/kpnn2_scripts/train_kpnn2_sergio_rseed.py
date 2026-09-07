"""kpnn2 port of train_kpnn_sergio_rseed.py.

Same inputs, same output layout, same hyperparameters -- but the network is
built with kpnn2 and trained by the loop in kpnn2_training.py instead of by
kpnn.train_kpnns(). See that module's docstring for what was mirrored from
kpnn v1 and where the two necessarily differ.

Reads train/val/test splits that are already preprocessed, from DATA_DIR, so
the SERGIO simulation, .h5ad conversion and KPNN preprocessing are all skipped.

Environment variables
---------------------
SERGIO_COND     condition to run (default TGFB1)
RESULTS_DIR     output root (default results_kpnn2)
DATA_DIR        folder holding the sergio_<COND> split folders
KPNN2_SEEDS     number of random seeds (default 20)
KPNN2_DETERMINISTIC  set to 0 to skip torch.use_deterministic_algorithms
"""
import os

# ============================================================
# Reproducibility settings that must be configured BEFORE
# importing numpy / torch / pandas / anndata
# ============================================================

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


# ===== imports =====

import random
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kpnn2_training


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

# Require deterministic PyTorch algorithms. Captum's gradient path can hit a
# kernel with no deterministic implementation; set KPNN2_DETERMINISTIC=0 to
# fall back rather than crash mid-run.
if os.environ.get("KPNN2_DETERMINISTIC", "1") != "0":
    torch.use_deterministic_algorithms(True)

# ===== paths =====
COND = os.environ.get("SERGIO_COND", "TGFB1")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results_kpnn2")
DATA_DIR = os.environ.get("DATA_DIR", "Sergio_Data_Karla_new")

project_root = Path(__file__).resolve().parents[2]
edgelist_path = project_root / f"src/Karla_kpnn_csv/{COND}_tf_only_edgelist_filtered.tsv"
config_yaml = (
    project_root / "src/Karla_kpnn_scripts/Sergio_simulation/kpnn_config.yaml"
)
output_path = (
    project_root / f"{RESULTS_DIR}/kpnn_output/sergio/{COND}_o_A_vs_B/alter_nothing"
)

# explicit split files (already decompressed/renamed to .h5ad).
# Filenames differ between conditions -- sergio_IFNB_nt_only.h5ad_train.h5ad vs
# sergio_INS_nt_only_train.h5ad -- so match on the common suffix.
split_dir = project_root / DATA_DIR / f"sergio_{COND}"


def _find_split(suffix: str) -> str:
    hits = sorted(split_dir.glob(f"*_{suffix}.h5ad"))
    if len(hits) != 1:
        raise FileNotFoundError(
            f"expected exactly one *_{suffix}.h5ad in {split_dir}, "
            f"found {[h.name for h in hits]}"
        )
    return str(hits[0])


train_fp = _find_split("train")
val_fp = _find_split("val")
test_fp = _find_split("test")
print(f"[{COND}] train={Path(train_fp).name}  val={Path(val_fp).name}  test={Path(test_fp).name}")

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
kpnn2_training.run(
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
