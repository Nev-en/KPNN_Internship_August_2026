# make_adata_from_sergio.py
import os
from pathlib import Path
import pandas as pd
import numpy as np
import anndata as ad

# === paths ===
# COND / RESULTS_DIR can be overridden via environment variables so the same
# script can be driven over all conditions without editing it each time.
COND = os.environ.get("SERGIO_COND", "TNFA")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results_t1_2")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sergio_dir = PROJECT_ROOT / RESULTS_DIR / "KO_simulation" / "sergio" / f"sergio_{COND}"
x_path = sergio_dir / "X_nt.csv"
y_path = sergio_dir / "y_nt.csv"   # will be optional

# === load ===
X = pd.read_csv(x_path)                  # cells × genes
genes = list(X.columns)

if y_path.exists():
    y = pd.read_csv(y_path)["y"].astype(int).to_numpy()
else:
    # fallback: assume 2 equal bins
    n = len(X)
    y = np.array([0]*(n//2) + [1]*(n - n//2), dtype=int)

# === labels ===
cell_type = np.where(y == 0, "A", "B")

# === make AnnData ===
adata = ad.AnnData(
    X=X.values,
    obs={"cell_type": cell_type, "y": y},
    var=pd.DataFrame(index=genes)
)

# === write ===
out_h5ad = sergio_dir / f"sergio_{COND}_nt_only.h5ad"
adata.write(out_h5ad, compression="gzip")
print("✅ Wrote:", out_h5ad)
