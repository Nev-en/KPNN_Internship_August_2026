# make_adata_from_sergio.py
from pathlib import Path
import pandas as pd
import numpy as np
import anndata as ad

# === paths ===
sergio_dir = Path("C:/Users/durie/PycharmProjects/PythonProject/results_t1/KO_simulation/sergio/sergio_TNFA")
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
out_h5ad = sergio_dir / "sergio_TNFA_nt_only.h5ad"
adata.write(out_h5ad, compression="gzip")
print("✅ Wrote:", out_h5ad)
