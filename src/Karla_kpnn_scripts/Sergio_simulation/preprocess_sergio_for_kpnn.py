# preprocess_sergio_for_kpnn.py
from pathlib import Path
import anndata as ad
from scipy.sparse import csr_matrix, issparse
import numpy as np
import kpnn

# ==== PATHS & SETTINGS (edit these) ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SERGIO_RUN_DIR = PROJECT_ROOT / "results_t1" / "KO_simulation" / "sergio" / "sergio_TNFA"

IN_H5AD   = _SERGIO_RUN_DIR / "sergio_TNFA_nt_only.h5ad"
OUT_DIR   = _SERGIO_RUN_DIR   # writes train/val/test here
OVERWRITE_INPUT = True   # if True, write the sparse-repaired h5ad back to IN_H5AD

SPLIT_RATIO     = "80:10:10"
RANDOM_SEED     = 9999
CLASS_COLUMN    = "cell_type"   # expects "A"/"B" strings
TPM_NORMALIZE   = False         # SERGIO: usually False
MINMAX_NORMALIZE = True
# ============================================================================


def ensure_sparse_and_labels(in_h5ad: Path, overwrite: bool = True) -> Path:
    """Make .X sparse if needed, ensure CLASS_COLUMN exists; write back."""
    print(f"📂 Loading: {in_h5ad}")
    adata = ad.read_h5ad(in_h5ad)

    # Ensure labels present
    if CLASS_COLUMN not in adata.obs.columns:
        if "y" in adata.obs.columns:
            y = adata.obs["y"].to_numpy()
            # Map 0→A, 1→B by default
            adata.obs[CLASS_COLUMN] = np.where(y == 0, "A", "B")
            print(f"🧭 Added '{CLASS_COLUMN}' from 'y'.")
        else:
            raise ValueError(f"Need '{CLASS_COLUMN}' in .obs or a numeric 'y' to derive it.")

    # Ensure sparse matrix
    if not issparse(adata.X):
        print("🧱 adata.X is dense → converting to CSR sparse …")
        adata.X = csr_matrix(adata.X)
        print("✅ Converted to CSR.")

    # Write back
    out_path = in_h5ad if overwrite else in_h5ad.with_name(in_h5ad.stem + "_sparse.h5ad")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write(out_path, compression="gzip")
    print(f"💾 Saved sparse .h5ad to: {out_path}")
    return out_path

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    repaired_path = ensure_sparse_and_labels(IN_H5AD, overwrite=OVERWRITE_INPUT)

    print("⚙️ Running KPNN preprocessing …")
    kpnn.preprocess_data(
        data_filepath=repaired_path,
        split_ratio=SPLIT_RATIO,
        tpm_normalize=TPM_NORMALIZE,
        min_max_normalize=MINMAX_NORMALIZE,
        compress=True,
        random_seed=RANDOM_SEED,
        class_column=CLASS_COLUMN,
        output_path=OUT_DIR,
    )
    print(f"✅ Preprocessing complete. Splits written to: {OUT_DIR}")

if __name__ == "__main__":
    main()
