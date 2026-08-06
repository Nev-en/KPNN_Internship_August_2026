
import pandas as pd
import numpy as np
from typing import Literal
import anndata as ad
import decoupler as dc
from pathlib import Path
import argparse
from predict_tf_ko_effect import predict_tf_ko_effect
# Main function definition -------------------------------------------------------------------------


def compute_tf_activities(
    X: pd.DataFrame,
    W: pd.DataFrame,
    method: Literal["ulm", "mlm", "auc"] = "ulm",
    tmin: int = 0,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute transcription factor (TF) activities from expression data using decoupler.

    Parameters
    ----------
    X : pandas.DataFrame, shape (N, G)
        Expression data (samples × genes). Columns are gene names.
    W : pandas.DataFrame, shape (G, T)
        Wiring matrix (genes × TFs).
        - The **index** must contain gene names that exactly match `X.columns`
          in both content and order.
        - The **columns** must contain the TF names (strings), which will be
          used as the TF identifiers in the output.
        - Values can be binary (0/1) or weighted (positive or signed weights).

    method : {"ulm", "mlm", "auc"}, default "ulm"
        Scoring method to use in decoupler.
    tmin : int, default 0
        Minimum number of targets per TF to keep.

    Returns
    -------
    A : np.ndarray, shape (N, T_kept)
        TF activity matrix.
    tf_names_out : list of str
        Names of the TFs kept (columns of A).
    """
    _validate_tf_activity_inputs(
        X=X,
        W=W,
        tmin=tmin
    )

    # --- Build net from W ---
    rows = []
    for tf in W.columns:
        idx = np.flatnonzero(W[tf].to_numpy())
        if len(idx) < tmin:
            continue
        rows.append(pd.DataFrame({
            "source": tf,
            "target": X.columns[idx],
            "weight": W.iloc[idx][tf].astype(float).to_numpy()
        }))
    if not rows:
        raise ValueError("No TFs have at least tmin targets in W.")
    net = pd.concat(rows, ignore_index=True)

    # --- Build AnnData ---
    adata = ad.AnnData(
        X=X.values,
        obs=pd.DataFrame(index=X.index.astype(str)),
        var=pd.DataFrame(index=X.columns.astype(str)),
    )

    # --- Score activities ---
    key_map = {"ulm": "score_ulm", "mlm": "score_mlm", "auc": "score_aucell"}
    method = method.lower()
    if method not in key_map:
        raise ValueError(f"method must be one of: {tuple(key_map)}")

    getattr(dc.mt, method)(data=adata, net=net, tmin=tmin, verbose=False)
    scores = dc.pp.get_obsm(adata, key=key_map[method])

    # Ensure dense output
    Xs = scores.X if isinstance(scores.X, np.ndarray) else scores.X.toarray()
    return Xs, list(scores.var_names)


# Level 1 function definitions ---------------------------------------------------------------------


def _validate_tf_activity_inputs(
    X: pd.DataFrame,
    W: pd.DataFrame,
    tmin: int
) -> None:
    """
    Validate inputs for compute_tf_activities when X and W are DataFrames.

    Parameters
    ----------
    X : pandas.DataFrame
        Expression data (samples × genes). Columns = gene names.
    W : pandas.DataFrame
        Wiring matrix (genes × TFs). Index = gene names, columns = TF names.
    tmin : int
        Minimum targets per TF to keep.

    Raises
    ------
    ValueError
        If any input check fails.
    """

    # ---------- Type checks ----------
    if not isinstance(X, pd.DataFrame):
        raise ValueError(f"X must be a pandas DataFrame (got {type(X).__name__}).")
    if not isinstance(W, pd.DataFrame):
        raise ValueError(f"W must be a pandas DataFrame (got {type(W).__name__}).")
    if not isinstance(tmin, (int, np.integer)) or tmin < 0:
        raise ValueError(f"tmin must be a non-negative integer (got {tmin!r}).")

    # ---------- Shape & dimension checks ----------
    if X.shape[1] == 0 or X.shape[0] == 0:
        raise ValueError(f"X must have at least 1 sample and 1 gene (got shape {X.shape}).")
    if W.shape[0] == 0 or W.shape[1] == 0:
        raise ValueError(f"W must have at least 1 gene and 1 TF (got shape {W.shape}).")

    # ---------- Name alignment checks ----------
    if not X.columns.equals(W.index):
        diff_X = set(X.columns) - set(W.index)
        diff_W = set(W.index) - set(X.columns)
        msg_parts = []
        if diff_X:
            msg_parts.append(
                f"{len(diff_X)} genes in X not found in W index: {sorted(list(diff_X))[:5]}..."
            )
        if diff_W:
            msg_parts.append(
                f"{len(diff_W)} genes in W index not found in X: {sorted(list(diff_W))[:5]}..."
            )
        raise ValueError(
            "Gene names in X.columns and W.index must match exactly and be in the same order.\n"
            + "\n".join(msg_parts)
        )

    # ---------- TF name checks ----------
    if W.columns.hasnans:
        raise ValueError("W.columns (TF names) contain NaN values.")
    if not all(isinstance(tf, str) for tf in W.columns):
        bad_types = {type(tf).__name__ for tf in W.columns if not isinstance(tf, str)}
        raise ValueError(f"All TF names (W.columns) must be strings (found types: {bad_types}).")
    if len(set(W.columns)) != len(W.columns):
        raise ValueError("W.columns (TF names) must be unique.")

    # ---------- Value checks ----------
    if not np.isfinite(W.to_numpy()).all():
        raise ValueError("W contains NaN or Inf values.")
    if not np.issubdtype(W.to_numpy().dtype, np.number):
        raise ValueError(f"W must be numeric (got dtype {W.dtypes}).")

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("X contains NaN or Inf values.")
    if not np.issubdtype(X.to_numpy().dtype, np.number):
        raise ValueError(f"X must be numeric (got dtypes: {X.dtypes.unique().tolist()}).")

    # ---------- tmin feasibility ----------
    tf_target_counts = (W != 0).sum(axis=0)
    if (tf_target_counts < tmin).all():
        raise ValueError(
            f"No TFs have at least tmin={tmin} nonzero targets.\n"
            f"Max target count among TFs = {tf_target_counts.max()}."
        )

    # Passed all checks
    return None

# -------------------------------------------------------------------------
# Driver block to run the analysis
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Compute EFFECT scores from TF activities.")
    parser.add_argument("--in_dir", required=True, type=Path,
                        help="Input dir containing X.csv, y.csv, W.csv")
    parser.add_argument("--out_csv", type=Path, default=None,
                        help="Output CSV path (default: in_dir/closed_form_ko_screen_decoupler_both.csv)")
    args = parser.parse_args()

    X = pd.read_csv(args.in_dir / "X.csv")
    y = pd.read_csv(args.in_dir / "y.csv").iloc[:, 0].astype(int)
    W = pd.read_csv(args.in_dir / "W.csv", index_col=0)
    W = W.drop(columns=[c for c in W.columns if c.lower() == "output"], errors="ignore")

    # Align genes
    if not X.columns.equals(W.index):
        common = [g for g in X.columns if g in W.index]
        X = X[common]
        W = W.loc[common]

    print(f"Loaded: X {X.shape}, W {W.shape}, y {y.shape}")

    A, tf_names = compute_tf_activities(X=X, W=W, method="ulm", tmin=0)
    print(f"A shape: {A.shape} (#TFs kept = {len(tf_names)})")

    df_all = predict_tf_ko_effect(
        A=A, y=y.to_numpy(), tf_names=tf_names,
        ko_scope="both", geometry="pooled",
        k_max=3, test_size=0.2, split_seed=777,
        base_lam=0.2, fallback_euclidean=True,
    )

    out_csv = args.out_csv or (args.in_dir / "closed_form_ko_screen_decoupler_both.csv")
    df_all.to_csv(out_csv, index=False)
    print(f"✅ Saved: {out_csv}")

if __name__ == "__main__":
    main()
