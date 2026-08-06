import itertools as it
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from math import ceil
from typing import TypedDict, Tuple, Optional, Sequence, Literal


# Main function definition -------------------------------------------------------------------------


def predict_tf_ko_effect_old(
    *,    # keyword-only
    A: Optional[np.ndarray] = None,
    X: Optional[np.ndarray] = None,
    W: Optional[np.ndarray] = None,
    use_relu: bool = True,
    y: np.ndarray,
    geometry: str = "tgt",
    tf_names: Optional[Sequence[str]] = None,
    k_max: int = 3,
    max_combos_per_k: Optional[int] = None,
    candidate_pool_multiplier: float = 3.0,
    use_euclidean_prescreen: bool = True,
    test_size: float = 0.2,
    split_seed: int = 777,
    base_lam: float = 0.2,
    fallback_euclidean: bool = True,
) -> pd.DataFrame:
    """
    Closed-form TF knockout (KO) importance without retraining — in TF-activity space.

    This function estimates how critical each transcription factor (TF), and
    combinations of TFs, are for separating two conditions (binary labels y∈{0,1})
    by measuring the drop in class separation when those TFs are “knocked out”.
    Everything is computed in closed form on a held-out test split;

    You can provide either:
      • per-sample TF activities A (N×T), e.g. from decoupler/DoRothEA, or
      • gene expression X (N×G) with a wiring matrix W (G×T), in which case the
        TF space is H = X @ W (optionally passed through ReLU).

    Separation is measured as a Mahalanobis distance between class means in TF space:
        d = sqrt( Δᵀ Σ⁻¹ Δ ),
    where Δ is the difference of test-set class means in TF space and Σ is a
    covariance matrix that defines the geometry. Which Σ is used is controlled by
    `geometry`:

      geometry = 'tgt'    → use Σ₁, the covariance of class 1 (target, e.g. drug). Directional.
      geometry = 'ref'    → use Σ₀, the covariance of class 0 (reference, e.g. control).
                                    Directional.
      geometry = 'pooled' → use pooled covariance across both classes. Symmetric.

    A KO of a TF set S is modeled by zeroing the corresponding coordinates of Δ:
    equivalently, you “set the target class mean to the reference mean” on those TFs.
    The KO effect is the drop in distance:
        effect(S) = d_base − d_S.
    Positive effect ⇒ KO blunts the phenotype of class 1 under the chosen geometry.
    (If you want enhancement scores, inspect d_S − d_base.)

    Parameters
    ----------
    A : Optional[np.ndarray], shape (N, T)
        Per-sample TF activity matrix. Provide this OR (X and W).
    X : Optional[np.ndarray], shape (N, G)
        Gene expression. Provide with W if A is not supplied.
    W : Optional[np.ndarray], shape (G, T)
        Gene→TF wiring (binary or weighted). Used to build H = X @ W.
    use_relu : bool, default True
        If True and using (X, W), apply ReLU to H (clips negative TF activities).
        Leave False if you expect repressors / signed activities.
    y : np.ndarray, shape (N,)
        Binary labels {0,1}. Class 1 is treated as the “target/drug” class.
    geometry : {'tgt','ref','pooled'}, default 'tgt'
        Which covariance to use for the metric:
          'tgt'  → Σ₁ (class-1 covariance), directional toward class 1
          'ref'  → Σ₀ (class-0 covariance), directional toward class 0
          'pooled' → pooled Σ, symmetric (original behavior)
    tf_names : Optional[Sequence[str]]
        Names for the T TFs (used in output labels). If None, uses 'TF_i'.
    k_max : int, default 3
        Evaluate all KO combinations up to size k_max (singles, pairs, triples…).
    max_combos_per_k : Optional[int], default None
        If None, evaluate all combinations of size k (can be expensive).
        If set (e.g., 200), evaluate at most that many per k after prescreening.
    candidate_pool_multiplier : float, default 3.0
        When max_combos_per_k is set, build a candidate pool of approximately
        ceil(multiplier * max_combos_per_k) TFs from top single-KO effects, and
        form k-combinations only from that pool. Keeps runtime manageable.
    use_euclidean_prescreen : bool, default True
        If too many combos remain for a given k, rank them cheaply using the
        Euclidean surrogate (depends only on Δ) and keep the top ones for the
        exact Mahalanobis evaluation.
    test_size : float, default 0.2
        Fraction of samples reserved for the test split (stratified).
    split_seed : int, default 777
        Random seed for the stratified split.
    base_lam : float, default 0.2
        Ridge/shrinkage intensity λ for Σ̂:  Σ̂_λ = (1−λ)Σ̂ + λ·τI, τ=trace(Σ̂)/T.
        Heuristics increase λ if class sizes are small or Σ̂ is ill-conditioned.
    fallback_euclidean : bool, default True
        If Σ̂_λ inversion fails, fall back to Euclidean distance.

    Returns
    -------
    pd.DataFrame
        One row for the baseline (k=0) and for each KO combo up to k_max with:
          - k : int
                Size of the KO set (0 for baseline).
          - combo_idx : tuple[int]
                TF indices in the KO set.
          - combo_label : str
                Human-readable label (e.g., "TF_A+TF_B").
          - sep : float
                Separation d_S after applying the KO.
          - effect : float
                Drop vs baseline: d_base − d_S (positive ⇒ KO reduces separation).
          - method : str
                Metric used (e.g., "mahalanobis[tgt] (λ=0.30)" or Euclidean fallback).
          - lambda_used : float
                Final λ used for shrinkage.
          - geometry : str
                Echo of the chosen geometry ('tgt'|'ref'|'pooled').

    Algorithm (high-level)
    ----------------------
    1) Build TF-space matrix:
         • Use A if provided; else compute H = X @ W (optionally ReLU).
    2) Stratified hold-out split; operate only on the test set.
    3) Compute class means μ₀, μ₁ and Δ = μ₁ − μ₀ (direction is fixed toward class 1).
    4) Choose covariance Σ according to `geometry`:
         • 'tgt' → cov of class-1 test samples; 'ref' → cov of class-0;
           'pooled' → pooled cov across both classes.
       Apply shrinkage Σ̂_λ for stability (λ auto-bumped when needed).
    5) Baseline separation: d_base = sqrt(Δᵀ Σ̂_λ⁻¹ Δ).
    6) For each KO set S up to k_max:
         • Zero Δ on coords in S (equivalently, set μ₁[S]=μ₀[S]).
         • Compute d_S with the same Σ̂_λ.
         • effect = d_base − d_S.
       Use candidate-pool + Euclidean prescreening if you limited combos per k.

    Interpretation
    --------------
    • Positive effect ⇒ those TFs are important for the *target phenotype*
      under the chosen geometry (their KO moves class-1 closer to class-0).
    • 'pooled' geometry is symmetric (class-agnostic). 'tgt' and 'ref' are
      class-specific and can yield different rankings.
    • This is the closed-form analogue of LDA-style separation in TF space; it
      avoids retraining and is extremely fast.

    Practical tips
    --------------
    • If using (X, W), prefer signed/weighted W and consider column-normalizing W
      (e.g., L1) to avoid “many-targets advantage”. Avoid ReLU if repressors matter.
    • If Σ̂_λ is ill-conditioned, increasing λ (e.g., 0.3–0.5) usually helps.
    • Complexity grows combinatorially with k when computing all combos; use
      `max_combos_per_k` + prescreening for large T.

    Notes
    -----
    This method estimates *directional importance toward class 1* when
    geometry='tgt' (or toward class 0 with 'ref'). To report enhancement effects,
    inspect (sep − baseline) instead of (baseline − sep).
    """
    _validate_predict_tf_ko_effect_inputs(
        A=A,
        X=X,
        W=W,
        use_relu=use_relu,
        y=y,
        geometry=geometry,
        tf_names=tf_names,
        k_max=k_max,
        max_combos_per_k=max_combos_per_k,
        candidate_pool_multiplier=candidate_pool_multiplier,
        use_euclidean_prescreen=use_euclidean_prescreen,
        test_size=test_size,
        split_seed=split_seed,
        base_lam=base_lam,
        fallback_euclidean=fallback_euclidean,
    )

    # ----- construct TF-space -----
    if A is None:
        if X is None or W is None:
            raise ValueError("Provide either A & y, OR X, W & y.")
        X = np.asarray(X, float)
        W = np.asarray(W, float)
        H_all = X @ W
        if use_relu:
            H_all = np.maximum(H_all, 0.0)
    else:
        H_all = np.asarray(A, float)

    y = np.asarray(y).astype(int)
    geometry = str(geometry).lower()

    N, T = H_all.shape
    if tf_names is None:
        tf_names = [f"TF_{i}" for i in range(T)]
    else:
        tf_names = list(tf_names)
        if len(tf_names) != T:
            raise ValueError(f"len(tf_names) ({len(tf_names)}) must equal T ({T}).")

    # ----- holdout -----
    _, te_idx = train_test_split(
        np.arange(N), test_size=test_size, random_state=split_seed, stratify=y
    )
    H = H_all[te_idx]
    yte = y[te_idx]

    # ----- class means & directional delta (class 1 is the target) -----
    mu0 = H[yte == 0].mean(axis=0)
    mu1 = H[yte == 1].mean(axis=0)
    delta = (mu1 - mu0).astype(float)  # direction: control → target

    if np.allclose(delta, 0.0, atol=1e-12):
        return pd.DataFrame([{
            "k": 0, "combo_idx": (), "combo_label": "No KO",
            "sep": 0.0, "effect": 0.0,
            "method": "degenerate (no separation)",
            "lambda_used": np.nan,
            "geometry": geometry,
        }])

    # ----- choose covariance (metric geometry) -----
    def _class_cov(Hc: np.ndarray) -> np.ndarray:
        if len(Hc) <= 1:
            return np.cov(H, rowvar=False)
        return np.cov(Hc, rowvar=False)

    if geometry == "pooled":
        C_base = _pooled_cov(H, yte)
    elif geometry == "tgt":
        C_base = _class_cov(H[yte == 1])
    elif geometry == "ref":
        C_base = _class_cov(H[yte == 0])
    else:
        C_base = _pooled_cov(H, yte)

    # ----- shrinkage -----
    lam = float(base_lam)
    n0 = int(np.sum(yte == 0))
    n1 = int(np.sum(yte == 1))
    if min(n0, n1) < 5 * T:
        lam = max(lam, 0.3)
    try:
        cond = np.linalg.cond(C_base)
        if np.isfinite(cond) and cond > 1e6:
            lam = max(lam, 0.5)
    except Exception:
        lam = max(lam, 0.5)
    C_shr = _shrink(C_base, lam)

    # ----- baseline distance & sep function -----
    keep_all = np.ones(T, dtype=bool)
    method_used = f"mahalanobis[{geometry}] (λ={lam:.2f})"
    try:
        d0 = _mahalanobis_sep(delta, C_shr, keep_all)
        sep_exact = lambda m: _mahalanobis_sep(delta, C_shr, m)
    except Exception:
        if not fallback_euclidean:
            raise
        d0 = _euclidean_sep(delta, keep_all)
        sep_exact = lambda m: _euclidean_sep(delta, m)
        method_used = f"euclidean (fallback; geometry={geometry})"

    class KORow(TypedDict):
        k: int
        combo_idx: Tuple[int, ...]
        combo_label: str
        sep: float
        effect: float
        method: str
        lambda_used: float
        geometry: str

    rows: list[KORow] = [{
        "k": 0,
        "combo_idx": (),
        "combo_label": "No KO",
        "sep": float(d0),
        "effect": 0.0,
        "method": method_used,
        "lambda_used": lam,
        "geometry": geometry,
    }]

    # Euclidean surrogate (for prescreen)
    delta_sq = delta ** 2

    # ----- k=1: exact for all -----
    single_effects = np.empty(T, dtype=float)
    for i in range(T):
        keep = np.ones(T, dtype=bool); keep[i] = False
        dS = sep_exact(keep)
        eff = float(d0 - dS)
        single_effects[i] = eff
        rows.append({
            "k": 1,
            "combo_idx": (i,),
            "combo_label": str(tf_names[i]),
            "sep": float(dS),
            "effect": eff,
            "method": method_used,
            "lambda_used": lam,
            "geometry": geometry,
        })

    # ----- candidate pool for k>=2 -----
    if max_combos_per_k is not None:
        pool_size = min(T, max(1, int(np.ceil(candidate_pool_multiplier * max_combos_per_k))))
        cand_idx = np.argsort(-single_effects)[:pool_size]
    else:
        cand_idx = np.arange(T)

    # ----- k=2..k_max -----
    for k in range(2, min(k_max, T) + 1):
        combos = list(it.combinations(cand_idx.tolist(), k))

        if max_combos_per_k is not None and use_euclidean_prescreen and len(combos) > max_combos_per_k:
            sums = np.array([delta_sq[list(c)].sum() for c in combos], dtype=float)
            top_idx = np.argsort(-sums)[:max_combos_per_k]
            combos = [combos[i] for i in top_idx]
        elif max_combos_per_k is not None and len(combos) > max_combos_per_k:
            combos = combos[:max_combos_per_k]

        for combo in combos:
            keep = np.ones(T, dtype=bool); keep[list(combo)] = False
            try:
                dS = sep_exact(keep)
                method_row = method_used
            except Exception:
                dS = _euclidean_sep(delta, keep)
                method_row = f"euclidean (combo fallback; geometry={geometry})"
            rows.append({
                "k": k,
                "combo_idx": combo,
                "combo_label": "+".join([str(tf_names[i]) for i in combo]),
                "sep": float(dS),
                "effect": float(d0 - dS),  # >0 ⇒ KO blunts separation toward class 1 under chosen geometry
                "method": method_row,
                "lambda_used": lam,
                "geometry": geometry,
            })

    return pd.DataFrame(rows)


def predict_tf_ko_effect(
    *,    # keyword-only
    A: Optional[np.ndarray] = None,
    X: Optional[np.ndarray] = None,
    W: Optional[np.ndarray] = None,
    use_relu: bool = True,
    y: np.ndarray,
    ko_scope: Literal["both", "target", "ref"] = "both",
    geometry: Literal["auto", "pooled", "tgt", "ref"] = "auto",
    tf_names: Optional[Sequence[str]] = None,
    k_max: int = 3,
    max_combos_per_k: Optional[int] = None,
    candidate_pool_multiplier: float = 3.0,
    use_euclidean_prescreen: bool = True,
    test_size: float = 0.2,
    split_seed: int = 777,
    base_lam: float = 0.2,
    fallback_euclidean: bool = True,
) -> pd.DataFrame:
    """
    Closed-form TF knockout (KO) influence without retraining — in TF-activity space.

    This function estimates how critical each transcription factor (TF), and
    combinations of TFs, are for separating two conditions (binary labels y∈{0,1})
    by measuring the drop in class separation when those TFs are “knocked out”.
    Everything is computed in closed form on a held-out test split.

    You can provide either:
      • per-sample TF activities A (N×T), e.g. from decoupler/DoRothEA, or
      • gene expression X (N×G) with a wiring matrix W (G×T), in which case the
        TF space is H = X @ W (optionally passed through ReLU).

    Separation is measured as a Mahalanobis distance between class means in TF space:
        d = sqrt( Δᵀ Σ⁻¹ Δ ),
    where Δ is the difference of test-set class means and Σ is a covariance matrix
    that defines the geometry. Geometry is either explicitly chosen (`geometry`)
    or inferred when `geometry="auto"`.

    KO scope
    --------
    The new `ko_scope` argument controls in which class(es) the KO is applied:
      • "both"   → remove TF activity in both classes before computing separation
      • "target" → remove TF activity only in the target class (y=1)
      • "ref"    → remove TF activity only in the reference class (y=0)

    KO effect is the drop in distance:
        effect(S) = d_base − d_S
    Positive effect ⇒ KO reduces the separation under the given scope/geometry.

    Parameters
    ----------
    A : Optional[np.ndarray], shape (N, T)
        Per-sample TF activity matrix. Provide this OR (X and W).
    X : Optional[np.ndarray], shape (N, G)
        Gene expression. Provide with W if A is not supplied.
    W : Optional[np.ndarray], shape (G, T)
        Gene→TF wiring (binary or weighted). Used to build H = X @ W.
    use_relu : bool, default True
        If True and using (X, W), apply ReLU to H. Set False if signed activities matter.
    y : np.ndarray, shape (N,)
        Binary labels {0,1}. Class 1 is the “target/drug” class.
    geometry : {'tgt','ref','pooled','auto'}, default 'auto'
        Covariance choice:
          'tgt'    → Σ₁ (class-1 covariance), directional toward class 1
          'ref'    → Σ₀ (class-0 covariance), directional toward class 0
          'pooled' → pooled Σ, symmetric
          'auto'   → pick based on `ko_scope` ('both' → pooled, 'target' → tgt, 'ref' → ref)
    ko_scope : Literal["both","target","ref"], default "both"
        Which class(es) receive the KO before measuring separation.
    tf_names : Optional[Sequence[str]]
        Names for the T TFs. If None, uses "TF_i".
    k_max : int, default 3
        Evaluate all KO combinations up to size k_max.
    max_combos_per_k : Optional[int], default None
        Limit the number of combos per k after prescreening.
    candidate_pool_multiplier : float, default 3.0
        Size multiplier for candidate pool when prescreening.
    use_euclidean_prescreen : bool, default True
        Use Euclidean surrogate to rank large combo sets before exact evaluation.
    test_size : float, default 0.2
        Fraction of samples reserved for test split (stratified).
    split_seed : int, default 777
        Random seed for the stratified split.
    base_lam : float, default 0.2
        Ridge/shrinkage intensity λ for Σ̂.
    fallback_euclidean : bool, default True
        Fall back to Euclidean distance if Σ̂ inversion fails.

    Returns
    -------
    pd.DataFrame
        One row for the baseline (k=0) and for each KO combo up to k_max with:
          - k : KO set size
          - combo_idx : TF indices in KO
          - combo_label : human-readable label
          - sep : separation after KO
          - effect : drop vs baseline
          - method : metric used
          - lambda_used : λ used for shrinkage
          - geometry : geometry actually used

    Interpretation
    --------------
    • Positive effect ⇒ those TFs contribute to the phenotype separation
      under the given KO scope and geometry.
    • 'both' KO scope isolates a TF’s *overall* contribution to separation,
      independent of which class it acts in.
    • 'target' and 'ref' KO scopes capture class-specific influence.

    Notes
    -----
    This is the closed-form analogue of an LDA-style separation analysis in TF space.
    It avoids retraining and runs in milliseconds for small–moderate T.
    """
    geometry = _resolve_geometry(
        geometry,
        ko_scope
    )

    _validate_predict_tf_ko_effect_inputs(
        A=A,
        X=X,
        W=W,
        use_relu=use_relu,
        y=y,
        geometry=geometry,
        tf_names=tf_names,
        k_max=k_max,
        max_combos_per_k=max_combos_per_k,
        candidate_pool_multiplier=candidate_pool_multiplier,
        use_euclidean_prescreen=use_euclidean_prescreen,
        test_size=test_size,
        split_seed=split_seed,
        base_lam=base_lam,
        fallback_euclidean=fallback_euclidean,
    )



    # ---------- construct TF-space ----------
    if A is None:
        if X is None or W is None:
            raise ValueError("Provide either A & y, OR X, W & y.")
        X = np.asarray(X, float)
        W = np.asarray(W, float)
        H_all = X @ W
        if use_relu:
            H_all = np.maximum(H_all, 0.0)
    else:
        H_all = np.asarray(A, float)

    y = np.asarray(y).astype(int)

    N, T = H_all.shape
    if tf_names is None:
        tf_names = [f"TF_{i}" for i in range(T)]
    else:
        tf_names = list(tf_names)
        if len(tf_names) != T:
            raise ValueError(f"len(tf_names) ({len(tf_names)}) must equal T ({T}).")

    # ---------- holdout ----------
    _, te_idx = train_test_split(
        np.arange(N), test_size=test_size, random_state=split_seed, stratify=y
    )
    H = H_all[te_idx]
    yte = y[te_idx]

    # ---------- baseline μ, Δ ----------
    H0 = H[yte == 0]
    H1 = H[yte == 1]
    mu0 = H0.mean(axis=0)
    mu1 = H1.mean(axis=0)
    delta = (mu1 - mu0).astype(float)

    if np.allclose(delta, 0.0, atol=1e-12):
        return pd.DataFrame([{
            "k": 0, "combo_idx": (), "combo_label": "No KO",
            "sep": 0.0, "effect": 0.0,
            "method": "degenerate (no separation)",
            "lambda_used": np.nan,
            "geometry": geometry,
        }])

    # Baseline covariance by geometry
    if geometry == "pooled":
        C_base = _pooled_cov(H, yte)
    elif geometry == "tgt":
        C_base = _class_cov(H1, H)
    else:                               # geometry == "ref"
        C_base = _class_cov(H0, H)

    # ---------- shrinkage (baseline) ----------
    lam = float(base_lam)
    n0 = int(np.sum(yte == 0))
    n1 = int(np.sum(yte == 1))
    if min(n0, n1) < 5 * T:
        lam = max(lam, 0.3)
    try:
        cond = np.linalg.cond(C_base)
        if np.isfinite(cond) and cond > 1e6:
            lam = max(lam, 0.5)
    except Exception:
        lam = max(lam, 0.5)
    C_shr = _shrink(C_base, lam)

    # ---------- baseline distance ----------
    keep_all = np.ones(T, dtype=bool)
    method_used_base = f"mahalanobis[{geometry}] (λ={lam:.2f})"
    try:
        d0 = _mahalanobis_sep(delta, C_shr, keep_all)
    except Exception:
        if not fallback_euclidean:
            raise
        d0 = _euclidean_sep(delta, keep_all)
        method_used_base = f"euclidean (fallback; geometry={geometry})"

    class KORow(TypedDict):
        k: int
        combo_idx: Tuple[int, ...]
        combo_label: str
        sep: float
        effect: float
        method: str
        lambda_used: float
        geometry: str

    rows: list[KORow] = [{
        "k": 0,
        "combo_idx": (),
        "combo_label": "No KO",
        "sep": float(d0),
        "effect": 0.0,
        "method": method_used_base,
        "lambda_used": lam,
        "geometry": geometry,
    }]

    # Cheap surrogate for prescreen (masking Δ, same as before)
    delta_sq = delta ** 2

    # ---------- function to compute separation after KO on selected class(es) ----------
    def _sep_after_ko(combo_idx: Tuple[int, ...]) -> Tuple[float, str]:
        # Copy class matrices
        H0p = H0.copy()
        H1p = H1.copy()
        if ko_scope in ("ref", "both"):
            H0p[:, list(combo_idx)] = 0.0
        if ko_scope in ("target", "both"):
            H1p[:, list(combo_idx)] = 0.0

        mu0p = H0p.mean(axis=0)
        mu1p = H1p.mean(axis=0)
        deltap = (mu1p - mu0p).astype(float)

        # Recompute covariance under chosen geometry on KO-adjusted data
        if geometry == "pooled":
            Hp = np.vstack([H0p, H1p])
            yp = np.concatenate([np.zeros(len(H0p), int), np.ones(len(H1p), int)])
            Cko = _pooled_cov(Hp, yp)
        elif geometry == "tgt":
            Cko = _class_cov(H1p, np.vstack([H0p, H1p]))
        else:  # 'ref'
            Cko = _class_cov(H0p, np.vstack([H0p, H1p]))

        Cko_shr = _shrink(Cko, lam)  # use same λ for fairness/consistency

        try:
            dS = _mahalanobis_sep(deltap, Cko_shr, keep_all)
            method_row = f"mahalanobis[{geometry}] (λ={lam:.2f}; Σ recomputed after KO)"
        except Exception:
            if not fallback_euclidean:
                raise
            dS = _euclidean_sep(deltap, keep_all)
            method_row = f"euclidean (fallback; geometry={geometry}; Σ recomputed after KO)"
        return float(dS), method_row

    # ---------- k=1 (exact for all) ----------
    single_effects = np.empty(T, dtype=float)
    for i in range(T):
        dS, method_row = _sep_after_ko((i,))
        eff = float(d0 - dS)
        single_effects[i] = eff
        rows.append({
            "k": 1,
            "combo_idx": (i,),
            "combo_label": str(tf_names[i]),
            "sep": dS,
            "effect": eff,
            "method": method_row,
            "lambda_used": lam,
            "geometry": geometry,
        })

    # ---------- candidate pool for k>=2 ----------
    if max_combos_per_k is not None:
        pool_size = min(T, max(1, int(np.ceil(candidate_pool_multiplier * max_combos_per_k))))
        cand_idx = np.argsort(-single_effects)[:pool_size]
    else:
        cand_idx = np.arange(T)

    # ---------- k=2..k_max ----------
    for k in range(2, min(k_max, T) + 1):
        combos = list(it.combinations(cand_idx.tolist(), k))

        # Prescreen if too many—use Δ-mask Euclidean surrogate as a fast proxy
        if max_combos_per_k is not None and len(combos) > max_combos_per_k:
            if use_euclidean_prescreen:
                surrogate = np.array([delta_sq[list(c)].sum() for c in combos], dtype=float)
                top_idx = np.argsort(-surrogate)[:max_combos_per_k]
                combos = [combos[i] for i in top_idx]
            else:
                combos = combos[:max_combos_per_k]

        for combo in combos:
            dS, method_row = _sep_after_ko(tuple(combo))
            rows.append({
                "k": k,
                "combo_idx": tuple(combo),
                "combo_label": "+".join([str(tf_names[i]) for i in combo]),
                "sep": dS,
                "effect": float(d0 - dS),
                "method": method_row,
                "lambda_used": lam,
                "geometry": geometry,
            })

    return pd.DataFrame(rows)


# Level 1 function definitions ---------------------------------------------------------------------


def _validate_predict_tf_ko_effect_inputs(
    *,
    A: Optional[np.ndarray],
    X: Optional[np.ndarray],
    W: Optional[np.ndarray],
    use_relu: bool,
    y: np.ndarray,
    geometry: str,
    tf_names: Optional[Sequence[str]],
    k_max: int,
    max_combos_per_k: Optional[int],
    candidate_pool_multiplier: float,
    use_euclidean_prescreen: bool,
    test_size: float,
    split_seed: int,
    base_lam: float,
    fallback_euclidean: bool,
) -> None:
    """
    Validate arguments for predict_tf_ko_effect. Raises ValueError with
    actionable messages if something is off. Returns None on success.
    """

    # ---------- helper checks ----------
    def _as_2d_float(name, arr):
        if arr is None:
            raise ValueError(f"{name} is None.")
        arr = np.asarray(arr)
        if arr.ndim != 2:
            raise ValueError(f"{name} must be 2D (got shape {arr.shape}).")
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError(f"{name} must be numeric (got dtype {arr.dtype}).")
        if not np.isfinite(arr).all():
            bad = np.argwhere(~np.isfinite(arr))
            i, j = bad[0] if bad.size else (-1, -1)
            raise ValueError(f"{name} contains NaN/Inf at position [{i}, {j}].")
        return arr.astype(float)

    def _as_1d_int(name, vec):
        vec = np.asarray(vec)
        if vec.ndim != 1:
            raise ValueError(f"{name} must be 1D (got shape {vec.shape}).")
        if not np.isfinite(vec).all():
            idx = int(np.argwhere(~np.isfinite(vec))[0]) if (~np.isfinite(vec)).any() else -1
            raise ValueError(f"{name} contains NaN/Inf at index {idx}.")
        try:
            vec_i = vec.astype(int)
        except Exception:
            raise ValueError(f"{name} must be integer‑convertible (got dtype {vec.dtype}).")
        return vec_i

    # ---------- input source: A vs X@W ----------
    if (A is None) == (X is None or W is None):
        # True when both branches are invalid (all None) or both provided (ambiguous)
        if A is None:
            raise ValueError("Provide either A & y OR X, W & y (nothing provided).")
        else:
            raise ValueError("Provide either A & y OR X, W & y (not both).")

    # Scalars & ranges
    try:
        test_size = float(test_size)
    except Exception:
        raise ValueError(f"test_size must be a float in (0,1) (got {test_size!r}).")
    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size must be in (0, 1), e.g. 0.2 (got {test_size}).")

    try:
        base_lam = float(base_lam)
    except Exception:
        raise ValueError(f"base_lam must be a float in [0,1) (got {base_lam!r}).")
    if not (0.0 <= base_lam < 1.0):
        raise ValueError(
            f"base_lam must be in [0, 1); values ≥1 invert the shrinkage sign (got {base_lam})."
        )

    if not isinstance(k_max, (int, np.integer)) or k_max < 1:
        raise ValueError(f"k_max must be a positive int (got {k_max}).")

    if max_combos_per_k is not None:
        if not isinstance(max_combos_per_k, (int, np.integer)) or max_combos_per_k < 1:
            raise ValueError(
                f"max_combos_per_k must be a positive int or None (got {max_combos_per_k})."
            )

    try:
        candidate_pool_multiplier = float(candidate_pool_multiplier)
    except Exception:
        raise ValueError(
            f"candidate_pool_multiplier must be a positive float (got "
            f"{candidate_pool_multiplier!r})."
        )
    if candidate_pool_multiplier <= 0:
        raise ValueError(
            f"candidate_pool_multiplier must be > 0 (got {candidate_pool_multiplier})."
        )

    # Geometry
    geometry_norm = str(geometry).lower()
    if geometry_norm not in {"tgt", "ref", "pooled"}:
        raise ValueError("geometry must be one of {'tgt','ref','pooled'} "
                         f"(got {geometry!r}).")

    # y checks (before shape matching we need N)
    y_i = _as_1d_int("y", y)
    uniq = set(np.unique(y_i))
    if not uniq <= {0, 1}:
        raise ValueError(f"y must be binary {{0,1}} (observed labels: {sorted(uniq)}).")

    # Build‑path specific checks and shape compatibility for later operations
    if A is not None:
        A2 = _as_2d_float("A", A)
        N, T = A2.shape
        # tf_names length if provided
        if tf_names is not None:
            if len(tf_names) != T:
                raise ValueError(f"len(tf_names) ({len(tf_names)}) must equal A.shape[1] ({T}).")
    else:
        X2 = _as_2d_float("X", X)
        W2 = _as_2d_float("W", W)
        N, G = X2.shape
        Gw, T = W2.shape
        if G != Gw:
            raise ValueError(f"Shape mismatch for X@W: X is (N,{G}) but W is ({Gw},{T}). "
                             "Require X.shape[1] == W.shape[0].")
        if tf_names is not None and len(tf_names) != T:
            raise ValueError(f"len(tf_names) ({len(tf_names)}) must equal W.shape[1] ({T}).")

    # y length vs N
    if len(y_i) != N:
        raise ValueError(f"len(y) ({len(y_i)}) must equal number of rows N ({N}).")

    # Sample/class count sanity for stratified split
    n0 = int(np.sum(y_i == 0))
    n1 = int(np.sum(y_i == 1))
    if n0 == 0 or n1 == 0:
        raise ValueError(f"Both classes must be present in y for a stratified split "
                         f"(counts: class 0 → {n0}, class 1 → {n1}).")

    # For sklearn's stratified split, we need at least 2 per class overall AND
    # at least 1 sample of each class in both train and test partitions.
    n0_test = ceil(test_size * n0)
    n1_test = ceil(test_size * n1)
    n0_train = n0 - n0_test
    n1_train = n1 - n1_test
    if min(n0, n1) < 2 or min(n0_test, n1_test) < 1 or min(n0_train, n1_train) < 1:
        raise ValueError(
            "Stratified split is impossible with the current class counts and test_size.\n"
            f"Class counts: (n0={n0}, n1={n1}), test_size={test_size} → "
            f"test per-class ≈ ({n0_test}, {n1_test}), train per-class ≈ ({n0_train}, {n1_train}). "
            "Increase dataset size, adjust test_size, or relax stratification."
        )

    # Additional guardrails on T (number of TFs) vs k_max
    if k_max > T:
        # Not fatal (your code clamps with min), but be explicit to help users.
        raise ValueError(f"k_max ({k_max}) exceeds number of TFs T ({T}). "
                         "Lower k_max or provide more TFs.")

    # Everything looks good
    return None


def _resolve_geometry(
    geometry: Literal["auto", "pooled", "tgt", "ref"],
    ko_scope: Literal["both", "target", "ref"],
) -> Literal["pooled", "tgt", "ref"]:
    """
    Map geometry='auto' to a concrete geometry based on KO scope.

    Rules:
      - ko_scope='both'   → pooled (covariance from both classes)
      - ko_scope='target' → tgt    (covariance from target class only)
      - ko_scope='ref'    → ref    (covariance from reference class only)
    """
    if geometry != "auto":
        return geometry

    if ko_scope == "both":
        return "pooled"
    elif ko_scope == "target":
        return "tgt"
    else:                     # ko_scope == "ref"
        return "ref"


def _class_cov(
        Hc: np.ndarray,
        H_fallback: np.ndarray
) -> np.ndarray:
    """
    Compute the sample covariance matrix for a given set of samples,
    with a fallback if too few samples are available.

    Parameters
    ----------
    Hc : np.ndarray, shape (n_c, T)
        Data matrix for the class of interest, where n_c is the number
        of samples and T is the number of features/TFs.
    H_fallback : np.ndarray, shape (n_all, T)
        Fallback data matrix (e.g., all test samples) used when `Hc`
        has insufficient rows to compute a stable covariance.

    Returns
    -------
    np.ndarray, shape (T, T)
        Covariance matrix computed from `Hc` if it has >1 sample;
        otherwise computed from `H_fallback`.

    Notes
    -----
    - If `Hc` has 0 or 1 sample, its covariance is undefined; in this case,
      the covariance of `H_fallback` is returned instead.
    - Covariance is computed with `rowvar=False` so that columns correspond
      to features and rows to samples.
    """
    if len(Hc) <= 1:
        return np.cov(H_fallback, rowvar=False)
    return np.cov(Hc, rowvar=False)

def _pooled_cov(
        H: np.ndarray,
        y: np.ndarray
) -> np.ndarray:
    """
    Compute the pooled covariance matrix for two classes.

    This function estimates a pooled covariance matrix from two groups
    of samples labeled 0 and 1 in `y`. The pooled estimate is the
    weighted average of the per-class covariance matrices, weighted
    by their degrees of freedom (n - 1) in each class.

    If either class has fewer than two samples, the overall covariance
    of all samples in `H` is returned instead.

    Parameters
    ----------
    H : np.ndarray, shape (N, T)
        Data matrix with `N` samples (rows) and `T` features (columns).
    y : np.ndarray, shape (N,)
        Binary class labels for the samples in `H`. Must contain only
        values {0, 1}.

    Returns
    -------
    np.ndarray, shape (T, T)
        Pooled covariance matrix of the two classes.

    Notes
    -----
    The pooled covariance is computed as:

    .. math::
        \Sigma_{\text{pooled}} =
            \frac{(n_0 - 1) \Sigma_0 + (n_1 - 1) \Sigma_1}
                 {n_0 + n_1 - 2}

    where:
        - :math:`n_c` is the number of samples in class `c`
        - :math:`\Sigma_c` is the sample covariance matrix of class `c`

    If `n_0 <= 1` or `n_1 <= 1`, this reduces to `np.cov(H, rowvar=False)`.
    """
    H0, H1 = H[y == 0], H[y == 1]
    n0, n1 = len(H0), len(H1)
    if n0 <= 1 or n1 <= 1:
        return np.cov(H, rowvar=False)
    C0 = np.cov(H0, rowvar=False)
    C1 = np.cov(H1, rowvar=False)
    return ((n0 - 1) * C0 + (n1 - 1) * C1) / max(n0 + n1 - 2, 1)


def _shrink(
        C: np.ndarray,
        lam: float
) -> np.ndarray:
    """
    Apply linear shrinkage to a covariance matrix.

    This function returns a shrinkage estimator of a covariance matrix
    by blending it with a scaled identity matrix. The identity component
    is scaled by the mean variance (average of the diagonal entries in `C`).

    The shrinkage formula is:

    .. math::
        \hat{\Sigma}_\lambda = (1 - \lambda) C + \lambda \tau I_T

    where:
        - :math:`C` is the original covariance matrix
        - :math:`\tau` is the mean of the diagonal entries of `C`
        - :math:`I_T` is the :math:`T \\times T` identity matrix
        - :math:`\lambda` is the shrinkage intensity in [0, 1)

    Parameters
    ----------
    C : np.ndarray, shape (T, T)
        Covariance matrix to shrink. Must be square, symmetric, and numeric.
    lam : float
        Shrinkage intensity. Values closer to 0 keep more of the original
        covariance structure; values closer to 1 weight more toward the
        scaled identity.

    Returns
    -------
    np.ndarray, shape (T, T)
        Shrunk covariance matrix.

    Notes
    -----
    Shrinkage improves numerical stability when the covariance matrix
    is ill-conditioned or estimated from limited data. The resulting
    matrix is always symmetric and positive definite (for lam > 0).
    """
    T = C.shape[0]
    tau = np.trace(C) / T if T > 0 else 0.0
    return (1.0 - lam) * C + lam * tau * np.eye(T, dtype=C.dtype)


def _euclidean_sep(
        delta: np.ndarray,
        keep_mask: np.ndarray
) -> float:
    """
    Compute Euclidean separation between class means after feature masking.

    This function calculates the Euclidean distance between two class
    mean vectors represented by `delta`, restricted to the subset of
    features indicated by `keep_mask`.

    Parameters
    ----------
    delta : np.ndarray, shape (T,)
        Difference vector between the class means
        (e.g., μ₁ − μ₀ in feature space).
    keep_mask : np.ndarray of bool, shape (T,)
        Boolean mask indicating which features to keep in the calculation.
        Features where `keep_mask[i]` is False are excluded.

    Returns
    -------
    float
        Euclidean separation between the class means using only the
        selected features. Guaranteed to be non-negative.

    Notes
    -----
    The calculation is:

    .. math::
        d = \\sqrt{\\max(0, \\Delta_k^T \\Delta_k)}

    where:
        - :math:`\\Delta_k` is the subvector of `delta` corresponding
          to the kept features.

    The `max(0, ·)` guard ensures numerical stability against small
    negative values caused by floating-point precision errors.
    """
    d = float(np.dot(delta[keep_mask], delta[keep_mask]))
    return np.sqrt(max(0.0, d))


def _mahalanobis_sep(
        delta: np.ndarray,
        C: np.ndarray,
        keep_mask: np.ndarray
) -> float:
    """
    Compute Mahalanobis separation between class means after feature masking.

    This function calculates the Mahalanobis distance between two class
    mean vectors represented by `delta`, restricted to the subset of
    features indicated by `keep_mask`, using the corresponding submatrix
    of the covariance matrix `C`.

    Parameters
    ----------
    delta : np.ndarray, shape (T,)
        Difference vector between the class means
        (e.g., μ₁ − μ₀ in feature space).
    C : np.ndarray, shape (T, T)
        Covariance matrix in the same feature space as `delta`.
        Must be symmetric and positive semi-definite for meaningful results.
    keep_mask : np.ndarray of bool, shape (T,)
        Boolean mask indicating which features to keep in the calculation.
        Features where `keep_mask[i]` is False are excluded.

    Returns
    -------
    float
        Mahalanobis separation between the class means using only the
        selected features. Guaranteed to be non-negative.

    Notes
    -----
    The calculation is:

    .. math::
        d = \\sqrt{\\max(0, \\Delta_k^T \\Sigma_k^{-1} \\Delta_k)}

    where:
        - :math:`\\Delta_k` is the subvector of `delta` corresponding
          to the kept features.
        - :math:`\\Sigma_k` is the submatrix of `C` restricted to the
          kept features.
        - :math:`\\Sigma_k^{-1}` is computed via the Moore–Penrose
          pseudoinverse to handle singular matrices.

    The `max(0, ·)` guard ensures numerical stability against small
    negative values due to floating-point errors.
    """
    d = delta[keep_mask]
    Ck = C[np.ix_(keep_mask, keep_mask)]
    Pk = np.linalg.pinv(Ck)
    val = float(d @ Pk @ d)
    return np.sqrt(max(0.0, val))
