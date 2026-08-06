#!/usr/bin/env python3
"""
SERGIO two-cell-type simulation and TF knockout generation.

Purpose
-------
This script creates or reuses a SERGIO-based simulated dataset for two cell
classes, computes decoupler TF activity scores, predicts single-TF KO EFFECT
scores, and compares those scores to the preset |Δbasal| ground truth.

Important workflow notes
------------------------
1. Non-targeting (NT) data are always required.
   - If X_nt.csv and y_nt.csv already exist in --out_dir, they are reused.
   - Otherwise, the script simulates them and writes them to --out_dir.

2. Basal TF values are reused if possible.
   - If basals.csv already exists, it is loaded.
   - Otherwise, the script creates a preset A-vs-B basal gradient and saves it.

3. TF knockout (KO) data are optional.
   - KO datasets are only generated when --with_kos is supplied.
   - If X_ko.csv and y_ko.csv already exist, they are reused/skipped.
   - Otherwise, each TF is individually set to basal 0.0 in both classes,
     SERGIO is rerun, and all KO simulations are saved together.

Main outputs
------------
- W.csv                                  TF-target weight matrix
- basals.csv                            preset basal values per TF
- X_nt.csv / y_nt.csv                   non-targeting simulated data
- X_ko.csv / y_ko.csv                   optional TF KO simulated data
- tf_activity_samples.csv               per-cell decoupler TF activities
- tf_activity_means.csv                 mean TF activities per class
- activity_delta_vs_abs_deltabasal.csv  decoupler Δactivity vs |Δbasal|
- effect_full.csv                       predicted single-KO EFFECT scores
- effect_vs_basal.csv                   EFFECT vs basal comparison table

Example command
---------------
python /home/people/kmangano/masterthesis/src/KO_simulation/sergio_fake_KO.py \
  --edgelist /home/people/kmangano/masterthesis/data/edgelist/TGFB1/TGFB1_tf_only_edgelist_filtered_o.tsv \
  --out_dir /home/people/kmangano/masterthesis/results/KO_simulation/sergio/sergio_TGFB1_o \
  --sergio_root /home/people/kmangano/SERGIO \
  --scripts_dir /home/people/kmangano/masterthesis/src/KO_simulation \
  --n_cells 500 \
  --seed 1 \
  --n_tfs 16 \
  --method ulm \
  --geometry pooled \
  --with_kos \
  --loop_mode feedforward
"""

from __future__ import annotations

import argparse, csv, sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# -----------------------------------------------------------------------------
# Compatibility
# -----------------------------------------------------------------------------
# Older SERGIO code may still reference deprecated NumPy aliases.
if not hasattr(np, "int"):   np.int = int
if not hasattr(np, "float"): np.float = float
if not hasattr(np, "bool"):  np.bool = bool

# -----------------------------------------------------------------------------
# External imports loaded from user-provided paths
# -----------------------------------------------------------------------------
# SERGIO is not necessarily installed in the active environment, so it is loaded
# from --sergio_root when that argument is provided.
def _import_sergio(sergio_root: Path | None):
    if sergio_root:
        sys.path.append(str(sergio_root))
    from SERGIO.sergio import sergio
    return sergio

# The TF activity and EFFECT functions live in the local KO_simulation codebase
# passed through --scripts_dir.
def _import_user_pipeline(scripts_dir: Path):
    sys.path.insert(0, str(scripts_dir))
    from compute_tf_activities import compute_tf_activities
    from predict_tf_ko_effect import predict_tf_ko_effect
    return compute_tf_activities, predict_tf_ko_effect

# -----------------------------------------------------------------------------
# Edgelist and SERGIO input helpers
# -----------------------------------------------------------------------------
def parse_sign(x) -> float:
    # Convert edgelist signs into SERGIO-compatible +/- values.
    try:
        v = float(x); return 1.0 if v >= 0 else -1.0
    except Exception:
        s = str(x).strip().lower()
        return -1.0 if s in {"-", "-1", "rep", "repressor"} else 1.0

def read_edgelist(tsv: Path, strip_child_suffix: str = "_leaf") -> pd.DataFrame:
    # Standardize parent/child/sign columns and remove self-edges.
    df = pd.read_csv(tsv, sep="\t", dtype=str)
    cols = {c.lower(): c for c in df.columns}
    df = df.rename(columns={cols["parent"]: "parent", cols["child"]: "child"})
    df["parent"] = df["parent"].astype(str).str.strip()
    df["child"]  = df["child"].astype(str).str.strip()
    if strip_child_suffix:
        df["child"] = df["child"].str.replace(f"{strip_child_suffix}$", "", regex=True)
    if "sign" in cols:
        df["sign"] = df[cols["sign"]].map(parse_sign)
    else:
        df["sign"] = 1.0
    # IMPORTANT: keep parent != child for SERGIO stability
    df = df[df["parent"] != df["child"]].drop_duplicates()
    return df[["parent","child","sign"]]


def rename_top_tfs(
    edges: pd.DataFrame,
    n_tfs: int,
    loop_mode: str,
) -> Tuple[pd.DataFrame, List[str], dict]:
    tfs_all = sorted(edges["parent"].unique())
    pick = tfs_all[:n_tfs] if len(tfs_all) >= n_tfs else tfs_all
    mapping = {old: f"TF{i+1}" for i, old in enumerate(pick)}

    edges2 = edges.copy()
    # always rename parents (TFs)
    edges2["parent"] = edges2["parent"].map(lambda x: mapping.get(x, x))

    if loop_mode == "loops":
        # in loop mode, also rename children *if* they are TF genes
        edges2["child"] = edges2["child"].map(lambda x: mapping.get(x, x))

    return edges2, [mapping[o] for o in pick], mapping


def build_W(edges: pd.DataFrame, genes_order: List[str], tfs_order: List[str]) -> pd.DataFrame:
    # Rows are genes, columns are selected TFs. Non-edges stay 0.
    W = pd.DataFrame(0.0, index=genes_order, columns=tfs_order)
    for p, c, s in edges.itertuples(index=False):
        if p in W.columns and c in W.index:
            W.at[c, p] = float(s)
    return W


# -----------------------------------------------------------------------------
# SERGIO regulator/target file construction
# -----------------------------------------------------------------------------
# SERGIO expects two temporary CSV files:
#   - targets_csv: regulated genes and their regulators
#   - regs_csv: regulator genes and their basal values in each bin/class

def build_targets_regs(
    edges: pd.DataFrame,
    regs_csv: Path,
    targets_csv: Path,
    basal_A: dict[str,float],
    basal_B: dict[str,float],
    force_regulators: List[str],
    genes_order: List[str],
    K: float=1.0,
    hill: float=2.0,
    allow_tf_targets: bool = False,
) -> None:
    genes = list(genes_order)
    gid   = {g:i for i,g in enumerate(genes)}

    forced = {r for r in force_regulators if r in gid}
    incoming = {g: [] for g in genes}
    for p, c, s in edges.itertuples(index=False):
        if p in gid and c in gid and p != c:   # keep no self-loops
            incoming[c].append((p, 1.0 if parse_sign(s) > 0 else -1.0))

    # -------- targets_csv --------
    with open(targets_csv, "w", newline="") as f:
        w = csv.writer(f)
        for tgt in genes:
            # 🔑 KEY DIFFERENCE:
            # in feedforward mode we skip TFs as targets; in loops mode we don't
            if (not allow_tf_targets) and (tgt in forced):
                continue
            regs = incoming[tgt]
            if not regs:
                continue
            reg_ids = [gid[r] for r, _ in regs]
            Ks      = [K * sg for _, sg in regs]
            hills   = [hill] * len(regs)
            w.writerow([gid[tgt], len(regs)] + reg_ids + Ks + hills)

    # -------- regs_csv (unchanged original logic) --------
    inv_gid  = {i:g for g,i in gid.items()}
    zero_ind = [g for g in genes if not incoming[g]]
    zero_ind = [g for g in genes if not incoming[g]]

    if allow_tf_targets:
        # loop mode: only true root nodes are regulators
        regs_base = sorted(set(zero_ind), key=lambda x: gid[x])
    else:
        # feedforward mode: TFs are forced regulators
        regs_base = sorted(set(zero_ind) | forced, key=lambda x: gid[x])

    with open(regs_csv,"w",newline="") as f:
        w = csv.writer(f)
        reg_ids = set()
        for rname in regs_base:
            rid = gid[rname]; reg_ids.add(rid)
            w.writerow([rid,
                        float(basal_A.get(rname,1.0)),
                        float(basal_B.get(rname,1.0))])
        all_ids = set(range(len(genes)))
        used    = set(reg_ids)
        with open(targets_csv) as f_in:
            for row in csv.reader(f_in):
                if row:
                    used.add(int(float(row[0])))
        missing = sorted(all_ids - used)
        for mid in missing:
            gname = inv_gid[mid]
            w.writerow([mid,
                        float(basal_A.get(gname,1.0)),
                        float(basal_B.get(gname,1.0))])

"""Removed for looping edgelists:     
forced={r for r in force_regulators if r in gid}
incoming={g:[] for g in genes}
    for p,c,s in edges.itertuples(index=False):
            if p in gid and c in gid and p!=c:
                incoming[c].append((p,1.0 if parse_sign(s)>0 else -1.0))
    with open(targets_csv,"w",newline="") as f:
        w=csv.writer(f)
        for tgt in genes:
            if tgt in forced: continue
            regs=incoming[tgt]
            if not regs: continue
            reg_ids=[gid[r] for r,_ in regs]
            Ks=[K*sg for _,sg in regs]
            hills=[hill]*len(regs)
            w.writerow([gid[tgt], len(regs)] + reg_ids + Ks + hills)"""

def simulate_two_types(
    sergio_cls,
    targets_csv: Path,
    regs_csv: Path,
    n_genes: int,
    n_cells: int,
    seed: int,
    noise_q: float,
    decay: float,
    sampling_state: int,
    dt: float,
    coop: float,
) -> np.ndarray:
    np.random.seed(seed)
    print(f"[DEBUG] starting SERGIO: n_genes={n_genes}, n_cells={n_cells}")
    sim=sergio_cls(number_genes=n_genes,number_bins=2,number_sc=n_cells,
                   noise_params=noise_q,noise_type="dpd",decays=decay,
                   dynamics=False,sampling_state=sampling_state,dt=dt)
    print("[DEBUG] calling build_graph")
    sim.build_graph(str(targets_csv),str(regs_csv),shared_coop_state=coop)
    print("[DEBUG] calling simulate")
    sim.simulate()
    print("[DEBUG] simulate() finished, getting expressions")
    expr3d=sim.getExpressions()
    expr2d=np.concatenate(expr3d,axis=1) # genes, 2*cells
    return expr2d.T


# -----------------------------------------------------------------------------
# Plotting helper
# -----------------------------------------------------------------------------
def pretty_scatter(df,x,y,label_col,title,out_png):
    plt.figure(figsize=(10,6))
    ax=sns.scatterplot(data=df,x=x,y=y,
        s=120,color="darkviolet",edgecolor="black",linewidth=1.2)
    for _,r in df.iterrows():
        ax.text(r[x],r[y],r[label_col],fontsize=10,ha="right",va="bottom")
    ax.set_xlabel(x); ax.set_ylabel(y)
    ax.set_title(title,fontsize=11,fontweight='bold')
    plt.tight_layout(); plt.savefig(out_png,dpi=300); plt.close()


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--edgelist",type=Path,required=True)
    ap.add_argument("--out_dir",type=Path,required=True)
    ap.add_argument("--sergio_root",type=Path,default=None)
    ap.add_argument("--scripts_dir",type=Path,required=True)
    ap.add_argument("--n_cells",type=int,default=500)
    ap.add_argument("--seed",type=int,default=1)
    ap.add_argument("--n_tfs",type=int,default=20)
    ap.add_argument("--scale_max",type=float,default=16.0)
    ap.add_argument("--noise_q",type=float,default=0.2)
    ap.add_argument("--decay",type=float,default=1.0)
    ap.add_argument("--sampling_state",type=int,default=5)
    ap.add_argument("--dt",type=float,default=0.01)
    ap.add_argument("--coop",type=float,default=2.0)
    ap.add_argument("--hill",type=float,default=2.0)
    ap.add_argument("--K",type=float,default=1.0)
    ap.add_argument("--method",choices=["ulm","mlm","auc"],default="ulm")
    ap.add_argument("--geometry",choices=["auto","pooled","tgt","ref"],default="auto")
    ap.add_argument("--ko_scope",choices=["both","target","ref"],default="both")
    ap.add_argument("--test_size",type=float,default=0.2)
    ap.add_argument("--split_seed",type=int,default=777)
    ap.add_argument("--base_lam",type=float,default=0.2)
    ap.add_argument("--with_kos",action="store_true",help="Also generate KO dataset")
    ap.add_argument("--loop_mode", choices=["feedforward", "loops"], default="feedforward", help="feedforward = original behaviour (TFs only regulators). "
         "loops = allow TF→TF regulation (TFs can be targets).")
    args=ap.parse_args()
    out_dir=args.out_dir; out_dir.mkdir(parents=True,exist_ok=True)

    sergio_cls=_import_sergio(args.sergio_root)
    compute_tf_activities, predict_tf_ko_effect=_import_user_pipeline(args.scripts_dir)

    # ------------------------------------------------------------------
    # 1) Read edgelist, rename selected TFs to TF1..TFN, and build W
    # ------------------------------------------------------------------
    edges = read_edgelist(args.edgelist)
    edges2, tfs, _ = rename_top_tfs(edges, args.n_tfs, args.loop_mode)
    genes_order=sorted(set(edges2["parent"]).union(edges2["child"]))
    W=build_W(edges2,genes_order,tfs); W.to_csv(out_dir/"W.csv")

    # ------------------------------------------------------------------
    # 2) Load or create preset basal TF values for the two SERGIO classes
    # ------------------------------------------------------------------
    basals_csv=out_dir/"basals.csv"
    if basals_csv.exists():
        df_basals=pd.read_csv(basals_csv)
        basal_A=dict(zip(df_basals["tf"],df_basals["basal_A"]))
        basal_B=dict(zip(df_basals["tf"],df_basals["basal_B"]))
        print(f"ℹ️ using existing {basals_csv}")
    else:
        N=len(tfs)
        vals_A=np.linspace(args.scale_max,1.0,N)
        vals_B=np.linspace(1.0,args.scale_max,N)
        basal_A={t:float(vals_A[i]) for i,t in enumerate(tfs)}
        basal_B={t:float(vals_B[i]) for i,t in enumerate(tfs)}
        df_basals=pd.DataFrame({"tf":tfs,
            "basal_A":[basal_A[t] for t in tfs],
            "basal_B":[basal_B[t] for t in tfs]})
        df_basals["abs_delta_basal"]=(df_basals["basal_A"]-df_basals["basal_B"]).abs()
        df_basals["delta_basal"]=df_basals["basal_A"]-df_basals["basal_B"]
        df_basals.to_csv(basals_csv,index=False)
        print(f"✅ wrote {basals_csv}")

    allow_tf_targets = (args.loop_mode == "loops")

    # ------------------------------------------------------------------
    # 3) Load or simulate non-targeting data
    # ------------------------------------------------------------------
    X_nt_path=out_dir/"X_nt.csv"; y_nt_path=out_dir/"y_nt.csv"
    if X_nt_path.exists() and y_nt_path.exists():
        print("ℹ️ using existing NT dataset")
        X=pd.read_csv(X_nt_path); y_vec=pd.read_csv(y_nt_path)["y"].to_numpy()
    else:
        regs_csv=out_dir/"_tmp_regs.csv"; tars_csv=out_dir/"_tmp_targets.csv"

        build_targets_regs(
            edges2, regs_csv, tars_csv,
            basal_A, basal_B, tfs, genes_order,
            K=args.K, hill=args.hill,
            allow_tf_targets=allow_tf_targets)
        
        # 🔎 DEBUG: check consistency before calling SERGIO
        N = len(genes_order)
        print(f"[DEBUG] len(genes_order) = {N}")

        import csv
        def max_id_in_csv(path, col_idx):
            mx = -1
            with open(path) as f:
                for row in csv.reader(f):
                    if not row:
                        continue
                    try:
                        gid = int(float(row[col_idx]))
                        mx = max(mx, gid)
                    except Exception:
                        pass
            return mx

        max_tgt = max_id_in_csv(tars_csv, 0)      # first col = target gene id
        max_reg = max_id_in_csv(regs_csv, 0)      # first col = regulator id
        print(f"[DEBUG] max target id = {max_tgt}, max regulator id = {max_reg}")

        # sanity condition that SERGIO expects:
        if max(max_tgt, max_reg) >= N:
            raise RuntimeError("IDs in regs/targets exceed number_genes")
        
        X_arr=simulate_two_types(sergio_cls,tars_csv,regs_csv,
                                 len(genes_order),args.n_cells,args.seed,
                                 args.noise_q,args.decay,
                                 args.sampling_state,args.dt,args.coop)
        X=pd.DataFrame(X_arr,columns=genes_order)
        y_vec=np.array([0]*args.n_cells+[1]*args.n_cells,int)
        X.to_csv(X_nt_path,index=False)
        pd.DataFrame({"y":y_vec}).to_csv(y_nt_path,index=False)
        print("✅ NT dataset written")

    # ------------------------------------------------------------------
    # 4) Optionally generate single-TF KO datasets
    # ------------------------------------------------------------------
    # Only runs when --with_kos is passed. For each TF, its basal value is
    # set to 0.0 in both SERGIO classes before resimulating.
    if args.with_kos:
        X_ko_path=out_dir/"X_ko.csv"; y_ko_path=out_dir/"y_ko.csv"
        if X_ko_path.exists() and y_ko_path.exists():
            print("ℹ️ KO dataset already exists, skipping")
        else:
            print("▶️ Simulating KO datasets...")
            X_blocks=[]; y_rows=[]
            for i,tf in enumerate(tfs):
                bA=dict(basal_A); bB=dict(basal_B)
                bA[tf]=0.0; bB[tf]=0.0
                regs_tf=out_dir/f"_tmp_regs_ko_{tf}.csv"
                tars_tf=out_dir/f"_tmp_targets_ko_{tf}.csv"

                build_targets_regs(
                    edges2, regs_tf, tars_tf,
                    bA, bB, tfs, genes_order,
                    K=args.K, hill=args.hill,
                    allow_tf_targets=allow_tf_targets) 
                
                X_tf=simulate_two_types(sergio_cls,tars_tf,regs_tf,
                                        len(genes_order),args.n_cells,args.seed+100+i,
                                        args.noise_q,args.decay,
                                        args.sampling_state,args.dt,args.coop)
                X_blocks.append(pd.DataFrame(X_tf,columns=genes_order))
                y_rows.extend([{"tf":tf,"y":0}]*args.n_cells+[{"tf":tf,"y":1}]*args.n_cells)
            X_ko=pd.concat(X_blocks,ignore_index=True)
            y_ko=pd.DataFrame(y_rows)
            X_ko.to_csv(X_ko_path,index=False)
            y_ko.to_csv(y_ko_path,index=False)
            print(f"✅ KO dataset written: {X_ko.shape}")

    # ------------------------------------------------------------------
    # 5) Compute decoupler TF activities from the NT expression matrix
    # ------------------------------------------------------------------
    common=[g for g in X.columns if g in W.index]
    X_al=X[common]; W_al=W.loc[common]
    A_mat, tf_names = compute_tf_activities(X=X_al, W=W_al, method=args.method, tmin=0)
    pd.DataFrame(A_mat, columns=tf_names).assign(y=y_vec).to_csv(out_dir/"tf_activity_samples.csv", index=False)

    # ------------------------------------------------------------------
    # 6) Average TF activities per class and compare Δactivity to |Δbasal|
    # ------------------------------------------------------------------
    A_df=pd.DataFrame(A_mat,columns=tf_names); A_df["y"]=y_vec
    mean_A=A_df.query("y==0").drop(columns="y").mean(axis=0)
    mean_B=A_df.query("y==1").drop(columns="y").mean(axis=0)
    df_means=pd.DataFrame({
        "tf":tf_names,
        "mean_activity_A":[mean_A[t] for t in tf_names],
        "mean_activity_B":[mean_B[t] for t in tf_names],
    })
    df_means["delta_activity"]=df_means["mean_activity_A"]-df_means["mean_activity_B"]
    df_means["abs_delta_activity"]=df_means["delta_activity"].abs()
    df_means.to_csv(out_dir/"tf_activity_means.csv", index=False)

    df_act_vs_basal=pd.merge(
        df_basals[["tf","abs_delta_basal"]],
        df_means[["tf","delta_activity","abs_delta_activity"]],
        on="tf", how="inner"
    ).sort_values("tf")
    df_act_vs_basal.to_csv(out_dir/"activity_delta_vs_abs_deltabasal.csv", index=False)

    rho_act, p_act = spearmanr(df_act_vs_basal["abs_delta_basal"], df_act_vs_basal["abs_delta_activity"])
    print(f"Spearman(|Δactivity|, |Δbasal|): ρ={rho_act:.3f} (p={p_act:.2e})")

    title_act=f"decoupler avg TF activity |Δ| vs |Δbasal| — Spearman ρ={rho_act:.2f} (p={p_act:.1e})"
    df_plot_act=df_act_vs_basal.rename(columns={
        "abs_delta_basal":"|Δ basal|",
        "abs_delta_activity":"|Δ activity (avg)|"
    })
    pretty_scatter(df_plot_act, x="|Δ basal|", y="|Δ activity (avg)|",
                   label_col="tf", title=title_act,
                   out_png=out_dir/"activity_delta_vs_abs_deltabasal.png")

    # ------------------------------------------------------------------
    # 7) Predict single-KO EFFECT scores from TF activities
    # ------------------------------------------------------------------
    df_effect = predict_tf_ko_effect(
        A=A_mat, y=y_vec, tf_names=tf_names,
        ko_scope=args.ko_scope, geometry=args.geometry,
        k_max=1,
        test_size=args.test_size, split_seed=args.split_seed,
        base_lam=args.base_lam, fallback_euclidean=True,
    )
    df_effect.to_csv(out_dir/"effect_full.csv", index=False)

    df_single=df_effect.query("k==1").copy()
    df_single=df_single.rename(columns={"combo_label":"tf"})[["tf","effect","sep"]]

    # ------------------------------------------------------------------
    # 8) Compare predicted EFFECT scores against the preset basal signal
    # ------------------------------------------------------------------
    merged=pd.merge(df_basals,df_single,on="tf",how="inner").sort_values("tf")
    merged.to_csv(out_dir/"effect_vs_basal.csv", index=False)

    rho,pval=spearmanr(merged["abs_delta_basal"],merged["effect"])
    print(f"Spearman(EFFECT, |Δbasal|): ρ={rho:.3f} (p={pval:.2e})")

    title_abs=f"SERGIO EFFECT (single-KO) vs |Δbasal| — Spearman ρ={rho:.2f} (p={pval:.1e})"
    y_label="SERGIO EFFECT (single-KO)"
    df_plot=merged.rename(columns={"abs_delta_basal":"|Δ basal|","effect":y_label})
    pretty_scatter(df_plot, x="|Δ basal|", y=y_label,
                   label_col="tf", title=title_abs,
                   out_png=out_dir/"effect_vs_abs_deltabasal.png")

    # ------------------------------------------------------------------
    # 9) Clean up temporary SERGIO input files
    # ------------------------------------------------------------------
    (out_dir/"_tmp_regs.csv").unlink(missing_ok=True)
    (out_dir/"_tmp_targets.csv").unlink(missing_ok=True)

    print("\n✅ Done. Wrote:")
    for p in [
        out_dir/"W.csv",
        out_dir/"basals.csv",
        out_dir/"X_nt.csv",
        out_dir/"y_nt.csv",
        out_dir/"tf_activity_samples.csv",
        out_dir/"tf_activity_means.csv",
        out_dir/"activity_delta_vs_abs_deltabasal.csv",
        out_dir/"activity_delta_vs_abs_deltabasal.png",
        out_dir/"effect_full.csv",
        out_dir/"effect_vs_basal.csv",
        out_dir/"effect_vs_abs_deltabasal.png",
    ]:
        print("  ", p)

if __name__=="__main__":
    main()