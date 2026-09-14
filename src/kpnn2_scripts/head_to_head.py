"""Train kpnn v1 and kpnn2 back to back, in one process, on the same data.

Why one script
--------------
Every earlier comparison read Karla's export, produced months ago on another
machine. That leaves room for doubt: package versions, BLAS build, thread count,
even the .h5ad files could have drifted. This script removes that doubt by
running both packages in a single process, from one set of loaded AnnData
objects, one after the other -- v1 first, then kpnn2.

Whatever the two still disagree on afterwards is a property of the packages, not
of the environment they ran in.

What is held fixed
------------------
The same edgelist, the same .h5ad splits (deep-copied per package, since v1
slices and reorders what it is handed), the same kpnn_config.yaml values, the
same seed list, and the same filter_classes={0: 1, 1: 0} that Karla's call uses.

Output
------
One timestamped directory holding both runs in the layout src/comp reads::

    <out>/run_<timestamp>/
        kpnn_v1/<COND>_o_A_vs_B/alter_nothing/kpnn_output_*/importance_scores/...
        kpnn2_<variant>/<COND>_o_A_vs_B/alter_nothing/kpnn_output_*/...

Compare and plot them with, from the project root::

    $env:RUN_A="results_kpnn2/results_headtohead/run_<ts>/kpnn_v1"
    $env:RUN_B="results_kpnn2/results_headtohead/run_<ts>/kpnn2_port"
    $env:LABEL_A="kpnn v1"; $env:LABEL_B="kpnn2 port"
    $env:COMPARISON_OUT="headtohead"
    .\.venv310\Scripts\python.exe src\comp\compare_runs.py

Usage
-----
    python head_to_head.py                              # all five, 20 seeds
    python head_to_head.py --condition INS --seeds 3    # a quick check
    python head_to_head.py --variant port2

v1 is the slow half by a wide margin; expect it to dominate the runtime.
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Dict, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import yaml

import kpnn

PROJECT = Path(__file__).resolve().parents[2]
CONDITIONS = ["INS", "TNFA", "IFNG", "TGFB1", "IFNB"]
CONFIG_YAML = PROJECT / "src/Karla_kpnn_scripts/Sergio_simulation/kpnn_config.yaml"

# Karla's kpnn v1 call relabels 0 <-> 1; both halves must use it or the class
# folders come out mirrored. See FEHLER.txt 2.1.
FILTER_CLASSES = {0: 1, 1: 0}


def load_splits(data_dir: Path, condition: str) -> Dict[str, ad.AnnData]:
    """Read train/val/test once. Both packages get deep copies of these."""
    split_dir = data_dir / f"sergio_{condition}"
    splits = {}
    for name in ("train", "val", "test"):
        hits = sorted(split_dir.glob(f"*_{name}.h5ad")) or sorted(
            split_dir.glob(f"*.h5ad_{name}")
        )
        if len(hits) != 1:
            raise FileNotFoundError(
                f"expected one {name} file in {split_dir}, found {[h.name for h in hits]}"
            )
        splits[name] = ad.read_h5ad(hits[0])
    return splits


def ensure_label(adata: ad.AnnData, column: str) -> None:
    """Derive the 0/1 label column when the file does not carry it."""
    if column in adata.obs.columns:
        return
    if "y" in adata.obs.columns and set(np.unique(adata.obs["y"])) <= {0, 1}:
        adata.obs[column] = adata.obs["y"].astype("Int64")
        return
    if "cell_type" in adata.obs.columns:
        as_str = adata.obs["cell_type"].astype(str)
        if set(as_str.unique()) <= {"A", "B"}:
            adata.obs[column] = as_str.map({"A": 0, "B": 1}).astype("Int64")
            return
    raise KeyError(f"no {column!r} and could not derive it from 'y' or 'cell_type'")


def run_v1(edgelist, splits, hyper, settings, seeds, out, steps) -> float:
    """kpnn v1, called exactly as Karla's train_kpnn_sergio_rseed.py calls it.

    The config dicts are copied because v1 writes into them: it adds a
    'class_priors' key to nn_hyperparameters, and its own validator then rejects
    that key on the next call. Without the copy, train_kpnns can only be called
    once per process -- which is exactly what this script needs to do twice.
    """
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    kpnn.train_kpnns(
        edgelist=edgelist,
        datasets=copy.deepcopy(splits),
        nn_hyperparameters=copy.deepcopy(hyper),
        additional_settings=copy.deepcopy(settings),
        remove_layers=None,
        random_seeds=list(seeds),
        network_bias_method=None,
        set_feature_values=None,
        save_models=False,
        compute_attr=True,
        captum_n_steps=steps,
        get_all_attributions=False,
        get_activations=False,
        compute_infidelity=False,
        compute_max_sensitivity=False,
        filter_classes=FILTER_CLASSES,
        check_edgelist=False,
        output_path=out,
    )
    return time.perf_counter() - started


def run_kpnn2(variant, edgelist, splits, hyper, settings, seeds, out, steps) -> float:
    """The kpnn2 port. Both port and port2 expose the same run() signature."""
    module = __import__(f"kpnn2_{variant}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    module.run(
        edgelist=edgelist,
        datasets=copy.deepcopy(splits),
        nn_hyperparameters=copy.deepcopy(hyper),
        additional_settings=copy.deepcopy(settings),
        output_path=out,
        random_seeds=list(seeds),
        captum_n_steps=steps,
        filter_classes=FILTER_CLASSES,
        condition="",
    )
    return time.perf_counter() - started


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Run kpnn v1 and kpnn2 back to back on identical inputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--condition", default=None,
                        help="one condition; default runs all five")
    parser.add_argument("--seeds", type=int, default=20, help="train seeds 1..N")
    parser.add_argument("--variant", default="port", choices=("port", "port2"),
                        help="which kpnn2 port to put against v1")
    parser.add_argument("--ig-steps", type=int, default=50)
    parser.add_argument("--data-dir", type=Path,
                        default=PROJECT / "Sergio_Data_Karla_new")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT / "results_kpnn2/results_headtohead")
    args = parser.parse_args(argv)

    config = yaml.safe_load(CONFIG_YAML.read_text())
    hyper = config.get("nn_hyperparameters", {})
    settings = config.get("additional_settings", {})
    label_column = settings.get("data_label_column", "cell_type_class_int")

    seeds = range(1, args.seeds + 1)
    conditions = [args.condition] if args.condition else CONDITIONS
    run_dir = args.output_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}"

    print(f"Output {run_dir}")
    print(f"kpnn v1 first, then kpnn2_{args.variant}; {args.seeds} seeds each\n")

    timings = []
    for condition in conditions:
        edgelist = pd.read_csv(
            PROJECT / f"src/Karla_kpnn_csv/{condition}_tf_only_edgelist_filtered.tsv",
            sep="\t",
        )
        splits = load_splits(args.data_dir, condition)
        for adata in splits.values():
            ensure_label(adata, label_column)
        cells = {k: v.n_obs for k, v in splits.items()}
        print(f"[{condition}] {cells} cells, {len(edgelist)} edges")

        tail = f"{condition}_o_A_vs_B/alter_nothing"
        v1_seconds = run_v1(edgelist, splits, hyper, settings, seeds,
                            run_dir / "kpnn_v1" / tail, args.ig_steps)
        v2_seconds = run_kpnn2(args.variant, edgelist, splits, hyper, settings,
                               seeds, run_dir / f"kpnn2_{args.variant}" / tail,
                               args.ig_steps)
        timings.append((condition, v1_seconds, v2_seconds))
        print(f"[{condition}] v1 {v1_seconds:7.1f}s   kpnn2 {v2_seconds:7.1f}s   "
              f"factor {v1_seconds / max(v2_seconds, 1e-9):.1f}x\n")

    print(f"{'condition':10s} {'v1 [s]':>9s} {'kpnn2 [s]':>10s}")
    for condition, a, b in timings:
        print(f"{condition:10s} {a:9.1f} {b:10.1f}")
    # compare_runs.py takes RUN_A/RUN_B relative to the project root, so print
    # them that way when possible; an --output-dir elsewhere stays absolute.
    try:
        shown = run_dir.relative_to(PROJECT).as_posix()
    except ValueError:
        shown = str(run_dir)
    print(f"\nRuns are in {run_dir}")
    print(f"  RUN_A = {shown}/kpnn_v1")
    print(f"  RUN_B = {shown}/kpnn2_{args.variant}")


if __name__ == "__main__":
    main()
