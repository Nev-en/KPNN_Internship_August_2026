"""Knowledge-primed neural network training, written the way kpnn2 intends.

This is not a port of kpnn v1. Where the port in kpnn2_training.py mirrors v1's
choices so the numbers stay comparable, this script makes the choices that are
actually right for the job and follows the module shape CONTEXT.md prescribes:
one MaskedLinear per hop, a `saved` dict fed through `gather_hop_inputs`, and
`map_node_attributions` to put names back onto the attribution tensors.

Differences from the v1 port, and why
-------------------------------------
Logits, not probabilities
    The network returns logits and the loss is `BCEWithLogitsLoss`. v1 ends in
    `nn.Sigmoid` and uses `BCELoss`, which is the same function mathematically
    but loses the log-sum-exp trick and can underflow. It also means the
    attribution path needs no separate `forward_logits` -- `forward` already is
    that path.
Library init
    `MaskedLinear` initialises each row from its own fan-in
    (U(±1/sqrt(row_degree))). v1 instead uses one global bound from the average
    fan-in, which under-scales rows with few parents and over-scales dense ones.
    The library default is the better one, so nothing is overridden here.
Any depth
    Layers come from `spec.hops`, so a three-layer or ten-layer edgelist works
    unchanged. The port hardcodes genes -> TFs -> output.
Best epoch wins
    Validation loss is tracked and the best state is restored before scoring.
    v1 keeps whatever the last epoch produced.
Named attributions
    Captum tensors go through `kpnn2.map_node_attributions` into xarray, and
    class aggregation is an xarray groupby rather than boolean-mask indexing.

Configuration
-------------
`TrainConfig` carries the values from kpnn_config.yaml as defaults, transcribed
rather than loaded -- the script has no YAML dependency and no path to a file
that lives in another pipeline's directory. Override from the command line.

One thing worth knowing about those values: `lr_step_size` is 10 while `epochs`
is 5, so the StepLR decay never actually fires. That is what the YAML says, so
it is what the defaults say; it is left intact rather than quietly "fixed".

Usage
-----
    python kpnn2_native.py --condition INS
    python kpnn2_native.py --condition INS --epochs 20 --seeds 5
    python kpnn2_native.py --condition INS --swap-labels   # match Karla's run
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import torch
import xarray as xr
from captum.attr import IntegratedGradients, LayerConductance
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import kpnn2

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Network leaves carry this suffix; AnnData var names do not.
LEAF_SUFFIX = "_leaf"


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters, transcribed from kpnn_config.yaml.

    Frozen so a run cannot mutate its own configuration halfway through; use
    `dataclasses.replace` for variants.
    """

    # --- architecture ---
    dropout: float = 0.2
    prelu_init: float = 0.25

    # --- optimisation ---
    lr: float = 0.02
    weight_decay: float = 0.0
    l1_regularization: float = 0.0
    lr_step_size: int = 10  # >= epochs, so StepLR never fires at the default
    lr_gamma: float = 0.1
    batch_size: int = 64
    epochs: int = 5

    # --- data ---
    label_column: str = "cell_type_class_int"
    # Karla's kpnn v1 call passes filter_classes={0: 1, 1: 0}, which relabels
    # 0<->1. Off by default because swapping labels is not a sensible default;
    # turn it on to line up with that reference run.
    swap_labels: bool = False

    # --- attribution ---
    ig_steps: int = 50
    attr_batch_size: int = 32

    # --- run ---
    seeds: Tuple[int, ...] = tuple(range(1, 21))
    device: str = "cpu"

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)


# --- graph ---------------------------------------------------------------------


def read_edgelist(path: Path) -> pd.DataFrame:
    """Read a parent/child edgelist and hand kpnn2 the source/target form.

    The file is written top-down (parent = regulator, child = gene) with
    explicit layer columns; kpnn2 wants two columns in the direction of
    computation and derives depth itself.

    Hidden nodes with no children are dropped, repeatedly, since removing one
    can orphan another. Without this they have in-degree 0 and would parse as
    input nodes -- landing in layer 0 beside the genes.
    """
    raw = pd.read_csv(path, sep="\t")
    required = {"parent", "child", "layer_parent", "layer_child"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    el = raw.rename(columns={"child": "source", "parent": "target"})
    el = el[["source", "target", "layer_child", "layer_parent"]]

    while True:
        has_incoming = set(el["target"])
        childless = {
            row.source
            for row in el.itertuples()
            if row.layer_child > 0 and row.source not in has_incoming
        }
        if not childless:
            break
        el = el[~el["source"].isin(childless)]

    out = el[["source", "target"]].drop_duplicates().reset_index(drop=True)
    if out.empty:
        raise ValueError(f"{path.name}: no edges left after pruning childless nodes")
    return out


# --- model ---------------------------------------------------------------------


class KpnnNet(nn.Module):
    """One MaskedLinear per hop, wired through a `saved` dict.

    Returns **logits**. Every hop's mask already carries all of its target's
    parents, including skip edges, so no per-skip bookkeeping is needed here.
    """

    def __init__(self, spec: kpnn2.LayeredSpec, config: TrainConfig) -> None:
        super().__init__()
        self.spec = spec
        self.hops = nn.ModuleList(kpnn2.MaskedLinear(hop.mask) for hop in spec.hops)
        # One activation per hidden hop; the final hop emits logits unactivated.
        self.activations = nn.ModuleList(
            nn.PReLU(init=config.prelu_init) for _ in spec.hops[:-1]
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        saved: Dict[int, torch.Tensor] = {0: self.dropout(x)}
        last = len(self.hops) - 1
        hidden = x
        for index, hop in enumerate(self.spec.hops):
            sources = kpnn2.gather_hop_inputs(saved, hop)
            hidden = self.hops[index](sources)
            if index < last:
                hidden = self.dropout(self.activations[index](hidden))
            saved[hop.target_layer] = hidden
        return hidden


# --- data ----------------------------------------------------------------------


@dataclass
class Split:
    x: torch.Tensor
    y: torch.Tensor

    def __len__(self) -> int:
        return self.x.shape[0]

    @property
    def labels(self) -> np.ndarray:
        return self.y.numpy().astype(int)


def load_split(path: Path, spec: kpnn2.LayeredSpec, config: TrainConfig) -> Split:
    """AnnData on disk -> aligned float32 tensor plus labels.

    kpnn2 does not accept AnnData (CONTEXT.md: "Not AnnData"), so the matrix is
    materialised into a DataFrame and `align_inputs` puts the columns into
    `spec.input_nodes` order. That ordering step is the whole point -- it is
    what guarantees column *i* of the tensor is the node the spec says it is.
    """
    adata = ad.read_h5ad(path)

    matrix = adata.X
    if hasattr(matrix, "toarray"):  # scipy sparse
        matrix = matrix.toarray()
    frame = pd.DataFrame(
        np.asarray(matrix, dtype=np.float32),
        columns=[f"{name}{LEAF_SUFFIX}" for name in adata.var.index],
    )

    labels = _labels(adata, config)
    if config.swap_labels:
        labels = 1 - labels

    return Split(x=kpnn2.align_inputs(frame, spec), y=torch.tensor(labels, dtype=torch.float32))


def _labels(adata: ad.AnnData, config: TrainConfig) -> np.ndarray:
    """The 0/1 label column, derived from `y` or `cell_type` when absent."""
    obs = adata.obs
    if config.label_column in obs.columns:
        values = obs[config.label_column].to_numpy()
    elif "y" in obs.columns:
        values = obs["y"].to_numpy()
    elif "cell_type" in obs.columns:
        mapping = {"A": 0, "B": 1}
        as_str = obs["cell_type"].astype(str)
        if not set(as_str.unique()) <= set(mapping):
            raise KeyError(f"cell_type holds {sorted(as_str.unique())}, expected A/B")
        values = as_str.map(mapping).to_numpy()
    else:
        raise KeyError(
            f"no {config.label_column!r}, 'y' or 'cell_type' column in obs"
        )

    labels = np.asarray(values, dtype=np.int64)
    unexpected = set(np.unique(labels)) - {0, 1}
    if unexpected:
        raise ValueError(f"labels must be binary 0/1, found {sorted(unexpected)}")
    return labels


def find_splits(data_dir: Path, condition: str) -> Dict[str, Path]:
    """Locate train/val/test .h5ad files, whichever naming this delivery uses."""
    split_dir = data_dir / f"sergio_{condition}"
    if not split_dir.is_dir():
        raise FileNotFoundError(f"no split directory {split_dir}")

    found = {}
    for split in ("train", "val", "test"):
        # Names differ between conditions (sergio_IFNB_nt_only.h5ad_train.h5ad
        # vs sergio_INS_nt_only_train.h5ad), so match the common suffix.
        hits = sorted(split_dir.glob(f"*_{split}.h5ad")) or sorted(
            split_dir.glob(f"*.h5ad_{split}")
        )
        if len(hits) != 1:
            raise FileNotFoundError(
                f"expected exactly one {split} file in {split_dir}, "
                f"found {[h.name for h in hits]}"
            )
        found[split] = hits[0]
    return found


# --- training ------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class Metrics:
    loss: float
    accuracy: float
    auc: float

    def as_dict(self, prefix: str) -> Dict[str, float]:
        return {f"{prefix}_{k}": v for k, v in asdict(self).items()}


@torch.no_grad()
def evaluate(model: KpnnNet, split: Split, device: torch.device) -> Metrics:
    model.eval()
    logits = model(split.x.to(device)).squeeze(-1)
    target = split.y.to(device)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, target).item()
    probs = torch.sigmoid(logits).cpu().numpy()
    labels = split.labels

    accuracy = float(((probs >= 0.5).astype(int) == labels).mean())
    # A split with one class present has no defined ROC curve.
    auc = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else float("nan")
    return Metrics(loss=loss, accuracy=accuracy, auc=auc)


def train(
    spec: kpnn2.LayeredSpec,
    splits: Dict[str, Split],
    config: TrainConfig,
    seed: int,
) -> Tuple[KpnnNet, Dict[str, float]]:
    """Train one model, restoring the epoch with the lowest validation loss."""
    seed_everything(seed)
    device = config.torch_device

    model = KpnnNet(spec, config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma
    )
    loader = DataLoader(
        TensorDataset(splits["train"].x, splits["train"].y),
        batch_size=config.batch_size,
        shuffle=True,
    )

    best_loss = float("inf")
    best_state: Optional[dict] = None
    best_epoch = -1

    for epoch in range(config.epochs):
        model.train()
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x).squeeze(-1)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, batch_y)
            if config.l1_regularization:
                penalty = sum(p.abs().sum() for p in model.parameters())
                loss = loss + config.l1_regularization * penalty
            loss.backward()
            optimizer.step()
        scheduler.step()

        val = evaluate(model, splits["val"], device)
        if val.loss < best_loss:
            best_loss, best_epoch = val.loss, epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = {"best_epoch": best_epoch}
    for name in ("train", "val", "test"):
        metrics.update(evaluate(model, splits[name], device).as_dict(name))
    return model, metrics


# --- attribution ---------------------------------------------------------------


def attribute(
    model: KpnnNet, split: Split, spec: kpnn2.LayeredSpec, config: TrainConfig
) -> Dict[str, xr.DataArray]:
    """Attribute every named layer, aggregated per class.

    IntegratedGradients answers "which input genes moved the logit"; the network
    has no other named inputs, so it is the right tool for layer 0.
    LayerConductance answers the same question for a hidden layer's units, which
    IG cannot address because those units are not inputs. Both run against a
    zero baseline on the logit output.

    Returns one (class, node) DataArray per layer, keyed by layer name. Not an
    `xr.Dataset`: each layer has its own node axis (783 genes vs 10 TFs vs 1
    output) and its own scalar `layer` coordinate, so there is nothing for a
    Dataset's shared dimensions to align.
    """
    device = config.torch_device
    model.eval().to(device)
    x = split.x.to(device)
    baseline = torch.zeros_like(x)

    ig = IntegratedGradients(model)
    per_layer: Dict[str, xr.DataArray] = {}

    # Layer 0: the inputs themselves.
    per_layer[_layer_name(spec, 0)] = _aggregate(
        _batched(
            lambda xb, bb: ig.attribute(xb, baselines=bb, n_steps=config.ig_steps),
            x, baseline, config.attr_batch_size,
        ),
        spec, 0, split.labels,
    )

    # Every hop target: the units that hop produces.
    for index, hop in enumerate(spec.hops):
        conductance = LayerConductance(model, model.hops[index])
        per_layer[_layer_name(spec, hop.target_layer)] = _aggregate(
            _batched(
                lambda xb, bb: conductance.attribute(
                    xb, baselines=bb, n_steps=config.ig_steps
                ),
                x, baseline, config.attr_batch_size,
            ),
            spec, hop.target_layer, split.labels,
        )

    return per_layer


def _batched(fn, x: torch.Tensor, baseline: torch.Tensor, size: int) -> torch.Tensor:
    """Run an attribution in minibatches; full-batch IG allocates n_steps copies."""
    parts: List[torch.Tensor] = []
    for start in range(0, x.shape[0], size):
        chunk = x[start : start + size].clone().detach().requires_grad_(True)
        parts.append(fn(chunk, baseline[start : start + size]).detach().cpu())
    return torch.cat(parts)


def _aggregate(
    attributions: torch.Tensor,
    spec: kpnn2.LayeredSpec,
    layer: int,
    labels: np.ndarray,
) -> xr.DataArray:
    """Name the node axis, then average the cells of each class."""
    named = kpnn2.map_node_attributions(
        attributions=attributions,
        spec=spec,
        layer=layer,
        coords={"observation": labels},
    )
    # groupby on the observation coord is the class split; no boolean masks.
    return named.groupby("observation").mean().rename(observation="class")


def _layer_name(spec: kpnn2.LayeredSpec, layer: int) -> str:
    if layer == 0:
        return "layer_0_input"
    if layer == len(spec.layer_nodes) - 1:
        return f"layer_{layer}_output"
    return f"layer_{layer}_hidden"


# --- output --------------------------------------------------------------------


def tidy(per_layer: Dict[str, xr.DataArray], seed: int) -> pd.DataFrame:
    """One long table: seed, layer, class, node, score."""
    frames = []
    for name, array in per_layer.items():
        part = array.to_dataframe(name="score").reset_index()
        part["layer_name"] = name
        part["layer"] = int(array.coords["layer"])
        frames.append(part)
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "seed", seed)
    return out[["seed", "layer", "layer_name", "class", "node", "score"]]


def save_checkpoint(path: Path, spec: kpnn2.LayeredSpec, model: KpnnNet) -> None:
    """Save weights next to the spec they belong to.

    A MaskedLinear state_dict is not self-describing: it holds
    `parametrizations.weight.original` and a `mask_digest`, but no node names
    and no layout, and the mask itself stays out. Pickling the module raises
    (the mask is a parametrization) and `torch.save(spec)` would break whenever
    a spec field moves, so `to_dict()` is the interchange form.

    Rebuild with::

        blob = torch.load(path, weights_only=False)
        spec = kpnn2.LayeredSpec.from_dict(blob["spec"])
        model = KpnnNet(spec, TrainConfig(**blob["config"]))
        model.load_state_dict(blob["state_dict"])

    `load_state_dict` raises Kpnn2Error if the digest does not match the mask
    the rebuilt layer has, which catches a silently rewired edgelist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"spec": spec.to_dict(), "state_dict": model.state_dict()}, path)


def write_run(
    output_dir: Path,
    spec: kpnn2.LayeredSpec,
    config: TrainConfig,
    metrics: List[dict],
    attributions: List[pd.DataFrame],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # spec.to_dict() is the interchange form; state_dict alone cannot rebuild
    # node names or layout.
    (output_dir / "spec.json").write_text(json.dumps(spec.to_dict(), indent=2))
    (output_dir / "config.json").write_text(
        json.dumps({**asdict(config), "seeds": list(config.seeds)}, indent=2)
    )
    (output_dir / "spec_fingerprint.txt").write_text(spec.fingerprint)

    pd.DataFrame(metrics).to_csv(output_dir / "metrics.csv", index=False)

    long = pd.concat(attributions, ignore_index=True)
    long.to_parquet(output_dir / "attributions.parquet", index=False)
    long.to_csv(output_dir / "attributions.csv.gz", index=False, compression="gzip")

    _write_wide(long, output_dir / "aggregated_scores")


def _write_wide(long: pd.DataFrame, root: Path) -> None:
    """Also emit the wide per-layer CSVs the comparison scripts in comp read.

    One row per seed, one column per node -- the layout kpnn v1 writes, so the
    existing comparison tooling works against this run unchanged.
    """
    filenames = {"layer_0_input": "input_layer_feature_attributions.csv"}
    for name in long["layer_name"].unique():
        if name.endswith("_hidden"):
            filenames[name] = f"hidden_layer_{name.split('_')[1]}_attributions.csv"

    for (layer_name, class_index), part in long.groupby(["layer_name", "class"]):
        if layer_name not in filenames:
            continue  # the single output unit needs no wide table
        wide = part.pivot(index="seed", columns="node", values="score")
        wide.insert(0, "model_id", [f"{s}_native" for s in wide.index])
        directory = root / f"class_{int(class_index)}"
        directory.mkdir(parents=True, exist_ok=True)
        wide.to_csv(directory / filenames[layer_name], index=False)


# --- entry point ---------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(
        description="Train a KPNN with kpnn2 and attribute it.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--condition", default="INS", help="e.g. INS, IFNB, TGFB1")
    parser.add_argument(
        "--edgelist",
        type=Path,
        default=None,
        help="defaults to src/Karla_kpnn_csv/<condition>_tf_only_edgelist_filtered.tsv",
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "Sergio_Data_Karla_new")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results_kpnn2_native")
    parser.add_argument("--seeds", type=int, default=len(defaults.seeds),
                        help="train seeds 1..N")
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--ig-steps", type=int, default=defaults.ig_steps)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--swap-labels", action="store_true",
                        help="relabel 0<->1, matching Karla's kpnn v1 run")
    parser.add_argument("--save-models", action="store_true",
                        help="write a spec+state_dict checkpoint per seed")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    config = replace(
        TrainConfig(),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dropout=args.dropout,
        ig_steps=args.ig_steps,
        device=args.device,
        swap_labels=args.swap_labels,
        seeds=tuple(range(1, args.seeds + 1)),
    )

    edgelist_path = args.edgelist or (
        PROJECT_ROOT
        / f"src/Karla_kpnn_csv/{args.condition}_tf_only_edgelist_filtered.tsv"
    )
    spec = kpnn2.parse_layered(read_edgelist(edgelist_path))
    print(
        f"[{args.condition}] layers {spec.layer_dims}  "
        f"edges {sum(int(h.mask.sum()) for h in spec.hops)}  "
        f"fingerprint {spec.fingerprint[:12]}"
    )

    paths = find_splits(args.data_dir, args.condition)
    splits = {name: load_split(path, spec, config) for name, path in paths.items()}
    for name, split in splits.items():
        counts = np.bincount(split.labels, minlength=2)
        print(f"   {name:5s} {len(split):5d} cells   class 0/1: {counts[0]}/{counts[1]}")
    if config.swap_labels:
        print("   labels swapped (0<->1) to match Karla's kpnn v1 run")

    run_dir = (
        args.output_dir
        / args.condition
        / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    metrics, attributions = [], []
    for seed in config.seeds:
        model, seed_metrics = train(spec, splits, config, seed)
        attributions.append(tidy(attribute(model, splits["test"], spec, config), seed))
        metrics.append({"seed": seed, **seed_metrics})
        if args.save_models:
            save_checkpoint(run_dir / "models" / f"seed_{seed}.pt", spec, model)
        print(
            f"   seed {seed:2d}  epoch {seed_metrics['best_epoch']}  "
            f"test loss {seed_metrics['test_loss']:.4f}  "
            f"acc {seed_metrics['test_accuracy']:.4f}  "
            f"auc {seed_metrics['test_auc']:.4f}"
        )

    write_run(run_dir, spec, config, metrics, attributions)
    print(f"\nWrote {len(metrics)} seeds to {run_dir}")


if __name__ == "__main__":
    main()
