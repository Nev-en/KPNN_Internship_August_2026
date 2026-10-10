# kpnn2_port_alt.py -- the kpnn2 port as it was BEFORE apply_v1_init reseeded.
#
# Verbatim copy of src/kpnn2_scripts/kpnn2_training.py at commit 85d9809
# (2026-09-07); only this comment block is new. Kept to reproduce the early
# port run (comp/t2/plots/plots_karla_new_vs_kpnn2_port/*.png, 2026-09-06),
# where v1's init rule was applied to weights already drawn at a shifted
# position in the random stream: correlation 0.005 to v1's starting weights,
# TF-level r 0.81-0.94 against Karla. Do not fix anything here -- use
# kpnn2_port.py for real work.
#
# Run it through train_kpnn2_sergio_rseed.py with KPNN2_PORT=kpnn2_port_alt.

"""KPNN training on top of kpnn2, reproducing the behaviour of kpnn v1.

Why this module exists
----------------------
Karla's scripts call ``kpnn.train_kpnns(...)``, a single entry point that hides
~6000 lines: it builds the network from the edgelist, trains it over N random
seeds, runs Captum, and writes the score CSVs. kpnn2 deliberately does none of
that -- its CONTEXT.md states it is "not a trainer" and "not Captum". It parses
an edgelist into a ``LayeredSpec`` and gives you ``MaskedLinear``; the training
loop is ordinary PyTorch that the caller owns.

So this module is that missing loop, written to match what kpnn v1 does, so the
resulting scores are comparable to Karla's reference export.

What was mirrored from kpnn v1 (verified against the installed 0.1.12 source)
----------------------------------------------------------------------------
Architecture   Dropout -> MaskedLinear(genes->TFs) -> PReLU -> Dropout
               -> Linear(TFs->1) -> Sigmoid.  Only the first layer is sparse;
               v1's output layer is a dense ``nn.Linear``, because every TF
               feeds the single output node anyway.
Init           Sparse layer: weights ~ U(+-sqrt(6 / average_fan_in)) where
               average_fan_in = n_edges / n_output_nodes (a single global bound,
               not per row), bias constant 0.01.  This differs from kpnn2's own
               ``reset_parameters`` (per-row U(+-1/sqrt(row_degree))), so it is
               applied explicitly -- see ``_apply_v1_init``.
               Output layer: default PyTorch init, bias zeroed.
Loss           ``nn.BCELoss`` on the sigmoid output (not BCEWithLogits).
Attribution    IntegratedGradients on the inputs and LayerConductance on the
               sparse layer's *pre-activation* output, both against a zero
               baseline on the logits path (sigmoid skipped), n_steps=50.
Aggregation    Mean over the test cells of each class; one CSV row per seed.
Labels         ``filter_classes={0: 1, 1: 0}`` in Karla's call filters to
               classes {0,1} and *swaps* them.  The comment in her script says
               "dataset already binary" but the mapping really does relabel, so
               class_0 in the output holds the originally-labelled-1 cells.
               ``FILTER_CLASSES`` below keeps that behaviour.

Childless hidden nodes
----------------------
The edgelists contain TFs that feed ``output`` but have no gene children (e.g.
ELK1 and SREBF1 for INS).  Karla's reference CSVs do not contain them, and in
kpnn2 they would parse as in-degree-0 *input* nodes and land in layer 0 next to
the genes.  ``to_kpnn2_edgelist`` prunes them, which reproduces the reference
column sets exactly for all five conditions.

Output layout
-------------
Written to match kpnn v1, so ``src/comp`` can read it unchanged::

    <output_path>/kpnn_output_<timestamp>/
        model_metrics.csv
        importance_scores/aggregated_scores/class_{0,1}/
            input_layer_feature_attributions.csv
            hidden_layer_1_attributions.csv
"""
from __future__ import annotations

import math
import random
import secrets
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients, LayerConductance
from torch.utils.data import DataLoader, TensorDataset

import kpnn2

# kpnn v1 is called with filter_classes={0: 1, 1: 0}: keep both classes, swap
# the labels. Kept as a module constant so a run that should NOT swap can pass
# filter_classes={0: 0, 1: 1} instead of editing the loop.
FILTER_CLASSES: Dict[int, int] = {0: 1, 1: 0}

# The network's leaf nodes carry this suffix; the AnnData var names do not.
LEAF_SUFFIX = "_leaf"


# --- edgelist ------------------------------------------------------------------


def to_kpnn2_edgelist(edgelist: pd.DataFrame) -> pd.DataFrame:
    """Convert Karla's parent/child edgelist into kpnn2's source/target form.

    Karla's file is written top-down (``parent`` = TF, ``child`` = gene) with
    explicit ``layer_parent`` / ``layer_child`` columns. kpnn2 wants two columns
    named ``source`` and ``target`` in the direction of computation, and derives
    the layering itself. So the columns are renamed *and* swapped.

    Hidden nodes without children are dropped, repeatedly, because dropping one
    can orphan another. Without this they would parse as input nodes.
    """
    required = {"parent", "child", "layer_parent", "layer_child"}
    missing = required - set(edgelist.columns)
    if missing:
        raise ValueError(f"edgelist is missing columns: {sorted(missing)}")

    el = edgelist.rename(columns={"child": "source", "parent": "target"})
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
        raise ValueError("no edges left after pruning childless hidden nodes")
    return out


# --- model ---------------------------------------------------------------------


class Kpnn2Model(nn.Module):
    """kpnn v1's ``KpnnModel`` rebuilt on kpnn2's ``MaskedLinear``.

    Only binary classification is implemented, which is all the Sergio
    comparison needs and all ``kpnn_config.yaml`` configures.
    """

    def __init__(self, spec: kpnn2.LayeredSpec, hyperparameters: dict) -> None:
        super().__init__()
        if len(spec.layer_dims) != 3:
            raise ValueError(
                f"expected genes -> TFs -> output (3 layers), got "
                f"{spec.layer_dims}. Deeper networks would need one "
                f"MaskedLinear per hop here."
            )

        self.spec = spec
        self.dropout_in = nn.Dropout(p=hyperparameters["dropout"])
        # hops[0] is the gene -> TF mask; this is the only sparse layer.
        self.masked = kpnn2.MaskedLinear(spec.hops[0].mask)
        self.activation = _activation_from_config(
            hyperparameters.get("activation_function", {})
        )
        self.dropout_hidden = nn.Dropout(p=hyperparameters["dropout"])
        # v1 uses a dense Linear for the output layer, not a sparse one.
        self.output = nn.Linear(spec.layer_dims[1], 1)
        self.sigmoid = nn.Sigmoid()

        _apply_v1_init(self.masked, spec.hops[0].mask)
        with torch.no_grad():
            self.output.bias.zero_()

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """The attribution path: everything except the final sigmoid.

        Attributing through the sigmoid saturates the gradients, which is why
        v1 exposes the same separate path.
        """
        h = self.dropout_in(x)
        h = self.masked(h)
        h = self.activation(h)
        h = self.dropout_hidden(h)
        return self.output(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.forward_logits(x))


def _activation_from_config(config: dict) -> nn.Module:
    name = str(config.get("name", "prelu")).lower()
    params = config.get("params", {}) or {}
    if name == "prelu":
        return nn.PReLU(init=params.get("init", 0.25))
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"unsupported activation function: {name!r}")


def _apply_v1_init(layer: kpnn2.MaskedLinear, mask: torch.Tensor) -> None:
    """Reproduce ``SparseLinear.initialize_weights`` from kpnn v1.

    v1 draws every edge weight from one uniform distribution whose bound comes
    from the *average* fan-in over output neurons, and fills the bias with 0.01.
    kpnn2's own init is per-row instead, so it is overwritten here.

    The trainable tensor is ``parametrizations.weight.original``; assigning to
    ``layer.weight`` copies into it. Masked-out entries are drawn too but never
    reach the output, exactly as v1's dense-init-then-mask would behave.
    """
    n_connections = int(mask.sum().item())
    n_output_neurons = int((mask.sum(dim=1) > 0).sum().item())
    if n_connections == 0 or n_output_neurons == 0:
        raise ValueError("mask has no connections")

    average_fan_in = n_connections / n_output_neurons
    bound = math.sqrt(6.0 / average_fan_in)

    original = layer.parametrizations.weight.original
    with torch.no_grad():
        original.uniform_(-bound, bound)
        if layer.bias is not None:
            layer.bias.fill_(0.01)


# --- data ----------------------------------------------------------------------


def anndata_to_frame(adata, input_nodes: Sequence[str]) -> pd.DataFrame:
    """Build the DataFrame ``align_inputs`` needs from an AnnData object.

    kpnn2 v1 does not accept AnnData (CONTEXT.md: "Not AnnData"), so the matrix
    is materialised here. The ``_leaf`` suffix is added back onto the var names
    because that is how the nodes are called in the edgelist.
    """
    matrix = adata.X
    if hasattr(matrix, "toarray"):  # scipy sparse
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=np.float32)

    columns = [f"{name}{LEAF_SUFFIX}" for name in adata.var.index]
    frame = pd.DataFrame(matrix, columns=columns)

    missing = [node for node in input_nodes if node not in frame.columns]
    if missing:
        raise ValueError(
            f"{len(missing)} network leaves are absent from the data, e.g. "
            f"{missing[:5]}. kpnn v1 would drop them from the network; here the "
            f"edgelist and the data are expected to already match."
        )
    return frame


def labels_from_obs(adata, label_column: str) -> np.ndarray:
    if label_column not in adata.obs.columns:
        raise KeyError(f"missing label column {label_column!r}")
    return adata.obs[label_column].to_numpy().astype(np.int64)


def filter_and_relabel(
    frame: pd.DataFrame,
    labels: np.ndarray,
    filter_classes: Optional[Dict[int, int]],
) -> Tuple[pd.DataFrame, np.ndarray]:
    """kpnn v1's ``filter_and_relabel``: keep the mapped classes, relabel them."""
    if not filter_classes:
        return frame, labels
    keep = np.isin(labels, list(filter_classes.keys()))
    frame = frame.loc[keep].reset_index(drop=True)
    labels = np.array([filter_classes[int(v)] for v in labels[keep]], dtype=np.int64)
    return frame, labels


# --- training ------------------------------------------------------------------


def _build_optimizer(model: nn.Module, hyperparameters: dict):
    name = str(hyperparameters.get("optimizer", "Adam"))
    optimizers = {"Adam": torch.optim.Adam, "SGD": torch.optim.SGD}
    if name not in optimizers:
        raise ValueError(f"unsupported optimizer: {name!r}")
    return optimizers[name](
        model.parameters(),
        lr=hyperparameters["lr"],
        weight_decay=hyperparameters.get("weight_decay", 0),
    )


def _build_scheduler(optimizer, hyperparameters: dict):
    config = hyperparameters.get("lr_scheduler")
    if not config:
        return None
    if config.get("type") != "StepLR":
        raise ValueError(f"unsupported lr_scheduler: {config.get('type')!r}")
    return torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config["step_size"], gamma=config["gamma"]
    )


def train_one_model(
    spec: kpnn2.LayeredSpec,
    tensors: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    hyperparameters: dict,
    seed: int,
    device: torch.device,
) -> Tuple[Kpnn2Model, Dict[str, float]]:
    """Train a single model for one seed and return it with its metrics."""
    set_global_seed(seed)

    model = Kpnn2Model(spec, hyperparameters).to(device)
    criterion = nn.BCELoss()
    optimizer = _build_optimizer(model, hyperparameters)
    scheduler = _build_scheduler(optimizer, hyperparameters)
    lambda_l1 = float(hyperparameters.get("l1_regularization", 0) or 0)
    batch_size = int(hyperparameters["minibatch_size"])

    train_x, train_y = tensors["train"]
    val_x, val_y = tensors["val"]
    loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True
    )

    metrics: Dict[str, float] = {}
    for _ in range(int(hyperparameters["num_epochs"])):
        model.train()
        running = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).float()
            optimizer.zero_grad()
            out = model(batch_x).squeeze(-1)
            loss = criterion(out, batch_y)
            if lambda_l1 > 0:
                l1 = sum(p.abs().sum() for p in model.parameters() if p.requires_grad)
                loss = loss + lambda_l1 * l1
            loss.backward()
            optimizer.step()
            running += loss.item() * batch_x.size(0)
        if scheduler is not None:
            scheduler.step()
        metrics["train_loss"] = running / len(train_x)

    model.eval()
    with torch.no_grad():
        for split, (x, y) in (("val", (val_x, val_y)), ("test", tensors["test"])):
            out = model(x.to(device)).squeeze(-1)
            target = y.to(device).float()
            metrics[f"{split}_loss"] = criterion(out, target).item()
            metrics[f"{split}_accuracy"] = ((out >= 0.5).float() == target).float().mean().item()

    return model, metrics


def set_global_seed(seed: int) -> None:
    """kpnn v1's ``set_global_seed``, minus the CUDA-only knobs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- attribution ---------------------------------------------------------------


def compute_attributions(
    model: Kpnn2Model,
    test_x: torch.Tensor,
    device: torch.device,
    n_steps: int = 50,
    minibatch_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """IntegratedGradients on the inputs, LayerConductance on the sparse layer.

    Both use a zero baseline and the logits path, matching kpnn v1. ``target``
    stays None because the output is a single logit.

    Returned in cell order: ``(n_cells, n_genes)`` and ``(n_cells, n_TFs)``.
    """
    model.eval()
    model.to(device)
    test_x = test_x.to(device)
    baseline = torch.zeros_like(test_x)

    ig = IntegratedGradients(model.forward_logits)
    conductance = LayerConductance(model.forward_logits, model.masked)

    feature_parts: List[torch.Tensor] = []
    hidden_parts: List[torch.Tensor] = []

    for start in range(0, test_x.size(0), minibatch_size):
        batch_x = test_x[start : start + minibatch_size].clone().detach().requires_grad_(True)
        batch_baseline = baseline[start : start + minibatch_size]

        attr = ig.attribute(
            inputs=batch_x, baselines=batch_baseline, target=None, n_steps=n_steps
        )
        feature_parts.append(attr.detach().cpu())

        cond = conductance.attribute(
            inputs=batch_x, baselines=batch_baseline, target=None, n_steps=n_steps
        )
        hidden_parts.append(cond.detach().cpu())

    return torch.cat(feature_parts), torch.cat(hidden_parts)


def aggregate_by_class(
    attributions: torch.Tensor, labels: np.ndarray, class_index: int
) -> Optional[np.ndarray]:
    """Mean over the test cells of one class, as kpnn v1 aggregates."""
    selection = labels == class_index
    if not selection.any():
        return None
    return attributions.numpy()[selection].mean(axis=0)


# --- output --------------------------------------------------------------------


def generate_model_id(seed: int) -> str:
    """``<seed>_<random hex>``, as kpnn v1 names its models.

    The comparison scripts split on ``_`` and keep the seed, so the random part
    only has to be unique.
    """
    return f"{seed}_{secrets.token_hex(4)}"


def append_scores_row(
    scores: np.ndarray, node_names: Sequence[str], output_file: Path, model_id: str
) -> None:
    """Append one seed's aggregated scores, creating the file on first write."""
    if len(scores) != len(node_names):
        raise ValueError(
            f"{output_file.name}: {len(scores)} scores for {len(node_names)} nodes"
        )
    frame = pd.DataFrame([scores], columns=list(node_names))
    frame.insert(0, "model_id", model_id)
    if output_file.exists():
        frame = pd.concat([pd.read_csv(output_file), frame], ignore_index=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_file, index=False)


def run(
    edgelist: pd.DataFrame,
    datasets: Dict[str, object],
    nn_hyperparameters: dict,
    additional_settings: dict,
    output_path: Path,
    random_seeds: Sequence[int] = tuple(range(1, 21)),
    captum_n_steps: int = 50,
    filter_classes: Optional[Dict[int, int]] = None,
    condition: str = "",
) -> Path:
    """Train over all seeds and write the score CSVs. Returns the run directory."""
    if filter_classes is None:
        filter_classes = FILTER_CLASSES

    device = torch.device(additional_settings.get("device", "cpu"))
    label_column = additional_settings.get("data_label_column", "cell_type_class_int")

    spec = kpnn2.parse_layered(to_kpnn2_edgelist(edgelist))
    gene_names = spec.layer_nodes[0]
    tf_names = spec.layer_nodes[1]
    print(
        f"[{condition or 'run'}] network: {len(gene_names)} genes -> "
        f"{len(tf_names)} TFs -> {len(spec.layer_nodes[2])} output "
        f"({int(spec.hops[0].mask.sum())} edges)"
    )

    tensors: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    test_labels: Optional[np.ndarray] = None
    for split in ("train", "val", "test"):
        frame = anndata_to_frame(datasets[split], spec.input_nodes)
        labels = labels_from_obs(datasets[split], label_column)
        frame, labels = filter_and_relabel(frame, labels, filter_classes)
        tensors[split] = (
            kpnn2.align_inputs(frame, spec),
            torch.tensor(labels, dtype=torch.float32),
        )
        if split == "test":
            test_labels = labels
        print(f"   {split}: {len(frame)} cells, class counts {np.bincount(labels).tolist()}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_path) / f"kpnn_output_{timestamp}"
    aggregated_dir = run_dir / "importance_scores" / "aggregated_scores"
    run_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    for seed in random_seeds:
        model, metrics = train_one_model(
            spec, tensors, nn_hyperparameters, seed, device
        )
        model_id = generate_model_id(seed)

        feature_attr, hidden_attr = compute_attributions(
            model, tensors["test"][0], device, n_steps=captum_n_steps
        )

        for class_index in (0, 1):
            class_dir = aggregated_dir / f"class_{class_index}"
            feature_scores = aggregate_by_class(feature_attr, test_labels, class_index)
            hidden_scores = aggregate_by_class(hidden_attr, test_labels, class_index)
            if feature_scores is None or hidden_scores is None:
                print(f"   seed {seed}: no test cells in class {class_index} — skipped")
                continue
            append_scores_row(
                feature_scores,
                gene_names,
                class_dir / "input_layer_feature_attributions.csv",
                model_id,
            )
            append_scores_row(
                hidden_scores,
                tf_names,
                class_dir / "hidden_layer_1_attributions.csv",
                model_id,
            )

        metric_rows.append({"model_id": model_id, "seed": seed, **metrics})
        print(
            f"   seed {seed:2d} ({model_id}): "
            f"test_loss={metrics['test_loss']:.4f} "
            f"test_acc={metrics['test_accuracy']:.4f}"
        )

    pd.DataFrame(metric_rows).to_csv(run_dir / "model_metrics.csv", index=False)
    print(f"\nWrote {len(metric_rows)} models to {run_dir}")
    return run_dir
