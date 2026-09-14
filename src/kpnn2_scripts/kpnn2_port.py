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
               applied explicitly -- see ``apply_v1_init``, which reseeds
               first so the draws land at v1's position in the stream.
               Output layer: default PyTorch init, then the bias is overwritten
               with log(p1 / p0) of the training labels, as v1 does via
               ``check_class_counts`` (``rng="torch"`` keeps the old 0).
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

Random numbers during training (``rng="v1"``, the default)
----------------------------------------------------------
Matching the starting weights is not enough: v1 and a plain PyTorch loop
consume different random numbers from the first batch on, and that alone
scatters the scores as much as a new seed does (rng_isolation.py shows it, and
that with v1's draws kpnn2 reproduces v1 to r = 1.000). So the loop draws what
v1 draws (kpnn 0.1.12 on anndata 0.11 / torch 2.4):

* Batch order from ``np.random.permutation`` -- v1's AnnLoader shuffles with
  the *numpy* stream, not with torch's RandomSampler.
* One torch int64 whenever v1 creates a DataLoader iterator
  (``_BaseDataLoaderIter._base_seed``): before the train pass and again for the
  val pass v1 runs every epoch, plus that val pass's numpy permutation.
* Dropout masks in v1's node order. v1 numbers genes by first appearance in the
  edgelist, kpnn2 alphabetically; a mask is drawn by position, so the model
  permutes into v1's order around each Dropout (``v1_node_order``).

``rng="torch"`` is the loop as it was before, for reproducing older runs.

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
import parameter_tables

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


def v1_edge_order(edgelist: pd.DataFrame, spec: kpnn2.LayeredSpec):
    """The gene -> TF edges in the order kpnn v1 numbers them.

    v1 builds its connection index from the rows of the TSV, so edge k is row k
    of the file, and its flat ``uniform_`` gives edge k the random number at
    position k of the stream. kpnn2 sorts nodes alphabetically, which numbers
    the same edges differently -- for INS only 7 of 845 slots coincide.

    Indices point into ``spec.layer_nodes``; only the *edge order* is v1's, the
    node indices stay kpnn2's, so the aligned input tensor still fits.
    """
    genes = {n: i for i, n in enumerate(spec.layer_nodes[0])}
    tfs = {n: i for i, n in enumerate(spec.layer_nodes[1])}
    rows = edgelist[edgelist["layer_parent"] == 1]
    pairs = [
        (genes[r.child], tfs[r.parent])
        for r in rows.itertuples()
        if r.child in genes and r.parent in tfs
    ]
    expected = int(spec.hops[0].mask.sum())
    if len(pairs) != expected:
        raise ValueError(f"v1 order has {len(pairs)} edges, the mask has {expected}")
    return pairs


def v1_node_order(edgelist: pd.DataFrame, spec: kpnn2.LayeredSpec) -> Dict[str, torch.Tensor]:
    """Where v1's i-th gene and i-th TF sit in kpnn2's (alphabetical) order.

    v1's ``get_nodes_by_layer`` lists the children of each layer's rows in
    order of first appearance in the file. ``x[:, order["genes"]]`` turns a
    kpnn2-ordered tensor into v1's column order; ``*_back`` undoes it.
    """
    def order(layer_parent: int, names: Sequence[str]) -> torch.Tensor:
        position = {n: i for i, n in enumerate(names)}
        children = edgelist.loc[edgelist["layer_parent"] == layer_parent, "child"]
        in_v1_order = [position[c] for c in children.drop_duplicates() if c in position]
        if len(in_v1_order) != len(names):
            raise ValueError(f"layer {layer_parent - 1}: v1 order has {len(in_v1_order)} "
                             f"nodes, the spec has {len(names)}")
        return torch.tensor(in_v1_order)

    genes = order(1, spec.layer_nodes[0])
    tfs = order(2, spec.layer_nodes[1])
    return {"genes": genes, "genes_back": torch.argsort(genes),
            "tfs": tfs, "tfs_back": torch.argsort(tfs)}


# --- model ---------------------------------------------------------------------


class Kpnn2Model(nn.Module):
    """kpnn v1's ``KpnnModel`` rebuilt on kpnn2's ``MaskedLinear``.

    Only binary classification is implemented, which is all the Sergio
    comparison needs and all ``kpnn_config.yaml`` configures.
    """

    def __init__(
        self,
        spec: kpnn2.LayeredSpec,
        hyperparameters: dict,
        v1_order: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
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

        # With v1_order, each Dropout sees its nodes in v1's positions, so a
        # mask drawn from v1's stream hits the same gene or TF it does in v1.
        # Everything else, and every input and output, stays in kpnn2's order.
        self.v1_order = v1_order is not None
        if self.v1_order:
            for name, index in v1_order.items():
                self.register_buffer(f"order_{name}", index, persistent=False)

    def _dropout_in_v1_order(self, dropout: nn.Dropout, h: torch.Tensor, layer: str) -> torch.Tensor:
        if not self.v1_order:
            return dropout(h)
        to_v1 = getattr(self, f"order_{layer}")
        back = getattr(self, f"order_{layer}_back")
        return dropout(h[:, to_v1])[:, back]

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """The attribution path: everything except the final sigmoid.

        Attributing through the sigmoid saturates the gradients, which is why
        v1 exposes the same separate path.
        """
        h = self._dropout_in_v1_order(self.dropout_in, x, "genes")
        h = self.masked(h)
        h = self.activation(h)
        h = self._dropout_in_v1_order(self.dropout_hidden, h, "tfs")
        return self.output(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.forward_logits(x))

    @property
    def parameter_layers(self):
        """The weight-bearing layers, in spec.hops order.

        The output layer is a plain nn.Linear rather than a kpnn2 layer, but it
        carries the same (out, in) weight, so parameter_tables handles it via
        the hop mask like any other.
        """
        return [self.masked, self.output]


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


def apply_v1_init(
    model: "Kpnn2Model",
    seed: int,
    edges,
    tf_order: Optional[torch.Tensor] = None,
    output_bias: float = 0.0,
) -> None:
    """Redraw every parameter in exactly the order kpnn v1 draws them.

    ``tf_order`` (from ``v1_node_order``) places the output weights by v1's TF
    numbering; ``output_bias`` is the value v1 writes over the drawn bias,
    log(p1 / p0) of the training labels.

    A seed does not label weights; it sets the start of one shared stream, and
    every draw pops the next values. Building the modules already consumed part
    of that stream, so the seed is set again here and v1's sequence replayed:

      1. sparse layer -- one flat ``uniform_`` over nnz entries in v1's *edge*
         order, bound sqrt(6 / average_fan_in). The bias is not drawn, it is
         filled with the constant 0.01.
      2. nn.Linear(n_tfs, 1) -- PyTorch's reset_parameters, kaiming_uniform_ on
         the weight then uniform_ on the bias. v1 overwrites that bias, but the
         draw has happened, so it must happen here too.

    Without the reseed this function only reproduced v1's *distribution*: same
    bound, same spread, correlation 0.005 to v1's actual values, because
    MaskedLinear had already drawn 7830 dense entries at a shifted position.
    The flat draw plus scatter is what makes edge k receive stream[k].

    nn.Dropout and nn.PReLU draw nothing, so they do not appear.
    """
    torch.manual_seed(seed)

    n_connections = len(edges)
    n_output_neurons = model.spec.layer_dims[1]
    bound = math.sqrt(6.0 / (n_connections / n_output_neurons))

    with torch.no_grad():
        # Flat, in v1's edge order, then scattered into the dense weight.
        flat = torch.empty(n_connections).uniform_(-bound, bound)
        dense = torch.zeros_like(model.spec.hops[0].mask)
        for position, (gene, tf) in enumerate(edges):
            dense[tf, gene] = flat[position]
        model.masked.weight = dense
        if model.masked.bias is not None:
            model.masked.bias.fill_(0.01)

        if tf_order is None:
            nn.init.kaiming_uniform_(model.output.weight, a=math.sqrt(5))
        else:
            # Drawn in v1's TF order, then placed at kpnn2's positions.
            drawn = torch.empty_like(model.output.weight)
            nn.init.kaiming_uniform_(drawn, a=math.sqrt(5))
            model.output.weight[:, tf_order] = drawn
        out_bound = 1.0 / math.sqrt(n_output_neurons)
        nn.init.uniform_(model.output.bias, -out_bound, out_bound)
        model.output.bias.fill_(output_bias)


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
    edges,
    v1_order: Optional[Dict[str, torch.Tensor]] = None,
    rng: str = "v1",
) -> Tuple[Kpnn2Model, Dict[str, float], list]:
    """Train a single model for one seed, with the pre-training parameters.

    The snapshot is taken before the first optimiser step so the caller can
    report how far each weight and bias moved.

    ``rng="v1"`` consumes random numbers as kpnn v1 does (see the module
    docstring) and needs ``v1_order``; ``rng="torch"`` is the earlier loop.
    """
    if rng not in ("v1", "torch"):
        raise ValueError(f"rng must be 'v1' or 'torch', got {rng!r}")
    if rng == "v1" and v1_order is None:
        raise ValueError("rng='v1' needs v1_order (see v1_node_order)")

    set_global_seed(seed)

    train_x, train_y = tensors["train"]
    val_x, val_y = tensors["val"]

    if rng == "v1":
        model = Kpnn2Model(spec, hyperparameters, v1_order).to(device)
        # v1's KpnnModel: bias.fill_(torch.log(torch.tensor(p1 / p0))), with
        # the priors from check_class_counts as Python floats.
        n1 = int((train_y == 1).sum())
        p1, p0 = n1 / len(train_y), (len(train_y) - n1) / len(train_y)
        output_bias = float(torch.log(torch.tensor(p1 / p0)))
        tf_order = v1_order["tfs"]
    else:
        model = Kpnn2Model(spec, hyperparameters).to(device)
        output_bias, tf_order = 0.0, None
    # Building the modules already consumed random numbers; replay the stream
    # from the seed so it matches v1 draw for draw.
    apply_v1_init(model, seed, edges, tf_order, output_bias)
    initial = parameter_tables.snapshot(model.parameter_layers)
    criterion = nn.BCELoss()
    optimizer = _build_optimizer(model, hyperparameters)
    scheduler = _build_scheduler(optimizer, hyperparameters)
    lambda_l1 = float(hyperparameters.get("l1_regularization", 0) or 0)
    batch_size = int(hyperparameters["minibatch_size"])

    if rng == "v1":
        def batches():
            # v1's AnnLoader: creating the iterator draws a torch int64, the
            # batch order is a numpy permutation.
            torch.empty((), dtype=torch.int64).random_()
            order = np.random.permutation(len(train_x))
            for start in range(0, len(train_x), batch_size):
                index = order[start : start + batch_size]
                yield train_x[index], train_y[index]
    else:
        batches = DataLoader(
            TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True
        ).__iter__

    metrics: Dict[str, float] = {}
    for _ in range(int(hyperparameters["num_epochs"])):
        model.train()
        running = 0.0
        for batch_x, batch_y in batches():
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
        if rng == "v1":
            # v1's per-epoch val pass: its iterator's int64 and its shuffle.
            # The pass itself runs in eval mode and draws nothing.
            torch.empty((), dtype=torch.int64).random_()
            np.random.permutation(len(val_x))
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

    return model, metrics, initial


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
    rng: str = "v1",
) -> Path:
    """Train over all seeds and write the score CSVs. Returns the run directory.

    ``rng`` is passed to ``train_one_model``: "v1" draws as kpnn v1 does,
    "torch" reproduces runs made before that was added.
    """
    if filter_classes is None:
        filter_classes = FILTER_CLASSES

    device = torch.device(additional_settings.get("device", "cpu"))
    label_column = additional_settings.get("data_label_column", "cell_type_class_int")

    spec = kpnn2.parse_layered(to_kpnn2_edgelist(edgelist))
    edges = v1_edge_order(edgelist, spec)
    v1_order = v1_node_order(edgelist, spec)
    gene_names = spec.layer_nodes[0]
    tf_names = spec.layer_nodes[1]
    print(
        f"[{condition or 'run'}] network: {len(gene_names)} genes -> "
        f"{len(tf_names)} TFs -> {len(spec.layer_nodes[2])} output "
        f"({int(spec.hops[0].mask.sum())} edges), rng={rng}"
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

    metric_rows, edge_rows, bias_rows = [], [], []
    for seed in random_seeds:
        model, metrics, initial = train_one_model(
            spec, tensors, nn_hyperparameters, seed, device, edges, v1_order, rng
        )
        model_id = generate_model_id(seed)
        layers = model.parameter_layers
        edge_rows.append(parameter_tables.edge_table(layers, spec, initial, seed))
        bias_rows.append(parameter_tables.bias_table(layers, spec, initial, seed))

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
    # What the network learned, named: one row per edge, one per biased node.
    parameter_tables.write_tables(
        run_dir,
        pd.concat(edge_rows, ignore_index=True),
        pd.concat(bias_rows, ignore_index=True),
    )
    print(f"\nWrote {len(metric_rows)} models to {run_dir}")
    return run_dir
