"""Is "different random numbers" the whole reason kpnn v1 and kpnn2 disagree?

The question
------------
kpnn2_port reproduces v1's starting weights exactly (apply_v1_init), yet its
TF scores still reach only r = 0.94-0.99 per seed against Karla's run. The
suspicion: the remaining scatter is not a modelling difference but simply
other random numbers being used during training. This script tests that on a
small synthetic task, in one process, with v1 and kpnn2 sharing everything that
can be shared -- data, network, config, seeding, optimiser, training loop,
attribution code -- so that whatever still differs has nowhere to hide.

The task
--------
A made-up regulatory network: N_TFS transcription factors, each regulating a
handful of N_GENES genes with a random sign. Every cell gets a latent activity
per TF; the genes are a noisy, rectified sum of their regulators. Class 1 cells
have DRIVERS TFs shifted upwards, class 0 cells do not. The splits are exactly
balanced, so v1's class-prior output bias is log(1) = 0 and cannot confound.
The task has its own numpy Generator and never touches the global streams.

What v1 does with the streams (read from kpnn 0.1.12, torch 2.4, anndata 0.11)
--------------------------------------------------------------------------------
* temp_seed(seed) seeds torch, numpy and random, then KpnnModel draws
  SparseLinear's flat uniform_ and nn.Linear's kaiming_uniform_ + bias uniform_.
* Batches come from anndata's BatchIndexSampler: np.random.permutation --
  the *numpy* stream, not torch.
* Every `for batch in loader` makes a DataLoader iterator, which draws one int64
  from the *torch* stream (_BaseDataLoaderIter._base_seed). v1 iterates the val
  loader every epoch too, so that is two torch draws per epoch.
* Dropout draws its masks from the torch stream, batch by batch.
kpnn2_port instead shuffles with torch's RandomSampler (one extra torch draw per
epoch, and a different order) and has no val pass. From the first batch on, the
two runs therefore see other batches and other dropout masks.

The arms
--------
Every arm is compared, seed by seed, with the reference `v1`:

    v1_repeat                   v1 again, same seed. Must be 0: determinism.
    v1_other_seed               v1 with seed + 1000. Pure random-number scatter,
                                the yardstick for "as different as a new seed".
    kpnn2_infused_v1loop        kpnn2 model, weights copied from v1's freshly
                                built model, trained by v1's own ModelTrainer on
                                v1's own AnnLoader. Only the layer class differs.
    kpnn2_owninit_v1loop        as above, but kpnn2's own reset_parameters.
                                Only the random numbers of the init differ.
    kpnn2_infused_portloop      v1's weights, kpnn2_port's training loop
                                (torch DataLoader shuffle, no val pass).
    kpnn2_infused_portloop_v1rng  the port loop, but drawing exactly what v1
                                draws: numpy batch order and the two torch int64
                                draws per epoch. Nothing else changed.
    kpnn2_port_torch_rng        kpnn2_port.train_one_model with rng="torch",
                                the port as it was before.
    kpnn2_port_v1_rng           kpnn2_port.train_one_model with rng="v1".

How to read the result
----------------------
If the hypothesis holds: kpnn2_infused_v1loop, kpnn2_infused_portloop_v1rng and
kpnn2_port_v1_rng agree with v1 to float precision (r = 1.000), while
kpnn2_owninit_v1loop, kpnn2_infused_portloop and kpnn2_port_torch_rng scatter
about as much as v1_other_seed. Any arm that fits neither pattern points at a
real difference.

The node names of this task make v1's and kpnn2's node order coincide, so the
reordering around Dropout in kpnn2_port is not exercised here; on the SERGIO
edgelists it is, and head_to_head.py checks that case.

Output
------
    results_kpnn2/results_rng_isolation/run_<timestamp>/
        per_seed.csv    one row per arm and seed
        summary.csv     median per arm

Usage (from the project root)
-----------------------------
    .\\.venv310\\Scripts\\python.exe src\\kpnn2_scripts\\rng_isolation.py
    .\\.venv310\\Scripts\\python.exe src\\kpnn2_scripts\\rng_isolation.py --seeds 3
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import logging
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import anndata as ad
import numpy as np
import pandas as pd
import torch
import yaml
from captum.attr import IntegratedGradients, LayerConductance
from scipy import sparse
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import kpnn2
import kpnn2_port as port
from kpnn.training.core.train_kpnns.match_data_and_network import match_data_and_network
from kpnn.training.core.train_kpnns.model_training.ModelTrainer import ModelTrainer
from kpnn.training.core.train_kpnns.model_training.setup_training import (
    ModelTrainerInitializer,
    SparseLinear,
    check_class_counts,
    create_all_data_loaders,
)
from kpnn.training.utils.train_kpnns.ConfigManager import ConfigManager
from kpnn.training.utils.train_kpnns.control_inputs import check_and_convert_labels
from kpnn.training.utils.train_kpnns.misc import temp_seed

PROJECT = Path(__file__).resolve().parents[2]
CONFIG_YAML = PROJECT / "src/Karla_kpnn_scripts/Sergio_simulation/kpnn_config.yaml"
LABEL = "cell_type_class_int"

N_GENES = 400
N_TFS = 40
DRIVERS = 3
CELLS = {"train": 2000, "val": 500, "test": 500}
TASK_SEED = 20260914
OTHER_SEED_OFFSET = 1000

LOG = logging.getLogger("rng_isolation")
LOG.addHandler(logging.NullHandler())
LOG.propagate = False


# --- the synthetic task --------------------------------------------------------


def make_task(rng: np.random.Generator):
    """Edgelist in v1's format plus balanced train/val/test AnnData.

    Node names are zero-padded and the rows sorted, so v1's first-appearance
    order and kpnn2's alphabetical order coincide. That lets both models read
    the very same input columns without any reindexing.
    """
    genes = [f"g{i:03d}" for i in range(N_GENES)]
    tfs = [f"TF{i:02d}" for i in range(N_TFS)]

    # Every gene gets one regulator round-robin, plus a few random extra ones.
    signs = np.zeros((N_TFS, N_GENES), dtype=np.float32)
    for g in range(N_GENES):
        signs[g % N_TFS, g] = 1.0
    extra = rng.random((N_TFS, N_GENES)) < 0.02
    signs[extra] = 1.0
    signs *= rng.choice([-1.0, 1.0], size=signs.shape)

    rows = [
        (tfs[t], f"{genes[g]}{port.LEAF_SUFFIX}", 1, 0)
        for g in range(N_GENES)
        for t in range(N_TFS)
        if signs[t, g] != 0
    ]
    rows += [("output", tf, 2, 1) for tf in tfs]
    edgelist = pd.DataFrame(rows, columns=["parent", "child", "layer_parent", "layer_child"])

    drivers = sorted(rng.choice(N_TFS, size=DRIVERS, replace=False).tolist())
    degree = np.abs(signs).sum(axis=0)

    datasets = {}
    for split, n in CELLS.items():
        labels = np.array([0, 1] * (n // 2))
        activity = rng.normal(size=(n, N_TFS))
        activity[np.ix_(labels == 1, drivers)] += 1.0
        expression = 1.0 + activity @ signs / np.sqrt(degree) + 0.5 * rng.normal(size=(n, N_GENES))
        expression = np.maximum(expression, 0.0).astype(np.float32)
        obs = pd.DataFrame({LABEL: labels}, index=[f"{split}_{i}" for i in range(n)])
        datasets[split] = ad.AnnData(
            X=sparse.csr_matrix(expression), obs=obs, var=pd.DataFrame(index=genes)
        )
    return edgelist, datasets, [tfs[d] for d in drivers]


# --- v1, assembled from its own parts ------------------------------------------


@dataclass
class V1:
    """Everything train_kpnns builds before its seed loop, built the same way."""

    config: ConfigManager
    initializer: ModelTrainerInitializer
    loaders: tuple
    datasets: Dict[str, ad.AnnData]
    genes: List[str]
    tfs: List[str]
    nr_classes: int


def prepare_v1(edgelist, datasets, hyper, settings, out_dir: Path) -> V1:
    """train_kpnns, lines 261-386, minus logging and file output.

    The config dicts are copied because v1 writes 'class_priors' into them.
    """
    config = ConfigManager(copy.deepcopy(hyper), copy.deepcopy(settings), out_dir)
    datasets = check_and_convert_labels(copy.deepcopy(datasets), config, LOG)
    nr_classes = len(datasets["train"].obs[LABEL].unique())
    datasets, network = match_data_and_network(datasets=datasets, edgelist=edgelist)
    check_class_counts(config=config, datasets=datasets, logger=LOG)
    initializer = ModelTrainerInitializer(
        network=network,
        hyperparameters=config.nn_hyperparameters,
        classification_mode=config.additional_settings["classification_mode"],
        nr_classes=nr_classes,
    )
    loaders = create_all_data_loaders(
        datasets=datasets,
        minibatch_size=config.nn_hyperparameters["minibatch_size"],
        use_cuda=False,
        data_label_column=LABEL,
    )
    nodes = network.get_nodes_by_layer()
    return V1(config, initializer, loaders, datasets, nodes[0], nodes[1], nr_classes)


def train_v1_loop(v1: V1, model: nn.Module) -> List[float]:
    """v1's train_model up to trainer.train -- the part that moves weights.

    The loss plot and test pass that follow in train_model run after training
    and inside no RNG-relevant scope, so they are left out.
    """
    v1.initializer.model = model
    criterion, optimizer, scheduler = v1.initializer.initialize_training_components()
    trainer = ModelTrainer(model, criterion, optimizer, scheduler, v1.config, v1.nr_classes)
    train_loader, val_loader, _ = v1.loaders
    trainer.train(train_loader, val_loader, LOG)
    return trainer.train_loss_history


# --- kpnn2 ---------------------------------------------------------------------


class SqueezedOutput(nn.Module):
    """v1's ModelTrainer expects (N,) from a binary model; Kpnn2Model gives (N, 1)."""

    def __init__(self, inner: port.Kpnn2Model) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner(x).squeeze(-1)


def v1_sparse_layer(model) -> SparseLinear:
    return next(layer for layer in model.layers if isinstance(layer, SparseLinear))


def v1_output_layer(model) -> nn.Linear:
    return next(layer for layer in model.layers if isinstance(layer, nn.Linear))


def v1_prelu(model) -> nn.PReLU:
    return next(layer for layer in model.layers if isinstance(layer, nn.PReLU))


def infuse(target: port.Kpnn2Model, donor) -> None:
    """Copy every parameter of a v1 KpnnModel into a Kpnn2Model.

    SparseLinear stores (gene, tf) index pairs and a flat weight vector;
    MaskedLinear stores the dense (tf, gene) matrix. Node orders are identical
    by construction (see make_task), so the indices can be used directly.
    """
    sparse_layer = v1_sparse_layer(donor)
    gene_index, tf_index = sparse_layer.indices
    dense = torch.zeros_like(target.masked.mask)
    dense[tf_index, gene_index] = sparse_layer.weights.detach()
    with torch.no_grad():
        target.masked.parametrizations.weight.original.copy_(dense)
        target.masked.bias.copy_(sparse_layer.bias)
        target.activation.weight.copy_(v1_prelu(donor).weight)
        target.output.weight.copy_(v1_output_layer(donor).weight)
        target.output.bias.copy_(v1_output_layer(donor).bias)


def train_port_loop(model, tensors, hyper, v1_rng: bool) -> List[float]:
    """kpnn2_port.train_one_model's loop (lines 356-386), optionally drawing like v1.

    v1_rng=False is the port as written. v1_rng=True swaps in exactly the draws
    v1's loop makes, and nothing else: the numpy permutation for the batch
    order, and one torch int64 each time a train or val DataLoader iterator is
    created. The val pass itself runs in eval mode and draws nothing.
    """
    criterion = nn.BCELoss()
    optimizer = port._build_optimizer(model, hyper)
    scheduler = port._build_scheduler(optimizer, hyper)
    batch_size = int(hyper["minibatch_size"])
    train_x, train_y = tensors["train"]
    n_val = len(tensors["val"][0])

    if v1_rng:
        def batches():
            torch.empty((), dtype=torch.int64).random_()   # train iterator's _base_seed
            order = np.random.permutation(len(train_x))
            for start in range(0, len(train_x), batch_size):
                index = order[start : start + batch_size]
                yield train_x[index], train_y[index]
    else:
        loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
        batches = loader.__iter__

    history = []
    for _ in range(int(hyper["num_epochs"])):
        model.train()
        batch_losses = []
        for batch_x, batch_y in batches():
            optimizer.zero_grad()
            loss = criterion(model(batch_x).squeeze(-1), batch_y.float())
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        if v1_rng:
            torch.empty((), dtype=torch.int64).random_()   # val iterator's _base_seed
            np.random.permutation(n_val)                   # val loader shuffles too
        if scheduler is not None:
            scheduler.step()
        # Mean over batches, as v1 reports it (the port weights by cell count),
        # so epoch losses are comparable across arms. Training is unaffected.
        history.append(sum(batch_losses) / len(batch_losses))
    return history


# --- shared measurement --------------------------------------------------------


@dataclass
class Outcome:
    init: Dict[str, np.ndarray]
    trained: Dict[str, np.ndarray]
    predictions: np.ndarray
    tf_scores: np.ndarray      # (class, TF)
    gene_scores: np.ndarray    # (class, gene)
    rng_after_init: Optional[str] = None
    losses: List[float] = field(default_factory=list)


def rng_fingerprint() -> str:
    torch_state = torch.get_rng_state().numpy().tobytes()
    numpy_state = np.random.get_state()
    digest = hashlib.sha1(torch_state + numpy_state[1].tobytes() + bytes([numpy_state[2] % 256]))
    return digest.hexdigest()[:12]


def params_v1(model) -> Dict[str, np.ndarray]:
    sparse_layer = v1_sparse_layer(model)
    gene_index, tf_index = sparse_layer.indices
    dense = torch.zeros(sparse_layer.out_features, sparse_layer.in_features)
    dense[tf_index, gene_index] = sparse_layer.weights.detach()
    return {
        "sparse_weight": dense.numpy().copy(),
        "sparse_bias": sparse_layer.bias.detach().numpy().copy(),
        "prelu": v1_prelu(model).weight.detach().numpy().copy(),
        "out_weight": v1_output_layer(model).weight.detach().numpy().copy(),
        "out_bias": v1_output_layer(model).bias.detach().numpy().copy(),
    }


def params_kpnn2(model: port.Kpnn2Model) -> Dict[str, np.ndarray]:
    return {
        "sparse_weight": model.masked.weight.detach().numpy().copy(),
        "sparse_bias": model.masked.bias.detach().numpy().copy(),
        "prelu": model.activation.weight.detach().numpy().copy(),
        "out_weight": model.output.weight.detach().numpy().copy(),
        "out_bias": model.output.bias.detach().numpy().copy(),
    }


def measure(model, forward_logits, sparse_layer, test_x, labels, n_steps) -> tuple:
    """Predictions plus class-mean IG (genes) and LayerConductance (TFs).

    One function for every arm, so attribution code cannot differ between them.
    Batched by 64 like v1 and the port; zero baseline, logits path.
    """
    model.eval()
    with torch.no_grad():
        predictions = torch.sigmoid(forward_logits(test_x)).squeeze(-1).numpy()

    ig = IntegratedGradients(forward_logits)
    conductance = LayerConductance(forward_logits, sparse_layer)
    genes, tfs = [], []
    for start in range(0, len(test_x), 64):
        batch = test_x[start : start + 64].clone().requires_grad_(True)
        baseline = torch.zeros_like(batch)
        genes.append(ig.attribute(batch, baselines=baseline, n_steps=n_steps).detach())
        tfs.append(conductance.attribute(batch, baselines=baseline, n_steps=n_steps).detach())
    genes, tfs = torch.cat(genes).numpy(), torch.cat(tfs).numpy()
    by_class = lambda a: np.stack([a[labels == c].mean(axis=0) for c in (0, 1)])
    return predictions, by_class(tfs), by_class(genes)


# --- the arms ------------------------------------------------------------------


class Experiment:
    def __init__(self, edgelist, datasets, hyper, settings, out_dir: Path, n_steps: int):
        self.hyper = hyper
        self.n_steps = n_steps
        self.parts = prepare_v1(edgelist, datasets, hyper, settings, out_dir)

        self.spec = kpnn2.parse_layered(port.to_kpnn2_edgelist(edgelist))
        self.edges = port.v1_edge_order(edgelist, self.spec)
        self.v1_order = port.v1_node_order(edgelist, self.spec)
        if list(self.spec.layer_nodes[0]) != self.parts.genes or list(self.spec.layer_nodes[1]) != self.parts.tfs:
            raise RuntimeError("v1 and kpnn2 order the nodes differently; make_task must prevent this")

        # One tensor set, taken from v1's matched AnnData, for every arm.
        self.tensors = {}
        for split, data in self.parts.datasets.items():
            x = torch.tensor(data.X.toarray(), dtype=torch.float32)
            y = torch.tensor(data.obs[LABEL].to_numpy(), dtype=torch.float32)
            self.tensors[split] = (x, y)
        self.test_labels = self.tensors["test"][1].numpy().astype(int)

    def _finish_v1(self, model, init, fingerprint, losses) -> Outcome:
        x = self.tensors["test"][0]
        predictions, tf_scores, gene_scores = measure(
            model, model.forward_logits, v1_sparse_layer(model), x, self.test_labels, self.n_steps
        )
        return Outcome(init, params_v1(model), predictions, tf_scores, gene_scores, fingerprint, losses)

    def _finish_kpnn2(self, model, init, fingerprint, losses) -> Outcome:
        x = self.tensors["test"][0]
        predictions, tf_scores, gene_scores = measure(
            model, model.forward_logits, model.masked, x, self.test_labels, self.n_steps
        )
        return Outcome(init, params_kpnn2(model), predictions, tf_scores, gene_scores, fingerprint, losses)

    def _kpnn2_without_drawing(self) -> port.Kpnn2Model:
        """Build the kpnn2 model inside fork_rng, so its own init leaves the stream untouched."""
        with torch.random.fork_rng(devices=[]):
            return port.Kpnn2Model(self.spec, self.hyper)

    def _infused(self) -> port.Kpnn2Model:
        """kpnn2 model carrying the weights v1 draws for the current seed.

        The v1 model is really built here, not replayed, so the stream moves on
        exactly as far as it does in the reference run.
        """
        model = self._kpnn2_without_drawing()
        infuse(model, self.parts.initializer.initialize_model(device="cpu"))
        return model

    # Each arm: seed -> Outcome.

    def v1(self, seed: int) -> Outcome:
        with temp_seed(seed):
            model = self.parts.initializer.initialize_model(device="cpu")
            init, fingerprint = params_v1(model), rng_fingerprint()
            losses = train_v1_loop(self.parts, model)
        return self._finish_v1(model, init, fingerprint, losses)

    def v1_repeat(self, seed: int) -> Outcome:
        return self.v1(seed)

    def v1_other_seed(self, seed: int) -> Outcome:
        return self.v1(seed + OTHER_SEED_OFFSET)

    def kpnn2_infused_v1loop(self, seed: int) -> Outcome:
        with temp_seed(seed):
            model = self._infused()
            init, fingerprint = params_kpnn2(model), rng_fingerprint()
            losses = train_v1_loop(self.parts, SqueezedOutput(model))
        return self._finish_kpnn2(model, init, fingerprint, losses)

    def kpnn2_owninit_v1loop(self, seed: int) -> Outcome:
        with temp_seed(seed):
            model = port.Kpnn2Model(self.spec, self.hyper)
            init, fingerprint = params_kpnn2(model), rng_fingerprint()
            losses = train_v1_loop(self.parts, SqueezedOutput(model))
        return self._finish_kpnn2(model, init, fingerprint, losses)

    def kpnn2_infused_portloop(self, seed: int) -> Outcome:
        with temp_seed(seed):
            model = self._infused()
            init, fingerprint = params_kpnn2(model), rng_fingerprint()
            losses = train_port_loop(model, self.tensors, self.hyper, v1_rng=False)
        return self._finish_kpnn2(model, init, fingerprint, losses)

    def kpnn2_infused_portloop_v1rng(self, seed: int) -> Outcome:
        with temp_seed(seed):
            model = self._infused()
            init, fingerprint = params_kpnn2(model), rng_fingerprint()
            losses = train_port_loop(model, self.tensors, self.hyper, v1_rng=True)
        return self._finish_kpnn2(model, init, fingerprint, losses)

    def _port(self, seed: int, rng: str) -> Outcome:
        with temp_seed(seed):   # only to restore the outer state afterwards; the port seeds itself
            model, _, _ = port.train_one_model(
                self.spec, self.tensors, self.hyper, seed, torch.device("cpu"),
                self.edges, self.v1_order, rng,
            )
        # The port keeps its initial weights and stream position to itself.
        return self._finish_kpnn2(model, {}, None, [])

    def kpnn2_port_torch_rng(self, seed: int) -> Outcome:
        return self._port(seed, "torch")

    def kpnn2_port_v1_rng(self, seed: int) -> Outcome:
        return self._port(seed, "v1")


ARMS = [
    "v1_repeat",
    "v1_other_seed",
    "kpnn2_infused_v1loop",
    "kpnn2_owninit_v1loop",
    "kpnn2_infused_portloop",
    "kpnn2_infused_portloop_v1rng",
    "kpnn2_port_torch_rng",
    "kpnn2_port_v1_rng",
]


# --- comparison ----------------------------------------------------------------


def max_diff(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> float:
    shared = [k for k in a if k in b]
    if not shared:
        return float("nan")
    return max(float(np.abs(a[k] - b[k]).max()) for k in shared)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def drivers_found(outcome: Outcome, drivers_index: List[int]) -> int:
    """How many true driver TFs rank in the top len(drivers) by |conductance|."""
    top = np.argsort(-np.abs(outcome.tf_scores).sum(axis=0))[: len(drivers_index)]
    return len(set(top.tolist()) & set(drivers_index))


def compare(ref: Outcome, other: Outcome, drivers_index: List[int], labels: np.ndarray) -> dict:
    return {
        "init_max_diff": max_diff(ref.init, other.init),
        "rng_after_init_equal": (
            None if other.rng_after_init is None else ref.rng_after_init == other.rng_after_init
        ),
        "epoch1_loss_diff": (
            abs(ref.losses[0] - other.losses[0]) if ref.losses and other.losses else float("nan")
        ),
        "trained_max_diff": max_diff(ref.trained, other.trained),
        "pred_max_diff": float(np.abs(ref.predictions - other.predictions).max()),
        "tf_r": pearson(ref.tf_scores, other.tf_scores),
        "tf_max_diff": float(np.abs(ref.tf_scores - other.tf_scores).max()),
        "gene_r": pearson(ref.gene_scores, other.gene_scores),
        "test_acc": float(((other.predictions >= 0.5) == labels).mean()),
        "drivers_found": drivers_found(other, drivers_index),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Train v1 and kpnn2 side by side on a synthetic task and isolate RNG effects.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seeds", type=int, default=10, help="seeds 1..N")
    parser.add_argument("--ig-steps", type=int, default=50)
    parser.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "results_kpnn2/results_rng_isolation")
    args = parser.parse_args(argv)

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    config = yaml.safe_load(CONFIG_YAML.read_text())
    hyper = config["nn_hyperparameters"]
    settings = config["additional_settings"]

    run_dir = args.output_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    edgelist, datasets, drivers = make_task(np.random.default_rng(TASK_SEED))
    experiment = Experiment(edgelist, datasets, hyper, settings, run_dir, args.ig_steps)
    drivers_index = [experiment.parts.tfs.index(d) for d in drivers]
    print(f"task: {N_GENES} genes -> {N_TFS} TFs, {len(edgelist)} edges, drivers {drivers}")
    print(f"arms: {', '.join(args.arms)}; seeds 1..{args.seeds}\n")

    rows = []
    for seed in range(1, args.seeds + 1):
        started = time.perf_counter()
        labels = experiment.test_labels
        reference = experiment.v1(seed)
        rows.append({"arm": "v1", "seed": seed,
                     "test_acc": float(((reference.predictions >= 0.5) == labels).mean()),
                     "drivers_found": drivers_found(reference, drivers_index)})
        for arm in args.arms:
            outcome = getattr(experiment, arm)(seed)
            rows.append({"arm": arm, "seed": seed,
                         **compare(reference, outcome, drivers_index, labels)})
        print(f"seed {seed:2d} done in {time.perf_counter() - started:5.1f}s")

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(run_dir / "per_seed.csv", index=False)

    numeric = per_seed.drop(columns=["seed", "rng_after_init_equal"]).groupby("arm", sort=False)
    summary = numeric.median(numeric_only=True)
    summary["rng_after_init_equal"] = per_seed.groupby("arm", sort=False)["rng_after_init_equal"].agg(
        lambda s: "n/a" if s.isna().all() else f"{int(s.fillna(False).sum())}/{s.notna().sum()}"
    )
    summary.to_csv(run_dir / "summary.csv")

    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.3g}".format):
        print("\nmedian over seeds, every arm against v1 with the same seed:\n")
        print(summary)
    print(f"\nWrote {run_dir}")


if __name__ == "__main__":
    main()
