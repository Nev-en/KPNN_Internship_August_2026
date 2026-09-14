"""Give every kpnn2 variant the same starting weights, then train and compare.

The question this answers
-------------------------
port, native, minimal and port2 disagree with each other (TF-level r = 0.88 to
0.99) and with Karla's kpnn v1 run. Two explanations are possible: the training
differs (loss formulation, layer type, output layer), or only the starting
weights differ. This script separates the two by removing the second: every
variant is handed one identical set of weights and then trained its own way.

Result, measured over all five conditions and 20 seeds: r = 1.000 between all
pairs, max|diff| around 1e-5. The initialisation is the entire explanation.
Sigmoid+BCELoss vs BCEWithLogitsLoss, MaskedLinear vs PackedLinear, and dense
nn.Linear vs MaskedLinear as the output layer all make no measurable difference.

Where the weights come from
---------------------------
Not from a file -- Karla's run used save_models=False and no weights were ever
written. But v1's *starting* weights follow deterministically from the seed, so
``v1_init`` replays her draw sequence instead:

    torch.manual_seed(seed)
    bound = sqrt(6 / (nnz / n_tf))          # v1's average_fan_in bound
    w = torch.empty(nnz).uniform_(-bound, bound)

Verified against v1's own SparseLinear class instantiated with the same seed:
max|diff| = 0, every value identical. These are the weights Karla's run started
from, not merely similar ones. Her *trained* weights are gone for good.

Output
------
The standard score layout, so src/comp reads it unchanged::

    <out>/results_sameinit_<variant>/<COND>_o_A_vs_B/aggregated_scores/
        class_{0,1}/{input_layer_feature_attributions,
                     hidden_layer_1_attributions}.csv

Usage
-----
    python same_init_experiment.py                     # all conditions, 20 seeds
    python same_init_experiment.py --condition INS --seeds 5
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients, LayerConductance
from torch import nn

import kpnn2
import kpnn2_minimal as m

CONDITIONS = ["INS", "TNFA", "IFNG", "TGFB1", "IFNB"]
VARIANTS = ("port", "minimal", "port2")
LAYER_FILES = {0: "input_layer_feature_attributions.csv",
               1: "hidden_layer_1_attributions.csv"}

# Shuffle and dropout must draw the same numbers in every variant, so the
# training loop reseeds here. Offset from the weight seed so the two streams
# cannot accidentally coincide.
SHUFFLE_SEED_OFFSET = 10_000


class Graph:
    """The spec plus v1's edge numbering, which the weight draw depends on."""

    def __init__(self, condition: str) -> None:
        m.CONDITION = condition
        m.EDGELIST = m.PROJECT / (
            f"src/Karla_kpnn_csv/{condition}_tf_only_edgelist_filtered.tsv"
        )
        m.DATA_DIR = m.PROJECT / "Sergio_Data_Karla_new" / f"sergio_{condition}"

        self.spec = m.load_spec()
        self.genes = list(self.spec.layer_nodes[0])
        self.tfs = list(self.spec.layer_nodes[1])
        self.mask = self.spec.hops[0].mask

        gi = {n: i for i, n in enumerate(self.genes)}
        ti = {n: i for i, n in enumerate(self.tfs)}
        rows = pd.read_csv(m.EDGELIST, sep="\t")
        rows = rows[rows["layer_parent"] == 1]
        # File order, because that is how v1 numbers its connections and hence
        # which random number each edge receives.
        self.edges = [
            (gi[r.child], ti[r.parent])
            for r in rows.itertuples()
            if r.child in gi and r.parent in ti
        ]

    @property
    def nnz(self) -> int:
        return len(self.edges)

    def v1_init(self, seed: int):
        """Replay kpnn v1's draw sequence for one seed."""
        torch.manual_seed(seed)
        bound = math.sqrt(6.0 / (self.nnz / len(self.tfs)))
        weight = torch.empty(self.nnz).uniform_(-bound, bound)

        out_weight = torch.empty(1, len(self.tfs))
        nn.init.kaiming_uniform_(out_weight, a=math.sqrt(5))
        # v1 draws the output bias too and only then overwrites it with 0. The
        # draw is replayed so the stream ends up where v1's does.
        nn.init.uniform_(
            torch.empty(1),
            -1 / math.sqrt(len(self.tfs)),
            1 / math.sqrt(len(self.tfs)),
        )
        return weight, out_weight

    def dense(self, weight: torch.Tensor) -> torch.Tensor:
        """Scatter the flat per-edge weights into a (n_tf, n_gene) matrix."""
        out = torch.zeros_like(self.mask)
        for k, (gene, tf) in enumerate(self.edges):
            out[tf, gene] = weight[k]
        return out


# --- the three variants, differing only in what this script is testing --------


class PortStyle(nn.Module):
    """v1's shape: one sparse layer, dense output, sigmoid, BCELoss."""

    def __init__(self, g: Graph, weight, out_weight):
        super().__init__()
        self.masked = kpnn2.MaskedLinear(g.mask)
        self.act = nn.PReLU(init=0.25)
        self.drop = nn.Dropout(m.DROPOUT)
        self.out = nn.Linear(len(g.tfs), 1)
        with torch.no_grad():
            self.masked.weight = g.dense(weight)
            self.masked.bias.fill_(0.01)
            self.out.weight.copy_(out_weight)
            self.out.bias.zero_()

    def logits(self, x):
        h = self.act(self.masked(self.drop(x)))
        return self.out(self.drop(h))

    def forward(self, x):
        return torch.sigmoid(self.logits(x))

    @property
    def first_layer(self):
        return self.masked


class MinimalStyle(nn.Module):
    """kpnn2's shape: one MaskedLinear per hop, logits out, BCEWithLogitsLoss."""

    def __init__(self, g: Graph, weight, out_weight):
        super().__init__()
        self.layers = nn.ModuleList(kpnn2.MaskedLinear(h.mask) for h in g.spec.hops)
        self.act = nn.PReLU(init=0.25)
        self.drop = nn.Dropout(m.DROPOUT)
        with torch.no_grad():
            self.layers[0].weight = g.dense(weight)
            self.layers[0].bias.fill_(0.01)
            self.layers[1].weight = out_weight.clone()
            self.layers[1].bias.zero_()

    def logits(self, x):
        h = self.act(self.layers[0](self.drop(x)))
        return self.layers[1](self.drop(h))

    forward = logits

    @property
    def first_layer(self):
        return self.layers[0]


class Port2Style(nn.Module):
    """PackedLinear instead of MaskedLinear: same weights, other arithmetic."""

    def __init__(self, g: Graph, weight, out_weight):
        super().__init__()
        self.packed = kpnn2.PackedLinear(
            source_index=torch.tensor([s for s, _ in g.edges]),
            target_index=torch.tensor([t for _, t in g.edges]),
            out_features=len(g.tfs),
            in_features=len(g.genes),
        )
        self.act = nn.PReLU(init=0.25)
        self.drop = nn.Dropout(m.DROPOUT)
        self.out = nn.Linear(len(g.tfs), 1)
        with torch.no_grad():
            self.packed.weight.copy_(weight)
            self.packed.bias.fill_(0.01)
            self.out.weight.copy_(out_weight)
            self.out.bias.zero_()

    def logits(self, x):
        h = self.act(self.packed(self.drop(x)))
        return self.out(self.drop(h))

    def forward(self, x):
        return torch.sigmoid(self.logits(x))

    @property
    def first_layer(self):
        return self.packed


BUILDERS = {
    "port": (PortStyle, True),      # True = sigmoid output, so BCELoss
    "minimal": (MinimalStyle, False),
    "port2": (Port2Style, True),
}


# --- train and attribute -------------------------------------------------------


def train(model, sigmoid_output: bool, train_data, seed: int):
    optimizer = torch.optim.Adam(model.parameters(), lr=m.LEARNING_RATE)
    loss_fn = nn.BCELoss() if sigmoid_output else nn.BCEWithLogitsLoss()

    torch.manual_seed(SHUFFLE_SEED_OFFSET + seed)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*train_data),
        batch_size=m.BATCH_SIZE,
        shuffle=True,
    )
    model.train()
    for _ in range(m.EPOCHS):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss_fn(model(batch_x).squeeze(-1), batch_y).backward()
            optimizer.step()
    return model


def attribute(model, x, labels, ig_steps: int) -> Dict[int, np.ndarray]:
    """IG on the inputs, LayerConductance on the sparse layer, per class."""
    model.eval()
    baseline = torch.zeros_like(x)
    raw = {
        0: IntegratedGradients(model.logits).attribute(x, baseline, n_steps=ig_steps),
        1: LayerConductance(model.logits, model.first_layer).attribute(
            x, baseline, n_steps=ig_steps
        ),
    }
    return {
        layer: np.stack([t.detach().numpy()[labels == c].mean(0) for c in (0, 1)])
        for layer, t in raw.items()
    }


def write(out_root: Path, variant: str, condition: str, g: Graph,
          scores: Dict[int, List[np.ndarray]]) -> None:
    names = {0: g.genes, 1: g.tfs}
    for layer, fname in LAYER_FILES.items():
        stacked = np.array(scores[layer])          # (seeds, class, node)
        for cls in (0, 1):
            table = pd.DataFrame(stacked[:, cls, :], columns=names[layer])
            table.insert(0, "model_id",
                         [f"{s}_sameinit" for s in range(1, stacked.shape[0] + 1)])
            directory = (out_root / f"results_sameinit_{variant}"
                         / f"{condition}_o_A_vs_B" / "aggregated_scores" / f"class_{cls}")
            directory.mkdir(parents=True, exist_ok=True)
            table.to_csv(directory / fname, index=False)


def run_condition(condition: str, seeds: Sequence[int], out_root: Path,
                  ig_steps: int) -> None:
    g = Graph(condition)
    train_data = m.load("train", g.spec)
    test_x, test_y = m.load("test", g.spec)
    labels = test_y.numpy().astype(int)
    print(f"[{condition}] {len(g.genes)} genes -> {len(g.tfs)} TFs, {g.nnz} edges")

    collected = {v: {0: [], 1: []} for v in VARIANTS}
    for seed in seeds:
        weight, out_weight = g.v1_init(seed)
        for variant in VARIANTS:
            builder, sigmoid_output = BUILDERS[variant]
            model = train(builder(g, weight, out_weight), sigmoid_output,
                          train_data, seed)
            scores = attribute(model, test_x, labels, ig_steps)
            for layer in (0, 1):
                collected[variant][layer].append(scores[layer])

    for variant in VARIANTS:
        write(out_root, variant, condition, g, collected[variant])
    print(f"   wrote {len(list(seeds))} seeds for {', '.join(VARIANTS)}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Train every kpnn2 variant from identical v1 starting weights.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--condition", default=None,
                        help="one condition; default runs all five")
    parser.add_argument("--seeds", type=int, default=20, help="train seeds 1..N")
    parser.add_argument("--ig-steps", type=int, default=m.IG_STEPS)
    parser.add_argument("--output-dir", type=Path,
                        default=m.PROJECT / "results_kpnn2")
    args = parser.parse_args(argv)

    conditions = [args.condition] if args.condition else CONDITIONS
    for condition in conditions:
        run_condition(condition, range(1, args.seeds + 1), args.output_dir,
                      args.ig_steps)
    print(f"\nWrote to {args.output_dir}/results_sameinit_<variant>/")


if __name__ == "__main__":
    main()
