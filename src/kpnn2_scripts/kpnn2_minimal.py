"""Train a knowledge-primed neural network with kpnn2 -- the short version.

A KPNN is an ordinary neural network whose connections are not "everything to
everything", but exactly the edges of a biological graph: genes feed the
transcription factors that regulate them, and those feed one output neuron that
predicts the cell's class. kpnn2 turns that graph into connection masks; the
rest is plain PyTorch, and this file is all of it.

Read it top to bottom. Roughly:

    edgelist  --kpnn2.parse_layered-->  spec  (which node is in which layer,
                                               and a mask per layer telling us
                                               which connections may exist)
    spec      --kpnn2.MaskedLinear--->  the layers of the model
    .h5ad     --kpnn2.align_inputs--->  a tensor whose columns are in the
                                        order the spec expects
    training  --Captum------------->    per-gene and per-TF importance scores

Change the settings below and run it:

    python kpnn2_minimal.py

For the fuller version -- command line options, checkpoints, several output
formats, best-epoch restore -- see kpnn2_native.py.
"""
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients, LayerConductance
from torch import nn

import kpnn2

# --- settings -----------------------------------------------------------------

PROJECT = Path(__file__).resolve().parents[2]

CONDITION = "INS"
EDGELIST = PROJECT / f"src/Karla_kpnn_csv/{CONDITION}_tf_only_edgelist_filtered.tsv"
DATA_DIR = PROJECT / "Sergio_Data_Karla_new" / f"sergio_{CONDITION}"
OUT_DIR = PROJECT / "results_kpnn2" / "results_kpnn2_minimal" / CONDITION

SEEDS = range(1, 21)  # one trained model per seed, to see how much they vary
EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.02
DROPOUT = 0.2
IG_STEPS = 50  # how finely Captum integrates; higher = slower, more accurate

# kpnn v1 is called with filter_classes={0: 1, 1: 0}, which relabels 0 <-> 1.
# True here so every script in this folder follows v1; set False for the
# natural labels from class_mapping.txt.
SWAP_LABELS = True


# --- the graph ----------------------------------------------------------------


def load_spec():
    """Read the edgelist and let kpnn2 work out the layers.

    The file lists edges top-down (parent = TF, child = gene). kpnn2 wants them
    in the direction the data flows, so child becomes `source` and parent
    becomes `target`.

    A few TFs feed the output but have no genes of their own. They would look
    like inputs to kpnn2 and land in layer 0 next to the genes, so drop them.
    """
    raw = pd.read_csv(EDGELIST, sep="\t")
    edges = raw.rename(columns={"child": "source", "parent": "target"})

    while True:
        targets = set(edges["target"])
        childless = {
            row.source
            for row in edges.itertuples()
            if row.layer_child > 0 and row.source not in targets
        }
        if not childless:
            break
        edges = edges[~edges["source"].isin(childless)]

    return kpnn2.parse_layered(edges[["source", "target"]].drop_duplicates())


# --- the model ----------------------------------------------------------------


class Kpnn(nn.Module):
    """One masked layer per hop in the graph.

    `nn.Linear` connects every input to every output. `kpnn2.MaskedLinear` is
    the same thing with a mask laid over the weights, so only the connections
    the graph allows can ever be non-zero. `spec.hops` gives one mask per layer
    transition, so this works for a graph of any depth.

    The model returns *logits* -- raw scores, not probabilities. Squashing them
    into 0..1 is left to the loss function, which does it more accurately.
    """

    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.layers = nn.ModuleList(kpnn2.MaskedLinear(hop.mask) for hop in spec.hops)
        # PReLU is the non-linearity; without one, stacked layers would collapse
        # into a single linear function. One per layer except the last.
        self.activations = nn.ModuleList(nn.PReLU(init=0.25) for _ in spec.hops[:-1])
        # Dropout randomly zeroes activations during training, which discourages
        # the model from leaning on any single gene.
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        h = self.dropout(x)
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:  # last layer emits logits, unchanged
                h = self.dropout(self.activations[i](h))
        return h


# --- the data -----------------------------------------------------------------


def load(split, spec):
    """Read one .h5ad file into a tensor of inputs and a vector of labels."""
    adata = ad.read_h5ad(next(DATA_DIR.glob(f"*_{split}.h5ad")))

    matrix = adata.X
    if hasattr(matrix, "toarray"):  # some files store a sparse matrix
        matrix = matrix.toarray()

    # The graph calls genes "TP53_leaf"; the data file calls them "TP53".
    frame = pd.DataFrame(
        np.asarray(matrix, dtype=np.float32),
        columns=[f"{name}_leaf" for name in adata.var.index],
    )
    # align_inputs puts the columns into spec.input_nodes order. That ordering
    # is the whole point: column i of the tensor is then the node the spec says.
    x = kpnn2.align_inputs(frame, spec)

    # Written as a mapping rather than "1 - y" so that a third class raises a
    # KeyError here instead of quietly turning into -1 further downstream.
    relabel = {0: 1, 1: 0} if SWAP_LABELS else {0: 0, 1: 1}
    y = np.array([relabel[int(v)] for v in adata.obs["cell_type_class_int"]], dtype=np.int64)
    return x, torch.tensor(y, dtype=torch.float32)


# --- training -----------------------------------------------------------------


def train(spec, train_data, seed):
    """Train one model. This is the standard PyTorch loop, nothing kpnn2-specific."""
    torch.manual_seed(seed)  # same seed -> same starting weights -> same result
    model = Kpnn(spec)

    # Adam decides how far to move the weights after each batch.
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # This loss expects logits and measures how wrong a yes/no prediction is.
    loss_fn = nn.BCEWithLogitsLoss()

    x, y = train_data
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=BATCH_SIZE, shuffle=True
    )

    model.train()  # switch dropout on
    for _ in range(EPOCHS):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()  # forget the previous batch's gradients
            loss = loss_fn(model(batch_x).squeeze(-1), batch_y)
            loss.backward()  # work out which way each weight should move
            optimizer.step()  # move them
    return model


# --- importance scores --------------------------------------------------------


def importance(model, spec, x, y):
    """Ask Captum which genes and which TFs drove the prediction.

    Two methods, because the questions differ:
      * IntegratedGradients works on the model's *inputs*, so it scores genes.
      * LayerConductance works on a layer *inside* the model, so it scores the
        TF neurons, which are not inputs and so out of reach for the first.

    Both compare each cell against an all-zero baseline cell.

    Returns {layer index: DataFrame of class x node scores}.
    """
    model.eval()  # switch dropout off, so scores are deterministic
    baseline = torch.zeros_like(x)

    raw = {
        0: IntegratedGradients(model).attribute(x, baseline, n_steps=IG_STEPS),
        1: LayerConductance(model, model.layers[0]).attribute(
            x, baseline, n_steps=IG_STEPS
        ),
    }

    out = {}
    for layer, scores in raw.items():
        rows = {}
        for cls in (0, 1):
            mean = scores[y == cls].mean(dim=0).detach()
            # map_node_attributions attaches the node names from the spec, so we
            # never have to track which column was which gene.
            rows[cls] = kpnn2.map_node_attributions(mean, spec, layer=layer).to_pandas()
        out[layer] = rows
    return out


# --- run ----------------------------------------------------------------------


def main():
    spec = load_spec()
    print(f"[{CONDITION}] layers {spec.layer_dims}")

    train_data = load("train", spec)
    test_x, test_y = load("test", spec)
    print(f"  {len(train_data[0])} training cells, {len(test_x)} test cells")
    print("  labels swapped 0<->1, as kpnn v1 does" if SWAP_LABELS
          else "  labels kept as in class_mapping.txt -- NOT comparable to a v1 run")

    # collected[layer][class] becomes one row per seed
    collected = {0: {0: [], 1: []}, 1: {0: [], 1: []}}

    for seed in SEEDS:
        model = train(spec, train_data, seed)
        scores = importance(model, spec, test_x, test_y.numpy().astype(int))
        for layer in (0, 1):
            for cls in (0, 1):
                collected[layer][cls].append(scores[layer][cls].rename(seed))

        with torch.no_grad():
            correct = ((model(test_x).squeeze(-1) > 0) == test_y.bool()).float().mean()
        print(f"  seed {seed:2d}  test accuracy {correct:.4f}")

    # One CSV per layer and class: rows are seeds, columns are node names.
    names = {0: "input_layer_feature_attributions.csv", 1: "hidden_layer_1_attributions.csv"}
    for layer in (0, 1):
        for cls in (0, 1):
            table = pd.DataFrame(collected[layer][cls])
            table.insert(0, "model_id", [f"{s}_minimal" for s in table.index])
            path = OUT_DIR / "aggregated_scores" / f"class_{cls}" / names[layer]
            path.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(path, index=False)

    print(f"\nWrote scores to {OUT_DIR}")


if __name__ == "__main__":
    main()
