"""Named tables of what a KPNN actually learned: one row per edge, one per node.

The attribution CSVs say which nodes mattered for a prediction. They say nothing
about the parameters themselves -- what weight sits on the STAT1 -> IRF9 edge,
how far it moved during training, what bias each node carries. This module turns
those parameters into tables with real names, so they can be inspected, joined
against the edgelist, or plotted like any other result.

Two tables
----------
edge_table    source, target, source_layer, target_layer, weight
              plus initial_weight and delta when a snapshot is supplied
bias_table    node, layer, bias
              plus initial_bias and delta likewise

Both layer types are handled. `MaskedLinear` stores a dense (out, in) weight and
the edge is one cell of it; `PackedLinear` stores a 1-D vector indexed by its
own source/target arrays. Which one a model uses is detected, not configured.

Typical use::

    before = snapshot(model.layers)          # right after building the model
    ...train...
    edges = edge_table(model.layers, spec, initial=before)
    nodes = bias_table(model.layers, spec, initial=before)

    edges.sort_values("delta", key=abs, ascending=False).head(20)

Masked-out cells are not edges and never appear: the row count equals the number
of ones in the hop masks, which is the edge count `parse_layered` reports.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import pandas as pd
import torch

import kpnn2


def source_node_names(spec: kpnn2.LayeredSpec, hop) -> List[str]:
    """Column names of a hop's mask.

    A hop reads every source layer it draws from, concatenated in the order
    `source_layers` lists them -- that is also the column order of the mask, so
    a skip edge lands in the right place without special handling here.
    """
    names: List[str] = []
    for layer in hop.source_layers:
        names.extend(spec.layer_nodes[layer])
    return names


def source_layer_of_column(spec: kpnn2.LayeredSpec, hop) -> List[int]:
    """Which layer each mask column comes from, parallel to source_node_names."""
    layers: List[int] = []
    for layer in hop.source_layers:
        layers.extend([layer] * len(spec.layer_nodes[layer]))
    return layers


def snapshot(layers: Sequence[torch.nn.Module]) -> List[Dict[str, torch.Tensor]]:
    """Detached copies of every layer's weight and bias, for a later diff.

    Take this immediately after building the model and before the first
    optimiser step; the tables then carry initial_weight and delta.
    """
    out = []
    for layer in layers:
        entry = {"weight": layer.weight.detach().clone()}
        if getattr(layer, "bias", None) is not None:
            entry["bias"] = layer.bias.detach().clone()
        out.append(entry)
    return out


def _edges_of_layer(layer, spec: kpnn2.LayeredSpec, hop, weight: torch.Tensor):
    """(source index, target index, weight) for every real edge of one hop."""
    if isinstance(layer, kpnn2.PackedLinear):
        # Already one entry per edge, in the order the indices were given.
        return (
            layer.source_index.tolist(),
            layer.target_index.tolist(),
            weight.tolist(),
        )
    # MaskedLinear: the mask says which cells of the dense weight are edges.
    rows, cols = hop.mask.nonzero(as_tuple=True)
    return cols.tolist(), rows.tolist(), weight[rows, cols].tolist()


def edge_table(
    layers: Sequence[torch.nn.Module],
    spec: kpnn2.LayeredSpec,
    initial: Optional[Sequence[Dict[str, torch.Tensor]]] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """One row per edge, named by node.

    `layers` must line up with `spec.hops`. Pass the `snapshot` taken before
    training as `initial` to get initial_weight and delta alongside.
    """
    if len(layers) != len(spec.hops):
        raise ValueError(
            f"{len(layers)} layers for {len(spec.hops)} hops -- they must line up"
        )

    frames = []
    for index, (layer, hop) in enumerate(zip(layers, spec.hops)):
        sources = source_node_names(spec, hop)
        source_layers = source_layer_of_column(spec, hop)
        targets = spec.layer_nodes[hop.target_layer]

        src_idx, tgt_idx, values = _edges_of_layer(layer, spec, hop, layer.weight.detach())
        frame = pd.DataFrame({
            "source": [sources[i] for i in src_idx],
            "target": [targets[i] for i in tgt_idx],
            "source_layer": [source_layers[i] for i in src_idx],
            "target_layer": hop.target_layer,
            "weight": values,
        })

        if initial is not None:
            _, _, before = _edges_of_layer(layer, spec, hop, initial[index]["weight"])
            frame.insert(4, "initial_weight", before)
            frame["delta"] = frame["weight"] - frame["initial_weight"]

        frames.append(frame)

    table = pd.concat(frames, ignore_index=True)
    if seed is not None:
        table.insert(0, "seed", seed)
    return table


def bias_table(
    layers: Sequence[torch.nn.Module],
    spec: kpnn2.LayeredSpec,
    initial: Optional[Sequence[Dict[str, torch.Tensor]]] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """One row per node that carries a bias.

    Input-layer nodes have none -- a bias belongs to the layer that *computes* a
    node, and nothing computes the inputs. So the table covers layers 1 upward.
    """
    frames = []
    for index, (layer, hop) in enumerate(zip(layers, spec.hops)):
        bias = getattr(layer, "bias", None)
        if bias is None:
            continue
        frame = pd.DataFrame({
            "node": list(spec.layer_nodes[hop.target_layer]),
            "layer": hop.target_layer,
            "bias": bias.detach().tolist(),
        })
        if initial is not None and "bias" in initial[index]:
            frame.insert(2, "initial_bias", initial[index]["bias"].tolist())
            frame["delta"] = frame["bias"] - frame["initial_bias"]
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["node", "layer", "bias"])

    table = pd.concat(frames, ignore_index=True)
    if seed is not None:
        table.insert(0, "seed", seed)
    return table


def write_tables(directory, edges: pd.DataFrame, biases: pd.DataFrame) -> None:
    """Write both tables next to a run's other output."""
    from pathlib import Path

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    edges.to_csv(directory / "edge_weights.csv", index=False)
    biases.to_csv(directory / "node_bias.csv", index=False)
