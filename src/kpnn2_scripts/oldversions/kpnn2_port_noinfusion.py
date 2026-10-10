"""kpnn2 port with kpnn v1's training random numbers but WITHOUT weight infusion.

The fourth cell of the 2 x 2 behind the port's history:

                         torch training draws     v1 training draws
    own starting weights   kpnn2_port_alt           this module
    v1 starting weights    kpnn2_port rng="torch"   kpnn2_port rng="v1"

Everything is kpnn2_port.py with rng="v1" -- model, v1 batch order and DataLoader
draws, dropout in v1's node order, output bias log(p1/p0), attribution, output
layout -- except the one step that infuses v1's starting weights. That step is
swapped for ``init_without_infusion``:

1. Weights: exactly the early port's (kpnn2_port_alt): kpnn2's constructor draws,
   then v1's bound sqrt(6 / average fan-in) applied with ``uniform_`` wherever
   the stream stands, sparse bias 0.01. Same values as that run, seed for seed.
2. Stream: reseeded and advanced by exactly the draws v1's model construction
   makes, into throwaway tensors. Training therefore starts at v1's position and
   sees v1's batch order and dropout masks, while the weights are not v1's.

kpnn2_port.py itself is not modified; ``run`` swaps the init step only for the
duration of the call.

Run it through train_kpnn2_sergio_rseed.py with KPNN2_PORT=kpnn2_port_noinfusion.
"""
from __future__ import annotations

import math
from unittest import mock

import torch
from torch import nn

import kpnn2_port as port
import kpnn2_port_alt as alt


def init_without_infusion(model, seed, edges, tf_order=None, output_bias=0.0) -> None:
    """Drop-in for kpnn2_port.apply_v1_init: own weights, v1's stream position."""
    alt._apply_v1_init(model.masked, model.spec.hops[0].mask)
    with torch.no_grad():
        model.output.bias.fill_(output_bias)

    # Consume what v1's KpnnModel draws -- flat uniform_ over the edges, then
    # nn.Linear's kaiming_uniform_ and bias uniform_ -- without keeping it.
    # The number of values drawn, not their bounds, sets the stream position.
    torch.manual_seed(seed)
    n_tfs = model.spec.layer_dims[1]
    with torch.no_grad():
        torch.empty(len(edges)).uniform_()
        nn.init.kaiming_uniform_(torch.empty(1, n_tfs), a=math.sqrt(5))
        torch.empty(1).uniform_()


def run(*args, rng: str = "v1", **kwargs):
    """kpnn2_port.run with rng="v1" and the weight infusion switched off."""
    if rng != "v1":
        raise ValueError("kpnn2_port_noinfusion only exists for rng='v1'")
    with mock.patch.object(port, "apply_v1_init", init_without_infusion):
        return port.run(*args, rng="v1", **kwargs)
