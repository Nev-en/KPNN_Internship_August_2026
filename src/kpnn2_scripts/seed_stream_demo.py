import random
import numpy as np
import torch
import torch.nn as nn
torch.set_printoptions(precision=4)

def header(t):
    print("\n" + "=" * 70); print(t); print("=" * 70)

header("0. Look at the raw stream: the first 6 numbers after seed 42")
torch.manual_seed(42); stream = torch.rand(6)
print("stream[0:6] =", stream)

header("1. Same seed, same order -> identical results (the baseline)")
torch.manual_seed(42); a1 = torch.rand(3); b1 = torch.rand(3)
torch.manual_seed(42); a2 = torch.rand(3); b2 = torch.rand(3)
print("a identical?", torch.equal(a1, a2), "| b identical?", torch.equal(b1, b2))

header("2. Flip the order, SAME sizes -> the values swap")
torch.manual_seed(42); a_first = torch.rand(3); b_second = torch.rand(3)
torch.manual_seed(42); b_first = torch.rand(3); a_second = torch.rand(3)
print("a same in both scripts?        ", torch.equal(a_first, a_second))
print("script B's b == script A's a?  ", torch.equal(b_first, a_first))

header("3. Flip the order, DIFFERENT sizes -> a is simply different")
torch.manual_seed(42); a_A = torch.rand(2); b_A = torch.rand(3)
torch.manual_seed(42); b_B = torch.rand(3); a_B = torch.rand(2)
print("script A: a =", a_A); print("script B: a =", a_B)
print("a same in both scripts?", torch.equal(a_A, a_B))

header("4. A hidden consumer: building an unused layer shifts everything after it")
torch.manual_seed(42); layer_clean = nn.Linear(4, 2)
torch.manual_seed(42); _unused = nn.Linear(10, 10); layer_shifted = nn.Linear(4, 2)
print("weights identical?", torch.equal(layer_clean.weight, layer_shifted.weight))

header("5. Even a tiny draw shifts it (one torch.rand(1) before the layer)")
torch.manual_seed(42); _ = torch.rand(1); layer_one_draw = nn.Linear(4, 2)
print("weights identical to clean?", torch.equal(layer_clean.weight, layer_one_draw.weight))

header("6. Fix #1: re-seed immediately before the layer")
torch.manual_seed(42); _unused = nn.Linear(10, 10); _ = torch.rand(100)
torch.manual_seed(42); layer_reseeded = nn.Linear(4, 2)
print("weights identical to clean?", torch.equal(layer_clean.weight, layer_reseeded.weight))

header("7. Fix #2: a dedicated generator is immune to the global stream")
def make_weights_with_own_generator():
    g = torch.Generator().manual_seed(123)
    return torch.rand(2, 4, generator=g) * 2 - 1
torch.manual_seed(42); w_quiet = make_weights_with_own_generator()
torch.manual_seed(999); _ = torch.randn(1000); w_noisy = make_weights_with_own_generator()
print("weights identical despite global chaos?", torch.equal(w_quiet, w_noisy))

header("8. fork_rng: randomness inside the block does NOT advance the outer stream")
torch.manual_seed(42); _ = torch.rand(3); after_plain = torch.rand(3)
torch.manual_seed(42); _ = torch.rand(3)
with torch.random.fork_rng(devices=[]):
    _ = torch.randn(500)
after_fork = torch.rand(3)
print("identical?", torch.equal(after_plain, after_fork))

header("9. torch.manual_seed does NOT seed Python's random or NumPy")
torch.manual_seed(42); r1, n1 = random.random(), np.random.rand()
torch.manual_seed(42); r2, n2 = random.random(), np.random.rand()
print("python random reproduced by torch seed?", r1 == r2)
print("numpy         reproduced by torch seed?", n1 == n2)
