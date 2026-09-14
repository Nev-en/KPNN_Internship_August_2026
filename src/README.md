# KPNN with kpnn2 — how to run it

Train a knowledge-primed neural network on the SERGIO data with `kpnn2`, then
check the importance scores against Karla's original kpnn v1 run.

All commands are **PowerShell**, run from the project root
(`C:\Users\durie\PycharmProjects\PythonProject`) — not from `kpnn2\`.

## Workflow

```
edgelist + .h5ad  ->  training script  ->  results_*/  ->  compare_*  ->  comp/t2/  ->  plot_*  ->  comp/t2/plots/
 src/Karla_kpnn_csv   src/kpnn2_scripts    scores CSV    src/comp      table CSV     src/comp    PNG
```

Two questions are worth asking of any run, and there is one comparison for each:

| Question | Script | Expected |
|---|---|---|
| Is the pipeline reproducible? | `compare_runs.py` | max\|diff\| = 0 |
| Does it find the same biology as kpnn v1? | `compare_karla_kpnn2.py` | port: r = 1.000 per seed; native/minimal: TF-level r ≈ 0.99 |

Only `kpnn2_port.py` (via `train_kpnn2_sergio_rseed.py`) reproduces Karla's run
seed by seed: it draws its random numbers exactly as kpnn v1 does
(`rng="v1"`, the default), and then matches her export to max|diff| ≈ 3e-5 in
all five conditions. Every other script consumes the random number stream
differently, which alone scatters the scores as much as a new seed would
(`rng_isolation.py` isolates this). Judge those on the TF means across the
20 seeds, not per seed.

## Which training script

| Script | Use it for |
|---|---|
| `kpnn2_minimal.py` | reading and understanding — 94 lines of code, no CLI |
| `kpnn2_native.py` | real work — CLI, checkpoints, all layers, several output formats |
| `train_kpnn2_sergio_rseed.py` | reproducing kpnn v1 exactly (runs `kpnn2_port.py`; `KPNN2_RNG=torch` for the older, non-exact loop) |
| `rng_isolation.py` | showing that the remaining v1/kpnn2 differences are only random numbers |

`minimal` and `native` agree with each other to r ≈ 0.997.

## 1. Train

Pick one. Both write 20 seeds per condition.

**Minimal** — edit `CONDITION` at the top of the file first, one condition per run:

```powershell
.\.venv310\Scripts\python.exe src\kpnn2_scripts\kpnn2_minimal.py
```

**Native** — all five conditions:

```powershell
foreach ($c in "INS","TNFA","IFNG","TGFB1","IFNB") { .\.venv310\Scripts\python.exe src\kpnn2_scripts\kpnn2_native.py --condition $c }
```

Every script relabels 0 <-> 1 by default, because kpnn v1 is called with
`filter_classes={0: 1, 1: 0}` and the whole folder follows v1. So `class_0/`
holds the B cells and the signs are flipped -- see FEHLER.txt 2.1 before
reading any score biologically. Pass `--no-swap-labels` (native) or set
`SWAP_LABELS = False` (minimal) for the natural labels; results are then no
longer comparable to Karla's export.

## 2. Compare

Against Karla's kpnn v1 run:

```powershell
$env:RESULTS_DIR="results_kpnn2/results_kpnn2_native"; $env:COMPARISON_OUT="karla_new_vs_kpnn2_native"; $env:COMPARISON_LABEL="kpnn2 native"
```

```powershell
.\.venv310\Scripts\python.exe src\comp\compare_karla_kpnn2.py
```

Or two of your own runs against each other, to confirm determinism:

```powershell
$env:RUN_A="results_kpnn2/results_kpnn2_minimal"; $env:RUN_B="results_kpnn2/results_kpnn2_minimal_b"; $env:LABEL_A="run A"; $env:LABEL_B="run B"; $env:COMPARISON_OUT="selfcheck_native_a_vs_b"
```

```powershell
.\.venv310\Scripts\python.exe src\comp\compare_runs.py
```

## 3. Plot

```powershell
$env:COMPARISON_IN="karla_new_vs_kpnn2_native"; $env:PLOT_OUT="plots/plots_karla_new_vs_kpnn2_native"
```

```powershell
.\.venv310\Scripts\python.exe src\comp\plot_karla_benjamin_scatter.py
```

```powershell
.\.venv310\Scripts\python.exe src\comp\plot_karla_kpnn2_tf_agreement.py
```

The first gives scatter plots per seed and disagreement boxplots; the second the
TF-level agreement.

## Environment variables

| Name | Default | Meaning |
|---|---|---|
| `RESULTS_DIR` | `results_kpnn2/results_kpnn2_native` | which run to compare |
| `KARLA_ROOT` | `results_kpnn/KPNN_results_karla_new` | the kpnn v1 reference |
| `COMPARISON_OUT` / `COMPARISON_IN` | — | table folder under `comp/<COMP_SUBDIR>/` |
| `PLOT_OUT` | — | figure folder, same root |
| `COMP_SUBDIR` | `t2` | `t1` = kpnn v1 era, `t2` = kpnn2 |
| `COMPARISON_LABEL` | `kpnn2` | second column name, i.e. the plot axis |

PowerShell keeps `$env:` values for the whole session — set them again before
each run, or a stale one silently sends output to the wrong folder.

## Where things live

```
results_kpnn/    kpnn v1 runs      (KPNN_results_karla_new, results_karla_data_4, ...)
results_kpnn2/   kpnn2 runs        (results_kpnn2_v1 = port, _native, _minimal, ...)
comp/t1/         kpnn v1 era comparisons and figures
comp/t2/         kpnn2 comparisons and figures
```

Both run roots take a nested path, e.g. `RESULTS_DIR=results_kpnn2/results_kpnn2_minimal`.

## Notes

- Use `.venv310` (Python 3.10). `kpnn2` is installed editable from `kpnn2\`, so
  edits there take effect immediately.
- Layer 0 is the input layer (genes), layer 1 the first hidden layer (TFs).
- The five edgelists contain TFs with no gene children. They are pruned before
  parsing, which is what makes the column sets match Karla's export.
- `FEHLER.txt` in the project root lists the open questions found along the way.
  Read 2.1 (the label swap) before interpreting any score biologically.
