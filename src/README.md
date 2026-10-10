# KPNN with kpnn2 — how to run it

Train a knowledge-primed neural network on the SERGIO data with `kpnn2`, then
check the importance scores against Karla's original kpnn v1 run.

All commands are **PowerShell**, run from the project root
(`C:\Users\durie\PycharmProjects\PythonProject`) — not from `kpnn2\`.

## 0. Before anything

Use `.venv310` (Python 3.10). `kpnn2` is installed editable from `kpnn2\`, so
edits there take effect immediately.

The scripts live in nested folders but import each other by plain module name,
so **put the three script folders on the import path once per console tab**:

```powershell
$env:PYTHONPATH="src\kpnn2_scripts;src\kpnn2_scripts\oldversions;src\comp"
```

Without it `kpnn2_port.py` cannot find `parameter_tables`, the scripts in
`oldversions\` cannot find `kpnn2_port`, and the figures in `comp\old\` cannot
find `figure_style`.

## Where the code lives

```
src/kpnn2_scripts/      kpnn2_port.py              the port: v1's training loop on kpnn2
                        kpnn2_port2.py             same, on PackedLinear instead of MaskedLinear
                        train_kpnn2_sergio_rseed.py   entry point, Karla's prepared data
                        train_kpnn2_sergio.py      entry point, locally simulated data
    oldversions/        parameter_tables.py        weight/bias tables, used by the port
                        kpnn2_port_alt.py          the port before the weight infusion
                        kpnn2_port_noinfusion.py   v1's draws, but not v1's starting weights
                        rng_isolation.py           the controlled experiment, synthetic task
                        head_to_head.py            v1 and the port in one process
                        kpnn2_native.py            kpnn2 used as intended, not a v1 replica
                        kpnn2_minimal.py           94 lines, for reading
                        same_init_experiment.py    older variant of rng_isolation
                        seed_stream_demo.py        the random stream, nine small examples

src/comp/               figure_style.py            house style, imported by every figure
                        compare_karla_kpnn2.py     a run against Karla's export
                        compare_runs.py            two of your own runs
                        plot_karla_benjamin_scatter.py   scatter + boxplots
    old/                plot_karla_kpnn2_tf_agreement.py  TF means and ranking
                        plot_stream_basics.py      how the random stream works
                        plot_rng_implications.py   what it means for the two scripts
                        plot_layer_types(presentation).py  SparseLinear vs MaskedLinear
                        plot_overlay_scatter.py    one run in grey behind another
                        plot_rng_consumption.py, plot_rng_symbolic.py, plot_seed_stream.py
                                                   earlier drafts of the three above
                        compare_karla_benjamin.py  the kpnn v1 era comparison
```

`oldversions\` and `old\` are not dead code: `parameter_tables.py` is a
dependency of the port, and the figures for the talk live in `old\`.

## Which script trains what

| Script | Use it for |
|---|---|
| `train_kpnn2_sergio_rseed.py` | the normal case: Karla's data, reproduces kpnn v1 exactly |
| `train_kpnn2_sergio.py` | locally simulated data (`<RESULTS_DIR>\KO_simulation\...`) |
| `kpnn2_native.py` | new work with kpnn2, not a v1 replica — CLI, checkpoints, all layers |
| `kpnn2_minimal.py` | reading and understanding, no CLI |

The rseed script picks its training module with `KPNN2_PORT` and the port's
random numbers with `KPNN2_RNG`. The four combinations are the four stages of
the port's history:

| `KPNN2_PORT` | `KPNN2_RNG` | v1 starting weights | v1 draws in training | TF-level r vs Karla |
|---|---|---|---|---|
| `kpnn2_port_alt` | — | no | no | 0.81–0.94 |
| `kpnn2_port_noinfusion` | — | no | yes | 0.85–0.95 |
| `kpnn2_port` | `torch` | yes | no | 0.94–0.99 |
| **`kpnn2_port`** | **`v1`** (default) | **yes** | **yes** | **1.000** |

Every script relabels 0 <-> 1 by default, because kpnn v1 is called with
`filter_classes={0: 1, 1: 0}` and the whole folder follows v1. So `class_0\`
holds the B cells and the signs are flipped — see FEHLER.txt 2.1 before reading
any score biologically.

## 1. Train

All five conditions, the exact reproduction:

```powershell
foreach ($c in "INS","TNFA","IFNG","TGFB1","IFNB") { $env:SERGIO_COND=$c; $env:RESULTS_DIR="results_kpnn2/results_kpnn2_port_v1rng"; .\.venv310\Scripts\python.exe src\kpnn2_scripts\train_kpnn2_sergio_rseed.py }
```

One of the earlier stages, e.g. without the weight infusion:

```powershell
$env:KPNN2_PORT="kpnn2_port_alt"; $env:RESULTS_DIR="results_kpnn2/results_kpnn2_port_alt"
```

Afterwards remove it again, or later runs silently use the old port:

```powershell
Remove-Item Env:KPNN2_PORT
```

## 2. Compare

Against Karla's kpnn v1 run:

```powershell
$env:RESULTS_DIR="results_kpnn2/results_kpnn2_port_v1rng"; $env:COMPARISON_OUT="karla_new_vs_kpnn2_port_v1rng"; $env:COMPARISON_LABEL="kpnn2 port (v1 rng)"
```

```powershell
.\.venv310\Scripts\python.exe src\comp\compare_karla_kpnn2.py
```

Or two of your own runs against each other, to confirm determinism:

```powershell
$env:RUN_A="results_kpnn2/results_kpnn2_port_v1rng"; $env:RUN_B="results_kpnn2/results_kpnn2_port_v1rng_20260917"; $env:LABEL_A="run A"; $env:LABEL_B="run B"; $env:COMPARISON_OUT="selfcheck_v1rng"
```

```powershell
.\.venv310\Scripts\python.exe src\comp\compare_runs.py
```

Both write `all_conditions_comparison.csv` (`;` separated, `,` decimal) into
`comp\<COMP_SUBDIR>\<COMPARISON_OUT>\`.

**Careful:** the comparison always takes the *newest* `kpnn_output_*` folder of
a condition. Two runs in the same `RESULTS_DIR` means the older one is ignored.

## 3. Plot

```powershell
$env:COMPARISON_IN="karla_new_vs_kpnn2_port_v1rng"; $env:PLOT_OUT="plots/plots_karla_new_vs_kpnn2_port_v1rng"
```

```powershell
.\.venv310\Scripts\python.exe src\comp\plot_karla_benjamin_scatter.py
```

```powershell
.\.venv310\Scripts\python.exe "src\comp\old\plot_karla_kpnn2_tf_agreement.py"
```

| Figure script | What it draws | Needs a table? |
|---|---|---|
| `plot_karla_benjamin_scatter.py` | scatter per condition, overview, diff boxplots | yes |
| `old\plot_karla_kpnn2_tf_agreement.py` | TF means ±1 SD, TF ranking | yes |
| `old\plot_overlay_scatter.py` | two runs in one panel, grey behind coloured | yes, two (`COMPARISON_IN`, `COMPARISON_BG`) |
| `old\plot_stream_basics.py` | how the random stream works | no |
| `old\plot_rng_implications.py` | what the stream means for the two scripts | no |
| `old\plot_layer_types(presentation).py` | SparseLinear vs MaskedLinear, network + matrix | no |

All figures follow `figure_style.py`: at most 160 × 160 mm, every text 6 pt
Liberation Sans in black, transparent **SVG**, colour = condition, marker =
layer, class 0 filled / class 1 open. Run names come from
`figure_style.DISPLAY_NAMES`, `X_LABEL` / `Y_LABEL` / `BG_LABEL` override them
for one call.

Liberation Sans is not installed on this machine, so the layout is measured with
Arial (same metrics) and the SVG still asks for Liberation Sans. Install the
font if the figures should display exactly as intended.

## Environment variables

| Name | Default | Meaning |
|---|---|---|
| `PYTHONPATH` | — | the three script folders, see section 0 |
| `SERGIO_COND` | `TGFB1` | condition to train |
| `RESULTS_DIR` | varies | output root of a run, also the run to compare |
| `DATA_DIR` | `Sergio_Data_Karla_new` | the prepared train/val/test splits |
| `KPNN2_SEEDS` | `20` | number of seeds |
| `KPNN2_RNG` | `v1` | `v1` draws like kpnn v1, `torch` is the older loop |
| `KPNN2_PORT` | `kpnn2_port` | which training module the rseed script uses |
| `KPNN2_DETERMINISTIC` | `1` | `0` skips `torch.use_deterministic_algorithms` |
| `KARLA_ROOT` | `results_kpnn/KPNN_results_karla_new` | the kpnn v1 reference |
| `COMPARISON_OUT` / `COMPARISON_IN` | — | table folder under `comp\<COMP_SUBDIR>\` |
| `COMPARISON_BG` | — | the second table for the overlay figure |
| `COMPARISON_LABEL` | `kpnn2` | name of the second column in the table |
| `PLOT_OUT` | — | figure folder, same root |
| `COMP_SUBDIR` | `t2` | `t1` = kpnn v1 era, `t2` = kpnn2 |
| `X_LABEL` / `Y_LABEL` / `BG_LABEL` | — | rename the runs in one figure |

PowerShell keeps `$env:` values for the whole session — set them again before
each run, or a stale one silently sends output to the wrong folder.

## Where the results live

```
results_kpnn/     kpnn v1 runs
    KPNN_results_karla_new      Karla's export, the reference
    results_karla_data_4        local v1 run on Karla's data — matches her exactly
    results_t1, _t1_2, _t1_3    local v1 runs on locally simulated data — do not match
results_kpnn2/    kpnn2 runs
    results_kpnn2_port_v1rng, _port_v1rng_20260917   the exact reproduction (r = 1.000)
    results_kpnn2_v1            the port with rng="torch"
    results_kpnn2_port_alt      the port before the weight infusion
    results_kpnn2_port_noinfusion   v1's draws without v1's weights
    results_kpnn2_native, _minimal, _port2, _headtohead, results_rng_isolation
comp/t1/          kpnn v1 era comparisons and figures
comp/t2/          kpnn2 comparisons and figures
```

Both run roots take a nested path, e.g.
`RESULTS_DIR=results_kpnn2/results_kpnn2_port_v1rng`.

## Notes

- Layer 0 is the input layer (genes), layer 1 the first hidden layer (TFs).
- The five edgelists contain TFs with no gene children. They are pruned before
  parsing, which is what makes the column sets match Karla's export.
- The port reproduces Karla seed by seed to max|diff| ≈ 3e-5. What is left is
  floating point: v1 computes with `torch.sparse.mm`, kpnn2 with dense
  `F.linear`.
- `t1_2` and `t1_3` differ from Karla because of their **input data**, not the
  package versions: the local simulation produced other values for IFNG, TNFA
  and INS (INS even half the cells).
- `FEHLER.txt` in the project root lists the open questions found along the way.
  Read 2.1 (the label swap) before interpreting any score biologically.
