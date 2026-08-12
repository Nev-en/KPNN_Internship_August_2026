# SERGIO -> KPNN workflow: raw PowerShell commands

The individual commands, for running steps one at a time or inspecting exactly
what gets executed. To run the whole thing in one go instead, use
`run_workflow.ps1` in this folder:

```powershell
.\src\Karla_kpnn_scripts\Sergio_simulation\run_workflow.ps1 -ResultsDir results_t1_5
```

Run from the project root: `C:\Users\durie\PycharmProjects\PythonProject`
(`Set-Location C:\Users\durie\PycharmProjects\PythonProject` first if needed).

Windows PowerShell 5.1 has no `&&` — run each command on its own, or join with `;`.

---

## 0. Check the interpreter (once per console tab)

```powershell
.\.venv310\Scripts\python.exe -c "import sys,torch,numpy; print(sys.executable); print('torch',torch.__version__,'numpy',numpy.__version__)"
```

Expect `.venv310`, torch `2.4.0+cpu`, numpy `1.26.4`.
PyCharm's terminal activates `.venv` (old Python 3.14) by default, which is why
every command below calls `.venv310\Scripts\python.exe` explicitly.

---

## A. One condition (change the two variables, then run steps 1-5)

```powershell
$cond = "TNFA"
$results = "results_t1_5"
$env:SERGIO_COND = $cond; $env:RESULTS_DIR = $results; $env:PYTHONIOENCODING = "utf-8"
```

### 1. Simulate the non-targeting dataset  (seed 1)

```powershell
.\.venv310\Scripts\python.exe .\src\Karla_kpnn_scripts\Sergio_simulation\sergio_simulate_data.py --edgelist ".\src\Karla_kpnn_csv\${cond}_tf_only_edgelist_filtered.tsv" --out_dir ".\$results\KO_simulation\sergio\sergio_$cond" --sergio_root .\SERGIO --scripts_dir .\src\Karla_kpnn_scripts\Sergio_simulation --n_cells 500 --seed 1 --n_tfs 16 --method ulm --geometry pooled --loop_mode feedforward
```

### 2. Convert to .h5ad

```powershell
.\.venv310\Scripts\python.exe .\src\Karla_kpnn_scripts\Sergio_simulation\make_adata_from_sergio.py
```

### 3. KPNN preprocessing (writes the .xz splits)

```powershell
.\.venv310\Scripts\python.exe .\src\Karla_kpnn_scripts\Sergio_simulation\preprocess_sergio_for_kpnn.py
```

### 4. Unzip the splits (the manual step)

```powershell
Get-ChildItem ".\$results\KO_simulation\sergio\sergio_$cond\preprocessed_*\*.xz" | ForEach-Object { .\.venv310\Scripts\python.exe -c "import lzma,sys; open(sys.argv[2],'wb').write(lzma.open(sys.argv[1],'rb').read())" $_.FullName $_.FullName.Substring(0,$_.FullName.Length-3); Write-Output "unzipped $($_.Name)" }
```

### 5. Train KPNN (20 seeds)

```powershell
.\.venv310\Scripts\python.exe .\src\Karla_kpnn_scripts\Sergio_simulation\train_kpnn_sergio.py
```

Then repeat from the `$cond = ...` line for the next condition.

---

## B. Fully literal, no variables — all five conditions

Target folder is `results_t1_5` throughout; replace it if you want another.

```powershell
$env:RESULTS_DIR = "results_t1_5"; $env:PYTHONIOENCODING = "utf-8"
```

### IFNB

```powershell
$env:SERGIO_COND = "IFNB"
.\.venv310\Scripts\python.exe .\src\Karla_kpnn_scripts\Sergio_simulation\sergio_simulate_data.py --edgelist .\src\Karla_kpnn_csv\IFNB_tf_only_edgelist_filtered.tsv --out_dir .\results_t1_5\KO_simulation\sergio\sergio_IFNB --sergio_root .\SERGIO --scripts_dir .\src\Karla_kpnn_scripts\Sergio_simulation --n_cells 500 --seed 1 --n_tfs 16 --method ulm --geometry pooled --loop_mode feedforward
.\.venv310\Scripts\python.exe .\src\Karla_kpnn_scripts\Sergio_simulation\make_adata_from_sergio.py
.\.venv310\Scripts\python.exe .\src\Karla_kpnn_scripts\Sergio_simulation\preprocess_sergio_for_kpnn.py
Get-ChildItem .\results_t1_5\KO_simulation\sergio\sergio_IFNB\preprocessed_*\*.xz | ForEach-Object { .\.venv310\Scripts\python.exe -c "import lzma,sys; open(sys.argv[2],'wb').write(lzma.open(sys.argv[1],'rb').read())" $_.FullName $_.FullName.Substring(0,$_.FullName.Length-3) }
.\.venv310\Scripts\python.exe .\src\Karla_kpnn_scripts\Sergio_simulation\train_kpnn_sergio.py
```

### IFNG

```powershell
$env:SERGIO_COND = "IFNG"
.\.venv310\Scripts\python.exe .\src\Karla_kpnn_scripts\Sergio_simulation\sergio_simulate_data.py --edgelist .\src\Karla_kpnn_csv\IFNG_tf_only_edgelist_filtered.tsv --out_dir .\results_t1_5\KO_simulation\sergio\sergio_IFNG --sergio_root .\SERGIO --scripts_dir .\src\Karla_kpnn_scripts\Sergio_simulation --n_cells 500 --seed 1 --n_tfs 16 --method ulm --geometry pooled --loop_mode feedforward
.\.venv310\Scripts\python.exe .\src\Karla_kpnn_scripts\Sergio_simulation\make_adata_from_sergio.py
.\.venv310\Scripts\python.exe .\src\Karla_kpnn_scripts\Sergio_simulation\preprocess_sergio_for_kpnn.py
Get-ChildItem .\results_t1_5\KO_simulation\sergio\sergio_IFNG\preprocessed_*\*.xz | ForEach-Object { .\.venv310\Scripts\python.exe -c "import lzma,sys; open(sys.argv[2],'wb').write(lzma.open(sys.argv[1],'rb').read())" $_.FullName $_.FullName.Substring(0,$_.FullName.Length-3) }
.\.venv310\Scripts\python.exe .\src\Karla_kpnn_scripts\Sergio_simulation\train_kpnn_sergio.py
```

### INS

```powershell
$env:SERGIO_COND = "INS"
.\.venv310\Scripts\python.exe .\src\Karla_kpnn_scripts\Sergio_simulation\sergio_simulate_data.py --edgelist .\src\Karla_kpnn_csv\INS_tf_only_edgelist_filtered.tsv --out_dir .\results_t1_5\KO_simulation\sergio\sergio_INS --sergio_root .\SERGIO --scripts_dir .\src\Karla_kpnn_scripts\Sergio_simulation --n_cells 500 --seed 1 --n_tfs 16 --method ulm --geometry pooled --loop_mode feedforward
.\.venv310\Scripts\python.exe .\src\Karla_kpnn_scripts\Sergio_simulation\make_adata_from_sergio.py
.\.venv310\Scripts\python.exe .\src\Karla_kpnn_scripts\Sergio_simulation\preprocess_sergio_for_kpnn.py
Get-ChildItem .\results_t1_5\KO_simulation\sergio\sergio_INS\preprocessed_*\*.xz | ForEach-Object { .\.venv310\Scripts\python.exe -c "import lzma,sys; open(sys.argv[2],'wb').write(lzma.open(sys.argv[1],'rb').read())" $_.FullName $_.FullName.Substring(0,$_.FullName.Length-3) }
.\.venv310\Scripts\python.exe .\src\Karla_kpnn_scripts\Sergio_simulation\train_kpnn_sergio.py
```

### TGFB1

```powershell
$env:SERGIO_COND = "TGFB1"
.\.venv310\Scripts\python.exe .\src\Karla_kpnn_scripts\Sergio_simulation\sergio_simulate_data.py --edgelist .\src\Karla_kpnn_csv\TGFB1_tf_only_edgelist_filtered.tsv --out_dir .\results_t1_5\KO_simulation\sergio\sergio_TGFB1 --sergio_root .\SERGIO --scripts_dir .\src\Karla_kpnn_scripts\Sergio_simulation --n_cells 500 --seed 1 --n_tfs 16 --method ulm --geometry pooled --loop_mode feedforward
.\.venv310\Scripts\python.exe .\src\Karla_kpnn_scripts\Sergio_simulation\make_adata_from_sergio.py
.\.venv310\Scripts\python.exe .\src\Karla_kpnn_scripts\Sergio_simulation\preprocess_sergio_for_kpnn.py
Get-ChildItem .\results_t1_5\KO_simulation\sergio\sergio_TGFB1\preprocessed_*\*.xz | ForEach-Object { .\.venv310\Scripts\python.exe -c "import lzma,sys; open(sys.argv[2],'wb').write(lzma.open(sys.argv[1],'rb').read())" $_.FullName $_.FullName.Substring(0,$_.FullName.Length-3) }
.\.venv310\Scripts\python.exe .\src\Karla_kpnn_scripts\Sergio_simulation\train_kpnn_sergio.py
```

### TNFA

```powershell
$env:SERGIO_COND = "TNFA"
.\.venv310\Scripts\python.exe .\src\Karla_kpnn_scripts\Sergio_simulation\sergio_simulate_data.py --edgelist .\src\Karla_kpnn_csv\TNFA_tf_only_edgelist_filtered.tsv --out_dir .\results_t1_5\KO_simulation\sergio\sergio_TNFA --sergio_root .\SERGIO --scripts_dir .\src\Karla_kpnn_scripts\Sergio_simulation --n_cells 500 --seed 1 --n_tfs 16 --method ulm --geometry pooled --loop_mode feedforward
.\.venv310\Scripts\python.exe .\src\Karla_kpnn_scripts\Sergio_simulation\make_adata_from_sergio.py
.\.venv310\Scripts\python.exe .\src\Karla_kpnn_scripts\Sergio_simulation\preprocess_sergio_for_kpnn.py
Get-ChildItem .\results_t1_5\KO_simulation\sergio\sergio_TNFA\preprocessed_*\*.xz | ForEach-Object { .\.venv310\Scripts\python.exe -c "import lzma,sys; open(sys.argv[2],'wb').write(lzma.open(sys.argv[1],'rb').read())" $_.FullName $_.FullName.Substring(0,$_.FullName.Length-3) }
.\.venv310\Scripts\python.exe .\src\Karla_kpnn_scripts\Sergio_simulation\train_kpnn_sergio.py
```

---

## Afterwards: compare to Karla and plot

```powershell
$env:RESULTS_DIR = "results_t1_5"; $env:COMPARISON_OUT = "results_t1_5"
.\.venv310\Scripts\python.exe .\src\comp_t1\compare_karla_benjamin.py
```

```powershell
$env:COMPARISON_IN = "results_t1_5"; $env:PLOT_OUT = "plots_t1_5"
.\.venv310\Scripts\python.exe .\src\comp_t1\plot_karla_benjamin_scatter.py
```

Compare two of your own runs instead:

```powershell
$env:RUN_A = "results_t1_2"; $env:RUN_B = "results_t1_3"; $env:LABEL_A = "t1_2 KPNN"; $env:LABEL_B = "t1_3 KPNN"; $env:COMPARISON_OUT = "t1_2_vs_t1_3"
.\.venv310\Scripts\python.exe .\src\comp_t1\compare_runs.py
```

---

## Notes

- `$env:...` variables live only in that console tab; re-set them in a new tab.
- `PYTHONIOENCODING=utf-8` is required — without it Python 3.10 on Windows
  crashes on the emoji in the scripts' `print` statements (cp1252).
- Steps 2, 3 and 5 take the condition and results folder from `SERGIO_COND` /
  `RESULTS_DIR`; only step 1 takes them as command-line arguments. Setting the
  variables and then running step 1 for a *different* condition will mismatch.
- `--seed 1` controls the SERGIO simulation only. KPNN training always uses
  seeds 1-20 (`random_seeds` in `train_kpnn_sergio.py`).
- Step 1 for one condition is ~4 min, training ~2 min; ~8 min per condition.
