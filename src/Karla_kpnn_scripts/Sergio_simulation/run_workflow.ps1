<#
Runs the full SERGIO -> KPNN workflow for one or more conditions.

Steps per condition:
  1. sergio_simulate_data.py          simulate the non-targeting dataset
  2. make_adata_from_sergio.py        X_nt.csv / y_nt.csv  ->  .h5ad
  3. preprocess_sergio_for_kpnn.py    train/val/test split, writes .xz files
  4. unzip the .xz split files        (the manual step train_kpnn needs)
  5. train_kpnn_sergio.py             train 20 seeds and compute attributions

Steps 2, 3 and 5 read SERGIO_COND and RESULTS_DIR from the environment; this
script sets them per condition so the Python files need no editing.

Run it from a PowerShell console. It calls .venv310 by absolute path, so it does
not matter which interpreter the console has activated -- relevant because the
PyCharm project interpreter is currently .venv (the old Python 3.14 env).

Examples
--------
  # all five conditions into results_t1_5
  .\src\Karla_kpnn_scripts\Sergio_simulation\run_workflow.ps1 -ResultsDir results_t1_5

  # one condition
  .\src\Karla_kpnn_scripts\Sergio_simulation\run_workflow.ps1 -Conditions TNFA -ResultsDir results_t1_5

  # basal gradient of 9 instead of the default 16
  .\src\Karla_kpnn_scripts\Sergio_simulation\run_workflow.ps1 -Conditions TNFA -ResultsDir results_test -ScaleMax 9

  # print the commands without running them
  .\src\Karla_kpnn_scripts\Sergio_simulation\run_workflow.ps1 -ResultsDir results_t1_5 -DryRun
#>
param(
    [string[]] $Conditions = @("IFNB", "IFNG", "INS", "TGFB1", "TNFA"),
    [string]   $ResultsDir = "results_t1_5",
    [int]      $Seed       = 1,
    [int]      $NCells     = 500,
    [int]      $NTfs       = 16,
    [double]   $ScaleMax   = 0,     # 0 = leave at the script's default (16)
    [string]   $Venv       = ".venv310",
    [switch]   $DryRun
)

$proj    = (Resolve-Path "$PSScriptRoot\..\..\..").Path
$py      = Join-Path $proj "$Venv\Scripts\python.exe"
$scripts = $PSScriptRoot

if (-not (Test-Path $py)) { throw "interpreter not found: $py" }

$env:PYTHONIOENCODING = "utf-8"   # Py3.10 on Windows defaults stdout to cp1252
$env:PYTHONUTF8       = "1"
$env:RESULTS_DIR      = $ResultsDir

Write-Output "project     : $proj"
Write-Output "interpreter : $py"
Write-Output "results dir : $ResultsDir"
Write-Output "conditions  : $($Conditions -join ', ')"
Write-Output "seed        : $Seed"
if (-not $DryRun) {
    & $py -c "import sys,torch,numpy; print('exe         :',sys.executable); print('versions    : torch',torch.__version__,'numpy',numpy.__version__)"
    if ($LASTEXITCODE -ne 0) { throw "interpreter check failed" }
}

function Invoke-Step {
    param([string] $Name, [string[]] $Arguments)
    if ($DryRun) {
        Write-Output "  [dry-run] $py $($Arguments -join ' ')"
        return
    }
    & $py @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Name failed (exit $LASTEXITCODE)" }
}

foreach ($cond in $Conditions) {
    Write-Output ""
    Write-Output "=== $cond ==="
    $env:SERGIO_COND = $cond

    $outDir   = Join-Path $proj "$ResultsDir\KO_simulation\sergio\sergio_$cond"
    $edgelist = Join-Path $proj "src\Karla_kpnn_csv\${cond}_tf_only_edgelist_filtered.tsv"
    if (-not (Test-Path $edgelist)) { throw "$cond : edgelist not found: $edgelist" }

    # 1) SERGIO simulation
    $simArgs = @(
        (Join-Path $scripts "sergio_simulate_data.py"),
        "--edgelist", $edgelist,
        "--out_dir", $outDir,
        "--sergio_root", (Join-Path $proj "SERGIO"),
        "--scripts_dir", $scripts,
        "--n_cells", $NCells,
        "--seed", $Seed,
        "--n_tfs", $NTfs,
        "--method", "ulm",
        "--geometry", "pooled",
        "--loop_mode", "feedforward"
    )
    if ($ScaleMax -gt 0) { $simArgs += @("--scale_max", $ScaleMax) }
    Invoke-Step "$cond : sergio_simulate_data" $simArgs

    # 2) CSV -> .h5ad
    Invoke-Step "$cond : make_adata_from_sergio" @((Join-Path $scripts "make_adata_from_sergio.py"))

    # 3) KPNN preprocessing
    Invoke-Step "$cond : preprocess_sergio_for_kpnn" @((Join-Path $scripts "preprocess_sergio_for_kpnn.py"))

    # 4) manual unzip of the split files
    if ($DryRun) {
        Write-Output "  [dry-run] expand $outDir\preprocessed_*\*.xz"
    } else {
        $xz = Get-ChildItem "$outDir\preprocessed_*\*.xz" -ErrorAction SilentlyContinue
        if (-not $xz) { throw "$cond : no .xz split files found under $outDir" }
        foreach ($f in $xz) {
            $src = $f.FullName
            $dst = $src.Substring(0, $src.Length - 3)
            & $py -c "import lzma,sys; open(sys.argv[2],'wb').write(lzma.open(sys.argv[1],'rb').read())" $src $dst
            if ($LASTEXITCODE -ne 0) { throw "$cond : unzip failed on $($f.Name)" }
            Write-Output "  unzipped $($f.Name)"
        }
    }

    # 5) KPNN training
    Invoke-Step "$cond : train_kpnn_sergio" @((Join-Path $scripts "train_kpnn_sergio.py"))

    Write-Output "=== $cond DONE ==="
}

Write-Output ""
Write-Output "ALL DONE ($($Conditions.Count) condition(s) -> $ResultsDir)"
