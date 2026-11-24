Param(
    [string]$Seeds = "11,23,37",
    [string]$DataDir = "data/processed/study_s12_fists_feet"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot
$seedList = $Seeds.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }

foreach($seed in $seedList) {
  $shared = @(
    "--data-dir", $DataDir,
    "--epochs", "14",
    "--batch-size", "32",
    "--val-size", "0.2",
    "--lr", "0.001",
    "--weight-decay", "0.0005",
    "--patience", "8",
    "--seed", $seed,
    "--label-smoothing", "0.02",
    "--augment-std", "0.01",
    "--clip-grad", "1.0",
    "--compact-gat-dropout", "0.35",
    "--compact-gat-dropedge", "0.10",
    "--electrode-path", "assets/electrode_positions.txt"
  )

  py -3.7 train_compare.py @shared --models cnn --graph-mode distance --graph-k 10 --graph-sigma 0.65 --results-dir "results/ablation_cnn_seed$seed"
  py -3.7 train_compare.py @shared --models graph_enhanced_cnn --graph-mode distance --graph-k 5 --graph-sigma 0.65 --results-dir "results/ablation_distance_k5_seed$seed"
  py -3.7 train_compare.py @shared --models graph_enhanced_cnn --graph-mode distance --graph-k 10 --graph-sigma 0.65 --results-dir "results/ablation_distance_k10_seed$seed"
  py -3.7 train_compare.py @shared --models graph_enhanced_cnn --graph-mode distance --graph-k 15 --graph-sigma 0.65 --results-dir "results/ablation_distance_k15_seed$seed"
  py -3.7 train_compare.py @shared --models graph_enhanced_cnn --graph-mode random --graph-k 10 --graph-sigma 0.65 --results-dir "results/ablation_random_k10_seed$seed"
}

Write-Host "Ablation runs complete. Keep each configuration separate when aggregating; do not pool test trials across seeds."
