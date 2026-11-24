Param(
    [string]$Seeds = "11,23,37",
    [string]$DataDir = "data/processed/study_s12_fists_feet"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot
$seedList = $Seeds.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }

foreach($seed in $seedList) {
  $results = "results/clean_seed$seed"
  py -3.7 train_compare.py `
    --data-dir $DataDir `
    --results-dir $results `
    --models cnn,eegnet,lstm,cnn_lstm,gcn_hybrid,gat_hybrid,graph_enhanced_cnn `
    --epochs 14 `
    --batch-size 32 `
    --val-size 0.2 `
    --lr 0.001 `
    --weight-decay 0.0005 `
    --patience 8 `
    --electrode-path assets/electrode_positions.txt `
    --graph-mode distance `
    --graph-k 10 `
    --graph-sigma 0.65 `
    --seed $seed `
    --label-smoothing 0.02 `
    --augment-std 0.01 `
    --clip-grad 1.0 `
    --compact-gat-dropout 0.35 `
    --compact-gat-dropedge 0.10

  py -3.7 scripts/audit_results.py --results-dir $results --val-size 0.2
}

py -3.7 scripts/analyze_study_results.py --results-root results --run-prefix clean_seed --seeds $Seeds --out-dir results/study_analysis
Write-Host "Leakage-controlled benchmark complete. Inspect results/study_analysis before reporting results."
