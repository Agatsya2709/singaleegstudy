Param(
    [string]$Seeds = "11,23,37"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$seedList = $Seeds.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }
$dataDir = "data/processed/tune_s12_ff"

foreach($s in $seedList){
  $results = "results_clean_all_models_s12_ff_seed$s"
  Write-Host "Running seed $s -> $results"
  py -3.7 train_compare.py `
    --data-dir $dataDir `
    --results-dir $results `
    --models cnn,lstm,cnn_lstm,gcn_hybrid,gat_hybrid,graph_enhanced_cnn `
    --epochs 14 `
    --batch-size 32 `
    --val-size 0.2 `
    --lr 0.001 `
    --weight-decay 0.0005 `
    --patience 8 `
    --electrode-path assets/electrode_positions.txt `
    --graph-k 10 `
    --graph-sigma 0.65 `
    --seed $s `
    --label-smoothing 0.02 `
    --augment-std 0.01 `
    --clip-grad 1.0 `
    --compact-gat-dropout 0.35 `
    --compact-gat-dropedge 0.10

  py -3.7 scripts/audit_results.py --results-dir $results --val-size 0.2
  py -3.7 scripts/export_separate_graphs.py --results-dir $results --out-dir "$results/separate_graphs"
}

Write-Host "Done. You can compare summaries using:"
Write-Host "results_clean_all_models_s12_ff_multiseed_summary.csv"
