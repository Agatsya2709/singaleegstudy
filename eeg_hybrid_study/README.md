# EEG Hybrid Follow-Up Study

This is the active codebase for the leakage-controlled follow-up study. It is intentionally separate from the published audit snapshot and contains no inherited performance table.

## Reproducible workflow

1. Download or point `--raw-dir` to EEGBCI EDF recordings.
2. Create unscaled epochs with `scripts/prepare_dataset.py --scaling none`.
3. Run `run_clean_study.ps1` for the seven-model benchmark.
4. Run `run_graph_ablation.ps1` for the geometry-versus-random control.
5. Inspect `results/study_analysis/` before discussing outcomes.

## Source map

| Path | Responsibility |
| --- | --- |
| `train_compare.py` | Fold-local scaling, model training, prediction exports, and plots |
| `models/baselines.py` | CNN, LSTM, CNN-LSTM, and EEGNet baselines |
| `models/hybrid_cnn_lstm_gnn.py` | GCN, GAT, and graph-enhanced variants |
| `models/graph_utils.py` | RBF spatial graph, correlation graph, randomized control, normalization |
| `scripts/prepare_dataset.py` | EDF filtering, epoching, split creation, and unscaled matrix export |
| `scripts/analyze_study_results.py` | Seed aggregation, participant metrics, and within-seed McNemar tests |
| `run_clean_study.ps1` | Corrected seven-model benchmark |
| `run_graph_ablation.ps1` | Controlled topology ablation |

## Data policy

Raw recordings, processed matrices, checkpoints, and generated results are ignored by Git. This prevents accidental publication of downloaded data and ensures reported results always link to a documented complete run.

Read [the study protocol](docs/STUDY_PROTOCOL.md) before running or reporting the experiment.
