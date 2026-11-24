# Preliminary Seed-11 Evidence

This directory is a transparent snapshot of one **complete** run of the corrected leakage-controlled protocol. It is not a final, three-seed result and must not be averaged with the historical `cnnlstmeeg` outputs.

## Scope

- Seed: `11`
- Task: fists versus feet motor imagery
- Participants: 1 and 2
- Outer split: 72 training trials and 18 untouched test trials
- Inner split: 57 training trials and 15 validation trials
- Scaling: fit on the 57 inner-training trials only
- Graph: RBF electrode-distance graph, `k=10`, `sigma=0.65`
- Models: CNN, EEGNet, LSTM, CNN-LSTM, GCN hybrid, GAT hybrid, graph-enhanced CNN

## Files

- `summary.csv`: all test metrics for the seven models.
- `run_info.json`: complete parameter and preprocessing provenance.
- `integrity_report.json`: overlap and class-distribution checks.
- `*_predictions.csv`: test-trial labels and predictions; participant IDs are included.
- `*_per_subject_metrics.csv`: participant-level performance for this split.
- `*_history.csv`, `*_curve.png`: training trajectories.
- `*_cm.png`: normalized confusion matrices with count annotations.
- `adjacency.png` and `learned_adj.png`: graph visualizations.
- `model_comparison.png`: within-seed comparison only.

## Interpretation

CNN reached 17/18 correct trials (94.44%) and the GCN hybrid 16/18 (88.89%) on this seed. With only 18 test trials, one prediction changes accuracy by 5.56 percentage points. These values are therefore evidence that the corrected pipeline runs end to end, not proof that one model outperforms another.
