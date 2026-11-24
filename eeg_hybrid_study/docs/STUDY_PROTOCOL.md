# Leakage-Controlled Follow-Up Protocol

## Research question

Does a scalp-geometry graph improve motor-imagery EEG classification over convolutional and recurrent baselines in a low-data setting?

## Dataset and primary benchmark

The initial benchmark uses EEGBCI participants 1 and 2, motor-imagery runs 6, 10, and 14, and the binary fists-versus-feet task. The outer split is a stratified 80/20 trial split. This is a within-cohort pilot only; it is not a participant-independent generalization study.

## Leakage-controlled preprocessing

`prepare_dataset.py --scaling none` saves filtered, epoched trial matrices without standardization. `train_compare.py` then creates the internal validation split from outer-training trials, fits `StandardScaler` on the inner training fold only, and applies it to validation and outer-test trials. The run metadata records this as `fit_on_inner_training_fold`.

The trainer refuses datasets that lack `scaling_scope: unscaled_epochs` unless `--allow-legacy-prescaled-input` is explicitly supplied. That compatibility option must not be used for the new study's reported results.

## Models

The clean benchmark includes CNN, EEGNet, LSTM, CNN-LSTM, CNN-LSTM-GCN, CNN-LSTM-GAT, and graph-enhanced CNN. EEGNet is a compact EEG-specific reference baseline. It is a seventh model in this follow-up study, not a model retroactively added to the original six-model result table.

## Graph ablation

`run_graph_ablation.ps1` holds the graph-enhanced CNN architecture and optimizer fixed while varying graph topology:

- geometry-based RBF graph with `k=5`;
- geometry-based RBF graph with `k=10`;
- geometry-based RBF graph with `k=15`;
- degree-preserving randomized control derived from the `k=10` graph.

The distance graph uses `w_ij = exp(-d_ij^2 / (2 sigma^2))` with `sigma=0.65`. Neighbourhoods are union-symmetrized and the final adjacency receives self-loops plus symmetric degree normalization. The random control preserves the undirected degree sequence through double-edge swaps and shuffles the original edge weights.

## Evaluation and reporting

For each seed, record accuracy, balanced accuracy, macro precision, macro recall, macro F1, MCC, confusion matrices, test predictions, and participant-level metrics. Aggregate model scores across seeds with mean and sample standard deviation; do not pool repeated predictions from the same test trials as independent observations. `analyze_study_results.py` also performs exact McNemar tests for CNN versus GCN separately within each seed.

## Subject-disjoint extension

The preparation script supports participant-disjoint test partitions with `--test-subjects`. Use a larger participant set before making subject-generalization claims. For example, after downloading EDFs for participants 1-20, reserve `--test-subjects 17-20` and ensure all preprocessing fitting remains within the training participants.
