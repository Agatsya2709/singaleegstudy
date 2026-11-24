# Leakage-Controlled EEG Motor-Imagery Study

This repository is the active follow-up study to the preserved `cnnlstmeeg` audit snapshot. It contains a redesigned, reproducible protocol for evaluating whether scalp-geometry graphs add value to low-data EEG motor-imagery classification.

## What changed from the audit snapshot

- The dataset writer now stores **unscaled epochs** by default.
- The trainer performs the train/validation split first and fits `StandardScaler` only on the inner training fold.
- The outer test set is transformed only with those fold-local training statistics.
- Every model exports test-trial predictions, participant-level metrics, macro precision/recall, balanced accuracy, MCC, and confusion matrices.
- A controlled graph ablation compares geometry graphs (`k=5, 10, 15`) to a degree-preserving randomized graph.
- EEGNet is added as an EEG-specific compact benchmark.

The original reported results are not copied or relabelled as results from this protocol. New claims may be made only after complete runs for all configured seeds.

## Current public evidence

The repository includes one complete, leakage-controlled seed-11 run under [`eeg_hybrid_study/reported_runs/seed11_preliminary`](eeg_hybrid_study/reported_runs/seed11_preliminary). It is included so readers can inspect the exact predictions, metrics, curves, confusion matrices, graph figures, and integrity metadata used in the smoke benchmark.

It is **preliminary**, not the final result table: seeds 23 and 37, the three-seed aggregate, and the graph-topology ablation remain to be completed before drawing comparative conclusions.

## Start here

```powershell
cd eeg_hybrid_study
py -3.7 -m pip install -r requirements_py37.txt

# Generate the raw, unscaled dataset from local EDF downloads.
py -3.7 scripts/prepare_dataset.py --raw-dir data/raw_edf --out-dir data/processed/study_s12_fists_feet --subjects 1,2 --runs 6,10,14 --task fists_feet --tmin 0.5 --tmax 3.5 --scaling none

# Run the corrected seven-model benchmark and aggregate the three seeds.
.\run_clean_study.ps1 -Seeds "11,23,37"

# Run the controlled graph ablation.
.\run_graph_ablation.ps1 -Seeds "11,23,37"
```

See [the experiment documentation](eeg_hybrid_study/docs/STUDY_PROTOCOL.md) for the scope, controls, and limitations.
