#!/usr/bin/env python
"""Aggregate per-seed metrics and run within-seed paired CNN-vs-GCN comparisons."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score


def metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def mcnemar_exact(left, right):
    left_correct = left["y_pred"].to_numpy() == left["y_true"].to_numpy()
    right_correct = right["y_pred"].to_numpy() == right["y_true"].to_numpy()
    b = int(np.sum(left_correct & ~right_correct))
    c = int(np.sum(~left_correct & right_correct))
    discordant = b + c
    p_value = 1.0 if discordant == 0 else float(binomtest(min(b, c), n=discordant, p=0.5, alternative="two-sided").pvalue)
    return {"cnn_only_correct": b, "gcn_only_correct": c, "discordant_trials": discordant, "exact_p_value": p_value}


def main():
    parser = argparse.ArgumentParser(description="Summarize leakage-controlled EEG study outputs.")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--run-prefix", default="clean_seed")
    parser.add_argument("--seeds", default="11,23,37")
    parser.add_argument("--out-dir", default="results/study_analysis")
    args = parser.parse_args()

    root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(token.strip()) for token in args.seeds.split(",") if token.strip()]
    rows = []
    paired_rows = []
    participant_rows = []

    for seed in seeds:
        run_dir = root / "{}_{}".format(args.run_prefix, seed)
        for prediction_path in sorted(run_dir.glob("*_predictions.csv")):
            model = prediction_path.name.replace("_predictions.csv", "")
            frame = pd.read_csv(prediction_path)
            row = metrics(frame["y_true"], frame["y_pred"])
            row.update({"seed": seed, "model": model, "n_test_trials": len(frame)})
            rows.append(row)
            if "subject_id" in frame.columns:
                for subject_id, subset in frame.groupby("subject_id"):
                    subject_row = metrics(subset["y_true"], subset["y_pred"])
                    subject_row.update({"seed": seed, "model": model, "subject_id": int(subject_id), "n_test_trials": len(subset)})
                    participant_rows.append(subject_row)

        cnn_path = run_dir / "cnn_predictions.csv"
        gcn_path = run_dir / "gcn_hybrid_predictions.csv"
        if cnn_path.exists() and gcn_path.exists():
            cnn = pd.read_csv(cnn_path).sort_values("trial_index").reset_index(drop=True)
            gcn = pd.read_csv(gcn_path).sort_values("trial_index").reset_index(drop=True)
            if not np.array_equal(cnn["y_true"].to_numpy(), gcn["y_true"].to_numpy()):
                raise RuntimeError("CNN and GCN labels differ in {}".format(run_dir))
            paired = mcnemar_exact(cnn, gcn)
            paired["seed"] = seed
            paired_rows.append(paired)

    if not rows:
        raise RuntimeError("No prediction files found. Run train_compare.py first.")
    raw = pd.DataFrame(rows).sort_values(["model", "seed"])
    raw.to_csv(out_dir / "per_seed_metrics.csv", index=False)
    numeric = ["accuracy", "balanced_accuracy", "macro_f1", "macro_precision", "macro_recall", "mcc"]
    aggregate = raw.groupby("model")[numeric].agg(["mean", "std"])
    aggregate.columns = ["{}_{}".format(metric, statistic) for metric, statistic in aggregate.columns]
    aggregate.reset_index().to_csv(out_dir / "model_summary_mean_sd.csv", index=False)
    if participant_rows:
        pd.DataFrame(participant_rows).sort_values(["model", "seed", "subject_id"]).to_csv(
            out_dir / "per_participant_metrics.csv", index=False
        )
    if paired_rows:
        pd.DataFrame(paired_rows).sort_values("seed").to_csv(out_dir / "cnn_vs_gcn_mcnemar_by_seed.csv", index=False)


if __name__ == "__main__":
    main()
