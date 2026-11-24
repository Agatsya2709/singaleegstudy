#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_compare import EEGDataset, create_model, evaluate, load_csv_dataset


def summarize_distribution(name: str, arr: np.ndarray):
    return {
        "set": name,
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "mean_abs": float(np.mean(np.abs(arr))),
        "num_samples": int(arr.shape[0]),
        "num_features": int(arr.shape[1]),
    }


def main():
    parser = argparse.ArgumentParser(description="Audit consistency of one results folder.")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation split used inside train_compare.py")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    run_info_path = results_dir / "run_info.json"
    summary_path = results_dir / "summary.csv"
    integrity_path = results_dir / "integrity_report.json"
    if not run_info_path.exists() or not summary_path.exists():
        raise FileNotFoundError("run_info.json or summary.csv missing in {}".format(results_dir))

    with run_info_path.open("r", encoding="utf-8") as f:
        run_info = json.load(f)
    summary_df = pd.read_csv(summary_path)
    integrity = {}
    if integrity_path.exists():
        with integrity_path.open("r", encoding="utf-8") as f:
            integrity = json.load(f)

    data_dir = Path(run_info["data_dir"]).resolve()
    seed = int(run_info["seed"])
    n_channels = int(run_info["n_channels"])
    n_times = int(run_info["n_times"])
    n_classes = int(run_info["n_classes"])

    x_train, y_train, x_test, y_test = load_csv_dataset(data_dir)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.val_size, random_state=seed)
    tr_idx, val_idx = next(splitter.split(x_train, y_train))
    x_tr = x_train[tr_idx]
    x_val = x_train[val_idx]
    y_val = y_train[val_idx]

    distribution_rows = [
        summarize_distribution("train_internal", x_tr),
        summarize_distribution("val", x_val),
        summarize_distribution("test", x_test),
    ]
    distribution_df = pd.DataFrame(distribution_rows)

    labels_df = pd.DataFrame(
        {
            "set": ["val", "test"],
            "num_samples": [int(len(y_val)), int(len(y_test))],
            "class_counts": [json.dumps(np.bincount(y_val, minlength=n_classes).tolist()), json.dumps(np.bincount(y_test, minlength=n_classes).tolist())],
        }
    )

    test_ds = EEGDataset(x_test, y_test, n_channels=n_channels, n_times=n_times)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss()

    adjacency = torch.tensor(np.load(results_dir / "adjacency.npy"), dtype=torch.float32)

    metrics_rows = []
    preds_by_model = {}
    for model_name in summary_df["model"].tolist():
        model = create_model(
            model_name=model_name,
            n_channels=n_channels,
            n_times=n_times,
            n_classes=n_classes,
            adjacency=adjacency,
            compact_gat_dropout=0.35,
            compact_gat_dropedge=0.10,
        ).to(device)

        ckpt = results_dir / "{}_best.pt".format(model_name)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        _, _, y_true, y_pred = evaluate(model, test_loader, criterion, device)
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        preds_by_model[model_name] = y_pred

        cm = confusion_matrix(y_true, y_pred).tolist()
        cm_path = results_dir / "{}_cm_values.audit.json".format(model_name)
        with cm_path.open("w", encoding="utf-8") as f:
            json.dump({"model": model_name, "confusion_matrix": cm}, f, indent=2)

        metrics_rows.append(
            {
                "model": model_name,
                "test_acc_recomputed": float(accuracy_score(y_true, y_pred)),
                "test_f1_macro_recomputed": float(f1_score(y_true, y_pred, average="macro")),
                "num_correct": int((y_true == y_pred).sum()),
                "num_test": int(len(y_true)),
            }
        )

    recomputed_df = pd.DataFrame(metrics_rows).sort_values("test_acc_recomputed", ascending=False)

    pair_rows = []
    model_names = list(preds_by_model.keys())
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            a = model_names[i]
            b = model_names[j]
            pa = preds_by_model[a]
            pb = preds_by_model[b]
            pair_rows.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "prediction_agreement": float(np.mean(pa == pb)),
                }
            )
    pair_df = pd.DataFrame(pair_rows).sort_values("prediction_agreement", ascending=False)

    distribution_df.to_csv(results_dir / "audit_distribution.csv", index=False)
    labels_df.to_csv(results_dir / "audit_label_counts.csv", index=False)
    recomputed_df.to_csv(results_dir / "audit_recomputed_metrics.csv", index=False)
    pair_df.to_csv(results_dir / "audit_prediction_agreement.csv", index=False)

    audit_summary = {
        "results_dir": str(results_dir),
        "seed": seed,
        "data_dir": str(data_dir),
        "integrity_overlap_count": int(integrity.get("train_test_exact_overlap_count", -1)),
        "test_size": int(len(y_test)),
        "test_class_counts": np.bincount(y_test, minlength=n_classes).tolist(),
        "notes": [
            "Recomputed metrics are generated from saved checkpoints on the exact test split.",
            "Validation statistics are reconstructed via the same StratifiedShuffleSplit seed and val_size.",
        ],
    }
    with (results_dir / "audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    print("Audit files written to:", results_dir)


if __name__ == "__main__":
    main()
