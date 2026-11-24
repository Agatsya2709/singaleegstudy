#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_compare import EEGDataset, create_model, evaluate, load_csv_dataset


def copy_if_exists(src: Path, dst: Path):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def save_confusion_csv(model_name: str, y_true, y_pred, out_dir: Path):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(np.float32) / np.clip(cm.sum(axis=1, keepdims=True), a_min=1, a_max=None)

    cm_df = pd.DataFrame(cm)
    cm_norm_df = pd.DataFrame(cm_norm)
    cm_df.to_csv(out_dir / f"{model_name}_cm_values.csv", index=False)
    cm_norm_df.to_csv(out_dir / f"{model_name}_cm_normalized.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Export separate PNG graphs and metric CSVs for report use.")
    parser.add_argument("--results-dir", default="results_college_report_ready")
    parser.add_argument("--out-dir", default="results_college_report_ready/separate_graphs")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_info_path = results_dir / "run_info.json"
    summary_path = results_dir / "summary.csv"
    if not run_info_path.exists() or not summary_path.exists():
        raise FileNotFoundError("run_info.json or summary.csv missing in {}".format(results_dir))

    with run_info_path.open("r", encoding="utf-8") as f:
        run_info = json.load(f)
    summary_df = pd.read_csv(summary_path)

    source_dir = Path(run_info.get("results_dir", str(results_dir))).resolve()
    if not source_dir.exists():
        source_dir = results_dir

    # Core tables
    summary_df.to_csv(out_dir / "summary_metrics.csv", index=False)
    table_cols = [c for c in ["model", "best_val_acc", "test_acc", "test_bal_acc", "test_f1_macro", "test_loss"] if c in summary_df.columns]
    summary_df[table_cols].to_csv(out_dir / "model_comparison_table.csv", index=False)

    # Copy all important shared PNGs
    png_names = [
        "model_comparison.png",
        "adjacency.png",
        "learned_adj.png",
    ]
    for name in png_names:
        if not copy_if_exists(results_dir / name, out_dir / name):
            copy_if_exists(source_dir / name, out_dir / name)

    # Copy per-model PNGs for every model in this run.
    for model_name in summary_df["model"].tolist():
        for suffix in ["_curve.png", "_cm.png"]:
            fname = "{}{}".format(model_name, suffix)
            if not copy_if_exists(source_dir / fname, out_dir / fname):
                copy_if_exists(results_dir / fname, out_dir / fname)

    # Optional aliases if present in packaged folders.
    for alias_name in ["baseline_curve.png", "baseline_cm.png", "gnn_curve.png", "gnn_cm.png"]:
        copy_if_exists(results_dir / alias_name, out_dir / alias_name)

    # Copy histories
    for model_name in summary_df["model"].tolist():
        copy_if_exists(source_dir / f"{model_name}_history.csv", out_dir / f"{model_name}_history.csv")

    # Recompute confusion matrix values from saved checkpoints for exact numeric tables
    data_dir = Path(run_info["data_dir"]).resolve()
    x_train, y_train, x_test, y_test = load_csv_dataset(data_dir)
    n_channels = int(run_info["n_channels"])
    n_times = int(run_info["n_times"])
    n_classes = int(run_info["n_classes"])

    test_ds = EEGDataset(x_test, y_test, n_channels=n_channels, n_times=n_times)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    adjacency_path = source_dir / "adjacency.npy"
    if not adjacency_path.exists():
        adjacency_path = results_dir / "adjacency.npy"
    adjacency = torch.tensor(np.load(adjacency_path), dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss()

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

        ckpt = source_dir / f"{model_name}_best.pt"
        if not ckpt.exists():
            ckpt = results_dir / f"{model_name}_best.pt"
        model.load_state_dict(torch.load(ckpt, map_location=device))
        _, _, y_true, y_pred = evaluate(model, test_loader, criterion, device)
        save_confusion_csv(model_name=model_name, y_true=y_true, y_pred=y_pred, out_dir=out_dir)

    notes = [
        "This folder contains separate PNG figures and CSV data tables.",
        "Figures are copied from the selected results directory.",
        "Confusion matrix CSV files are recomputed from saved model checkpoints on the test split.",
    ]
    (out_dir / "README.txt").write_text("\n".join(notes), encoding="utf-8")

    print("Exported separate graphs and data to:", out_dir)


if __name__ == "__main__":
    main()
