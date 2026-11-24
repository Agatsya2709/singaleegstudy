#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_compare import EEGDataset, build_adjacency, create_model, evaluate, get_class_weights, get_shape, load_csv_dataset, run_epoch


def main():
    parser = argparse.ArgumentParser(description="Run graph diagnostics for EEG hybrid models.")
    parser.add_argument("--data-dir", default="data/processed/EEG-Motor-Movement-Imagery-Dataset")
    parser.add_argument("--electrode-path", default="assets/electrode_positions.txt")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ablation-epochs", type=int, default=5)
    parser.add_argument("--graph-k", type=int, default=12)
    parser.add_argument("--graph-sigma", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = Path(".").resolve()
    data_dir = Path(args.data_dir).resolve()
    electrode_path = Path(args.electrode_path).resolve()
    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_test, y_test = load_csv_dataset(data_dir)
    n_channels, n_times, _ = get_shape(data_dir, x_train.shape[1], 0)
    n_classes = int(max(np.max(y_train), np.max(y_test)) + 1)

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    tr_idx, val_idx = next(split.split(x_train, y_train))
    x_tr, y_tr = x_train[tr_idx], y_train[tr_idx]
    x_val, y_val = x_train[val_idx], y_train[val_idx]

    adjacency, graph_source = build_adjacency(
        data_dir=data_dir,
        electrode_path=electrode_path,
        x_train_shaped=x_tr.reshape(-1, n_channels, n_times),
        graph_k=args.graph_k,
        sigma=args.graph_sigma,
    )

    print("=== 1) Adjacency usage check ===")
    print("graph_source:", graph_source)
    print("Adjacency matrix shape:", tuple(adjacency.shape))
    print("Non-zero entries:", int((adjacency > 0).sum().item()))
    print("Max/min weight:", float(adjacency.max().item()), float(adjacency.min().item()))

    train_ds = EEGDataset(x_tr, y_tr, n_channels=n_channels, n_times=n_times)
    val_ds = EEGDataset(x_val, y_val, n_channels=n_channels, n_times=n_times)
    test_ds = EEGDataset(x_test, y_test, n_channels=n_channels, n_times=n_times)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = get_class_weights(y_train=y_train, n_classes=n_classes)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    print("\n=== 2) Gradient flow check (gat_hybrid + gcn_hybrid) ===")
    for model_name in ["gat_hybrid", "gcn_hybrid"]:
        model = create_model(
            model_name=model_name,
            n_channels=n_channels,
            n_times=n_times,
            n_classes=n_classes,
            adjacency=adjacency,
            compact_gat_dropout=0.45,
            compact_gat_dropedge=0.15,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        x_batch, y_batch = next(iter(train_loader))
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()

        print("Model:", model_name)
        for name, param in model.named_parameters():
            lname = name.lower()
            if ("gat" in lname) or ("gcn" in lname):
                if param.grad is None:
                    print("  {}: grad_norm = NONE".format(name))
                else:
                    print("  {}: grad_norm = {:.6f}".format(name, float(param.grad.norm().item())))

    print("\n=== 3) Graph-only ablation (compact_gat) ===")
    graph_only = create_model(
        model_name="compact_gat",
        n_channels=n_channels,
        n_times=n_times,
        n_classes=n_classes,
        adjacency=adjacency,
        compact_gat_dropout=0.45,
        compact_gat_dropedge=0.15,
    ).to(device)

    graph_optimizer = torch.optim.AdamW(graph_only.parameters(), lr=1e-3, weight_decay=5e-4)
    for epoch in range(1, args.ablation_epochs + 1):
        train_loss, train_acc = run_epoch(
            model=graph_only,
            loader=train_loader,
            criterion=criterion,
            optimizer=graph_optimizer,
            device=device,
            augment_std=0.01,
            clip_grad=1.0,
        )
        val_loss, val_acc, _, _ = evaluate(graph_only, val_loader, criterion, device)
        print(
            "epoch={:02d} train_acc={:.4f} val_acc={:.4f} train_loss={:.4f} val_loss={:.4f}".format(
                epoch, train_acc, val_acc, train_loss, val_loss
            )
        )

    _, test_acc, _, _ = evaluate(graph_only, test_loader, criterion, device)
    print("graph-only compact_gat test_acc={:.4f}".format(test_acc))

    out = {
        "root_dir": str(root),
        "data_dir": str(data_dir),
        "graph_source": graph_source,
        "adj_shape": list(adjacency.shape),
        "adj_nonzero": int((adjacency > 0).sum().item()),
        "adj_max": float(adjacency.max().item()),
        "adj_min": float(adjacency.min().item()),
        "graph_only_test_acc": float(test_acc),
    }
    report_path = results_dir / "debug_diagnostics.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Saved diagnostics:", report_path)


if __name__ == "__main__":
    main()
