#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from models import CNNBaseline, CNNLSTMBaseline, CompactCNNGAT, EEGNetBaseline, GraphEnhancedCNNGAT, HybridCNNLSTMGCN, HybridCNNLSTMGAT, LSTMBaseline
from models.graph_utils import (
    build_correlation_graph,
    build_distance_graph,
    channels_to_positions,
    degree_preserving_random_graph,
    load_electrode_positions,
    normalize_adjacency,
)


class EEGDataset(Dataset):
    def __init__(self, x, y, n_channels, n_times):
        self.x = torch.tensor(x.reshape(-1, n_channels, n_times), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def set_plot_style():
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.dpi": 150,
        }
    )


def load_csv_dataset(data_dir: Path):
    x_train = pd.read_csv(data_dir / "training_set.csv", header=None).values.astype(np.float32)
    x_test = pd.read_csv(data_dir / "test_set.csv", header=None).values.astype(np.float32)

    y_train = pd.read_csv(data_dir / "training_label.csv", header=None).values
    y_test = pd.read_csv(data_dir / "test_label.csv", header=None).values

    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)
    if y_train.ndim > 1:
        y_train = np.argmax(y_train, axis=1)
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)
    return x_train, y_train, x_test, y_test


def load_subject_ids(data_dir: Path, split_name: str, expected_count: int):
    path = data_dir / "{}_subject.csv".format(split_name)
    if not path.exists():
        return None
    subject_ids = pd.read_csv(path, header=None).values.reshape(-1).astype(np.int64)
    if len(subject_ids) != expected_count:
        raise ValueError("{} has {} rows but expected {}".format(path.name, len(subject_ids), expected_count))
    return subject_ids


def _row_hashes(arr: np.ndarray):
    return [hashlib.sha1(np.ascontiguousarray(r).tobytes()).hexdigest() for r in arr]


def build_integrity_report(data_dir: Path, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray):
    train_hash = set(_row_hashes(x_train))
    test_hash = set(_row_hashes(x_test))
    overlap = train_hash.intersection(test_hash)

    report = {
        "num_train_samples": int(x_train.shape[0]),
        "num_test_samples": int(x_test.shape[0]),
        "num_features": int(x_train.shape[1]),
        "train_test_exact_overlap_count": int(len(overlap)),
        "train_class_distribution": {int(i): int(c) for i, c in enumerate(np.bincount(y_train))},
        "test_class_distribution": {int(i): int(c) for i, c in enumerate(np.bincount(y_test))},
        "train_feature_mean_abs": float(np.mean(np.abs(x_train))),
        "test_feature_mean_abs": float(np.mean(np.abs(x_test))),
    }

    train_sub = data_dir / "training_subject.csv"
    test_sub = data_dir / "test_subject.csv"
    if train_sub.exists() and test_sub.exists():
        train_subjects = np.squeeze(pd.read_csv(train_sub, header=None).values).astype(np.int64)
        test_subjects = np.squeeze(pd.read_csv(test_sub, header=None).values).astype(np.int64)
        report["train_subjects"] = sorted(np.unique(train_subjects).tolist())
        report["test_subjects"] = sorted(np.unique(test_subjects).tolist())
        report["subject_overlap_count"] = int(len(set(report["train_subjects"]).intersection(set(report["test_subjects"]))))

    return report


def get_shape(data_dir: Path, feature_dim: int, n_channels_arg: int):
    meta_path = data_dir / "dataset_meta.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        return int(meta["n_channels"]), int(meta["n_times"]), meta

    if n_channels_arg <= 0:
        raise ValueError("Could not infer shape. Provide --n-channels when dataset_meta.json is missing.")
    if feature_dim % n_channels_arg != 0:
        raise ValueError("Feature dimension {} not divisible by n_channels {}".format(feature_dim, n_channels_arg))
    return n_channels_arg, feature_dim // n_channels_arg, {}


def build_adjacency(
    data_dir: Path,
    electrode_path: Path,
    x_train_shaped: np.ndarray,
    graph_k: int,
    sigma: float,
    graph_mode: str = "distance",
    graph_seed: int = 42,
):
    ch_names_path = data_dir / "channel_names.csv"
    ch_names = None
    if ch_names_path.exists():
        ch_names = pd.read_csv(ch_names_path, header=None).values.squeeze().tolist()
        ch_names = [str(x) for x in ch_names]

    positions = load_electrode_positions(str(electrode_path))
    if ch_names and positions:
        coords = channels_to_positions(ch_names, positions)
    else:
        coords = None

    if graph_mode in {"distance", "random"} and coords is None:
        raise RuntimeError("Distance and random graph controls require channel_names.csv and valid electrode coordinates.")
    if graph_mode == "distance":
        adj = build_distance_graph(coords=coords, sigma=sigma, k=graph_k)
        source = "electrode_positions"
    elif graph_mode == "random":
        distance_adj = build_distance_graph(coords=coords, sigma=sigma, k=graph_k)
        adj = degree_preserving_random_graph(distance_adj, seed=graph_seed)
        source = "degree_preserving_randomized_electrode_graph"
    else:
        adj = build_correlation_graph(train_x=x_train_shaped, k=graph_k)
        source = "train_data_correlation"

    adj = normalize_adjacency(adj)
    return torch.tensor(adj, dtype=torch.float32), source


def get_class_weights(y_train: np.ndarray, n_classes: int):
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    inv = counts.sum() / counts
    weights = inv / inv.mean()
    return weights.astype(np.float32)


def run_epoch(model, loader, criterion, optimizer, device, augment_std: float, clip_grad: float):
    model.train()
    losses = []
    ys, preds = [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        if augment_std > 0:
            x = x + torch.randn_like(x) * augment_std

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        optimizer.step()

        losses.append(loss.item())
        ys.extend(y.detach().cpu().numpy().tolist())
        preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().tolist())
    return float(np.mean(losses)), accuracy_score(ys, preds)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    ys, preds = [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        losses.append(loss.item())
        ys.extend(y.detach().cpu().numpy().tolist())
        preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().tolist())
    return float(np.mean(losses)), accuracy_score(ys, preds), ys, preds


@torch.no_grad()
def collect_attention_matrix(model, loader, device):
    if not hasattr(model, "get_last_attention"):
        return None

    model.eval()
    matrices = []
    for x, _ in loader:
        x = x.to(device)
        _ = model(x)
        attn = model.get_last_attention()
        if attn is not None:
            matrices.append(attn.numpy())

    if not matrices:
        return None
    return np.mean(matrices, axis=0)


def plot_history(history_df: pd.DataFrame, model_name: str, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="train", linewidth=2)
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="val", linewidth=2)
    axes[0].set_title("{} loss".format(model_name))
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy")
    axes[0].legend(frameon=True)

    axes[1].plot(history_df["epoch"], history_df["train_acc"], label="train", linewidth=2)
    axes[1].plot(history_df["epoch"], history_df["val_acc"], label="val", linewidth=2)
    axes[1].set_title("{} accuracy".format(model_name))
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend(frameon=True)

    fig.tight_layout()
    fig.savefig(out_dir / "{}_curve.png".format(model_name), dpi=320, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(y_true, y_pred, model_name: str, out_dir: Path):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(np.float32) / np.clip(cm.sum(axis=1, keepdims=True), a_min=1, a_max=None)

    fig, ax = plt.subplots(figsize=(6, 5))
    labels = np.array([["{}\n({:.1%})".format(cm[i, j], cm_norm[i, j]) for j in range(cm.shape[1])] for i in range(cm.shape[0])])
    sns.heatmap(cm_norm, annot=labels, fmt="", cmap="Blues", cbar=True, square=True, ax=ax)
    ax.set_title("{} confusion matrix".format(model_name))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(out_dir / "{}_cm.png".format(model_name), dpi=320, bbox_inches="tight")
    plt.close(fig)


def plot_matrix(matrix: np.ndarray, title: str, save_path: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="viridis")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Weight", rotation=270, labelpad=15)
    ax.set_title(title)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Channel")
    fig.tight_layout()
    fig.savefig(save_path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def compute_metrics(y_true, y_pred):
    return {
        "test_acc": float(accuracy_score(y_true, y_pred)),
        "test_bal_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "test_f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "test_precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "test_recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "test_mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def per_subject_metrics(y_true, y_pred, subject_ids):
    if subject_ids is None:
        return []
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rows = []
    for subject_id in sorted(np.unique(subject_ids)):
        mask = subject_ids == subject_id
        row = compute_metrics(y_true[mask], y_pred[mask])
        row["subject_id"] = int(subject_id)
        row["n_trials"] = int(np.sum(mask))
        rows.append(row)
    return rows


def create_model(
    model_name: str,
    n_channels: int,
    n_times: int,
    n_classes: int,
    adjacency: torch.Tensor,
    compact_gat_dropout: float,
    compact_gat_dropedge: float,
):
    if model_name == "cnn":
        return CNNBaseline(n_channels=n_channels, n_classes=n_classes)
    if model_name == "eegnet":
        return EEGNetBaseline(n_channels=n_channels, n_classes=n_classes)
    if model_name == "cnn_lstm":
        return CNNLSTMBaseline(n_channels=n_channels, n_classes=n_classes)
    if model_name == "lstm":
        return LSTMBaseline(n_channels=n_channels, n_classes=n_classes)
    if model_name == "gcn_hybrid":
        return HybridCNNLSTMGCN(n_channels=n_channels, n_times=n_times, n_classes=n_classes, adjacency=adjacency)
    if model_name == "gat_hybrid":
        return HybridCNNLSTMGAT(n_channels=n_channels, n_times=n_times, n_classes=n_classes, adjacency=adjacency)
    if model_name == "compact_gat":
        return CompactCNNGAT(
            n_channels=n_channels,
            n_times=n_times,
            n_classes=n_classes,
            adjacency=adjacency,
            dropout=compact_gat_dropout,
            dropedge=compact_gat_dropedge,
        )
    if model_name == "graph_enhanced_cnn":
        return GraphEnhancedCNNGAT(
            n_channels=n_channels,
            n_times=n_times,
            n_classes=n_classes,
            adjacency=adjacency,
            dropout=compact_gat_dropout,
            dropedge=compact_gat_dropedge,
        )
    raise ValueError("Unsupported model: {}".format(model_name))


def train_one_model(
    model_name: str,
    n_channels: int,
    n_times: int,
    n_classes: int,
    train_loader,
    val_loader,
    test_loader,
    adjacency,
    class_weights,
    device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    label_smoothing: float,
    augment_std: float,
    clip_grad: float,
    compact_gat_dropout: float,
    compact_gat_dropedge: float,
):
    model = create_model(
        model_name=model_name,
        n_channels=n_channels,
        n_times=n_times,
        n_classes=n_classes,
        adjacency=adjacency,
        compact_gat_dropout=compact_gat_dropout,
        compact_gat_dropedge=compact_gat_dropedge,
    ).to(device)

    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(2, patience // 3), verbose=False
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    no_improve = 0
    history_rows = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            augment_std=augment_std,
            clip_grad=clip_grad,
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        print(
            "[{}] Epoch {:03d} | train_acc={:.4f} | val_acc={:.4f} | lr={:.2e}".format(
                model_name, epoch, train_acc, val_acc, optimizer.param_groups[0]["lr"]
            )
        )

        if no_improve >= patience:
            print("[{}] Early stopping at epoch {}.".format(model_name, epoch))
            break

    model.load_state_dict(best_state)
    test_loss, _, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    metrics = compute_metrics(y_true, y_pred)
    metrics["test_loss"] = float(test_loss)

    history_df = pd.DataFrame(history_rows)
    attention = collect_attention_matrix(model, test_loader, device)
    return model, history_df, best_val_acc, metrics, y_true, y_pred, attention


def parse_models(spec: str):
    if spec.strip().lower() == "all":
        return ["cnn", "eegnet", "cnn_lstm", "lstm", "gcn_hybrid", "gat_hybrid", "compact_gat", "graph_enhanced_cnn"]
    return [x.strip().lower() for x in spec.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Train EEG CNN/LSTM/GCN/GAT models with paper-ready outputs.")
    parser.add_argument("--data-dir", default="data/processed/EEG-Motor-Movement-Imagery-Dataset")
    parser.add_argument(
        "--electrode-path",
        default="assets/electrode_positions.txt",
        help="Electrode coordinates file from the referenced repos",
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--models",
        default="all",
        help="all or comma list: cnn,eegnet,cnn_lstm,lstm,gcn_hybrid,gat_hybrid,compact_gat,graph_enhanced_cnn",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--graph-k", type=int, default=12)
    parser.add_argument("--graph-sigma", type=float, default=0.55)
    parser.add_argument("--graph-mode", choices=["distance", "random", "correlation"], default="distance")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-channels", type=int, default=0, help="Only needed if dataset_meta.json is absent")
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--augment-std", type=float, default=0.01)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--compact-gat-dropout", type=float, default=0.45)
    parser.add_argument("--compact-gat-dropedge", type=float, default=0.15)
    parser.add_argument(
        "--allow-legacy-prescaled-input",
        action="store_true",
        help="Permit datasets standardized before validation splitting. Do not use for leakage-controlled results.",
    )
    args = parser.parse_args()

    set_plot_style()
    set_seed(args.seed)

    data_dir = Path(args.data_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_test, y_test = load_csv_dataset(data_dir)
    test_subject_ids = load_subject_ids(data_dir, "test", len(y_test))
    integrity_report = build_integrity_report(data_dir, x_train, y_train, x_test, y_test)
    with (results_dir / "integrity_report.json").open("w", encoding="utf-8") as f:
        json.dump(integrity_report, f, indent=2)
    print("Integrity report saved: {}".format(results_dir / "integrity_report.json"))
    print("Train/Test exact overlap count:", integrity_report["train_test_exact_overlap_count"])

    n_channels, n_times, meta = get_shape(data_dir, x_train.shape[1], args.n_channels)
    scaling_scope = (meta or {}).get("scaling_scope")
    if scaling_scope != "unscaled_epochs" and not args.allow_legacy_prescaled_input:
        raise RuntimeError(
            "This study requires an unscaled epoch dataset. Re-run prepare_dataset.py with --scaling none "
            "or use --allow-legacy-prescaled-input only for non-reporting compatibility."
        )
    n_classes = int(max(np.max(y_train), np.max(y_test)) + 1)
    class_weights = get_class_weights(y_train=y_train, n_classes=n_classes)

    split = StratifiedShuffleSplit(n_splits=1, test_size=args.val_size, random_state=args.seed)
    tr_idx, val_idx = next(split.split(x_train, y_train))
    x_tr_raw, y_tr = x_train[tr_idx], y_train[tr_idx]
    x_val_raw, y_val = x_train[val_idx], y_train[val_idx]

    # Fit only on the inner training fold; validation and outer-test trials never influence scaling.
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(x_tr_raw).astype(np.float32)
    x_val = scaler.transform(x_val_raw).astype(np.float32)
    x_test_scaled = scaler.transform(x_test).astype(np.float32)
    np.save(results_dir / "inner_train_scaler_mean.npy", scaler.mean_.astype(np.float32))
    np.save(results_dir / "inner_train_scaler_scale.npy", scaler.scale_.astype(np.float32))

    train_ds = EEGDataset(x_tr, y_tr, n_channels=n_channels, n_times=n_times)
    val_ds = EEGDataset(x_val, y_val, n_channels=n_channels, n_times=n_times)
    test_ds = EEGDataset(x_test_scaled, y_test, n_channels=n_channels, n_times=n_times)

    dl_gen = torch.Generator()
    dl_gen.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=dl_gen)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    x_tr_shaped = x_tr.reshape(-1, n_channels, n_times)
    electrode_path = Path(args.electrode_path).resolve()
    adjacency, graph_source = build_adjacency(
        data_dir=data_dir,
        electrode_path=electrode_path,
        x_train_shaped=x_tr_shaped,
        graph_k=args.graph_k,
        sigma=args.graph_sigma,
        graph_mode=args.graph_mode,
        graph_seed=args.seed,
    )

    adj_np = adjacency.numpy()
    np.save(results_dir / "adjacency.npy", adj_np)
    plot_matrix(adj_np, "Channel graph (normalized adjacency)", results_dir / "adjacency.png")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_names = parse_models(args.models)

    summary = []
    best_attention = None
    for model_name in model_names:
        (
            model,
            history_df,
            best_val_acc,
            metrics,
            y_true,
            y_pred,
            attention,
        ) = train_one_model(
            model_name=model_name,
            n_channels=n_channels,
            n_times=n_times,
            n_classes=n_classes,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            adjacency=adjacency,
            class_weights=class_weights,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            label_smoothing=args.label_smoothing,
            augment_std=args.augment_std,
            clip_grad=args.clip_grad,
            compact_gat_dropout=args.compact_gat_dropout,
            compact_gat_dropedge=args.compact_gat_dropedge,
        )

        history_df.to_csv(results_dir / "{}_history.csv".format(model_name), index=False)
        torch.save(model.state_dict(), results_dir / "{}_best.pt".format(model_name))
        plot_history(history_df, model_name=model_name, out_dir=results_dir)
        plot_confusion(y_true, y_pred, model_name=model_name, out_dir=results_dir)
        prediction_frame = pd.DataFrame({"trial_index": np.arange(len(y_true)), "y_true": y_true, "y_pred": y_pred})
        if test_subject_ids is not None:
            prediction_frame["subject_id"] = test_subject_ids
        prediction_frame.to_csv(results_dir / "{}_predictions.csv".format(model_name), index=False)
        subject_rows = per_subject_metrics(y_true, y_pred, test_subject_ids)
        if subject_rows:
            pd.DataFrame(subject_rows).to_csv(
                results_dir / "{}_per_subject_metrics.csv".format(model_name), index=False
            )

        if attention is not None:
            best_attention = attention
            np.save(results_dir / "learned_adj.npy", attention)
            plot_matrix(attention, "Learned adjacency (normalized)", results_dir / "learned_adj.png")

        row = {
            "model": model_name,
            "best_val_acc": float(best_val_acc),
            "graph_source": graph_source if (("hybrid" in model_name) or ("gat" in model_name) or ("graph_" in model_name)) else "",
        }
        row.update(metrics)
        summary.append(row)

        print(
            "[{}] test_acc={:.4f} | bal_acc={:.4f} | f1={:.4f}".format(
                model_name, metrics["test_acc"], metrics["test_bal_acc"], metrics["test_f1_macro"]
            )
        )

    summary_df = pd.DataFrame(summary).sort_values(by="test_acc", ascending=False)
    summary_df.to_csv(results_dir / "summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    plot_df = summary_df.set_index("model")[["test_acc", "test_f1_macro"]]
    plot_df.plot(kind="bar", ax=ax, width=0.75, color=["#2F5597", "#70AD47"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(results_dir / "model_comparison.png", dpi=320, bbox_inches="tight")
    plt.close(fig)

    run_info = {
        "data_dir": str(data_dir),
        "results_dir": str(results_dir),
        "n_channels": n_channels,
        "n_times": n_times,
        "n_classes": n_classes,
        "graph_source": graph_source,
        "models": model_names,
        "seed": args.seed,
        "class_weights": class_weights.tolist(),
        "graph_mode": args.graph_mode,
        "graph_k": args.graph_k,
        "graph_sigma": args.graph_sigma,
        "scaling_scope": "fit_on_inner_training_fold",
        "inner_train_samples": int(len(tr_idx)),
        "validation_samples": int(len(val_idx)),
        "outer_test_samples": int(len(y_test)),
    }
    if meta:
        run_info["dataset_meta"] = meta
    with (results_dir / "run_info.json").open("w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)

    print("\nTraining complete. Results saved in {}".format(results_dir))
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
