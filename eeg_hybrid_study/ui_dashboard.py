#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="EEG Research Dashboard", layout="wide")
st.title("EEG Motor Imagery Research Dashboard")
st.caption("Single-run and multi-seed audited benchmarking")


def show_png(path: Path, caption: str):
    try:
        data = path.read_bytes()
        st.image(data, caption=caption, use_column_width=True, output_format="PNG")
    except Exception as e:
        st.error("Failed to render {}: {}".format(path.name, e))


def detect_default_results_dir() -> Path:
    candidates = [
        Path("results_clean_all_models_s12_ff_seed23"),
        Path("results_college_report_ready"),
        Path("results_presentation_ready"),
        Path("results_presentation_final"),
        Path("results"),
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return Path("results").resolve()


def detect_default_summary_path(results_dir: Path) -> Path:
    multiseed = Path("results_clean_all_models_s12_ff_multiseed_summary.csv").resolve()
    if multiseed.exists():
        return multiseed
    return (results_dir / "summary.csv").resolve()


results_dir = Path(st.text_input("Results directory", str(detect_default_results_dir()))).resolve()
summary_path = Path(st.text_input("Summary CSV path", str(detect_default_summary_path(results_dir)))).resolve()

if not summary_path.exists():
    st.error("Summary CSV not found: {}".format(summary_path))
    st.stop()

summary_df = pd.read_csv(summary_path)
if summary_df.empty:
    st.error("Summary CSV is empty: {}".format(summary_path))
    st.stop()

multi_seed_mode = {"test_acc_mean", "test_acc_std", "f1_mean", "f1_std"}.issubset(set(summary_df.columns))

if multi_seed_mode:
    summary_df = summary_df.sort_values("test_acc_mean", ascending=False)
    leader = summary_df.iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best model", str(leader["model"]))
    c2.metric("Best test acc (mean)", "{:.2f}% +- {:.2f}".format(leader["test_acc_mean"] * 100.0, leader["test_acc_std"] * 100.0))
    c3.metric("Best macro-F1 (mean)", "{:.2f}% +- {:.2f}".format(leader["f1_mean"] * 100.0, leader["f1_std"] * 100.0))
    c4.metric("Models", str(summary_df.shape[0]))

    view_df = summary_df.copy()
    view_df["test_acc (mean±std)"] = view_df.apply(
        lambda r: "{:.2f}% +- {:.2f}".format(r["test_acc_mean"] * 100.0, r["test_acc_std"] * 100.0), axis=1
    )
    view_df["f1 (mean±std)"] = view_df.apply(
        lambda r: "{:.2f}% +- {:.2f}".format(r["f1_mean"] * 100.0, r["f1_std"] * 100.0), axis=1
    )

    st.subheader("Multi-seed summary")
    st.dataframe(
        view_df[["model", "test_acc (mean±std)", "f1 (mean±std)", "bal_acc_mean", "bal_acc_std"]],
        use_container_width=True,
    )

    st.subheader("Comparison (mean +- std)")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), dpi=140)
    x = np.arange(len(summary_df))

    axes[0].bar(x, summary_df["test_acc_mean"].values, yerr=summary_df["test_acc_std"].values, capsize=4, color="#2F5597")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(summary_df["model"].tolist(), rotation=30, ha="right")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Test accuracy")
    axes[0].set_ylabel("Score")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    axes[1].bar(x, summary_df["f1_mean"].values, yerr=summary_df["f1_std"].values, capsize=4, color="#70AD47")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(summary_df["model"].tolist(), rotation=30, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Macro-F1")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    st.pyplot(fig)
else:
    summary_df = summary_df.sort_values("test_acc", ascending=False)
    leader = summary_df.iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best model", str(leader["model"]))
    c2.metric("Best test accuracy", "{:.2f}%".format(leader["test_acc"] * 100.0))
    c3.metric("Best macro F1", "{:.2f}%".format(leader["test_f1_macro"] * 100.0))
    c4.metric("Best balanced accuracy", "{:.2f}%".format(leader["test_bal_acc"] * 100.0))

    st.subheader("Single-run summary")
    st.dataframe(summary_df, use_container_width=True)

    st.subheader("Comparison")
    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=140)
    plot_df = summary_df.set_index("model")[["test_acc", "test_f1_macro"]]
    plot_df.plot(kind="bar", ax=ax, color=["#2F5597", "#70AD47"], width=0.75)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    st.pyplot(fig)


st.subheader("Diagnostics source")
default_diag = results_dir
if multi_seed_mode and Path("results_clean_all_models_s12_ff_seed23").exists():
    default_diag = Path("results_clean_all_models_s12_ff_seed23").resolve()

diag_dir = Path(st.text_input("Diagnostics results directory", str(default_diag))).resolve()

integrity_path = diag_dir / "integrity_report.json"
if integrity_path.exists():
    with integrity_path.open("r", encoding="utf-8") as f:
        integrity = json.load(f)
    c1, c2, c3 = st.columns(3)
    c1.metric("Train/Test overlap rows", integrity.get("train_test_exact_overlap_count", "NA"))
    c2.metric("Train samples", integrity.get("num_train_samples", "NA"))
    c3.metric("Test samples", integrity.get("num_test_samples", "NA"))

    if integrity.get("train_test_exact_overlap_count", 0) > 0:
        st.error("Data leakage risk: exact overlapping rows detected between train and test.")


st.subheader("Per-model figures")
model_names = summary_df["model"].tolist()
selected_model = st.selectbox("Inspect model", model_names)

history_path = diag_dir / (selected_model + "_history.csv")
cm_path = diag_dir / (selected_model + "_cm.png")
curve_path = diag_dir / (selected_model + "_curve.png")

if history_path.exists():
    hist_df = pd.read_csv(history_path)
    if {"train_loss", "val_loss", "epoch"}.issubset(hist_df.columns):
        st.line_chart(hist_df.set_index("epoch")[["train_loss", "val_loss"]])
    if {"train_acc", "val_acc", "epoch"}.issubset(hist_df.columns):
        st.line_chart(hist_df.set_index("epoch")[["train_acc", "val_acc"]])

col1, col2 = st.columns(2)
with col1:
    if cm_path.exists():
        show_png(cm_path, "{} confusion matrix".format(selected_model))
    else:
        st.warning("Missing file: {}".format(cm_path.name))
with col2:
    if curve_path.exists():
        show_png(curve_path, "{} training curves".format(selected_model))
    else:
        st.warning("Missing file: {}".format(curve_path.name))


st.subheader("Graph explainability")
adj_path = diag_dir / "adjacency.png"
learned_adj_path = diag_dir / "learned_adj.png"
col3, col4 = st.columns(2)
with col3:
    if adj_path.exists():
        show_png(adj_path, "Initial channel adjacency")
    else:
        st.warning("Missing file: {}".format(adj_path.name))
with col4:
    if learned_adj_path.exists():
        show_png(learned_adj_path, "Learned graph adjacency / attention")
    else:
        st.info("learned_adj.png not found.")
