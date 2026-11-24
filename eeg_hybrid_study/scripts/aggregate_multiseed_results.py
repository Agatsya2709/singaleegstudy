#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate saved per-seed EEG model summaries using sample standard deviation."
    )
    parser.add_argument(
        "--results-root",
        default=".",
        help="Folder containing results_clean_all_models_s12_ff_seed<seed> directories.",
    )
    parser.add_argument("--seeds", default="11,23,37")
    parser.add_argument(
        "--prefix", default="results_clean_all_models_s12_ff_seed",
        help="Per-seed directory prefix.",
    )
    parser.add_argument(
        "--results-template",
        default=None,
        help="Optional directory template with {seed}, for example results/seed_{seed}.",
    )
    parser.add_argument(
        "--out-prefix", default="results_clean_all_models_s12_ff_multiseed",
        help="Output filename prefix, written under results-root.",
    )
    args = parser.parse_args()

    root = Path(args.results_root).resolve()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    rows = []
    for seed in seeds:
        if args.results_template:
            result_dir = root / args.results_template.format(seed=seed)
        else:
            result_dir = root / (args.prefix + str(seed))
        summary_path = result_dir / "summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError("Missing per-seed summary: {}".format(summary_path))
        frame = pd.read_csv(summary_path)
        frame["seed"] = seed
        rows.append(frame)

    raw = pd.concat(rows, ignore_index=True)
    raw_path = root / (args.out_prefix + "_raw.csv")
    raw.to_csv(raw_path, index=False)

    metrics = raw.groupby("model", as_index=False).agg(
        test_acc_mean=("test_acc", "mean"),
        test_acc_std=("test_acc", "std"),
        f1_mean=("test_f1_macro", "mean"),
        f1_std=("test_f1_macro", "std"),
        bal_acc_mean=("test_bal_acc", "mean"),
        bal_acc_std=("test_bal_acc", "std"),
    )
    metrics = metrics.sort_values("test_acc_mean", ascending=False)
    summary_path = root / (args.out_prefix + "_summary.csv")
    metrics.to_csv(summary_path, index=False)

    print("Wrote raw per-seed results: {}".format(raw_path))
    print("Wrote mean/std summary: {}".format(summary_path))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
