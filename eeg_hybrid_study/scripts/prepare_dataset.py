#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import mne
import numpy as np
from scipy.signal import resample
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


def parse_subjects(spec: str) -> List[int]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(list(range(int(lo), int(hi) + 1)))
        else:
            out.append(int(part))
    return sorted(set(out))


def parse_runs(spec: str) -> List[int]:
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def parse_optional_subjects(spec: str):
    if spec is None:
        return None
    spec = spec.strip()
    if not spec:
        return None
    return parse_subjects(spec)


def parse_subject_run(path: Path) -> Tuple[int, int]:
    m = re.search(r"S(\d+)R(\d+)", path.stem.upper())
    if m is None:
        return -1, -1
    return int(m.group(1)), int(m.group(2))


def four_class_map(run_id: int, ann: str):
    if run_id in (4, 8, 12):
        if ann == "T1":
            return 0
        if ann == "T2":
            return 1
    if run_id in (6, 10, 14):
        if ann == "T1":
            return 2
        if ann == "T2":
            return 3
    return None


def binary_lr_map(run_id: int, ann: str):
    if run_id not in (4, 8, 12):
        return None
    if ann == "T1":
        return 0
    if ann == "T2":
        return 1
    return None


def binary_ff_map(run_id: int, ann: str):
    if run_id not in (6, 10, 14):
        return None
    if ann == "T1":
        return 0
    if ann == "T2":
        return 1
    return None


def build_mapper(task: str):
    if task == "four_class":
        return four_class_map, ["left_fist", "right_fist", "both_fists", "both_feet"]
    if task == "left_right":
        return binary_lr_map, ["left_fist", "right_fist"]
    if task == "fists_feet":
        return binary_ff_map, ["both_fists", "both_feet"]
    raise ValueError("Unsupported task {}".format(task))


def find_edf_files(raw_dir: Path) -> List[Path]:
    return sorted(raw_dir.rglob("S*R*.edf"))


def extract_epochs(
    file_path: Path,
    mapper,
    l_freq: float,
    h_freq: float,
    tmin: float,
    tmax: float,
    timepoints: int,
):
    subject_id, run_id = parse_subject_run(file_path)
    raw = mne.io.read_raw_edf(str(file_path), preload=True, verbose=False)
    raw.pick_types(eeg=True, exclude="bads")
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    inv = {v: k for k, v in event_id.items()}

    keep_ids = [eid for eid, ann in inv.items() if mapper(run_id, ann) is not None]
    if not keep_ids:
        return None, None, None, None

    event_id_map = {inv[eid]: eid for eid in keep_ids}
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id_map,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False,
    )
    if len(epochs) == 0:
        return None, None, None, None

    x = epochs.get_data().astype(np.float32)
    x = resample(x, num=timepoints, axis=-1).astype(np.float32)

    y = []
    for eid in epochs.events[:, 2]:
        ann = inv[eid]
        label = mapper(run_id, ann)
        if label is None:
            continue
        y.append(label)

    y = np.asarray(y, dtype=np.int64)
    if len(y) != x.shape[0]:
        return None, None, None, None

    ch_names = list(epochs.ch_names)
    subjects = np.full(shape=(x.shape[0],), fill_value=subject_id, dtype=np.int64)
    return x, y, ch_names, subjects


def main():
    parser = argparse.ArgumentParser(description="Prepare CSV dataset with EEG-DL naming.")
    parser.add_argument("--raw-dir", default="data/raw_edf", help="Folder from MIND_Get_EDF.py")
    parser.add_argument("--out-dir", default="data/processed/EEG-Motor-Movement-Imagery-Dataset")
    parser.add_argument("--subjects", default="1-20")
    parser.add_argument("--runs", default="4,6,8,10,12,14")
    parser.add_argument("--task", choices=["four_class", "left_right", "fists_feet"], default="left_right")
    parser.add_argument("--timepoints", type=int, default=128)
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=4.0)
    parser.add_argument("--l-freq", type=float, default=8.0)
    parser.add_argument("--h-freq", type=float, default=30.0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--test-subjects", type=str, default=None, help="Optional fixed test subjects, e.g. 9-10")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scaling",
        choices=["none", "outer_train_legacy"],
        default="none",
        help="Use 'none' for leakage-controlled fold-local scaling during training. 'outer_train_legacy' is retained only to read older experiments.",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = set(parse_subjects(args.subjects))
    test_subjects = parse_optional_subjects(args.test_subjects)
    test_subjects = set(test_subjects) if test_subjects else None
    runs = set(parse_runs(args.runs))
    mapper, class_names = build_mapper(args.task)

    all_files = find_edf_files(raw_dir)
    selected = []
    for fp in all_files:
        subject, run_id = parse_subject_run(fp)
        if subject in subjects and run_id in runs:
            selected.append(fp)

    if not selected:
        raise RuntimeError("No EDF files found for the requested subjects/runs.")

    x_parts = []
    y_parts = []
    subject_parts = []
    ch_names = None
    for idx, fp in enumerate(selected, start=1):
        x, y, current_ch_names, sids = extract_epochs(
            file_path=fp,
            mapper=mapper,
            l_freq=args.l_freq,
            h_freq=args.h_freq,
            tmin=args.tmin,
            tmax=args.tmax,
            timepoints=args.timepoints,
        )
        if x is None:
            continue
        if ch_names is None:
            ch_names = current_ch_names
        if len(current_ch_names) != len(ch_names):
            continue
        x_parts.append(x)
        y_parts.append(y)
        subject_parts.append(sids)
        print("Processed {}/{}: {}".format(idx, len(selected), fp.name))

    if not x_parts:
        raise RuntimeError("No usable EEG epochs were extracted.")

    x_all = np.concatenate(x_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)
    subject_all = np.concatenate(subject_parts, axis=0)

    if test_subjects is not None:
        test_mask = np.isin(subject_all, list(test_subjects))
        train_mask = ~test_mask
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        if train_idx.size == 0 or test_idx.size == 0:
            raise RuntimeError("Subject-wise split produced empty train or test set.")
    else:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        train_idx, test_idx = next(splitter.split(x_all, y_all))

    x_train, x_test = x_all[train_idx], x_all[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]
    subject_train, subject_test = subject_all[train_idx], subject_all[test_idx]

    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)
    scaler = None
    if args.scaling == "outer_train_legacy":
        scaler = StandardScaler()
        x_train_flat = scaler.fit_transform(x_train_flat).astype(np.float32)
        x_test_flat = scaler.transform(x_test_flat).astype(np.float32)
    else:
        # The trainer splits these unscaled outer-training trials before fitting its scaler.
        x_train_flat = x_train_flat.astype(np.float32)
        x_test_flat = x_test_flat.astype(np.float32)

    np.savetxt(out_dir / "training_set.csv", x_train_flat, delimiter=",")
    np.savetxt(out_dir / "test_set.csv", x_test_flat, delimiter=",")
    np.savetxt(out_dir / "training_label.csv", y_train.reshape(-1, 1), delimiter=",", fmt="%d")
    np.savetxt(out_dir / "test_label.csv", y_test.reshape(-1, 1), delimiter=",", fmt="%d")
    np.savetxt(out_dir / "training_subject.csv", subject_train.reshape(-1, 1), delimiter=",", fmt="%d")
    np.savetxt(out_dir / "test_subject.csv", subject_test.reshape(-1, 1), delimiter=",", fmt="%d")

    np.savetxt(out_dir / "channel_names.csv", np.array(ch_names, dtype=object), delimiter=",", fmt="%s")
    if scaler is not None:
        np.save(out_dir / "scaler_mean.npy", scaler.mean_.astype(np.float32))
        np.save(out_dir / "scaler_scale.npy", scaler.scale_.astype(np.float32))

    meta = {
        "task": args.task,
        "class_names": class_names,
        "n_classes": int(len(np.unique(y_all))),
        "n_channels": int(x_all.shape[1]),
        "n_times": int(x_all.shape[2]),
        "num_samples_total": int(x_all.shape[0]),
        "num_samples_train": int(x_train.shape[0]),
        "num_samples_test": int(x_test.shape[0]),
        "subjects": sorted(subjects),
        "test_subjects": sorted(list(test_subjects)) if test_subjects is not None else None,
        "runs": sorted(runs),
        "scaling_scope": "unscaled_epochs" if args.scaling == "none" else "outer_train_legacy",
    }
    with (out_dir / "dataset_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Saved CSV files in {}".format(out_dir))
    print("training_set.csv shape: {}".format(x_train_flat.shape))
    print("test_set.csv shape: {}".format(x_test_flat.shape))


if __name__ == "__main__":
    main()
