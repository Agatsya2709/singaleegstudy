#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import re
from pathlib import Path
from typing import List

from mne.datasets import eegbci


def parse_subjects(spec: str) -> List[int]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            lo_i, hi_i = int(lo), int(hi)
            out.extend(list(range(lo_i, hi_i + 1)))
        else:
            out.append(int(part))
    return sorted(set(out))


def parse_runs(spec: str) -> List[int]:
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def parse_subject_run(path: Path):
    stem = path.stem.upper()
    m = re.search(r"S(\d+)R(\d+)", stem)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def main():
    parser = argparse.ArgumentParser(description="Download PhysioNet EEG Motor Imagery EDF files.")
    parser.add_argument("--subjects", default="1-20", help="Examples: 1-10 or 1,2,5,7")
    parser.add_argument("--runs", default="4,6,8,10,12,14", help="Motor imagery runs")
    parser.add_argument("--out-dir", default="data/raw_edf", help="Root download directory")
    args = parser.parse_args()

    subjects = parse_subjects(args.subjects)
    runs = parse_runs(args.runs)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for subject in subjects:
        files = eegbci.load_data(subject=subject, runs=runs, path=str(out_dir), update_path=False, verbose=False)
        for fp in files:
            path = Path(fp).resolve()
            sub_id, run_id = parse_subject_run(path)
            if sub_id is None:
                continue
            manifest_rows.append([str(path), sub_id, run_id])
        print("Downloaded subject {}".format(subject))

    manifest = out_dir / "edf_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_path", "subject", "run"])
        writer.writerows(sorted(manifest_rows, key=lambda x: (x[1], x[2], x[0])))

    print("Saved manifest: {}".format(manifest))
    print("Total EDF files: {}".format(len(manifest_rows)))


if __name__ == "__main__":
    main()

