#!/usr/bin/env python3
"""Build reproducible sample dataset archive for TabDPT Classifier pipeline."""
from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATA_DIR = ROOT / "examples" / "sample-data"
FIXED_ZIP_DATETIME = (2026, 1, 1, 0, 0, 0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_breast_cancer_archive() -> Path:
    SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = load_breast_cancer(as_frame=True)
    frame = data.frame.copy()

    # Stratified 60/40 split with seed 42
    train, val = train_test_split(
        frame,
        test_size=0.4,
        random_state=42,
        stratify=frame["target"],
    )
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)

    zip_path = SAMPLE_DATA_DIR / "breast-cancer-wisconsin.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Write train.csv
        train_buf = io.StringIO()
        train.to_csv(train_buf, index=False, lineterminator="\n")
        zinfo_train = zipfile.ZipInfo("train.csv", date_time=FIXED_ZIP_DATETIME)
        zf.writestr(zinfo_train, train_buf.getvalue().encode("utf-8"))

        # Write val.csv
        val_buf = io.StringIO()
        val.to_csv(val_buf, index=False, lineterminator="\n")
        zinfo_val = zipfile.ZipInfo("val.csv", date_time=FIXED_ZIP_DATETIME)
        zf.writestr(zinfo_val, val_buf.getvalue().encode("utf-8"))

    print(f"Created {zip_path.relative_to(ROOT)}")
    print(f"  train rows: {len(train)}, val rows: {len(val)}, columns: {train.shape[1]}")
    print(f"  SHA-256: {sha256_file(zip_path)}")
    return zip_path


if __name__ == "__main__":
    build_breast_cancer_archive()
