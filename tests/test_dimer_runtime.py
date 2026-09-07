import json
import os
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from tabdpt_classifier_pipeline.dimer_runtime import (
    DimerRuntimeConfig,
    SUPPORTED_HYPERPARAMETER_KEYS,
    SUPPORTED_PREPROCESSING_KEYS,
    TRANSPORT_HYPERPARAMETER_KEYS,
    load_dimer_tables,
    prepare_dimer_frames,
)


def _write_zip(path: Path, members: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)


def test_manifest_controls_match_runtime_contract():
    manifest = json.loads(Path("dimer-pipeline.json").read_text())
    assert set(manifest["datasetPreprocessing"]) == set(SUPPORTED_PREPROCESSING_KEYS)
    assert set(manifest["modelFinetuning"]) == set(SUPPORTED_HYPERPARAMETER_KEYS)
    assert "modelInference" not in manifest
    assert TRANSPORT_HYPERPARAMETER_KEYS == {"model_id"}
    assert "model_id" not in manifest["modelFinetuning"]


def test_runtime_accepts_transport_model_id_but_rejects_unknown_keys():
    config = DimerRuntimeConfig.from_payloads({}, {"model_id": "tabdpt-v1.2"})
    assert config == DimerRuntimeConfig()
    with pytest.raises(ValueError, match="Unsupported hyperparameter"):
        DimerRuntimeConfig.from_payloads({}, {"bogus": 1})
    with pytest.raises(ValueError, match="Unsupported preprocessing"):
        DimerRuntimeConfig.from_payloads({"bogus": 1}, {})


def test_runtime_rejects_gradient_fine_tuning():
    with pytest.raises(ValueError, match="fine_tune must be false"):
        DimerRuntimeConfig.from_payloads({}, {"fine_tune": True})


def test_stratified_holdout_happens_before_support_cap():
    frame = pd.DataFrame({
        "id": range(500),
        "target": [0, 1] * 250,
    })
    config = DimerRuntimeConfig(max_train_rows=200, validation_split=0.2, seed=7)
    train, val = prepare_dimer_frames(frame, None, config)
    assert len(train) == 200
    assert len(val) == 100
    assert set(train["target"]) == {0, 1}
    assert set(val["target"]) == {0, 1}
    assert set(train["id"]).isdisjoint(set(val["id"]))


def test_support_cap_preserves_ultra_rare_binary_class_deterministically():
    common = 59997
    frame = pd.DataFrame({
        "id": range(common + 3),
        "target": ["common"] * common + ["rare"] * 3,
    })
    val = pd.DataFrame({"id": [-1, -2], "target": ["common", "rare"]})
    config = DimerRuntimeConfig(max_train_rows=5000, seed=11)
    train_a, _ = prepare_dimer_frames(frame, val, config)
    train_b, _ = prepare_dimer_frames(frame, val, config)
    assert set(train_a["target"]) == {"common", "rare"}
    assert train_a["id"].tolist() == train_b["id"].tolist()


def test_support_cap_preserves_multiple_ultra_rare_classes():
    labels = ["A"] * 90000 + ["B"] * 9000 + ["C"] * 3 + ["D"] * 3
    frame = pd.DataFrame({"id": range(len(labels)), "target": labels})
    val = pd.DataFrame({"id": [-1, -2, -3, -4], "target": ["A", "B", "C", "D"]})
    config = DimerRuntimeConfig(max_train_rows=1000, seed=5)
    train, _ = prepare_dimer_frames(frame, val, config)
    assert set(train["target"]) == {"A", "B", "C", "D"}


def test_zip_rejects_archive_over_compressed_size_limit(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    archive = dataset / "dataset.zip"
    _write_zip(archive, {"train.csv": "x,target\n1,0\n2,1\n"})
    monkeypatch.setenv("DIMER_MAX_ARCHIVE_BYTES", "1")
    with pytest.raises(ValueError, match="DIMER_MAX_ARCHIVE_BYTES"):
        load_dimer_tables(dataset)


def test_zip_rejects_member_over_uncompressed_size_limit(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    archive = dataset / "dataset.zip"
    _write_zip(archive, {"train.csv": "x,target\n" + "1,0\n" * 100})
    monkeypatch.setenv("DIMER_MAX_MEMBER_BYTES", "32")
    with pytest.raises(ValueError, match="DIMER_MAX_MEMBER_BYTES"):
        load_dimer_tables(dataset)


def test_zip_rejects_aggregate_uncompressed_size_limit(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    archive = dataset / "dataset.zip"
    _write_zip(
        archive,
        {
            "train.csv": "x,target\n1,0\n2,1\n",
            "notes.txt": "N" * 64,
        },
    )
    monkeypatch.setenv("DIMER_MAX_MEMBER_BYTES", "1024")
    monkeypatch.setenv("DIMER_MAX_UNCOMPRESSED_BYTES", "64")
    with pytest.raises(ValueError, match="DIMER_MAX_UNCOMPRESSED_BYTES"):
        load_dimer_tables(dataset)


def test_zip_rejects_suspicious_compression_ratio(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    archive = dataset / "dataset.zip"
    _write_zip(archive, {"train.csv": "x,target\n" + ("A" * 1024 + ",0\n") * 64})
    monkeypatch.setenv("DIMER_MAX_COMPRESSION_RATIO", "2")
    with pytest.raises(ValueError, match="DIMER_MAX_COMPRESSION_RATIO"):
        load_dimer_tables(dataset)


def test_zip_rejects_too_many_files(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    archive = dataset / "dataset.zip"
    _write_zip(
        archive,
        {
            "train.csv": "x,target\n1,0\n2,1\n",
            "notes-1.txt": "a",
            "notes-2.txt": "b",
        },
    )
    monkeypatch.setenv("DIMER_MAX_DATASET_FILES", "2")
    with pytest.raises(ValueError, match="DIMER_MAX_DATASET_FILES"):
        load_dimer_tables(dataset)


def test_zip_rejects_path_traversal_member(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    archive = dataset / "dataset.zip"
    _write_zip(archive, {"../train.csv": "x,target\n1,0\n2,1\n"})
    with pytest.raises(ValueError, match="Unsafe dataset archive path"):
        load_dimer_tables(dataset)


def test_zip_rejects_absolute_and_backslash_traversal_paths(tmp_path):
    for member in ["/train.csv", r"..\train.csv", r"C:\train.csv"]:
        dataset = tmp_path / member.replace("/", "_").replace("\\", "_").replace(":", "_")
        dataset.mkdir()
        _write_zip(dataset / "dataset.zip", {member: "x,target\n1,0\n2,1\n"})
        with pytest.raises(ValueError, match="Unsafe dataset archive path"):
            load_dimer_tables(dataset)


def test_zip_rejects_duplicate_normalized_paths(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _write_zip(
        dataset / "dataset.zip",
        {
            "a/train.csv": "x,target\n1,0\n2,1\n",
            r"a\train.csv": "x,target\n3,0\n4,1\n",
        },
    )
    with pytest.raises(ValueError, match="Duplicate normalized archive path"):
        load_dimer_tables(dataset)


def test_dataset_rejects_multiple_zips_zip_csv_mix_and_malformed_zip(tmp_path):
    multiple = tmp_path / "multiple"
    multiple.mkdir()
    _write_zip(multiple / "a.zip", {"train.csv": "x,target\n1,0\n2,1\n"})
    _write_zip(multiple / "b.zip", {"train.csv": "x,target\n1,0\n2,1\n"})
    with pytest.raises(ValueError, match="exactly one ZIP"):
        load_dimer_tables(multiple)

    mixed = tmp_path / "mixed"
    mixed.mkdir()
    _write_zip(mixed / "dataset.zip", {"train.csv": "x,target\n1,0\n2,1\n"})
    (mixed / "train.csv").write_text("x,target\n1,0\n2,1\n")
    with pytest.raises(ValueError, match="either CSV files or exactly one ZIP"):
        load_dimer_tables(mixed)

    malformed = tmp_path / "malformed"
    malformed.mkdir()
    (malformed / "dataset.zip").write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="Invalid ZIP archive"):
        load_dimer_tables(malformed)


def test_direct_dataset_limits_are_enforced(tmp_path, monkeypatch):
    member = tmp_path / "member"
    member.mkdir()
    (member / "train.csv").write_text("x,target\n" + "1,0\n" * 20)
    monkeypatch.setenv("DIMER_MAX_MEMBER_BYTES", "16")
    with pytest.raises(ValueError, match="DIMER_MAX_MEMBER_BYTES"):
        load_dimer_tables(member)

    monkeypatch.delenv("DIMER_MAX_MEMBER_BYTES")
    aggregate = tmp_path / "aggregate"
    aggregate.mkdir()
    (aggregate / "train.csv").write_text("x,target\n1,0\n2,1\n")
    (aggregate / "notes.txt").write_text("N" * 64)
    monkeypatch.setenv("DIMER_MAX_UNCOMPRESSED_BYTES", "64")
    with pytest.raises(ValueError, match="DIMER_MAX_UNCOMPRESSED_BYTES"):
        load_dimer_tables(aggregate)

    monkeypatch.delenv("DIMER_MAX_UNCOMPRESSED_BYTES")
    count = tmp_path / "count"
    count.mkdir()
    (count / "train.csv").write_text("x,target\n1,0\n2,1\n")
    (count / "a.txt").write_text("a")
    (count / "b.txt").write_text("b")
    monkeypatch.setenv("DIMER_MAX_DATASET_FILES", "2")
    with pytest.raises(ValueError, match="DIMER_MAX_DATASET_FILES"):
        load_dimer_tables(count)


def test_direct_dataset_rejects_symlink_escape(tmp_path):
    if not hasattr(os, "symlink"):
        pytest.skip("symlinks unavailable")
    outside = tmp_path / "outside.csv"
    outside.write_text("x,target\n1,0\n2,1\n")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    link = dataset / "train.csv"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation not permitted")
    with pytest.raises(ValueError, match="must not contain symlinks"):
        load_dimer_tables(dataset)
