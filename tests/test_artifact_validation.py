import hashlib
import json
from pathlib import Path

import pytest

from tabdpt_classifier_pipeline import (
    TABDPT_HF_REPO,
    TABDPT_HF_REVISION,
    TABDPT_UPSTREAM_CODE_COMMIT,
    TABDPT_WEIGHT_FILENAME,
    TABDPT_WEIGHT_SHA256,
    validate_dimer_artifact,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_bundle(tmp_path: Path) -> tuple[Path, dict]:
    context = tmp_path / "training_context.parquet"
    context.write_bytes(b"PAR1-test-context")
    manifest = {
        "format": "tabdpt-dimer-context-v3",
        "taskType": "tabular_classification",
        "targetColumn": "target",
        "dropColumns": [],
        "classNames": ["no", "yes"],
        "runtimeConfig": {
            "target_column": "target",
            "drop_columns": [],
            "max_train_rows": 10000,
            "validation_split": 0.2,
            "fine_tune": False,
            "n_ensembles": 2,
            "context_size": 512,
            "batch_size": 512,
            "temperature": 1.0,
            "seed": 42,
        },
        "preprocessing": {
            "schemaVersion": 1,
            "targetColumn": "target",
            "dropColumns": [],
            "classLabels": ["no", "yes"],
            "seed": 42,
            "encoder": {
                "schemaVersion": 1,
                "featureColumns": ["feature"],
                "numericColumns": ["feature"],
                "categoryMaps": {},
            },
        },
        "baseModel": {
            "repo": TABDPT_HF_REPO,
            "revision": TABDPT_HF_REVISION,
            "filename": TABDPT_WEIGHT_FILENAME,
            "sha256": TABDPT_WEIGHT_SHA256,
            "upstreamCodeCommit": TABDPT_UPSTREAM_CODE_COMMIT,
        },
        "trainingContext": {
            "path": context.name,
            "sizeBytes": context.stat().st_size,
            "sha256": _sha256(context),
        },
    }
    manifest_path = tmp_path / "artifact.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, manifest


def test_validate_dimer_artifact_accepts_complete_v3_bundle(tmp_path):
    manifest_path, expected = _valid_bundle(tmp_path)
    manifest, context = validate_dimer_artifact(manifest_path)
    assert manifest == expected
    assert context == tmp_path / "training_context.parquet"


def test_validate_dimer_artifact_rejects_model_provenance_mismatch(tmp_path):
    manifest_path, manifest = _valid_bundle(tmp_path)
    manifest["baseModel"]["revision"] = "0" * 40
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="base-model provenance mismatch"):
        validate_dimer_artifact(manifest_path)


def test_validate_dimer_artifact_rejects_missing_runtime_config(tmp_path):
    manifest_path, manifest = _valid_bundle(tmp_path)
    manifest.pop("runtimeConfig")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="runtimeConfig"):
        validate_dimer_artifact(manifest_path)


def test_validate_dimer_artifact_rejects_reduced_runtime_schema(tmp_path):
    manifest_path, manifest = _valid_bundle(tmp_path)
    manifest["runtimeConfig"].pop("max_train_rows")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required fields"):
        validate_dimer_artifact(manifest_path)


def test_validate_dimer_artifact_rejects_runtime_target_mismatch(tmp_path):
    manifest_path, manifest = _valid_bundle(tmp_path)
    manifest["runtimeConfig"]["target_column"] = "other_target"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="runtimeConfig.target_column"):
        validate_dimer_artifact(manifest_path)


def test_validate_dimer_artifact_rejects_runtime_drop_columns_mismatch(tmp_path):
    manifest_path, manifest = _valid_bundle(tmp_path)
    manifest["runtimeConfig"]["drop_columns"] = ["id"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="runtimeConfig.drop_columns"):
        validate_dimer_artifact(manifest_path)


def test_validate_dimer_artifact_rejects_runtime_seed_mismatch(tmp_path):
    manifest_path, manifest = _valid_bundle(tmp_path)
    manifest["runtimeConfig"]["seed"] = 7
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="runtimeConfig.seed"):
        validate_dimer_artifact(manifest_path)


def test_validate_dimer_artifact_rejects_context_digest_mismatch(tmp_path):
    manifest_path, manifest = _valid_bundle(tmp_path)
    manifest["trainingContext"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        validate_dimer_artifact(manifest_path)


def test_validate_dimer_artifact_rejects_context_size_mismatch(tmp_path):
    manifest_path, manifest = _valid_bundle(tmp_path)
    manifest["trainingContext"]["sizeBytes"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="size mismatch"):
        validate_dimer_artifact(manifest_path)


def test_validate_dimer_artifact_rejects_traversal_path(tmp_path):
    manifest_path, manifest = _valid_bundle(tmp_path)
    manifest["trainingContext"]["path"] = "../training_context.parquet"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="trainingContext.path"):
        validate_dimer_artifact(manifest_path)


def test_validate_dimer_artifact_rejects_unlisted_files(tmp_path):
    manifest_path, _ = _valid_bundle(tmp_path)
    (tmp_path / "unexpected.bin").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="unexpected unlisted files"):
        validate_dimer_artifact(manifest_path)


def test_validate_dimer_artifact_can_allow_extra_files_explicitly(tmp_path):
    manifest_path, _ = _valid_bundle(tmp_path)
    (tmp_path / "release-note.txt").write_text("metadata", encoding="utf-8")
    manifest, _ = validate_dimer_artifact(manifest_path, strict_directory=False)
    assert manifest["format"] == "tabdpt-dimer-context-v3"
