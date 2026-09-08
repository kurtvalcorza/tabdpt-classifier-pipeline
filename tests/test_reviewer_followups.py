import json
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from tabdpt_classifier_pipeline.dimer_runtime import _dataset_limits, run_dimer_job
from tabdpt_classifier_pipeline.pipeline import _resolve_use_flash


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "1e400"])
def test_dataset_limits_reject_non_finite_compression_ratio(monkeypatch, value):
    monkeypatch.setenv("DIMER_MAX_COMPRESSION_RATIO", value)
    with pytest.raises(ValueError, match="must be finite"):
        _dataset_limits()


def test_dataset_limits_preserve_numeric_and_positive_validation(monkeypatch):
    monkeypatch.setenv("DIMER_MAX_ARCHIVE_BYTES", "abc")
    with pytest.raises(ValueError, match="must be numeric"):
        _dataset_limits()
    monkeypatch.delenv("DIMER_MAX_ARCHIVE_BYTES")

    monkeypatch.setenv("DIMER_MAX_DATASET_FILES", "0")
    with pytest.raises(ValueError, match="must be positive"):
        _dataset_limits()
    monkeypatch.delenv("DIMER_MAX_DATASET_FILES")

    monkeypatch.setenv("DIMER_MAX_MEMBER_BYTES", "-5")
    with pytest.raises(ValueError, match="must be positive"):
        _dataset_limits()


def _fake_torch(available: bool, capability: tuple[int, int]):
    cuda = SimpleNamespace(
        is_available=lambda: available,
        get_device_capability=lambda device=None: capability,
    )
    return SimpleNamespace(cuda=cuda)


def test_flash_auto_detection_disables_t4_and_enables_ampere(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True, (7, 5)))
    assert _resolve_use_flash(None, None) is False

    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True, (8, 0)))
    assert _resolve_use_flash(None, "cuda") is True


def test_flash_auto_detection_fails_safe_and_respects_explicit_override(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(False, (0, 0)))
    assert _resolve_use_flash(None, None) is False
    assert _resolve_use_flash(None, "cpu") is False
    assert _resolve_use_flash(True, "cpu") is True
    assert _resolve_use_flash(False, "cuda") is False


def test_artifact_drop_columns_use_fitted_preprocessing_state(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    output = tmp_path / "output"
    dataset.mkdir()
    pd.DataFrame({"id": [1, 2, 3, 4], "x": [1, 2, 3, 4], "target": [0, 1, 0, 1]}).to_csv(
        dataset / "train.csv", index=False
    )
    pd.DataFrame({"id": [5, 6], "x": [5, 6], "target": [0, 1]}).to_csv(
        dataset / "val.csv", index=False
    )

    class FakePipeline:
        def __init__(self, **kwargs):
            self.class_labels_ = ["0", "1"]
            self.drop_columns_ = []

        def fit(self, frame, target_column="target", drop_columns=None):
            self.drop_columns_ = [c for c in (drop_columns or []) if c != target_column]
            return self

        def evaluate(self, frame, **kwargs):
            return {"accuracy": 1.0, "log_loss": 0.0, "roc_auc": 1.0}

        def export_preprocessing_state(self):
            return {
                "schemaVersion": 1,
                "targetColumn": "target",
                "dropColumns": list(self.drop_columns_),
                "classLabels": list(self.class_labels_),
                "encoder": {"schemaVersion": 1, "featureColumns": ["x"], "numericColumns": ["x"], "categoryMaps": {}, "categoricalEncoding": {}},
            }

    import tabdpt_classifier_pipeline.dimer_runtime as runtime

    monkeypatch.setattr(runtime, "TabDPTClassificationPipeline", FakePipeline)
    monkeypatch.setenv("DIMER_DATASET_DIR", str(dataset))
    monkeypatch.setenv("DIMER_OUTPUT_DIR", str(output))
    monkeypatch.setenv(
        "DIMER_PREPROCESSING_ARGS_JSON",
        json.dumps({"target_column": "target", "drop_columns": "id,target"}),
    )

    run_dimer_job()
    artifact = json.loads((output / "artifacts" / "artifact.json").read_text())
    assert artifact["dropColumns"] == ["id"]
    assert artifact["preprocessing"]["dropColumns"] == ["id"]
