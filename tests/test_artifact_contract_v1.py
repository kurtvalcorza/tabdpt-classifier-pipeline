import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tabdpt_classifier_pipeline import TabDPTClassificationPipeline
from tabdpt_classifier_pipeline.artifact import (
    ARTIFACT_FORMAT,
    ARTIFACT_FORMAT_VERSION,
    EXPECTED_BASE_MODEL,
    export_artifact_bundle,
    load_verified_artifact,
    validate_artifact_bundle,
)


def _frame():
    return pd.DataFrame(
        {
            "numeric": [1.0, 2.0, 3.0, 4.0],
            "category": ["01", "02", "03", "01"],
            "target": ["alpha", "beta", "alpha", "beta"],
        }
    )


def _export_without_model_download(tmp_path, frame=None):
    frame = _frame() if frame is None else frame
    pipe = TabDPTClassificationPipeline(use_flash=False)
    pipe.target_column = "target"
    pipe.drop_columns_ = []
    pipe.class_labels_ = sorted(frame["target"].map(str).unique().tolist())
    encoded = pipe.feature_encoder.fit_transform(frame.drop(columns=["target"]))
    pipe.estimator = SimpleNamespace(
        V=None,
        imputer=SimpleNamespace(statistics_=np.nanmean(encoded, axis=0)),
        scaler=SimpleNamespace(mean_=np.nanmean(encoded, axis=0), scale_=np.ones(encoded.shape[1])),
    )
    return export_artifact_bundle(pipe, frame, tmp_path / "artifact")


def test_exported_artifact_validates_without_loading_model(tmp_path):
    manifest_path = _export_without_model_download(tmp_path)
    manifest, context_path = validate_artifact_bundle(manifest_path)

    assert manifest["format"] == ARTIFACT_FORMAT
    assert manifest["formatVersion"] == ARTIFACT_FORMAT_VERSION
    assert manifest["taskType"] == "tabular_classification"
    assert manifest["baseModel"] == EXPECTED_BASE_MODEL
    assert manifest["classLabels"] == ["alpha", "beta"]
    assert manifest["preprocessing"]["classLabels"] == ["alpha", "beta"]
    assert manifest["trainingContext"]["path"] == "training_context.parquet"
    assert manifest["trainingContext"]["size"] == context_path.stat().st_size
    assert context_path.is_file()
    context = pd.read_parquet(context_path)
    assert context["target"].tolist() == ["alpha", "beta", "alpha", "beta"]


def test_export_rejects_context_missing_a_fitted_class(tmp_path):
    frame = _frame()
    pipe = TabDPTClassificationPipeline(use_flash=False)
    pipe.target_column = "target"
    pipe.class_labels_ = ["alpha", "beta", "gamma"]
    encoded = pipe.feature_encoder.fit_transform(frame.drop(columns=["target"]))
    pipe.estimator = SimpleNamespace(
        V=None,
        imputer=SimpleNamespace(statistics_=np.nanmean(encoded, axis=0)),
        scaler=SimpleNamespace(mean_=np.nanmean(encoded, axis=0), scale_=np.ones(encoded.shape[1])),
    )
    with pytest.raises(ValueError, match="must match the fitted class labels"):
        export_artifact_bundle(pipe, frame, tmp_path / "artifact")


def test_established_v3_runtime_manifest_remains_accepted(tmp_path):
    manifest_path = _export_without_model_download(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("formatVersion")
    manifest.pop("artifactSemantics")
    manifest.pop("classLabels")
    manifest["trainingContext"].pop("size")
    manifest_path.write_text(json.dumps(manifest))

    validated, _ = validate_artifact_bundle(manifest_path)
    assert validated["format"] == ARTIFACT_FORMAT
    assert validated["baseModel"] == EXPECTED_BASE_MODEL


def test_explicit_artifact_requires_fitted_upstream_preprocessing_state(tmp_path):
    manifest_path = _export_without_model_download(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["preprocessing"].pop("upstream")
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="missing fitted upstream preprocessing state"):
        validate_artifact_bundle(manifest_path)


@pytest.mark.parametrize(
    "labels, message",
    [
        (["alpha"], "at least two classes"),
        (["beta", "alpha"], "sorted"),
        (["alpha", "alpha"], "duplicates"),
        (["alpha", 1], "string labels"),
    ],
)
def test_artifact_rejects_invalid_class_label_mapping(tmp_path, labels, message):
    manifest_path = _export_without_model_download(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["preprocessing"]["classLabels"] = labels
    manifest["classLabels"] = labels
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=message):
        validate_artifact_bundle(manifest_path)


def test_artifact_rejects_top_level_class_labels_disagreeing_with_preprocessing(tmp_path):
    manifest_path = _export_without_model_download(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["classLabels"] = ["alpha", "gamma"]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="classLabels disagree"):
        validate_artifact_bundle(manifest_path)


def _no_refit_module(monkeypatch):
    class NoRefitEstimator:
        def __init__(self, *args, **kwargs):
            self.device = kwargs.get("device") or "cpu"
            self.missing_indicators = False
            self.normalizer = "standard"
            self.feature_reduction = "pca"
            self.max_features = 128
            self.V = None

        def fit(self, X, y):
            raise AssertionError("verified artifact reload must not call estimator.fit")

        def predict_proba(self, X, **kwargs):
            transformed = self.scaler.transform(self.imputer.transform(X))
            score = 1.0 / (1.0 + np.exp(-transformed.sum(axis=1)))
            return np.column_stack([1.0 - score, score])

        def ensemble_predict_proba(self, X, **kwargs):
            return self.predict_proba(X, **kwargs)

    import tabdpt_classifier_pipeline.artifact as artifact_mod

    monkeypatch.setitem(sys.modules, "tabdpt", SimpleNamespace(TabDPTClassifier=NoRefitEstimator))
    monkeypatch.setattr(artifact_mod, "resolve_tabdpt_weights", lambda *args, **kwargs: Path("fake.safetensors"))


def test_verified_reload_restores_upstream_state_and_class_mapping_without_fit(tmp_path, monkeypatch):
    manifest_path = _export_without_model_download(tmp_path)
    _no_refit_module(monkeypatch)

    restored = load_verified_artifact(manifest_path, compile_model=False, use_flash=False)
    assert restored.preprocessing_restored_ is True
    assert restored.class_labels_ == ["alpha", "beta"]
    assert restored.estimator.num_classes == 2
    assert restored.estimator.y_train.tolist() == [0, 1, 0, 1]
    np.testing.assert_allclose(restored.estimator.imputer.statistics_, [2.5, 0.75])
    np.testing.assert_allclose(restored.estimator.scaler.mean_, [2.5, 0.75])
    np.testing.assert_allclose(restored.estimator.scaler.scale_, [1.0, 1.0])

    query = pd.DataFrame({"numeric": [4.0], "category": ["02"]})
    proba = restored.predict_proba(query, n_ensembles=1, context_size=128, batch_size=1, seed=42)
    assert list(proba.columns) == ["alpha", "beta"]
    prediction = restored.predict(query, n_ensembles=1, context_size=128, batch_size=1, seed=42)
    assert prediction.tolist() == ["beta"]


def test_verified_reload_rejects_context_with_unmapped_or_missing_classes(tmp_path, monkeypatch):
    manifest_path = _export_without_model_download(tmp_path)
    _no_refit_module(monkeypatch)
    context_path = manifest_path.parent / "training_context.parquet"

    def rewrite_context(labels):
        context = pd.read_parquet(context_path)
        context["target"] = labels
        context.to_parquet(context_path, index=False)
        manifest = json.loads(manifest_path.read_text())
        manifest["trainingContext"]["size"] = context_path.stat().st_size
        import hashlib

        manifest["trainingContext"]["sha256"] = hashlib.sha256(context_path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))

    rewrite_context(["alpha", "beta", "alpha", "gamma"])
    with pytest.raises(ValueError, match="labels not in classLabels"):
        load_verified_artifact(manifest_path, compile_model=False, use_flash=False)

    rewrite_context(["alpha", "alpha", "alpha", "alpha"])
    with pytest.raises(ValueError, match="missing fitted classes"):
        load_verified_artifact(manifest_path, compile_model=False, use_flash=False)


def test_artifact_rejects_base_model_revision_mismatch(tmp_path):
    manifest_path = _export_without_model_download(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["baseModel"]["revision"] = "mutable-or-wrong-revision"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=r"baseModel\.revision mismatch"):
        validate_artifact_bundle(manifest_path)


def test_artifact_rejects_task_type_mismatch(tmp_path):
    manifest_path = _export_without_model_download(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["taskType"] = "tabular_regression"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="taskType mismatch"):
        validate_artifact_bundle(manifest_path)


def test_artifact_rejects_unsupported_format_version(tmp_path):
    manifest_path = _export_without_model_download(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["formatVersion"] = 999
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="formatVersion"):
        validate_artifact_bundle(manifest_path)


def test_artifact_rejects_context_path_traversal(tmp_path):
    manifest_path = _export_without_model_download(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["trainingContext"]["path"] = "../training_context.parquet"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="Unsafe artifact member path"):
        validate_artifact_bundle(manifest_path)


def test_artifact_rejects_context_size_or_digest_mismatch(tmp_path):
    manifest_path = _export_without_model_download(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["trainingContext"]["size"] += 1
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="Training context size mismatch"):
        validate_artifact_bundle(manifest_path)

    manifest = json.loads(manifest_path.read_text())
    context_path = manifest_path.parent / "training_context.parquet"
    manifest["trainingContext"]["size"] = context_path.stat().st_size
    manifest["trainingContext"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="Training context SHA-256 mismatch"):
        validate_artifact_bundle(manifest_path)
