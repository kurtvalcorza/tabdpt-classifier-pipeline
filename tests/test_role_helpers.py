"""Offline tests for the public validation and evaluation stage helpers (DAT24 / EVAL21).

No weights, no model: the checks are the same private functions `fit` and `predict_proba` route
through, exercised on small in-memory tables.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tabdpt_classifier_pipeline import (
    DECISION_RULE,
    INPUT_SCHEMA,
    METRIC_IDS,
    MIN_CLASSES,
    MODEL_ID,
    MODEL_REVISION,
    TabDPTClassificationPipeline,
    evaluation_report,
    majority_class_baseline,
    validate_inputs,
)


def _table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [31, 45, np.nan, 52, 23, 38],
            "city": ["a", "b", None, "a", "c", "b"],
            "target": ["yes", "no", "yes", "no", "yes", "no"],
        }
    )


def test_validate_inputs_fit_mode_returns_manifest_with_schema_and_identity() -> None:
    manifest = validate_inputs(_table(), target_column="target", names=["sample"])
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["classes"] == [MIN_CLASSES, None]
    assert manifest["target_column"] == "target"
    assert manifest["drop_columns"] == []
    (entry,) = manifest["inputs"]
    assert entry["id"] == "sample"
    assert entry["mode"] == "fit"
    assert entry["rows"] == 6
    assert entry["feature_columns"] == ["age", "city"]
    assert entry["numeric_columns"] == ["age"]
    assert entry["categorical_columns"] == ["city"]
    assert entry["missing_value_columns"] == {"age": 1, "city": 1}
    assert entry["classes"] == ["no", "yes"]
    assert entry["class_counts"] == {"no": 3, "yes": 3}
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_validate_inputs_default_id_and_drop_columns() -> None:
    manifest = validate_inputs(_table(), "target", ["city", "target", "city"])
    assert manifest["inputs"][0]["id"] == "table-0"
    assert manifest["drop_columns"] == ["city"]
    assert manifest["inputs"][0]["feature_columns"] == ["age"]


def test_validate_inputs_rejects_like_fit() -> None:
    pipe = TabDPTClassificationPipeline()
    duplicated = pd.concat([_table(), _table()[["age"]]], axis=1)
    for bad, message in (
        (duplicated, "Duplicate column names"),
        (_table().rename(columns={"target": "label"}), "Target column 'target' not found"),
        (_table().assign(target=["yes", None, "yes", "no", "yes", "no"]), "contains missing values"),
        (_table().assign(target="yes"), "at least two classes"),
        (_table()[["target"]], "At least one feature column"),
    ):
        with pytest.raises(ValueError, match=message):
            validate_inputs(bad, target_column="target")
        with pytest.raises(ValueError, match=message):
            pipe.fit(bad, target_column="target")
    with pytest.raises(ValueError, match="names must have exactly one entry"):
        validate_inputs(_table(), names=["a", "b"])


def test_validate_inputs_inference_mode_rejects_like_predict_proba() -> None:
    fitted = ["age", "city"]
    manifest = validate_inputs(_table().drop(columns=["target"]), None, feature_columns=fitted, names=["new"])
    (entry,) = manifest["inputs"]
    assert entry == {
        "id": "new",
        "mode": "inference",
        "rows": 6,
        "feature_columns": fitted,
        "missing_value_columns": {"age": 1, "city": 1},
    }
    assert manifest["target_column"] is None
    pipe = TabDPTClassificationPipeline()
    pipe.feature_encoder.feature_columns = fitted
    pipe.feature_encoder.is_fitted = True
    pipe.feature_encoder.numeric_columns = {"age"}
    pipe.feature_encoder.category_maps = {"city": {"a": 0, "b": 1, "c": 2}}
    pipe.estimator = object()
    pipe.target_column = "target"
    wrong = _table().drop(columns=["target", "city"]).assign(extra=1)
    with pytest.raises(ValueError, match=r"missing=\['city'\], extra=\['extra'\]"):
        validate_inputs(wrong, None, feature_columns=fitted)
    with pytest.raises(ValueError, match=r"missing=\['city'\], extra=\['extra'\]"):
        pipe._feature_frame(wrong)
    with pytest.raises(ValueError, match="feature_columns is required"):
        validate_inputs(wrong, None)


def test_majority_class_baseline_matches_the_evaluate_metric_ids() -> None:
    support = ["no", "no", "no", "yes", "yes"]
    holdout = ["no", "yes", "yes", "no"]
    baseline = majority_class_baseline(support, holdout)
    assert set(baseline) == set(METRIC_IDS)
    assert baseline["accuracy"] == 0.5
    assert baseline["roc_auc"] == 0.5
    # p(no) = 0.6 for every row: -(2*ln 0.6 + 2*ln 0.4) / 4
    assert math.isclose(baseline["log_loss"], -(2 * math.log(0.6) + 2 * math.log(0.4)) / 4, rel_tol=1e-9)
    three = majority_class_baseline(["a", "b", "c", "c"], ["a", "c"])
    assert "roc_auc" not in three
    with pytest.raises(ValueError, match="unseen target classes"):
        majority_class_baseline(support, ["maybe"])
    with pytest.raises(ValueError, match="at least two classes"):
        majority_class_baseline(["no", "no"], ["no"])


def test_evaluation_report_not_measurable_without_metrics() -> None:
    report = evaluation_report(None, n_holdout=0, class_labels=["no", "yes"], sample_kind="BYOD")
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert report["baselines"] == []
    assert report["decision_rule"] == DECISION_RULE
    assert "labelled holdout" in report["needs"] and "majority_class_baseline" in report["needs"]
    assert report["class_labels"] == ["no", "yes"]
    assert report["sample_kind"] == "BYOD"
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_evaluation_report_sample_sanity_with_metrics_and_baseline() -> None:
    metrics = {"accuracy": 0.9, "log_loss": 0.3, "roc_auc": 0.95}
    baseline = {"accuracy": 0.6, "log_loss": 0.67, "roc_auc": 0.5}
    report = evaluation_report(metrics, baseline=baseline, n_holdout=10, sample_kind="sample", estimation="e")
    assert report["verdict"] == "sample-sanity"
    assert [m["id"] for m in report["metrics"]] == list(METRIC_IDS)
    assert {m["id"]: m["value"] for m in report["metrics"]} == metrics
    assert {m["id"]: m["higher_is_better"] for m in report["metrics"]} == {
        "accuracy": True,
        "log_loss": False,
        "roc_auc": True,
    }
    assert all(m["estimation"] == "e" for m in report["metrics"])
    assert report["baselines"] == [
        {"id": "majority_class", "metrics": [{"id": k, "value": baseline[k]} for k in METRIC_IDS]}
    ]
    assert report["n_holdout"] == 10
    assert "10 labelled holdout row(s)" in report["reason"]
    multiclass = evaluation_report({"accuracy": 0.5, "log_loss": 1.0}, n_holdout=3)
    assert [m["id"] for m in multiclass["metrics"]] == ["accuracy", "log_loss"]
    with pytest.raises(ValueError, match="unknown metric ids"):
        evaluation_report({"f1": 0.5})
