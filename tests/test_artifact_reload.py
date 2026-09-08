import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tabdpt_classifier_pipeline.pipeline import (
    TABDPT_HF_REPO,
    TABDPT_HF_REVISION,
    TABDPT_UPSTREAM_CODE_COMMIT,
    TABDPT_WEIGHT_FILENAME,
    TABDPT_WEIGHT_SHA256,
    TabDPTClassificationPipeline,
    TabularFeatureEncoder,
)


class FakeEstimator:
    def __init__(self, *args, **kwargs):
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        return self

    def predict(self, X, **kwargs):
        return np.zeros(len(X), dtype=np.int64)

    def predict_proba(self, X, **kwargs):
        probs = np.zeros((len(X), 2), dtype=np.float64)
        probs[:, 0] = 0.8
        probs[:, 1] = 0.2
        return probs

    def ensemble_predict_proba(self, X, **kwargs):
        return self.predict_proba(X, **kwargs)


@pytest.fixture
def mock_tabdpt(monkeypatch):
    import tabdpt_classifier_pipeline.pipeline as pipe_mod

    fake_module = SimpleNamespace(TabDPTClassifier=FakeEstimator)
    monkeypatch.setitem(__import__("sys").modules, "tabdpt", fake_module)
    monkeypatch.setattr(pipe_mod, "resolve_tabdpt_weights", lambda *a, **k: Path("fake.safetensors"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_artifact_reload_parity_preserves_numeric_looking_categories(tmp_path, mock_tabdpt):
    # 1. Prepare training table with numeric-looking string categories
    train = pd.DataFrame({
        "code": pd.Series(["01", "02", "03", "01"], dtype="object"),
        "val": [10.5, 20.5, 30.5, 40.5],
        "target": ["alpha", "beta", "alpha", "beta"],
    })

    pipe = TabDPTClassificationPipeline(compile_model=False, use_flash=False)
    pipe.fit(train, target_column="target")
    assert pipe.feature_encoder.numeric_columns == {"val"}
    assert pipe.feature_encoder.category_maps["code"] == {"01": 0, "02": 1, "03": 2}

    # 2. Export genuine tabdpt-dimer-context-v2 serving artifact bundle
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir(parents=True)
    context_path = artifact_dir / "training_context.csv"
    train.to_csv(context_path, index=False)

    preprocessing_state = pipe.export_preprocessing_state()
    manifest = {
        "format": "tabdpt-dimer-context-v2",
        "taskType": "tabular_classification",
        "targetColumn": "target",
        "dropColumns": list(preprocessing_state["dropColumns"]),
        "classNames": pipe.class_labels_,
        "preprocessing": preprocessing_state,
        "baseModel": {
            "repo": TABDPT_HF_REPO,
            "revision": TABDPT_HF_REVISION,
            "filename": TABDPT_WEIGHT_FILENAME,
            "sha256": TABDPT_WEIGHT_SHA256,
            "upstreamCodeCommit": TABDPT_UPSTREAM_CODE_COMMIT,
        },
        "trainingContext": {
            "path": context_path.name,
            "sha256": _sha256(context_path),
        },
    }
    manifest_path = artifact_dir / "artifact.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # 3. Prove that naive CSV re-fitting destroys categorical semantics
    naive_csv = pd.read_csv(context_path)
    assert pd.api.types.is_numeric_dtype(naive_csv["code"]), "read_csv misinterprets string codes as integer"
    naive_enc = TabularFeatureEncoder().fit(naive_csv.drop(columns=["target"]))
    assert "code" in naive_enc.numeric_columns, "Naive refit misidentifies code as numeric"
    assert "code" not in naive_enc.category_maps, "Naive refit destroyed category maps"

    # 4. Prove that load_artifact restores exact fitted preprocessing without refitting
    restored_pipe = TabDPTClassificationPipeline.load_artifact(
        manifest_path,
        compile_model=False,
        use_flash=False,
    )
    assert restored_pipe.target_column == "target"
    assert restored_pipe.class_labels_ == ["alpha", "beta"]
    assert restored_pipe.feature_encoder.numeric_columns == {"val"}
    assert restored_pipe.feature_encoder.category_maps["code"] == {"01": 0, "02": 1, "03": 2}

    # 5. Predict on new unlabelled test rows with string categories
    test_query = pd.DataFrame({
        "code": ["01", "02"],
        "val": [15.0, 25.0],
    })
    preds = restored_pipe.predict(test_query)
    assert len(preds) == 2
    assert list(preds) == ["alpha", "alpha"]

    probs = restored_pipe.predict_proba(test_query)
    assert probs.shape == (2, 2)


def test_artifact_reload_rejects_context_digest_mismatch(tmp_path, mock_tabdpt):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    context_path = artifact_dir / "training_context.csv"
    context_path.write_text("code,val,target\n01,1.0,0\n02,2.0,1\n", encoding="utf-8")

    manifest = {
        "format": "tabdpt-dimer-context-v2",
        "taskType": "tabular_classification",
        "preprocessing": {
            "schemaVersion": 1,
            "targetColumn": "target",
            "classLabels": ["0", "1"],
            "dropColumns": [],
            "encoder": {
                "schemaVersion": 1,
                "featureColumns": ["code", "val"],
                "numericColumns": ["val"],
                "categoryMaps": {"code": {"01": 0, "02": 1}},
            },
        },
        "trainingContext": {
            "path": "training_context.csv",
            "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
        },
    }
    manifest_path = artifact_dir / "artifact.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="digest mismatch"):
        TabDPTClassificationPipeline.load_artifact(manifest_path)


def test_artifact_reload_rejects_invalid_schemas_and_formats(tmp_path, mock_tabdpt):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    manifest_path = artifact_dir / "artifact.json"

    # Missing preprocessing
    manifest_path.write_text(json.dumps({"format": "tabdpt-dimer-context-v2", "taskType": "tabular_classification"}))
    with pytest.raises(ValueError, match="missing 'preprocessing' state"):
        TabDPTClassificationPipeline.load_artifact(manifest_path)

    # Wrong format
    manifest_path.write_text(json.dumps({"format": "wrong-format", "taskType": "tabular_classification"}))
    with pytest.raises(ValueError, match="Unsupported artifact format"):
        TabDPTClassificationPipeline.load_artifact(manifest_path)

    # Wrong taskType
    manifest_path.write_text(json.dumps({"format": "tabdpt-dimer-context-v2", "taskType": "tabular_regression"}))
    with pytest.raises(ValueError, match="Artifact taskType mismatch"):
        TabDPTClassificationPipeline.load_artifact(manifest_path)
