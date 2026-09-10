import json
from pathlib import Path

import pandas as pd

import tabdpt_classifier_pipeline.dimer_runtime as runtime
from tabdpt_classifier_pipeline import validate_dimer_artifact


class _StubPipeline:
    init_kwargs = None
    fit_kwargs = None

    def __init__(self, **kwargs):
        type(self).init_kwargs = kwargs
        self.class_labels_ = ["0", "1"]
        self.target_column = None
        self.drop_columns_ = []
        self.seed = kwargs.get("seed")

    def fit(self, frame, target_column="target", drop_columns=None, seed=None):
        type(self).fit_kwargs = {
            "target_column": target_column,
            "drop_columns": list(drop_columns or []),
            "seed": seed,
        }
        self.target_column = target_column
        self.drop_columns_ = list(drop_columns or [])
        self.seed = seed
        return self

    def evaluate(self, frame, **kwargs):
        return {"accuracy": 1.0}

    def export_preprocessing_state(self):
        return {
            "schemaVersion": 1,
            "targetColumn": self.target_column,
            "dropColumns": list(self.drop_columns_),
            "classLabels": list(self.class_labels_),
            "seed": self.seed,
            "encoder": {
                "schemaVersion": 1,
                "featureColumns": ["x"],
                "numericColumns": ["x"],
                "categoryMaps": {},
            },
        }


def test_run_dimer_job_propagates_seed_and_emits_full_runtime_contract(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    pd.DataFrame({"x": [1, 2, 3, 4], "target": [0, 1, 0, 1]}).to_csv(
        dataset / "train.csv", index=False
    )
    pd.DataFrame({"x": [5, 6], "target": [0, 1]}).to_csv(dataset / "val.csv", index=False)

    def fake_to_parquet(self, path, index=False):
        Path(path).write_bytes(b"PAR1-runtime-contract")

    monkeypatch.setattr(runtime, "TabDPTClassificationPipeline", _StubPipeline)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)
    monkeypatch.setenv("DIMER_DATASET_DIR", str(dataset))
    monkeypatch.setenv("DIMER_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv(
        "DIMER_PREPROCESSING_ARGS_JSON",
        json.dumps(
            {
                "target_column": "target",
                "drop_columns": [],
                "max_train_rows": 10000,
                "validation_split": 0.2,
            }
        ),
    )
    monkeypatch.setenv(
        "DIMER_HYPERPARAMETERS_JSON",
        json.dumps(
            {
                "fine_tune": False,
                "n_ensembles": 2,
                "context_size": 512,
                "batch_size": 512,
                "temperature": 1.0,
                "seed": 7,
            }
        ),
    )

    result = runtime.run_dimer_job()
    assert result["successful"] is True
    assert _StubPipeline.init_kwargs["seed"] == 7
    assert _StubPipeline.fit_kwargs["seed"] == 7

    manifest_path = tmp_path / "output" / "artifacts" / "artifact.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runtime_config = manifest["runtimeConfig"]

    assert runtime_config == {
        "target_column": "target",
        "drop_columns": [],
        "max_train_rows": 10000,
        "validation_split": 0.2,
        "fine_tune": False,
        "n_ensembles": 2,
        "context_size": 512,
        "batch_size": 512,
        "temperature": 1.0,
        "seed": 7,
    }
    assert manifest["preprocessing"]["seed"] == runtime_config["seed"]
    context_path = tmp_path / "output" / "artifacts" / "training_context.parquet"
    assert manifest["trainingContext"]["sizeBytes"] == context_path.stat().st_size

    validated, validated_context = validate_dimer_artifact(manifest_path)
    assert validated["runtimeConfig"] == runtime_config
    assert validated_context == context_path
