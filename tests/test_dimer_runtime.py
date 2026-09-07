import json
from pathlib import Path

import pandas as pd
import pytest

from tabdpt_classifier_pipeline.dimer_runtime import (
    DimerRuntimeConfig,
    SUPPORTED_HYPERPARAMETER_KEYS,
    SUPPORTED_PREPROCESSING_KEYS,
    prepare_dimer_frames,
)


def test_manifest_controls_match_runtime_contract():
    manifest = json.loads(Path("dimer-pipeline.json").read_text())
    assert set(manifest["datasetPreprocessing"]) == set(SUPPORTED_PREPROCESSING_KEYS)
    assert set(manifest["modelFinetuning"]) == set(SUPPORTED_HYPERPARAMETER_KEYS)
    assert "modelInference" not in manifest


def test_runtime_rejects_gradient_fine_tuning_and_unknown_keys():
    with pytest.raises(ValueError, match="fine_tune must be false"):
        DimerRuntimeConfig.from_payloads({}, {"fine_tune": True})
    with pytest.raises(ValueError, match="Unsupported hyperparameter"):
        DimerRuntimeConfig.from_payloads({}, {"bogus": 1})


def test_stratified_holdout_happens_before_support_cap():
    frame = pd.DataFrame({
        "x": range(500),
        "target": [0, 1] * 250,
    })
    config = DimerRuntimeConfig(max_train_rows=200, validation_split=0.2, seed=7)
    train, val = prepare_dimer_frames(frame, None, config)
    assert len(train) == 200
    assert len(val) == 100
    assert set(train["target"]) == {0, 1}
    assert set(val["target"]) == {0, 1}
