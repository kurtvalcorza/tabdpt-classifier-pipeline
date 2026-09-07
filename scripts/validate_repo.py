#!/usr/bin/env python3
import json
from pathlib import Path

from tabdpt_classifier_pipeline.dimer_runtime import (
    SUPPORTED_HYPERPARAMETER_KEYS,
    SUPPORTED_PREPROCESSING_KEYS,
)

ROOT = Path(__file__).resolve().parents[1]
required = [
    "README.md", "MODEL_CARD.md", "DIMER_CONTRACT.md", "LICENSE",
    "TABULAR_CLASSIFICATION_DATASET_SPEC.md", "dimer-pipeline.json",
    "dimer_entrypoint.py", "pyproject.toml", "tutorials/tabdpt_classifier_colab.ipynb"
]
missing = [name for name in required if not (ROOT / name).exists()]
if missing:
    raise SystemExit(f"Missing required files: {missing}")
manifest = json.loads((ROOT / "dimer-pipeline.json").read_text())
assert manifest["version"] == 1
assert manifest["taskType"] == "tabular_classification"
assert set(manifest["datasetPreprocessing"]) == set(SUPPORTED_PREPROCESSING_KEYS)
assert set(manifest["modelFinetuning"]) == set(SUPPORTED_HYPERPARAMETER_KEYS)
assert "modelInference" not in manifest
notebook = json.loads((ROOT / "tutorials/tabdpt_classifier_colab.ipynb").read_text())
assert notebook["nbformat"] == 4
assert len(notebook["cells"]) >= 5
print("TabDPT classifier repository contract: OK")
