from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

from .pipeline import (
    TABDPT_HF_REPO,
    TABDPT_HF_REVISION,
    TABDPT_UPSTREAM_CODE_COMMIT,
    TABDPT_WEIGHT_FILENAME,
    TABDPT_WEIGHT_SHA256,
    TabularFeatureEncoder,
)

ARTIFACT_FORMAT = "tabdpt-dimer-context-v3"
ARTIFACT_TASK = "tabular_classification"
DEFAULT_MAX_CONTEXT_BYTES = 512 * 1024**2
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")

_EXPECTED_BASE_MODEL = {
    "repo": TABDPT_HF_REPO,
    "revision": TABDPT_HF_REVISION,
    "filename": TABDPT_WEIGHT_FILENAME,
    "sha256": TABDPT_WEIGHT_SHA256,
    "upstreamCodeCommit": TABDPT_UPSTREAM_CODE_COMMIT,
}
_REQUIRED_RUNTIME_KEYS = {
    "target_column",
    "drop_columns",
    "max_train_rows",
    "validation_split",
    "fine_tune",
    "n_ensembles",
    "context_size",
    "batch_size",
    "temperature",
    "seed",
}
_REQUIRED_INFERENCE_CONFIG = {
    "n_ensembles": (int, 1, 16),
    "context_size": (int, 128, 16384),
    "batch_size": (int, 1, 131072),
    "temperature": ((int, float), 0.05, 5.0),
    "seed": (int, 0, 2147483647),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Artifact field {field!r} must be an object")
    return value


def _require_integer(value: Any, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Artifact {field} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"Artifact {field} must be between {minimum} and {maximum}")
    return value


def _require_number(value: Any, field: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Artifact {field} must be numeric")
    numeric = float(value)
    if not minimum <= numeric <= maximum:
        raise ValueError(f"Artifact {field} must be between {minimum} and {maximum}")
    return numeric


def _validate_runtime_config(runtime: dict[str, Any]) -> None:
    missing = sorted(_REQUIRED_RUNTIME_KEYS - set(runtime))
    unexpected = sorted(set(runtime) - _REQUIRED_RUNTIME_KEYS)
    if missing:
        raise ValueError(f"Artifact runtimeConfig is missing required fields: {missing}")
    if unexpected:
        raise ValueError(f"Artifact runtimeConfig contains unsupported fields: {unexpected}")

    target_column = runtime["target_column"]
    if not isinstance(target_column, str) or not target_column.strip() or len(target_column) > 128:
        raise ValueError("Artifact runtimeConfig.target_column must be a non-empty string of at most 128 characters")

    drop_columns = runtime["drop_columns"]
    if not isinstance(drop_columns, list) or not all(isinstance(value, str) for value in drop_columns):
        raise ValueError("Artifact runtimeConfig.drop_columns must be a list of strings")
    if len(drop_columns) != len(set(drop_columns)):
        raise ValueError("Artifact runtimeConfig.drop_columns must not contain duplicates")

    _require_integer(runtime["max_train_rows"], "runtimeConfig.max_train_rows", 200, 50000)
    _require_number(runtime["validation_split"], "runtimeConfig.validation_split", 0.05, 0.4)

    if runtime["fine_tune"] is not False:
        raise ValueError("Artifact runtimeConfig.fine_tune must be false for TabDPT")

    for key, (expected_type, minimum, maximum) in _REQUIRED_INFERENCE_CONFIG.items():
        value = runtime[key]
        if isinstance(value, bool) or not isinstance(value, expected_type):
            raise ValueError(f"Artifact runtimeConfig.{key} has the wrong type")
        numeric = float(value)
        if not minimum <= numeric <= maximum:
            raise ValueError(
                f"Artifact runtimeConfig.{key} must be between {minimum} and {maximum}"
            )


def validate_dimer_artifact(
    artifact_path: str | Path,
    *,
    max_context_bytes: int = DEFAULT_MAX_CONTEXT_BYTES,
    strict_directory: bool = True,
) -> tuple[dict[str, Any], Path]:
    """Validate a DIMER TabDPT v3 serving artifact before model reconstruction.

    The artifact is data-only (`artifact.json` plus `training_context.parquet`). This
    validates structure, exact model provenance, the complete production runtime
    contract, fitted-preprocessing consistency, path containment, file size, and
    support-context SHA-256. It establishes internal consistency, not sender
    authenticity.
    """
    if isinstance(max_context_bytes, bool) or not isinstance(max_context_bytes, int) or max_context_bytes <= 0:
        raise ValueError("max_context_bytes must be a positive integer")

    artifact_file = Path(artifact_path)
    if artifact_file.is_symlink():
        raise ValueError("Artifact manifest must not be a symlink")
    if not artifact_file.is_file():
        raise FileNotFoundError(f"Artifact manifest not found: {artifact_file}")

    try:
        manifest = json.loads(artifact_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Artifact manifest is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Artifact manifest must contain a JSON object")

    if manifest.get("format") != ARTIFACT_FORMAT:
        raise ValueError(
            f"Unsupported artifact format: {manifest.get('format')!r}; expected {ARTIFACT_FORMAT!r}"
        )
    if manifest.get("taskType") != ARTIFACT_TASK:
        raise ValueError(
            f"Artifact taskType mismatch: expected {ARTIFACT_TASK!r}, got {manifest.get('taskType')!r}"
        )

    base_model = _require_mapping(manifest.get("baseModel"), "baseModel")
    mismatches = {
        key: {"actual": base_model.get(key), "expected": expected}
        for key, expected in _EXPECTED_BASE_MODEL.items()
        if base_model.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Artifact base-model provenance mismatch: {mismatches}")

    preprocessing = _require_mapping(manifest.get("preprocessing"), "preprocessing")
    if preprocessing.get("schemaVersion") != 1:
        raise ValueError("Artifact preprocessing.schemaVersion must be 1")
    encoder_state = _require_mapping(preprocessing.get("encoder"), "preprocessing.encoder")
    TabularFeatureEncoder.from_state(encoder_state)

    target_column = preprocessing.get("targetColumn")
    if not isinstance(target_column, str) or not target_column:
        raise ValueError("Artifact preprocessing.targetColumn must be a non-empty string")
    if manifest.get("targetColumn") != target_column:
        raise ValueError("Artifact targetColumn disagrees with fitted preprocessing state")

    preprocessing_drop = preprocessing.get("dropColumns")
    if not isinstance(preprocessing_drop, list) or not all(isinstance(v, str) for v in preprocessing_drop):
        raise ValueError("Artifact preprocessing.dropColumns must be a list of strings")
    if manifest.get("dropColumns") != preprocessing_drop:
        raise ValueError("Artifact dropColumns disagree with fitted preprocessing state")

    class_labels = preprocessing.get("classLabels")
    if (
        not isinstance(class_labels, list)
        or len(class_labels) < 2
        or not all(isinstance(v, str) for v in class_labels)
        or len(class_labels) != len(set(class_labels))
    ):
        raise ValueError("Artifact preprocessing.classLabels must contain at least two unique strings")
    if manifest.get("classNames") != class_labels:
        raise ValueError("Artifact classNames disagree with fitted preprocessing classLabels")

    runtime = _require_mapping(manifest.get("runtimeConfig"), "runtimeConfig")
    _validate_runtime_config(runtime)
    if runtime["target_column"] != target_column:
        raise ValueError("Artifact runtimeConfig.target_column disagrees with fitted preprocessing state")
    effective_runtime_drop = [column for column in runtime["drop_columns"] if column != target_column]
    if effective_runtime_drop != preprocessing_drop:
        raise ValueError(
            "Artifact effective runtimeConfig.drop_columns disagree with fitted preprocessing state"
        )
    preprocessing_seed = preprocessing.get("seed")
    if isinstance(preprocessing_seed, bool) or not isinstance(preprocessing_seed, int):
        raise ValueError("Artifact preprocessing.seed must be an integer")
    if runtime["seed"] != preprocessing_seed:
        raise ValueError("Artifact runtimeConfig.seed disagrees with fitted preprocessing seed")

    training_context = _require_mapping(manifest.get("trainingContext"), "trainingContext")
    context_rel = training_context.get("path")
    if not isinstance(context_rel, str) or not context_rel or "\\" in context_rel:
        raise ValueError("Artifact trainingContext.path must be a portable POSIX relative path")
    posix_path = PurePosixPath(context_rel)
    if (
        posix_path.is_absolute()
        or ".." in posix_path.parts
        or len(posix_path.parts) != 1
        or posix_path.name != "training_context.parquet"
    ):
        raise ValueError("Artifact trainingContext.path must name training_context.parquet in the artifact directory")

    root = artifact_file.parent.resolve()
    context_file = artifact_file.parent / posix_path.name
    if context_file.is_symlink():
        raise ValueError("Artifact training context must not be a symlink")
    if not context_file.is_file():
        raise FileNotFoundError(f"Training context table not found: {context_file}")
    resolved_context = context_file.resolve()
    if resolved_context.parent != root:
        raise ValueError("Artifact training context resolves outside the artifact directory")

    context_size = context_file.stat().st_size
    if context_size > max_context_bytes:
        raise ValueError(
            f"Artifact training context exceeds maximum allowed bytes: {context_size} > {max_context_bytes}"
        )
    declared_size = training_context.get("sizeBytes")
    if declared_size is not None:
        if isinstance(declared_size, bool) or not isinstance(declared_size, int) or declared_size < 0:
            raise ValueError("Artifact trainingContext.sizeBytes must be a non-negative integer")
        if declared_size != context_size:
            raise ValueError(
                f"Artifact training-context size mismatch: expected {declared_size}, got {context_size}"
            )

    expected_sha = training_context.get("sha256")
    if not isinstance(expected_sha, str) or not _SHA256_RE.fullmatch(expected_sha):
        raise ValueError("Artifact trainingContext.sha256 must be a 64-character hexadecimal SHA-256")
    actual_sha = _sha256(context_file)
    if actual_sha.lower() != expected_sha.lower():
        raise RuntimeError(
            f"Artifact training-context digest mismatch: expected {expected_sha}, got {actual_sha}"
        )

    if strict_directory:
        files = {path.name for path in artifact_file.parent.iterdir() if path.is_file()}
        expected_files = {artifact_file.name, context_file.name}
        unexpected = sorted(files - expected_files)
        if unexpected:
            raise ValueError(f"Artifact directory contains unexpected unlisted files: {unexpected}")

    return manifest, context_file