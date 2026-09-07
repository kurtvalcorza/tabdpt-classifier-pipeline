from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

TABDPT_PACKAGE_VERSION = "1.2.0"
TABDPT_HF_REPO = "Layer6/TabDPT"
TABDPT_HF_REVISION = "4462ffbd1d8dea25d4862d30beed4b70cd596ae5"
TABDPT_WEIGHT_FILENAME = "tabdpt1_2.safetensors"
TABDPT_WEIGHT_SHA256 = "06680220fd66c4524051706b98c1c659a674d19d3a766cd0bb276505e99faccd"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_tabdpt_weights(model_weight_path: str | Path | None = None, cache_dir: str | Path | None = None) -> Path:
    if model_weight_path is None:
        model_weight_path = hf_hub_download(
            repo_id=TABDPT_HF_REPO,
            filename=TABDPT_WEIGHT_FILENAME,
            revision=TABDPT_HF_REVISION,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
    path = Path(model_weight_path)
    if not path.is_file():
        raise FileNotFoundError(f"TabDPT model weight not found: {path}")
    actual = sha256_file(path)
    if actual != TABDPT_WEIGHT_SHA256:
        raise RuntimeError(
            f"TabDPT weight SHA-256 mismatch: expected {TABDPT_WEIGHT_SHA256}, got {actual}"
        )
    return path


class TabularFeatureEncoder:
    """Train-fitted mixed-table encoder with explicit missing/unknown categorical codes."""

    def __init__(self) -> None:
        self.feature_columns: list[str] = []
        self.numeric_columns: set[str] = set()
        self.category_maps: dict[str, dict[str, int]] = {}
        self.is_fitted = False

    def fit(self, frame: pd.DataFrame) -> "TabularFeatureEncoder":
        if frame.columns.duplicated().any():
            raise ValueError("Duplicate feature column names are not supported")
        if frame.shape[1] == 0:
            raise ValueError("At least one feature column is required")
        self.feature_columns = list(frame.columns)
        self.numeric_columns = {
            col for col in self.feature_columns if pd.api.types.is_numeric_dtype(frame[col])
        }
        self.category_maps = {}
        for col in self.feature_columns:
            if col in self.numeric_columns:
                continue
            values = sorted({str(v) for v in frame[col].dropna().tolist()})
            self.category_maps[col] = {value: idx for idx, value in enumerate(values)}
        self.is_fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Feature encoder is not fitted")
        missing = [col for col in self.feature_columns if col not in frame.columns]
        extra = [col for col in frame.columns if col not in self.feature_columns]
        if missing or extra:
            raise ValueError(f"Feature schema mismatch; missing={missing}, extra={extra}")
        out = np.empty((len(frame), len(self.feature_columns)), dtype=np.float64)
        for idx, col in enumerate(self.feature_columns):
            series = frame[col]
            if col in self.numeric_columns:
                out[:, idx] = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
                continue
            mapping = self.category_maps[col]
            unknown_code = float(len(mapping))
            missing_code = float(len(mapping) + 1)
            encoded = []
            for value in series.tolist():
                if pd.isna(value):
                    encoded.append(missing_code)
                else:
                    encoded.append(float(mapping.get(str(value), unknown_code)))
            out[:, idx] = encoded
        return out

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)


class TabDPTClassificationPipeline:
    def __init__(
        self,
        model_weight_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        device: str | None = None,
        use_flash: bool = True,
        compile_model: bool = False,
        verbose: bool = False,
    ) -> None:
        self.model_weight_path = model_weight_path
        self.cache_dir = cache_dir
        self.device = device
        self.use_flash = use_flash
        self.compile_model = compile_model
        self.verbose = verbose
        self.feature_encoder = TabularFeatureEncoder()
        self.target_column: str | None = None
        self.class_labels_: list[str] = []
        self.estimator: Any | None = None

    def fit(self, frame: pd.DataFrame, target_column: str = "target", drop_columns: list[str] | None = None):
        if frame.columns.duplicated().any():
            raise ValueError("Duplicate column names are not supported")
        if target_column not in frame.columns:
            raise ValueError(f"Target column {target_column!r} not found")
        if frame[target_column].isna().any():
            raise ValueError("Classification target contains missing values")
        drop_columns = [c for c in (drop_columns or []) if c != target_column]
        features = frame.drop(columns=[target_column, *drop_columns], errors="ignore")
        labels = frame[target_column].map(str)
        self.class_labels_ = sorted(labels.unique().tolist())
        if len(self.class_labels_) < 2:
            raise ValueError("Classification requires at least two classes")
        label_to_id = {label: idx for idx, label in enumerate(self.class_labels_)}
        y = labels.map(label_to_id).to_numpy(dtype=np.int64)
        X = self.feature_encoder.fit_transform(features)
        weights = resolve_tabdpt_weights(self.model_weight_path, self.cache_dir)
        from tabdpt import TabDPTClassifier
        self.estimator = TabDPTClassifier(
            model_weight_path=str(weights),
            device=self.device,
            use_flash=self.use_flash,
            compile=self.compile_model,
            context_reduction="subsample-balanced",
            verbose=self.verbose,
        )
        self.estimator.fit(X, y)
        self.target_column = target_column
        return self

    def _require_fitted(self):
        if self.estimator is None or self.target_column is None:
            raise RuntimeError("Pipeline is not fitted")

    def predict_proba(
        self,
        frame: pd.DataFrame,
        n_ensembles: int = 4,
        context_size: int | None = 2048,
        batch_size: int | None = 4096,
        temperature: float = 1.0,
        seed: int = 42,
    ) -> pd.DataFrame:
        self._require_fitted()
        X = self.feature_encoder.transform(frame[self.feature_encoder.feature_columns])
        proba = self.estimator.ensemble_predict_proba(
            X,
            n_ensembles=n_ensembles,
            temperature=temperature,
            context_size=context_size,
            batch_size=batch_size,
            permute_classes=True,
            seed=seed,
        ) if n_ensembles > 1 else self.estimator.predict_proba(
            X,
            temperature=temperature,
            context_size=context_size,
            batch_size=batch_size,
            seed=seed,
        )
        return pd.DataFrame(proba, columns=self.class_labels_, index=frame.index)

    def predict(self, frame: pd.DataFrame, **kwargs) -> pd.Series:
        proba = self.predict_proba(frame, **kwargs)
        labels = proba.columns.to_numpy()[np.argmax(proba.to_numpy(), axis=1)]
        return pd.Series(labels, index=frame.index, name="prediction")

    def evaluate(self, frame: pd.DataFrame, **kwargs) -> dict[str, float]:
        self._require_fitted()
        if self.target_column not in frame.columns:
            raise ValueError(f"Evaluation target {self.target_column!r} not found")
        y_true = frame[self.target_column].map(str)
        unknown = sorted(set(y_true.unique()) - set(self.class_labels_))
        if unknown:
            raise ValueError(f"Evaluation contains unseen target classes: {unknown}")
        features = frame.drop(columns=[self.target_column])
        proba = self.predict_proba(features, **kwargs)
        pred = proba.idxmax(axis=1)
        metrics = {
            "accuracy": float(accuracy_score(y_true, pred)),
            "log_loss": float(log_loss(y_true, proba.to_numpy(), labels=self.class_labels_)),
        }
        if len(self.class_labels_) == 2 and y_true.nunique() == 2:
            positive = self.class_labels_[1]
            binary = (y_true == positive).astype(int)
            metrics["roc_auc"] = float(roc_auc_score(binary, proba[positive]))
        return metrics
