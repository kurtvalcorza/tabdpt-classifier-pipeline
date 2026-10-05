"""Stage runner for the standalone TabDPT classifier E2E tutorial (NOTEBOOK_SPEC 2.2 §25.13 isolated-environment pattern).

The tutorial notebook carries this file verbatim (as ``tutorial_stages.py`` in its run directory, beside the carried
package under ``src/``) and runs every stage with the interpreter of an isolated, hash-locked environment::

    python -u tutorial_stages.py --root RUN_DIR --outputs OUTPUTS --weights WEIGHTS --stage data --options '{...}'

Nothing is installed into the notebook kernel. Each stage is a separate process, so a stage starts from files only:
the verified checkpoint under ``--weights``, the table and split written by earlier stages under ``RUN_DIR/state``,
and the learner-facing exports under ``--outputs``. In-context conditioning (``fit``) is deterministic for a fixed
seed and support table, so a stage that needs the conditioned model conditions again from the saved split. On failure
a stage writes ``RUN_DIR/state/<stage>.error.json`` with the exception type and message, which the notebook re-raises
in the kernel.

Stages: weights → data → validate → condition → report → predict → export → reload, plus the optional ``activity``.
The ``data``, ``validate`` and ``report`` stages import no model library, so CI exercises them directly.
"""
# ruff: noqa: E501  -- the printed dictionaries are the learner-facing output; they are kept on one line each
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import io
import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

STEM = "tabdpt_classifier"
PACKAGE = "tabdpt_classifier_pipeline"
TARGET_DEFAULT = "target"
SEED = 42
BATCH_SIZE = 512
TEST_SIZE = 0.2
NEW_ROWS = 8
RELOAD_RTOL, RELOAD_ATOL = 1e-5, 1e-6
REFERENCE_SPLITS = 20  # seeded stratified splits used to measure the classical reference's split-to-split range
SAMPLE_NAME = "sklearn-breast-cancer.csv"
SAMPLE_ARTIFACT_RECORD = "sample-artifact/SAMPLE_ARTIFACT.json"  # carried: the pinned sample artifact's provenance
CONTROL_LIMITS = {"n_ensembles": (1, 16), "context_size": (16, 4096)}
ACTIVITY_DIR = "activity"


# --------------------------------------------------------------------------------------------------
# run context
# --------------------------------------------------------------------------------------------------


class Run:
    """Paths of one run: carried sources and state under ``root``; learner-facing files under ``outputs``."""

    def __init__(self, root: Path, outputs: Path, weights: Path, options: dict[str, Any]) -> None:
        self.root = root
        self.out = outputs
        self.weights = weights
        self.options = options
        self.state = root / "state"
        self.out.mkdir(parents=True, exist_ok=True)
        self.state.mkdir(parents=True, exist_ok=True)

    def write_state(self, name: str, value: Any) -> Path:
        path = self.state / name
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def read_state(self, name: str, needed_by: str) -> Any:
        path = self.state / name
        if not path.is_file():
            raise RuntimeError(f"{name} is missing: run the stage that writes it before '{needed_by}' (run the notebook from Section 4)")
        return json.loads(path.read_text(encoding="utf-8"))

    def read_frame(self, name: str, needed_by: str):
        import pandas as pd

        path = self.state / name
        if not path.is_file():
            raise RuntimeError(f"{name} is missing: run the stage that writes it before '{needed_by}' (run the notebook from Section 4)")
        return pd.read_parquet(path, engine="pyarrow")

    def write_output(self, name: str, value: Any) -> Path:
        path = self.out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
        return path


def package(root: Path):
    """The carried package (``ROOT/src``); in the repository, ``src/``."""
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return importlib.import_module(PACKAGE)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frame_sha256(frame) -> str:
    return hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()


def rounded(value: Any, digits: int = 4) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {k: rounded(v, digits) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [rounded(v, digits) for v in value]
    return value


# --------------------------------------------------------------------------------------------------
# data: the public sample and the BYOD reader (TDC-M3, TDC-m1)
# --------------------------------------------------------------------------------------------------


def load_sample_frame():
    """scikit-learn's bundled breast-cancer table with readable class names (no download)."""
    from sklearn.datasets import load_breast_cancer

    frame = load_breast_cancer(as_frame=True).frame.copy()
    frame[TARGET_DEFAULT] = frame[TARGET_DEFAULT].map({0: "malignant", 1: "benign"}).astype(str)
    return frame


def _missing_target_rows(series) -> list[int]:
    """0-based data-row positions whose label is missing or blank."""
    import pandas as pd

    blank = series.isna() | series.astype("string").str.strip().eq("").fillna(False)
    return [int(i) for i in pd.Series(blank.to_numpy()).to_numpy().nonzero()[0]]


def read_byod(path: str | Path, target: str, categorical_columns: list[str]):
    """Read one labelled CSV or Parquet file and refuse, naming the file and the rule, anything that would be
    silently rewritten later: duplicate or absent columns, and missing or blank target values (which a ``str``
    conversion would otherwise turn into a class called ``'<NA>'``)."""
    import pandas as pd

    path = Path(path)
    name = path.name
    if not path.is_file():
        raise FileNotFoundError(f"BYOD_PATH {str(path)!r} does not exist in this runtime: upload the file or correct the path.")
    suffix = path.suffix.lower()
    if not isinstance(target, str) or not target:
        raise ValueError("TARGET must name the label column of your table (a non-empty string).")
    if not isinstance(categorical_columns, list) or not all(isinstance(c, str) for c in categorical_columns):
        raise ValueError(f"CATEGORICAL_COLUMNS must be a list of column names, got {categorical_columns!r}.")
    raw = path.read_bytes()
    if suffix == ".csv":
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{name}: a BYOD CSV must be UTF-8 text ({exc}).") from None
        header = next(csv.reader(io.StringIO(text)), [])
        if not header:
            raise ValueError(f"{name}: the CSV is empty (no header row).")
        duplicates = sorted({x for x in header if header.count(x) > 1})
        if duplicates:
            raise ValueError(f"{name}: duplicate CSV columns {duplicates}; every column name must be unique.")
        if target not in header:
            raise ValueError(f"{name}: the target column {target!r} is not in the header {header}. Set TARGET to the name of your label column.")
        missing_declared = [c for c in categorical_columns if c not in header]
        if missing_declared:
            raise ValueError(f"{name}: CATEGORICAL_COLUMNS not present in CSV header: {missing_declared}; header is {header}.")
        frame = pd.read_csv(io.BytesIO(raw), dtype={c: "string" for c in [*categorical_columns, target]})
    elif suffix in (".parquet", ".pq"):
        frame = pd.read_parquet(io.BytesIO(raw), engine="pyarrow")
        header = [str(c) for c in frame.columns]
        if frame.columns.duplicated().any():
            raise ValueError(f"{name}: duplicate Parquet columns; every column name must be unique.")
        if target not in frame.columns:
            raise ValueError(f"{name}: the target column {target!r} is not among the columns {header}. Set TARGET to the name of your label column.")
        missing_declared = [c for c in categorical_columns if c not in frame.columns]
        if missing_declared:
            raise ValueError(f"{name}: CATEGORICAL_COLUMNS not present in the Parquet columns: {missing_declared}; columns are {header}.")
        for column in [*categorical_columns, target]:
            frame[column] = frame[column].astype("string")
    else:
        raise ValueError(f"{name}: BYOD must be one CSV (.csv) or Parquet (.parquet/.pq) file.")
    if frame.empty:
        raise ValueError(f"{name}: the table has a header but no data rows.")
    missing = _missing_target_rows(frame[target])
    if missing:
        shown = ", ".join(str(i + 2) for i in missing[:10]) + (" …" if len(missing) > 10 else "")
        raise ValueError(
            f"{name}: the target column {target!r} has {len(missing)} missing or blank value(s) out of {len(frame)} rows "
            f"(file lines {shown}). Every row needs a label: remove those rows or label them, then run Section 4 again. "
            "Nothing was written to outputs/."
        )
    frame[target] = frame[target].astype(str)
    return frame


def clear_outputs(run: Run) -> list[str]:
    """Remove this notebook's previous exports, so a refused input never leaves an earlier run's files looking current."""
    removed = []
    for path in sorted(run.out.glob(f"{STEM}_*")):
        if path.is_file():
            path.unlink()
            removed.append(path.name)
    for name in ("artifact", "artifact-reload"):
        if (run.out / name).is_dir():
            shutil.rmtree(run.out / name)
            removed.append(name + "/")
    return removed


# --------------------------------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------------------------------


def stage_weights(run: Run) -> None:
    P = package(run.root)
    snapshot = run.weights / P.MODEL_KEY
    snapshot.mkdir(parents=True, exist_ok=True)
    carried = run.root / "weights" / P.MODEL_KEY / P.MANIFEST_NAME
    manifest = json.loads(carried.read_text(encoding="utf-8"))
    if (manifest["modelId"], manifest["revision"]) != (P.MODEL_ID, P.MODEL_REVISION):
        raise RuntimeError("the carried manifest does not name the identity carried by the package; regenerate the notebook")
    shutil.copyfile(carried, snapshot / P.MANIFEST_NAME)
    print({"model_id": P.MODEL_ID, "revision": P.MODEL_REVISION, "license": P.MODEL_LICENSE, "files": len(manifest["files"]), "total_bytes": manifest["totalBytes"]})
    fetched = P.stage_missing_files(snapshot, allow_download=True)
    print({"weights_dir": str(snapshot), "fetched": fetched})
    verified = P.verify_snapshot(snapshot)
    print({"verified_files": len(verified["files"]), "revision": verified["revision"], "sha256": [f["sha256"] for f in verified["files"]]})
    run.write_state("weights.json", {"snapshot": str(snapshot), "weight": str(snapshot / P.TABDPT_WEIGHT_FILENAME)})


def stage_data(run: Run) -> None:
    opts = run.options
    use_byod = bool(opts.get("use_byod", False))
    target = opts.get("target", TARGET_DEFAULT) if use_byod else TARGET_DEFAULT
    categorical = list(opts.get("categorical_columns") or []) if use_byod else []
    removed = clear_outputs(run)
    for name in ("frame.parquet", "train.parquet", "test.parquet", "data.json", "split.json", "condition.json", "export_reference.json"):
        (run.state / name).unlink(missing_ok=True)
    if removed:
        print({"removed_previous_outputs": removed})
    if use_byod:
        byod_path = opts.get("byod_path") or ""
        if not byod_path:
            raise ValueError("USE_BYOD is True but no file was supplied: set BYOD_PATH, or (in Colab) leave it empty and choose a file in the upload dialog.")
        frame = read_byod(byod_path, target, categorical)
        kind, name = "BYOD", Path(byod_path).name
    else:
        frame = load_sample_frame()
        kind, name = "sample", SAMPLE_NAME
    digest = frame_sha256(frame)
    frame.to_parquet(run.state / "frame.parquet", index=True, engine="pyarrow")
    record = {"sample_kind": kind, "name": name, "rows": len(frame), "columns": frame.shape[1], "csv_sha256": digest, "target": target, "declared_categorical_columns": categorical if use_byod else "sample schema"}
    run.write_state("data.json", record)
    print(record)
    print({"class_counts": frame[target].value_counts().sort_index().to_dict()})


def stage_validate(run: Run) -> None:
    import numpy as np
    from sklearn.model_selection import train_test_split

    P = package(run.root)
    data = run.read_state("data.json", "validate")
    frame = run.read_frame("frame.parquet", "validate")
    target = data["target"]
    print({"ceilings": {"MIN_CLASSES": P.MIN_CLASSES, "DECISION_RULE": P.DECISION_RULE, "TEST_SIZE": TEST_SIZE, "SEED": SEED}})
    input_manifest = P.validate_inputs(frame, target_column=target, names=[data["name"]])
    # Demonstrate rejection on a probe that breaks the target contract; the finding is recorded, not swallowed.
    probe = frame.head(4).copy()
    probe[target] = probe[target].astype(object)
    probe.loc[probe.index[0], target] = None
    try:
        P.validate_inputs(probe, target_column=target)
    except ValueError as exc:
        input_manifest["findings"].append({"input": "missing-target-probe", "verdict": "rejected", "message": str(exc)})
    features = frame.drop(columns=[target])
    for column in features.select_dtypes(include=np.number).columns:
        values = features[column].dropna().to_numpy(dtype=float)
        if values.size and not np.isfinite(values).all():
            raise ValueError(f"{data['name']}: numeric feature {column!r} contains infinite values; replace them with a finite value or leave the cell empty (missing).")
    class_counts = frame[target].value_counts().sort_index()
    if (class_counts < 2).any():
        raise ValueError(f"{data['name']}: every class needs at least 2 rows for a stratified split: {class_counts.to_dict()}")
    train, test = train_test_split(frame, test_size=TEST_SIZE, random_state=SEED, stratify=frame[target])
    unseen = sorted(set(test[target]) - set(train[target]))
    if unseen:
        raise ValueError(f"Holdout classes absent from support data: {unseen}")
    run.write_output(f"{STEM}_input_manifest.json", input_manifest)
    print(json.dumps(input_manifest["inputs"][0], indent=2))
    print("findings:", input_manifest["findings"])
    train.to_parquet(run.state / "train.parquet", index=True, engine="pyarrow")
    test.to_parquet(run.state / "test.parquet", index=True, engine="pyarrow")
    split = {"method": "single seeded stratified 80/20 random holdout", "seed": SEED, "trainRows": len(train), "holdoutRows": len(test), "trainClassCounts": train[target].value_counts().sort_index().to_dict(), "holdoutClassCounts": test[target].value_counts().sort_index().to_dict()}
    run.write_state("split.json", split)
    print({"train": list(train.shape), "holdout": list(test.shape), "train_class_counts": split["trainClassCounts"], "holdout_class_counts": split["holdoutClassCounts"]})


def _controls(opts: dict[str, Any]) -> dict[str, int]:
    controls = {}
    for key, default in (("n_ensembles", 2), ("context_size", 512)):
        value = opts.get(key, default)
        low, high = CONTROL_LIMITS[key]
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"{key.upper()} must be an integer from {low} to {high}, got {value!r}.")
        controls[key] = value
    return {**controls, "batch_size": BATCH_SIZE, "seed": SEED}


def conditioned_pipeline(run: Run, train, target: str):
    """The pinned checkpoint, conditioned in context on the support split (deterministic for a fixed seed)."""
    P = package(run.root)
    pipe = P.TabDPTClassificationPipeline.from_pretrained(weights_dir=run.weights / P.MODEL_KEY, compile_model=False, use_flash=False, seed=SEED)
    pipe.fit(train, target_column=target, seed=SEED)
    return pipe


def classical_reference(train, test, target: str, frame=None) -> dict[str, Any]:
    """Standardised logistic regression on the package's own feature encoding — a one-line classical model that
    makes the in-context result readable (TDC-m2) — plus its split-to-split range over REFERENCE_SPLITS seeds."""
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from tabdpt_classifier_pipeline import TabularFeatureEncoder

    def fit_score(tr, te):
        encoder = TabularFeatureEncoder().fit(tr.drop(columns=[target]))
        model = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(), LogisticRegression(max_iter=5000))
        model.fit(encoder.transform(tr.drop(columns=[target])), tr[target].astype(str))
        x = encoder.transform(te.drop(columns=[target]))
        proba = model.predict_proba(x)
        labels = list(model.classes_)
        pred = np.asarray(labels)[proba.argmax(axis=1)]
        y = te[target].astype(str).to_numpy()
        out = {"accuracy": float(accuracy_score(y, pred)), "log_loss": float(log_loss(y, proba, labels=labels))}
        if len(labels) == 2 and len(set(y)) == 2:
            out["roc_auc"] = float(roc_auc_score((y == labels[1]).astype(int), proba[:, 1]))
        return out, int((pred != y).sum())

    metrics, errors = fit_score(train, test)
    result: dict[str, Any] = {"id": "standardised_logistic_regression", "metrics": metrics, "holdout_errors": errors}
    if frame is not None:
        accuracies = []
        for seed in range(REFERENCE_SPLITS):
            try:
                tr, te = train_test_split(frame, test_size=TEST_SIZE, random_state=seed, stratify=frame[target])
                accuracies.append(fit_score(tr, te)[0]["accuracy"])
            except ValueError:
                continue
        if accuracies:
            result["split_to_split_accuracy"] = {"splits": len(accuracies), "min": min(accuracies), "median": float(np.median(accuracies)), "max": max(accuracies)}
    return result


def stage_condition(run: Run) -> None:
    P = package(run.root)
    data = run.read_state("data.json", "condition")
    target = data["target"]
    frame = run.read_frame("frame.parquet", "condition")
    train = run.read_frame("train.parquet", "condition")
    test = run.read_frame("test.parquet", "condition")
    kw = _controls(run.options)
    print({"request": kw})
    print("Requested context_size:", kw["context_size"], "effective support rows <=", min(len(train), kw["context_size"]))
    if len(train) > kw["context_size"]:
        print("Support exceeds context_size; seeded class-balanced upstream context subsampling applies.")
    baseline = P.majority_class_baseline(train[target], test[target])
    pipe = conditioned_pipeline(run, train, target)
    encoded_feature_count = len(pipe.feature_encoder.feature_columns)
    model_feature_ceiling = int(pipe.estimator.max_features)
    feature_reduction = str(pipe.estimator.feature_reduction)
    numeric_missing_columns = [c for c in pipe.feature_encoder.feature_columns if c in pipe.feature_encoder.numeric_columns and train[c].isna().any()]
    capacity = {"classLabels": pipe.class_labels_, "encodedFeatureCount": encoded_feature_count, "modelFeatureCeiling": model_feature_ceiling, "featureReduction": feature_reduction, "featureReductionActive": encoded_feature_count > model_feature_ceiling, "numericMissingColumns": numeric_missing_columns, "upstreamImputer": type(pipe.estimator.imputer).__name__, "source": pipe.source}
    print(capacity)
    if encoded_feature_count > model_feature_ceiling:
        print(f"Feature count exceeds {model_feature_ceiling}; upstream {feature_reduction} reduction is active.")
    else:
        print("Feature reduction is not active for this dataset.")
    if numeric_missing_columns:
        print("Numeric missing values are mean-imputed using support/training-fitted statistics.")
    metrics = pipe.evaluate(test, **kw)
    pred = pipe.predict(test.drop(columns=[target]), **kw)
    errors = int((pred.to_numpy() != test[target].astype(str).to_numpy()).sum())
    reference = classical_reference(train, test, target, frame)
    baseline_errors = int((test[target].astype(str) != str(train[target].value_counts().sort_index().idxmax())).sum())
    print("tutorial TabDPT", rounded(metrics), {"holdout_errors": errors, "holdout_rows": len(test)})
    print("majority-class baseline", rounded(baseline), {"holdout_errors": baseline_errors})
    print("standardised logistic regression (classical reference)", rounded(reference["metrics"]), {"holdout_errors": reference["holdout_errors"]})
    if "split_to_split_accuracy" in reference:
        print("classical reference accuracy over", reference["split_to_split_accuracy"]["splits"], "seeded splits:", rounded(reference["split_to_split_accuracy"]))
    run.write_state("condition.json", {"controls": kw, "metrics": metrics, "holdout_errors": errors, "baseline": baseline, "baseline_errors": baseline_errors, "reference": reference, "capacity": capacity})


def build_report(P, data: dict[str, Any], split: dict[str, Any], cond: dict[str, Any]) -> dict[str, Any]:
    n = split["holdoutRows"]
    report = P.evaluation_report(cond["metrics"], baseline=cond["baseline"], n_holdout=n, class_labels=cond["capacity"]["classLabels"], sample_kind=data["sample_kind"], estimation="single seeded stratified 80/20 holdout; no dispersion estimate for the model")
    reference = cond["reference"]
    report["baselines"].append({"id": reference["id"], "metrics": [{"id": k, "value": float(v)} for k, v in reference["metrics"].items()], "holdout_errors": reference["holdout_errors"], **({"split_to_split_accuracy": reference["split_to_split_accuracy"]} if "split_to_split_accuracy" in reference else {})})
    report["baselines"][0]["holdout_errors"] = cond["baseline_errors"]
    report["holdout_errors"] = cond["holdout_errors"]
    spread = reference.get("split_to_split_accuracy")
    one_row = 1.0 / n if n else float("nan")
    report["interpretation"] = (
        f"One holdout row is {one_row:.4f} of accuracy ({n} rows). "
        + (
            f"The classical reference's accuracy ranged {spread['min']:.4f}–{spread['max']:.4f} over {spread['splits']} seeded splits of the same table, "
            "so a difference of one or two holdout rows between models is within split-to-split variation; "
            if spread
            else "No split-to-split dispersion was measured. "
        )
        + "no dispersion was measured for the in-context model itself."
    )
    return report


def stage_report(run: Run) -> None:
    P = package(run.root)
    data = run.read_state("data.json", "report")
    split = run.read_state("split.json", "report")
    cond = run.read_state("condition.json", "report")
    report = build_report(P, data, split, cond)
    run.write_output(f"{STEM}_evaluation_report.json", report)
    print(json.dumps(rounded(report), indent=2))
    if report["verdict"] == "not-measurable":
        print("No labelled holdout was scored; the metrics above are absent by construction.")


def runtime_identity() -> dict[str, Any]:
    import importlib.metadata
    import platform

    import numpy
    import pandas
    import sklearn
    import torch

    return {"python": platform.python_version(), "torch": torch.__version__, "tabdpt": importlib.metadata.version("tabdpt"), "numpy": numpy.__version__, "pandas": pandas.__version__, "sklearn": sklearn.__version__, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", "use_flash": False}


def _new_rows(test, target: str):
    return test.drop(columns=[target]).head(NEW_ROWS).copy()


def stage_predict(run: Run) -> None:
    import pandas as pd

    P = package(run.root)
    data = run.read_state("data.json", "predict")
    split = run.read_state("split.json", "predict")
    cond = run.read_state("condition.json", "predict")
    target = data["target"]
    train = run.read_frame("train.parquet", "predict")
    test = run.read_frame("test.parquet", "predict")
    kw = cond["controls"]
    pipe = conditioned_pipeline(run, train, target)
    new_rows = _new_rows(test, target)
    proba = pipe.predict_proba(new_rows, **kw)
    pred = pipe.predict(new_rows, **kw)
    out = pd.DataFrame({"row_id": new_rows.index.to_numpy(), "prediction": pred.to_numpy()})
    for label in pipe.class_labels_:
        out[f"proba_{label}"] = proba[label].to_numpy()
    out.to_csv(run.out / f"{STEM}_predictions.csv", index=False)
    # The same rows without their labels: the input the companion artifact-inference notebook accepts (TDC-S2).
    new_rows.reset_index(names="row_id").to_csv(run.out / f"{STEM}_new_rows.csv", index=False)
    report = json.loads((run.out / f"{STEM}_evaluation_report.json").read_text(encoding="utf-8"))
    input_manifest = json.loads((run.out / f"{STEM}_input_manifest.json").read_text(encoding="utf-8"))
    source = json.loads((run.root / "source.json").read_text(encoding="utf-8")) if (run.root / "source.json").is_file() else {}
    payload = {
        "predictions": out.to_dict(orient="records"),
        "metrics": cond["metrics"],
        "holdout_errors": cond["holdout_errors"],
        "majority_class_baseline": cond["baseline"],
        "classical_reference": cond["reference"],
        "evaluation_report": report,
        "input_manifest": input_manifest,
        "sample": {"kind": data["sample_kind"], "name": data["name"], "rows": data["rows"], "csv_sha256": data["csv_sha256"]},
        "split": split,
        "preprocessing": {"encodedFeatureCount": cond["capacity"]["encodedFeatureCount"], "modelFeatureCeiling": cond["capacity"]["modelFeatureCeiling"], "featureReduction": cond["capacity"]["featureReduction"], "featureReductionActive": cond["capacity"]["featureReductionActive"], "numericMissingPolicy": "support/training-fitted mean imputation", "categoricalMissingPolicy": "dedicated fitted missing code", "unknownCategoryPolicy": "dedicated fitted unknown code", "declaredCategoricalColumns": data["declared_categorical_columns"] if data["sample_kind"] == "BYOD" else []},
        "inference": {**kw, "decisionRule": P.DECISION_RULE + " over uncalibrated class probabilities", "threshold": None, "classLabels": pipe.class_labels_, "newRows": "the first 8 holdout rows with their labels removed; they were already scored in Section 6"},
        "notebook_source": source,
        "repository_revision": source.get("revision"),
        "model_id": P.MODEL_ID,
        "model_revision": P.MODEL_REVISION,
        "model_license": P.MODEL_LICENSE,
        "upstream_code_commit": P.TABDPT_UPSTREAM_CODE_COMMIT,
        "runtime": runtime_identity(),
    }
    run.write_output(f"{STEM}_result.json", payload)
    print(out.round(4).to_string(index=False))
    print(sorted(p.name for p in run.out.iterdir()))


def artifact_sha256(artifact_dir: Path) -> str:
    """The artifact's trusted digest: SHA-256 of ``artifact.json``, which pins ``training_context.parquet`` by size and
    SHA-256 (checked by ``validate_artifact_bundle``), so this one digest fixes both files."""
    return sha256_file(Path(artifact_dir) / "artifact.json")


def stage_export(run: Run) -> None:
    P = package(run.root)
    data = run.read_state("data.json", "export")
    cond = run.read_state("condition.json", "export")
    target = data["target"]
    train = run.read_frame("train.parquet", "export")
    test = run.read_frame("test.parquet", "export")
    pipe = conditioned_pipeline(run, train, target)
    art = run.out / "artifact"
    if art.exists():
        shutil.rmtree(art)
    manifest_path = P.export_artifact_bundle(pipe, train, art)
    digest = artifact_sha256(art)
    new_rows = _new_rows(test, target)
    proba = pipe.predict_proba(new_rows, **cond["controls"])
    pred = pipe.predict(new_rows, **cond["controls"])
    run.write_state("export_reference.json", {"labels": [str(x) for x in pred], "proba": proba.to_numpy().tolist(), "classLabels": pipe.class_labels_, "weight": str(pipe.model_weight_path)})
    record: dict[str, Any] = {"artifact": str(manifest_path), "artifact_files": sorted(p.name for p in art.iterdir()), "artifact_sha256": digest, "training_context_sha256": json.loads(manifest_path.read_text(encoding="utf-8"))["trainingContext"]["sha256"]}
    sample_record = run.root / SAMPLE_ARTIFACT_RECORD
    if data["sample_kind"] == "sample" and sample_record.is_file():
        pinned = json.loads(sample_record.read_text(encoding="utf-8"))
        record["matches_pinned_sample_artifact"] = digest == pinned["artifact_sha256"]
        record["pinned_sample_artifact_sha256"] = pinned["artifact_sha256"]
    result_path = run.out / f"{STEM}_result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["artifact"] = record
        run.write_output(result_path.name, result)
    print(record)
    print(f"Trusted digest for the companion notebook: set EXPECTED_ARTIFACT_SHA256 = '{digest}' when you hand it outputs/artifact/.")
    if record.get("matches_pinned_sample_artifact") is False:
        print("Note: this export differs from the companion notebook's pinned sample artifact; record both digests (see Troubleshooting).")


def stage_reload(run: Run) -> None:
    import numpy as np

    P = package(run.root)
    data = run.read_state("data.json", "reload")
    cond = run.read_state("condition.json", "reload")
    reference = run.read_state("export_reference.json", "reload")
    test = run.read_frame("test.parquet", "reload")
    art = run.out / "artifact"
    reload_dir = run.out / "artifact-reload"
    if reload_dir.exists():
        shutil.rmtree(reload_dir)
    shutil.copytree(art, reload_dir)
    P.validate_artifact_bundle(reload_dir / "artifact.json")
    reloaded = P.load_verified_artifact(reload_dir / "artifact.json", model_weight_path=reference["weight"], compile_model=False, use_flash=False, seed=SEED)
    if reloaded.preprocessing_restored_ is not True:
        raise RuntimeError("Verified reload must restore fitted preprocessing state without refit.")
    if reloaded.class_labels_ != reference["classLabels"]:
        raise RuntimeError(f"Reloaded class labels {reloaded.class_labels_} must match the exported class order {reference['classLabels']}.")
    new_rows = _new_rows(test, data["target"])
    proba2 = reloaded.predict_proba(new_rows, **cond["controls"])
    pred2 = reloaded.predict(new_rows, **cond["controls"])
    if [str(x) for x in pred2] != reference["labels"]:
        raise RuntimeError("Reloaded argmax labels must match exactly; they differ from the exporting process's labels.")
    np.testing.assert_allclose(np.asarray(reference["proba"]), proba2.to_numpy(), rtol=RELOAD_RTOL, atol=RELOAD_ATOL)
    max_diff = float(np.max(np.abs(np.asarray(reference["proba"]) - proba2.to_numpy())))
    print({"reload_dir": str(reload_dir), "process": "fresh (separate from the exporting stage)", "preprocessing_restored_": True, "labels_identical": True, "max_abs_probability_difference": max_diff})
    print(f"PASS: fitted preprocessing restored without refit; labels identical and probabilities equivalent (rtol={RELOAD_RTOL}, atol={RELOAD_ATOL}).")


def canonical_digests(run: Run) -> dict[str, str]:
    files = sorted(p for p in run.out.glob(f"{STEM}_*") if p.is_file()) + sorted((run.out / "artifact").glob("*"))
    return {str(p.relative_to(run.out)): sha256_file(p) for p in files}


def stage_activity(run: Run) -> None:
    """Optional activity: change the number of ensemble members, evaluate on the same holdout, write to outputs/activity/
    and prove the canonical outputs did not change."""
    data = run.read_state("data.json", "activity")
    cond = run.read_state("condition.json", "activity")
    target = data["target"]
    train = run.read_frame("train.parquet", "activity")
    test = run.read_frame("test.parquet", "activity")
    kw = _controls({"n_ensembles": run.options.get("n_ensembles", 8), "context_size": cond["controls"]["context_size"]})
    before = canonical_digests(run)
    pipe = conditioned_pipeline(run, train, target)
    metrics = pipe.evaluate(test, **kw)
    pred = pipe.predict(test.drop(columns=[target]), **kw)
    errors = int((pred.to_numpy() != test[target].astype(str).to_numpy()).sum())
    record = {"changed": {"n_ensembles": [cond["controls"]["n_ensembles"], kw["n_ensembles"]]}, "canonical": {"metrics": cond["metrics"], "holdout_errors": cond["holdout_errors"]}, "activity": {"metrics": metrics, "holdout_errors": errors}, "holdout_rows": len(test)}
    run.write_output(f"{ACTIVITY_DIR}/{STEM}_activity_n_ensembles_{kw['n_ensembles']}.json", record)
    after = canonical_digests(run)
    if before != after:
        raise RuntimeError("The activity changed a canonical output; it must write only to outputs/activity/.")
    print(rounded(record))
    print({"canonical_outputs_unchanged": True, "files_checked": len(after)})


STAGES = {
    "weights": stage_weights,
    "data": stage_data,
    "validate": stage_validate,
    "condition": stage_condition,
    "report": stage_report,
    "predict": stage_predict,
    "export": stage_export,
    "reload": stage_reload,
    "activity": stage_activity,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--options", default="{}")
    args = parser.parse_args(argv)
    run = Run(args.root.resolve(), args.outputs.resolve(), args.weights.resolve(), json.loads(args.options))
    error_file = run.state / f"{args.stage}.error.json"
    error_file.unlink(missing_ok=True)
    try:
        STAGES[args.stage](run)
    except BaseException as exc:  # noqa: BLE001 -- every failure is reported to the kernel with its own message
        traceback.print_exc()
        error_file.write_text(json.dumps({"type": type(exc).__name__, "message": str(exc)}), encoding="utf-8")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
