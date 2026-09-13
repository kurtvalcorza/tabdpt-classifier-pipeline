"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 1.1 §3.6 standalone carrier) — E2E.

Only the task-specific prose and stage cells live here. Runtime install, the embedded package
(three modules, dependency-ordered), and the model pin/stage/verify cell are produced by the generator
from repository sources so they cannot drift from the package. The ARTIFACT-INFERENCE companion has
its own template, ``tools/notebook_template_artifact_inference.py``.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

REPO = "tabdpt-classifier-pipeline"
BADGES = [
    (
        "GitHub",
        "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
        f"https://github.com/kurtvalcorza/{REPO}",
    ),
    (
        "Open In Colab",
        "https://colab.research.google.com/assets/colab-badge.svg",
        f"https://colab.research.google.com/github/kurtvalcorza/{REPO}/blob/main/tutorials/tabdpt_classifier_colab.ipynb",
    ),
    (
        "Hugging Face",
        "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Layer6%2FTabDPT-ffcc4d?style=flat",
        "https://huggingface.co/Layer6/TabDPT",
    ),
    (
        "Upstream",
        "https://img.shields.io/badge/Upstream-layer6ai--labs%2FTabDPT--inference-181717?style=flat&logo=github&logoColor=white",
        "https://github.com/layer6ai-labs/TabDPT-inference",
    ),
    ("arXiv", "https://img.shields.io/badge/arXiv-2608.01400-b31b1b.svg", "https://arxiv.org/abs/2608.01400"),
]

TEMPLATE = {
    "package": "tabdpt_classifier_pipeline",
    "repo_name": REPO,
    "stem": "tabdpt_classifier",
    "notebook_name": "tabdpt_classifier_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and digest-verifies the pinned TabDPT checkpoint, loads scikit-learn's bundled breast-cancer table (no download), validates it into an input manifest and splits it with class coverage preserved, fits a majority-class baseline, **adapts TabDPT by in-context conditioning on the training split** (the adaptation stage: TabDPT registers the support rows; it has no gradient fine-tuning route — `fine_tune` is a rejected hyper-parameter), reports the capacity used, evaluates on the held-out split and writes the evaluation report, scores new rows and writes machine-readable outputs, exports the serving artifact and reloads it from disk through `load_verified_artifact` to prove the fresh boundary. No repository clone, DIMER worker or service, credential, upload dialog or configuration edit is required (NOTEBOOK_SPEC 2.0 §5)."
    ),
    "byod": (
        "After the sample workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to upload one labelled CSV (declare `CATEGORICAL_COLUMNS` if any); it enters the same validation, split, baseline, in-context conditioning, evaluation, new-data inference, export and fresh-reload cells as the sample (DAT14). Expected schema, ceilings and privacy guidance are stated in the Prerequisites and in Section 4; the upload stays inside this runtime. BYOD is optional and never part of the default path."
    ),
    "pipeline_class": "TabDPTClassificationPipeline",
    "weights_key": "tabdpt-1.2",
    # generator /2: the whole package is carried — pipeline.py (identity constants, the only `__file__` use:
    # DEFAULT_WEIGHTS_DIR, rewritten by the default rule), artifact.py (serving-artifact contract) and
    # dimer_runtime.py (the DIMER worker entrypoint; nothing in the tutorials calls it, and the generator
    # rewrites its top-level `if __name__ == "__main__":` guard to `if False:` so it cannot run in the kernel).
    "modules": ["pipeline.py", "artifact.py", "dimer_runtime.py"],
    "entry_module": "pipeline.py",
    # `from_pretrained` stages + verifies the snapshot and pins the checkpoint path; it loads no model.
    # FlashAttention is disabled for Tesla T4 portability; the seed matches the tutorial's SEED.
    "model_load": "TabDPTClassificationPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, compile_model=False, use_flash=False, seed=42)",
    "runtime_imports": ["torch", "numpy", "pandas", "sklearn"],
    "title": "TabDPT Classifier — DIMER E2E tabular classification tutorial (standalone)",
    "badges": BADGES,
    "capability": "supervised tabular classification by in-context conditioning on labelled support rows with the pinned `Layer6/TabDPT` v1.2 checkpoint",
    "intro": (
        "TabDPT is an in-context tabular foundation model: `fit()` fits the repository's feature encoder, maps class "
        "labels to stable sorted ids and registers the labelled support rows as context; it does **not** gradient-train "
        "or fine-tune the pretrained weights. The upstream project supplies TabDPT and its checkpoint; the carried "
        "package adds immutable provenance, snapshot verification, schema-safe preprocessing, deterministic controls, "
        "the DIMER serving-artifact contract, classification evaluation, and the `validate_inputs`, "
        "`majority_class_baseline` and `evaluation_report` helpers. The default sample is scikit-learn's public "
        "breast-cancer table; its metrics are tutorial sanity evidence, not a benchmark or production claim, and "
        "upstream pretraining overlap with the public sample cannot be ruled out."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried package guarantees, resolve and digest-verify the immutable "
        "upstream checkpoint, load a public sample or a gated BYOD CSV and validate it into an input manifest, split it "
        "with class coverage preserved, compare a majority-class baseline, condition the model and inspect its capacity "
        "report, evaluate accuracy / log loss / ROC-AUC into an evaluation report, score new rows with class "
        "probabilities under an explicit `argmax` rule, export machine-readable outputs plus provenance, export the "
        "DIMER serving artifact, and reload it across a fresh filesystem boundary without refitting preprocessing."
    ),
    "exclusions": (
        "regression, gradient fine-tuning, calibrated probabilities, or any deployment threshold. `predict()` applies "
        "an implicit `argmax` over class probabilities that are raw ensemble outputs, **not calibrated probabilities**; "
        "the pipeline ships no acceptance threshold and any threshold is the caller's to set on labelled data."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.11+; the pins were executed locally on Python 3.12). GPU recommended, CPU supported but slower; FlashAttention is disabled (`use_flash=False`) for Tesla T4 portability. The pinned `torch==2.7.1` install is the largest download of the run.",
        "- **Knowledge:** basic pandas; what a stratified holdout, accuracy, log loss and ROC-AUC are.",
        "- **Data:** the default sample is scikit-learn's bundled breast-cancer table (569 rows, 30 numeric features, two classes), loaded from the installed package, so nothing is downloaded and no private data is needed. Optional BYOD upload is gated off by default so the sample path runs top-to-bottom without interaction. Expected BYOD input: one CSV with unique feature names and a column named `target` with at least two classes and no missing values. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Load the public sample or optional BYOD\n\n"
                "Default data is scikit-learn's public breast-cancer binary classification sample, loaded from the "
                "installed package (no download). For BYOD, the expected contract is one CSV with unique feature names "
                "and a categorical column named `target` with at least two classes and no missing values; set "
                "`USE_BYOD=True`. If identifier-like categorical fields contain numeric-looking strings such as `01`, "
                "list those fields in `CATEGORICAL_COLUMNS` **before upload** so CSV parsing preserves their string "
                "identity; every named categorical column must exist in the header. All other CSV columns use normal "
                "pandas inference. The cell prints the sample kind and the SHA-256 of the table's CSV serialisation so "
                "the exported provenance can be tied to the exact data."
            ),
            "code": (
                "import csv\n"
                "import hashlib\n"
                "import io\n\n"
                "from sklearn.datasets import load_breast_cancer\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "CATEGORICAL_COLUMNS = []  # @param {{type:\"raw\"}}\n"
                "TARGET, SEED, CONTEXT_SIZE, N_ENSEMBLES, BATCH_SIZE = 'target', 42, 512, 2, 512\n\n"
                "def read_csv_checked(raw, categorical_columns):\n"
                "    text = raw.decode('utf-8-sig')\n"
                "    header = next(csv.reader(io.StringIO(text)), [])\n"
                "    duplicates = sorted({{x for x in header if header.count(x) > 1}})\n"
                "    if duplicates:\n"
                "        raise ValueError(f'Duplicate CSV columns: {{duplicates}}')\n"
                "    missing_declared = [c for c in categorical_columns if c not in header]\n"
                "    if missing_declared:\n"
                "        raise ValueError(f'CATEGORICAL_COLUMNS not present in CSV header: {{missing_declared}}')\n"
                "    return pd.read_csv(io.BytesIO(raw), dtype={{c: 'string' for c in [*categorical_columns, TARGET] if c in header}})\n\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    if len(uploaded) != 1:\n"
                "        raise ValueError('Upload exactly one CSV.')\n"
                "    data_name, raw = next(iter(uploaded.items()))\n"
                "    if not data_name.lower().endswith('.csv'):\n"
                "        raise ValueError('BYOD must be CSV.')\n"
                "    frame = read_csv_checked(raw, CATEGORICAL_COLUMNS)\n"
                "    sample_kind = 'BYOD'\n"
                "else:\n"
                "    sample = load_breast_cancer(as_frame=True)\n"
                "    frame = sample.frame.copy()\n"
                "    frame[TARGET] = frame[TARGET].map({{0: 'malignant', 1: 'benign'}})\n"
                "    data_name = 'sklearn-breast-cancer.csv'\n"
                "    sample_kind = 'sample'\n"
                "frame[TARGET] = frame[TARGET].map(str)\n"
                "sample_sha256 = hashlib.sha256(frame.to_csv(index=False).encode('utf-8')).hexdigest()\n"
                "print({{'sample_kind': sample_kind, 'name': data_name, 'rows': len(frame), 'columns': frame.shape[1], 'csv_sha256': sample_sha256, 'declared_categorical_columns': CATEGORICAL_COLUMNS if USE_BYOD else 'sample schema'}})"
            ),
        },
        {
            "md": (
                "## 5. Validate the table → input manifest, then split with class coverage preserved\n\n"
                "`validate_inputs` is the pipeline's public validation stage: it applies exactly the checks `fit` "
                "applies — unique column names, the target column present with no missing values, at least "
                "`MIN_CLASSES` classes, at least one feature column — and returns an **input manifest** naming the "
                "schema, the observed feature/class structure (numeric vs categorical columns, missing-value counts, "
                "class counts) and the verdict. It is written to `outputs/{stem}_input_manifest.json`. To show what "
                "rejection looks like, the cell also validates a probe whose target has a missing value and records the "
                "pipeline's own error message as a finding. The named ceilings and the request parameters are printed "
                "before any model runs; the model's own feature ceiling (`max_features`) is only known after "
                "conditioning and is reported in Section 6.\n\n"
                "Numeric columns are kept as they are (numeric missing values stay NaN through the repository encoder "
                "and are mean-imputed by TabDPT's support-fitted imputer; infinite values fail here, before "
                "conditioning); categorical values are mapped to fitted codes with dedicated missing and unknown codes. "
                "The stratified 80/20 split assumes rows are sufficiently independent; use preserved temporal, group or "
                "spatial boundaries for leakage-sensitive data. Every class present in the holdout must also be present "
                "in the support set, which the split preserves and the cell checks. No model selection uses this holdout."
            ),
            "code": (
                "import json\n"
                "import os\n\n"
                "from sklearn.model_selection import train_test_split\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "print({{'ceilings': {{'MIN_CLASSES': MIN_CLASSES, 'DECISION_RULE': DECISION_RULE}}, 'request': {{'context_size': CONTEXT_SIZE, 'n_ensembles': N_ENSEMBLES, 'batch_size': BATCH_SIZE, 'seed': SEED}}}})\n"
                "input_manifest = validate_inputs(frame, target_column=TARGET, names=[data_name])\n"
                "# Demonstrate rejection on a probe that breaks the target contract; the finding is recorded, not swallowed.\n"
                "probe = frame.head(4).copy()\n"
                "probe[TARGET] = probe[TARGET].astype(object)\n"
                "probe.loc[probe.index[0], TARGET] = None\n"
                "try:\n"
                "    validate_inputs(probe, target_column=TARGET)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'missing-target-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest['inputs'][0], indent=2))\n"
                "print('findings:', input_manifest['findings'])\n"
                "features = frame.drop(columns=[TARGET])\n"
                "for column in features.select_dtypes(include=np.number).columns:\n"
                "    values = features[column].dropna().to_numpy(dtype=float)\n"
                "    if values.size and not np.isfinite(values).all():\n"
                "        raise ValueError(f'Numeric feature {{column!r}} contains infinite values.')\n"
                "class_counts = frame[TARGET].value_counts().sort_index()\n"
                "if (class_counts < 2).any():\n"
                "    raise ValueError(f'Every class needs at least 2 rows for a stratified split: {{class_counts.to_dict()}}')\n"
                "train, test = train_test_split(frame, test_size=0.2, random_state=SEED, stratify=frame[TARGET])\n"
                "unseen = sorted(set(test[TARGET]) - set(train[TARGET]))\n"
                "if unseen:\n"
                "    raise ValueError(f'Holdout classes absent from support data: {{unseen}}')\n"
                "print({{'train': train.shape, 'holdout': test.shape, 'train_class_counts': train[TARGET].value_counts().sort_index().to_dict()}})\n"
                "print('Requested context_size:', CONTEXT_SIZE, 'effective support rows <=', min(len(train), CONTEXT_SIZE))\n"
                "if len(train) > CONTEXT_SIZE:\n"
                "    print('Support exceeds context_size; seeded class-balanced upstream context subsampling applies.')"
            ),
        },
        {
            "md": (
                "## 6. Baseline, in-context conditioning, capacity report and evaluation\n\n"
                "`majority_class_baseline` is the trivial baseline: it always predicts the most frequent support class "
                "with that class's support frequency as its probability. `fit()` performs preprocessing fitting plus "
                "in-context conditioning, **not gradient training**; the model is loaded here, from the digest-verified "
                "checkpoint pinned in Section 3. After conditioning the cell prints the fitted class labels, the encoded "
                "feature count, the model feature ceiling, whether the upstream feature-reduction path is active, and the "
                "missing-value preprocessing in effect. `evaluate` then scores the holdout: accuracy is discrete "
                "correctness under the implicit `argmax` rule; log loss scores the class probabilities and penalises "
                "confident mistakes; ROC-AUC (binary targets only) scores ranking quality independent of any threshold. "
                "Reading accuracy alone hides both imbalance and probability-quality failures. Seeds control the "
                "supported stochastic paths (split, class-balanced context subsampling, ensemble class permutation); "
                "bitwise determinism across hardware kernels is not promised."
            ),
            "code": (
                "baseline = majority_class_baseline(train[TARGET], test[TARGET])\n"
                "pipe.fit(train, target_column=TARGET, seed=SEED)\n"
                "encoded_feature_count = len(pipe.feature_encoder.feature_columns)\n"
                "model_feature_ceiling = int(pipe.estimator.max_features)\n"
                "feature_reduction = str(pipe.estimator.feature_reduction)\n"
                "numeric_missing_columns = [c for c in pipe.feature_encoder.feature_columns if c in pipe.feature_encoder.numeric_columns and train[c].isna().any()]\n"
                "capacity = {{'classLabels': pipe.class_labels_, 'encodedFeatureCount': encoded_feature_count, 'modelFeatureCeiling': model_feature_ceiling, 'featureReduction': feature_reduction, 'featureReductionActive': encoded_feature_count > model_feature_ceiling, 'numericMissingColumns': numeric_missing_columns, 'upstreamImputer': type(pipe.estimator.imputer).__name__, 'source': pipe.source}}\n"
                "print(capacity)\n"
                "if encoded_feature_count > model_feature_ceiling:\n"
                "    print(f'Feature count exceeds {{model_feature_ceiling}}; upstream {{feature_reduction}} reduction is active.')\n"
                "else:\n"
                "    print('Feature reduction is not active for this dataset.')\n"
                "if numeric_missing_columns:\n"
                "    print('Numeric missing values are mean-imputed using support/training-fitted statistics.')\n"
                "kw = {{'n_ensembles': N_ENSEMBLES, 'context_size': CONTEXT_SIZE, 'batch_size': BATCH_SIZE, 'seed': SEED}}\n"
                "metrics = pipe.evaluate(test, **kw)\n"
                "print('tutorial TabDPT', metrics)\n"
                "print('majority-class baseline', baseline)"
            ),
        },
        {
            "md": (
                "## 7. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report. Here it "
                "carries the metrics `evaluate` returned (`accuracy`, `log_loss`, `roc_auc` — the repository's own "
                "metric ids) and the majority-class baseline with the verdict `sample-sanity`: a single seeded stratified "
                "holdout with no dispersion estimate, tutorial evidence rather than a benchmark. When no labelled holdout "
                "exists the verdict is `not-measurable` and the report states what would make the task measurable. The "
                "report is written to `outputs/{stem}_evaluation_report.json`. Expect the model to beat the baseline "
                "clearly on the public sample; a BYOD table can behave very differently."
            ),
            "code": (
                "report = evaluation_report(metrics, baseline=baseline, n_holdout=len(test), class_labels=pipe.class_labels_, sample_kind=sample_kind, estimation='single seeded stratified 80/20 holdout; no dispersion estimate')\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(report, indent=2))\n"
                "if report['verdict'] == 'not-measurable':\n"
                "    print('No labelled holdout was scored; the metrics above are absent by construction.')"
            ),
        },
        {
            "md": (
                "## 8. New-data inference and machine-readable outputs\n\n"
                "The target is removed before scoring. Training-fitted repository encoding and upstream "
                "imputation/scaling are reused; no preprocessing is fitted on the new rows. `predict_proba()` returns one "
                "column per fitted class **in the fitted (sorted) class order**, and `prediction` is the `argmax` label — "
                "an uncalibrated decision rule with no shipped threshold. `row_id` stays outside the model feature "
                "schema and maps exported predictions back to inputs. The prediction CSV keeps the `proba_<class>` "
                "columns in class order; the result JSON carries the metrics, the baseline, the evaluation report, the "
                "input manifest, the sample identity and digest, the split, the capacity report, the inference "
                "controls, the notebook's source (repository, revision, embedded module digests, generator), the model "
                "identifier, the immutable model revision and licence, and the runtime identity. No credentials are recorded."
            ),
            "code": (
                "new_rows = test.drop(columns=[TARGET]).head(8).copy()\n"
                "proba = pipe.predict_proba(new_rows, **kw)\n"
                "pred = pipe.predict(new_rows, **kw)\n"
                "out = pd.DataFrame({{'row_id': new_rows.index.to_numpy(), 'prediction': pred.to_numpy()}})\n"
                "for label in pipe.class_labels_:\n"
                "    out[f'proba_{{label}}'] = proba[label].to_numpy()\n"
                "out.to_csv('outputs/{stem}_predictions.csv', index=False)\n"
                "payload = {{\n"
                "    'predictions': out.to_dict(orient='records'),\n"
                "    'metrics': metrics,\n"
                "    'majority_class_baseline': baseline,\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': input_manifest,\n"
                "    'sample': {{'kind': sample_kind, 'name': data_name, 'rows': len(frame), 'csv_sha256': sample_sha256}},\n"
                "    'split': {{'method': 'single seeded stratified 80/20 random holdout', 'seed': SEED, 'trainRows': len(train), 'holdoutRows': len(test), 'trainClassCounts': train[TARGET].value_counts().sort_index().to_dict()}},\n"
                "    'preprocessing': {{'encodedFeatureCount': encoded_feature_count, 'modelFeatureCeiling': model_feature_ceiling, 'featureReduction': feature_reduction, 'featureReductionActive': encoded_feature_count > model_feature_ceiling, 'numericMissingPolicy': 'support/training-fitted mean imputation', 'categoricalMissingPolicy': 'dedicated fitted missing code', 'unknownCategoryPolicy': 'dedicated fitted unknown code', 'declaredCategoricalColumns': CATEGORICAL_COLUMNS if USE_BYOD else []}},\n"
                "    'inference': {{**kw, 'decisionRule': DECISION_RULE + ' over uncalibrated class probabilities', 'threshold': None, 'classLabels': pipe.class_labels_}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'upstream_code_commit': TABDPT_UPSTREAM_CODE_COMMIT,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'tabdpt': importlib.metadata.version('tabdpt'), 'numpy': numpy.__version__, 'pandas': pandas.__version__, 'sklearn': sklearn.__version__, 'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU', 'use_flash': False}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "print(out.head())\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
        {
            "md": (
                "## 9. Export the serving artifact and verify a fresh reload\n\n"
                "For this in-context model the deployable serving state is not the checkpoint alone: it includes the "
                "labelled support data, the fitted class-label mapping, the fitted repository/upstream preprocessing "
                "state and the exact pinned base-model contract. `export_artifact_bundle` writes `artifact.json` (with the "
                "support table's size and SHA-256 and the base-model identity) and `training_context.parquet`; the "
                "latter inherits the source data's confidentiality, licensing, retention and disclosure obligations.\n\n"
                "The artifact is copied to a fresh directory, validated before reconstruction, and loaded through the "
                "verified **no-preprocessing-refit** path with the base checkpoint taken from the digest-verified snapshot "
                "of Section 3 (no network fallback). Class labels must match exactly and class probabilities must match "
                "within `rtol=1e-5`, `atol=1e-6`; bitwise identity is not required. The companion notebook "
                "`tabdpt_classifier_artifact_inference_colab.ipynb` consumes this artifact from a separate execution."
            ),
            "code": (
                "import shutil\n\n"
                "ART = Path('outputs') / 'artifact'\n"
                "manifest_path = export_artifact_bundle(pipe, train, ART)\n"
                "RELOAD = Path('outputs') / 'artifact-reload'\n"
                "if RELOAD.exists():\n"
                "    shutil.rmtree(RELOAD)\n"
                "shutil.copytree(ART, RELOAD)\n"
                "reloaded = load_verified_artifact(RELOAD / 'artifact.json', model_weight_path=pipe.model_weight_path, compile_model=False, use_flash=False, seed=SEED)\n"
                "assert reloaded.preprocessing_restored_ is True, 'Verified reload must restore fitted preprocessing state without refit.'\n"
                "assert reloaded.class_labels_ == pipe.class_labels_, 'Reloaded class labels must match the exported class order.'\n"
                "proba2 = reloaded.predict_proba(new_rows, **kw)\n"
                "pred2 = reloaded.predict(new_rows, **kw)\n"
                "assert list(pred2) == list(pred), 'Reloaded argmax labels must match exactly.'\n"
                "np.testing.assert_allclose(proba.to_numpy(), proba2.to_numpy(), rtol=1e-5, atol=1e-6)\n"
                "print({{'artifact': str(manifest_path), 'artifact_files': sorted(os.listdir(ART)), 'reload_dir': str(RELOAD)}})\n"
                "print('PASS: fitted preprocessing restored without refit; labels identical and probabilities equivalent (rtol=1e-5, atol=1e-6).')"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "`prediction` is the `argmax` of uncalibrated class probabilities over the fitted class set; the pipeline ships "
        "no threshold, and any deployment cut-off must be chosen on the caller's own labelled, domain-representative data. "
        "The evaluation report's `sample-sanity` verdict names what it is: one seeded stratified holdout of the public "
        "sample with no dispersion estimate — tutorial evidence that must not be generalised to a domain, a class "
        "distribution or a data-collection process. Rows that are not independent (temporal, grouped, spatial), classes "
        "absent from the support rows, wide tables that trigger upstream feature reduction, and support sets larger than "
        "`context_size` (class-balanced subsampling) all change results in ways the metrics above do not measure.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this notebook, can "
        "acquire and digest-verify the pinned checkpoint, validate the demonstrated table, condition on classification "
        "support data with full class coverage, surface missing-value and capacity behaviour, compute sample metrics "
        "against a majority-class baseline, score new rows with class probabilities under an explicit `argmax` rule, emit "
        "the shown machine-readable outputs and the DIMER serving artifact, and reconstruct identical labels and "
        "equivalent probabilities from serialised fitted preprocessing without refitting it — without the repository being "
        "reachable. It does **not** establish benchmark superiority, domain generalisation, fairness, robustness, "
        "probability calibration, production safety, or deployment fitness.\n\n"
        "**Next experiments:** enable `USE_BYOD` with a labelled CSV from your own domain and compare `evaluate` against "
        "`majority_class_baseline` on a split that respects your data's structure; raise `N_ENSEMBLES` and watch log "
        "loss; supply a table with more features than the model ceiling to see the upstream feature-reduction path "
        "activate; feed the exported `outputs/artifact/` to the companion artifact-inference notebook in a separate session.\n\n"
        "## References\n\n"
        f"- Repository README: https://github.com/kurtvalcorza/{REPO}/blob/main/README.md\n"
        f"- Repository model card: https://github.com/kurtvalcorza/{REPO}/blob/main/MODEL_CARD.md\n"
        f"- Weight provenance: https://github.com/kurtvalcorza/{REPO}/blob/main/weights/README.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream inference code: https://github.com/layer6ai-labs/TabDPT-inference\n"
        "- TabDPT paper: https://arxiv.org/abs/2608.01400"
    ),
}
