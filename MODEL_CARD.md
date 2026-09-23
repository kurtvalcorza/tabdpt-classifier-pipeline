---
license: apache-2.0
model_card_spec: "1.1"
pipeline_tag: tabular-classification
task: "Others - Tabular Classification and Regression"
tags:
  - tabular-classification
  - tabular-foundation-model
  - in-context-learning
  - tabdpt
base_model: Layer6/TabDPT
date_published: "2025-06-18"
date_published_source: "Hugging Face Hub repository creation date of the exact hosted checkpoint (`createdAt`, https://huggingface.co/api/models/Layer6/TabDPT)"
---

# TabDPT v1.2 — Tabular Foundation Model (Classification)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Layer6%2FTabDPT-ffcc4d?style=flat)](https://huggingface.co/Layer6/TabDPT)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-layer6ai--labs%2FTabDPT--inference-181717?style=flat&logo=github&logoColor=white)](https://github.com/layer6ai-labs/TabDPT-inference)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2608.01400-b31b1b.svg)](https://arxiv.org/abs/2608.01400)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This repository ships standalone Google Colab tutorials that exercise its public pipeline API:

- **End-to-End Pipeline Tutorial**: \
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/tabdpt-classifier-pipeline/blob/main/tutorials/tabdpt_classifier_colab.ipynb) [`tabdpt_classifier_colab.ipynb`](https://github.com/kurtvalcorza/tabdpt-classifier-pipeline/blob/main/tutorials/tabdpt_classifier_colab.ipynb) \
  *Verify the pinned TabDPT snapshot, validate labelled tables and class coverage, fit preprocessing and in-context support, evaluate class predictions against the majority-class baseline, and export and reload the serving artifact.*

- **Serving Artifact Inference Tutorial**: \
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/tabdpt-classifier-pipeline/blob/main/tutorials/tabdpt_classifier_artifact_inference_colab.ipynb) [`tabdpt_classifier_artifact_inference_colab.ipynb`](https://github.com/kurtvalcorza/tabdpt-classifier-pipeline/blob/main/tutorials/tabdpt_classifier_artifact_inference_colab.ipynb) \
  *Validate an externally supplied serving artifact and its class-label mapping, reconstruct the predictor without refitting, validate new rows, and export class predictions and probabilities with provenance.*

> [!NOTE]
> The notebooks disable FlashAttention with `use_flash=False` for Colab Tesla T4 compatibility; `fit()` conditions on support data without gradient training. These tutorials remain release candidates; see [release verification](docs/release-verification.md) for execution records and promotion requirements.

---

#### Description

TabDPT v1.2, released as **TabDPT-Turbo**, is an open-weight tabular foundation model designed for in-context supervised classification on structured datasets. Rather than iteratively training neural network weights or tree ensembles on each new dataset via gradient descent or heuristic splits, TabDPT processes a labelled support table (the in-context prompt) alongside unlabelled test observations through a specialized tabular Transformer architecture. Task adaptation occurs entirely at inference time through in-context forward evaluation without gradient updates or per-dataset training loops. For single-context queries without ensembling, inference requires only a forward evaluation; when ensembling over multiple support subsets (`n_ensembles > 1`) or batching query chunks, predictions are aggregated across multiple forward passes. Pretrained on a diverse corpus of real-world tabular datasets and optimized with FlashAttention and key-value caching in v1.2 (Turbo), it delivers rapid, zero-shot tabular classification across binary and multiclass problems without per-dataset hyperparameter tuning. This repository packages the upstream classification estimator for reproducible, DIMER-ready deployment.

#### Intended Use and Limitations

###### Primary Intended Uses

Supervised tabular classification tasks, including binary and multiclass prediction on structured tables where features consist of numerical and categorical columns. Suitable target applications include tabular risk scoring, customer churn and retention prediction, diagnostic triage in exploratory research settings, equipment fault detection, marketing propensity, and general row-per-observation classification. It is designed to act as an out-of-the-box strong baseline and inference engine requiring zero hyperparameter optimization.

###### Primary Intended Users

Machine learning engineers, data scientists, researchers, and software engineers developing predictive pipelines for structured datasets in enterprise, scientific, or academic environments. The envisioned deployment setting is research or in-house enterprise use on infrastructure the user controls, not a public-facing service. Users are expected to understand data validation, leakage prevention, class imbalance, and standard classification evaluation methodology, and to recognise that `predict()` is an implicit `argmax` over `predict_proba()` whose probabilities have not been calibrated for their domain; a user who would ship the default decision rule into a cost-asymmetric setting without re-thresholding on held-out data is outside the assumed competency.

###### Out-of-scope use cases

- Continuous numerical regression (use TabDPT Regressor v1.2 instead).
- Unsupervised clustering, dimensionality reduction, density estimation, or tabular data synthesis.
- Raw unstructured modalities (unprocessed images, audio, video, free-form long text) without prior tabular feature extraction.
- Direct time-series forecasting or survival analysis requiring temporal dependency modeling without supervised lag feature engineering.
- Autonomous high-impact or safety-critical decisions without human oversight (e.g., autonomous clinical triage, judicial sentencing, loan/credit denials).
- Tables where linear feature compression is unsuitable: TabDPT-Turbo's native row encoder accommodates up to 128 features; for tables exceeding 128 features, the upstream estimator automatically applies PCA feature reduction down to its native 128-dimensional width. On very wide datasets where linear PCA discards critical non-linear signals, prior domain-specific feature selection is recommended. (Note that feature column width is distinct from transformer row-context capacity, which governs support row count `context_size`).

---

#### Factors

###### Groups

TabDPT v1.2 was pretrained on a broad corpus of public real-world tabular datasets and is not inherently tailored to, or debiased for, any specific demographic, phenotypic, or protected group (such as age, gender, race, ethnicity, or socioeconomic status). In human-centric applications, downstream users must rigorously audit group-level fairness metrics (e.g., equalized odds, demographic parity) on their specific application data.

###### Instrumentation

The model operates on normalized tabular data matrices (floating-point numbers and encoded categorical integers) rather than direct physical sensor streams. The "instruments" are upstream data collection systems, SQL databases, survey instruments, laboratory diagnostic assays, and ETL pipelines. Inaccuracies, sensor drifts, or calibration discrepancies in the underlying instruments directly propagate into model features.

###### Environment

TabDPT operates across generic computational environments (CPU, CUDA GPUs with `sm_80+` for FlashAttention, or `sm_75` like Tesla T4 with FlashAttention disabled). In terms of application environments, the model assumes that feature distributions between the in-context support set and the query test set are drawn from the same data-generating distribution; severe covariate shifts, concept drifts, or institutional data discrepancies will degrade predictive fidelity.

---

#### Metrics

###### Performance Measures

Model evaluation in the pipeline and upstream benchmarks reports:
- **Accuracy**: Proportion of correctly classified instances.
- **Log Loss (Cross-Entropy)**: Evaluates the quality and calibration of predicted class probability distributions.
- **Binary ROC-AUC**: Measures ranking discrimination across classification thresholds on binary tasks.

These metrics assess both discrete decision accuracy and probabilistic confidence calibration without arbitrary threshold selection. The standalone tutorials write them through the package's public `evaluation_report` helper (`outputs/<stem>_evaluation_report.json`, verdict `sample-sanity` with the `majority_class_baseline` comparison, or `not-measurable` when no labelled rows exist).

###### Decision thresholds

Default discrete predictions use argmax over predicted class probabilities (equivalent to a 0.5 probability threshold for binary classification). For operational deployments, users should calibrate application-specific decision thresholds based on the asymmetric real-world costs of False Positives versus False Negatives.

###### Approaches to uncertainty and variability

- **Predicted probabilities:** The pipeline exposes predicted class probabilities via `predict_proba()`. Note that raw softmax outputs reflect model confidence scores rather than formally calibrated Bayesian posterior probabilities.
- **Ensemble averaging:** When `n_ensembles > 1`, predicted probabilities are averaged across distinct support context subsamples.
- **Empirical validation:** Users requiring calibrated confidence scores should evaluate empirical reliability diagrams or apply post-hoc calibration methods (such as Platt scaling or isotonic regression) on independent holdout sets.

---

#### Ethical considerations and biases

###### Data

Pretrained by upstream authors on a broad collection of public tabular datasets sourced from open repositories. The pipeline distribution provides only model weights (`tabdpt1_2.safetensors`) and wrapper code, and does not distribute pretraining datasets. Downstream users should note that public pretraining tables may still reflect historical demographic skews, societal biases, or sensitive domain attributes present in their original sources. Operators deploying the model are responsible for auditing their own in-context support data for sensitive attributes, proprietary information, or PII before conditioning the model.

###### Human Life

The model is **not** certified, validated, or intended for autonomous decision-making in situations central to human life, health, or flourishing (such as clinical medical treatment, intensive care triage, criminal sentencing, or life-critical safety systems). Any application in sensitive domains requires human oversight and rigorous independent validation.

###### Mitigations

- Self-contained open weights (`tabdpt1_2.safetensors`) with cryptographic SHA-256 verification.
- Pinned upstream Hugging Face revision (`4462ffbd1d8dea25d4862d30beed4b70cd596ae5`).
- Balanced context subsampling during support set construction to protect against severe class imbalance.
- Strict input schema validation preventing silent column misalignment; the public `validate_inputs` helper applies the same checks and records the verdict and any rejection finding in an input manifest before any model execution.
- Deterministic random seed controls for reproducible sampling and ensembling.

###### Risks and harms

- **Overconfidence on out-of-distribution tables**: The model may produce confident misclassifications when inputs deviate drastically from pretraining patterns.
- **Amplification of dataset bias**: Conditioning on a biased support table will reproduce or amplify those biases in predictions.
- **Automation bias**: Users uncritically accepting model predictions without domain review.
- **Leakage**: Accidental inclusion of target-correlated artifacts in the feature set leading to spurious high performance.

###### Use cases

Distinct from the capability and decision boundaries listed under *Out-of-scope use cases*, the developers consider the following uses prohibited even where the model would produce a numerically plausible label:
- Mass surveillance, unauthorized biometric or demographic profiling, or social scoring systems.
- Unlawful discrimination in employment, housing, credit, insurance, education, or healthcare access, including classification on a target that proxies a protected attribute.
- Predictive scoring designed for predatory financial targeting or deceptive manipulation, or presenting an uncalibrated class probability as a certified risk estimate.
- Any use that violates the Apache-2.0 terms of the upstream Layer6/TabDPT weights or the terms of the DIMER deployment.

---

## Model Details

- **Task:** supervised tabular classification
- **Model family:** tabular transformer / in-context learner
- **Upstream release:** TabDPT v1.2 / TabDPT-Turbo
- **Upstream package:** `tabdpt==1.2.0`
- **Weight artifact:** `tabdpt1_2.safetensors`
- **Artifact size:** 254,098,072 bytes
- **Weight license:** Apache-2.0
- **Code license:** Apache-2.0

## Immutable provenance

- Upstream inference source commit: `9cfb05e0a6bc380ae6c99c08adc8d50dacd4f246` (`v1.2.0`)
- Hugging Face repository: `Layer6/TabDPT`
- Hugging Face revision: `4462ffbd1d8dea25d4862d30beed4b70cd596ae5`
- Filename: `tabdpt1_2.safetensors`
- SHA-256: `06680220fd66c4524051706b98c1c659a674d19d3a766cd0bb276505e99faccd`

The wrapper verifies SHA-256 before constructing the upstream estimator.

## Primary references

**Hosseinzadeh, R., Labach, A., Xue, Z., Han, S., Thomas, V., & Caterini, A. L. (2026). _TabDPT-Turbo: Efficient In-Context Learning for Tabular Prediction_. arXiv:2608.01400.**

The Turbo paper explicitly describes the released model as TabDPT v1.2. The earlier base work remains relevant background:

**Ma, J., et al. (2025). _TabDPT: Scaling Tabular Foundation Models_. NeurIPS 2025; arXiv:2410.18164.**

## How inference works

`fit(X_train, y_train)` prepares the labelled table as in-context support and fits preprocessing statistics; it does not update the pretrained TabDPT weights. Prediction conditions the frozen model on this support table. Classification uses balanced context subsampling when a requested context cap is smaller than the available support set.

## Input contract

- numeric and categorical/string columns are supported;
- learned categorical mappings are fitted only on training data;
- unseen and missing categorical values receive dedicated codes;
- `drop_columns` are excluded before the fitted schema is established and may be present in later raw tables;
- after configured drop columns are removed, inference must contain exactly the fitted feature names—missing or unconfigured extra columns are rejected;
- target labels are mapped to stable integer IDs and decoded back to string labels.

## Output contract

- `predict(...)`: one class label per row;
- `predict_proba(...)`: one probability column per fitted class;
- `evaluate(...)`: accuracy, log loss, and binary ROC-AUC when both classes are present.

## DIMER runtime status

The repository includes an executable local/on-prem DIMER adapter. `datasetPreprocessing` is consumed from `DIMER_PREPROCESSING_ARGS_JSON`; the platform's existing `modelFinetuning` transport is consumed from `DIMER_HYPERPARAMETERS_JSON`, but `fine_tune=true` is explicitly rejected because v1.2 uses ICL rather than gradient fine-tuning.

The adapter supports `train.csv` and optional `val.csv`, applies deterministic splitting/support capping, executes the model, and writes result/provenance/context artifacts. Routine CI tests this contract without downloading the model weight. A real checkpoint/GPU smoke test and on-platform deployment/reload test remain production-acceptance work.

## Training-data / benchmark caveat

TabDPT was pretrained on real-world tables. Common public benchmarks or tutorial datasets may overlap with upstream pretraining, so their metrics should be treated as smoke-test evidence rather than independent benchmark claims.

## License

This repository, the upstream TabDPT inference package, and the referenced public model weights are Apache-2.0 licensed. See `LICENSE` and preserve applicable upstream notices when redistributing upstream materials.
