# Model Card — TabDPT Classifier v1.2

## Model overview

**TabDPT** is an open-weight tabular foundation model based on in-context learning. The upstream project trains on real-world tabular data and exposes sklearn-style estimators for classification and regression. This repository packages the upstream **v1.2 classification estimator** for reproducible DIMER-oriented inference.

- **Task:** supervised tabular classification
- **Model family:** tabular transformer / in-context learner
- **Upstream release:** TabDPT v1.2 (TabDPT-Turbo)
- **Upstream package:** `tabdpt==1.2.0`
- **Weight artifact:** `tabdpt1_2.safetensors`
- **Artifact size:** 254,098,072 bytes
- **Weight license:** Apache-2.0
- **Code license:** Apache-2.0
- **Development source:** Layer 6 AI / The Toronto-Dominion Bank and contributors
- **Paper:** *TabDPT: Scaling Tabular Foundation Models on Real Data*, NeurIPS 2025, arXiv:2410.18164

## Immutable provenance

The integration pins the public model to:

- Hugging Face repository: `Layer6/TabDPT`
- revision: `4462ffbd1d8dea25d4862d30beed4b70cd596ae5`
- filename: `tabdpt1_2.safetensors`
- SHA-256: `06680220fd66c4524051706b98c1c659a674d19d3a766cd0bb276505e99faccd`

The wrapper verifies SHA-256 before constructing the upstream estimator. An operator-mounted `model_weight_path` is accepted only when it matches the same digest.

## How inference works

`fit(X_train, y_train)` does not perform gradient-based training. It prepares the labelled table as the in-context support set, fits preprocessing statistics on the training partition, and stores the context used by TabDPT at prediction time. `predict`/`predict_proba` then condition the pretrained transformer on that context.

TabDPT v1.2 supports long-context inference and uses no retrieval by default upstream. This integration uses `subsample-balanced` as the context-reduction strategy for classification so an operator-selected context cap does not trivially discard minority classes. Full context is still used when the requested context is at least the training-set size.

## Input contract

The repository wrapper accepts a pandas `DataFrame` containing a labelled target column for fit/evaluation and feature-only frames for prediction.

- numeric feature columns are converted with `pandas.to_numeric`; invalid values become missing values and are imputed by TabDPT using training-fitted statistics;
- categorical/string columns are encoded with mappings fitted **only on the training partition**;
- unseen categories receive a dedicated unknown code;
- missing categorical values receive a separate missing code;
- feature names and order are locked at fit time and checked at inference time;
- target labels are converted to stable integer IDs before being passed to the upstream estimator and converted back to strings on output.

## Output contract

- `predict(...)` returns one string class label per input row.
- `predict_proba(...)` returns a pandas `DataFrame` with one probability column per fitted class.
- `evaluate(...)` reports accuracy and log loss; binary ROC-AUC is included when both classes are present in the evaluation partition.

## Intended uses

Appropriate uses include supervised classification of conventional tabular datasets such as administrative records, business/operational tables, sensor-derived feature tables, scientific measurements, and public-sector structured datasets when the feature/target semantics are suitable.

The model is intended as a strong pretrained baseline or production candidate after task-specific validation. It is not a causal model and does not by itself establish policy, medical, financial, legal, or safety-critical validity.

## Out-of-scope uses

- raw images, audio, natural language, graphs, or unprocessed time series;
- unsupervised clustering or causal inference;
- use where a random/IID table split violates temporal, grouped, household, geographic, patient, or entity leakage constraints;
- high-stakes automated decisions without domain validation, error analysis, governance, and human oversight.

## Training data and contamination caveat

Unlike several synthetic-prior tabular foundation models, TabDPT was pretrained on **real-world tables**. This is a meaningful architectural/product distinction but it complicates benchmark interpretation: a public benchmark used as a DIMER tutorial or smoke test may have appeared in, or be closely related to, upstream pretraining data.

Accordingly:

1. tutorial metrics are sanity checks, not independent benchmark claims;
2. deployment decisions should use application-specific held-out data with provenance known to the deploying organization;
3. model comparison reports should flag possible pretraining contamination when using common public benchmarks.

## Performance and resource considerations

Upstream v1.2 is described as a major fitting/inference acceleration release and supports unrestricted context (`context_size=None`) when memory allows. Runtime cost still grows with context, evaluation rows, feature count, and ensemble count. DIMER therefore exposes context and ensemble controls and uses conservative defaults.

GPU execution is recommended for throughput. CPU inference is supported upstream but may not satisfy interactive service latency requirements for larger contexts.

## Fine-tuning status

**Not implemented in this pipeline.** The supported v1.2 inference package exposes pretrained ICL estimators. Layer 6 separately publishes full training code, but this repository does not present that training stack as a stable task-level fine-tuning API. A future DIMER fine-tuner should be treated as a separate implementation/review milestone.

## DIMER development status

**Initial integration / review candidate.** Repository-side acceptance covers weight identity, preprocessing semantics, estimator wrapping, static CI, and tutorial plumbing. Final production readiness requires an on-platform DIMER test using the deployed worker/runtime and representative data.

## License and attribution

The upstream TabDPT inference code and public model are Apache-2.0 licensed. Preserve upstream attribution and license terms when redistributing code or model artifacts.