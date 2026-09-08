# TabDPT Classifier Sample Datasets

This directory provides and documents sample datasets for the [TabDPT Classifier Colab Tutorials](../../tutorials/README.md).

TabDPT is an in-context learning tabular foundation model that consumes training rows as in-context support context (`training_context.csv`). The bundled sample dataset provides a reproducible, self-contained archive formatted for DIMER ingestion.

| Dataset | Modality / Task | Rows (Train / Val) | Features | Source & License | Archive SHA-256 |
|---|---|---|---|---|---|
| **Breast Cancer Wisconsin** | Binary classification (`0` / `1`) | 341 train / 228 val (569 total) | 30 numeric | scikit-learn / UCI (CC0 1.0 Universal) | `08d406c2c0fcb31c67db65b14dcf6ffb9f02b4b56717a1e58dd566465e0d3f3b` |

---

## 1. Breast Cancer Wisconsin (Diagnostic)

- **Purpose:** Quick smoke benchmark and DIMER contract validation. Pretrained TabDPT v1.2 scores ~0.97–0.98 accuracy out of the box, verifying GPU/CPU execution, in-context conditioning, and artifact export.
- **Archive:** `breast-cancer-wisconsin.zip` containing `train.csv` and `val.csv`.
- **Target:** `target` — binary diagnosis (`0` = malignant, `1` = benign).
- **Features:** 30 continuous numeric attributes derived from digitized cell nuclei images (mean, standard error, and "worst" values for radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension).
- **Split:** 60% train (341 rows) / 40% val (228 rows), stratified by `target`, random seed 42.
- **Specification Conformance:** Conforms strictly to [`TABULAR_CLASSIFICATION_DATASET_SPEC.md`](../../TABULAR_CLASSIFICATION_DATASET_SPEC.md). Contains non-missing, non-constant target column and finite numeric features.

---

## Deterministic Reproduction

The bundled archive is generated deterministically using [`examples/build_sample_datasets.py`](../build_sample_datasets.py):

```bash
python examples/build_sample_datasets.py
```
