# TabDPT Classifier Tutorials

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/kurtvalcorza/tabdpt-classifier-pipeline)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Layer6%2FTabDPT-ffcc4d?style=flat)](https://huggingface.co/Layer6/TabDPT)
[![Upstream](https://img.shields.io/badge/Upstream-layer6ai--labs%2FTabDPT--inference-181717?style=flat&logo=github&logoColor=white)](https://github.com/layer6ai-labs/TabDPT-inference)
[![arXiv](https://img.shields.io/badge/arXiv-2608.01400-b31b1b.svg)](https://arxiv.org/abs/2608.01400)

These notebooks implement the DIMER tutorial contract for TabDPT v1.2 classification.

**Notebook specification:** DIMER Notebook Specification `1.0`

## Tutorial registry

| Notebook | Profile | Capability | Default runtime | BYOD | Artifact contract | Release status |
|---|---|---|---|---|---|---|
| [`tabdpt_classifier_colab.ipynb`](tabdpt_classifier_colab.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/tabdpt-classifier-pipeline/blob/main/tutorials/tabdpt_classifier_colab.ipynb) | `E2E` | In-context tabular classification, evaluation, new-data inference, artifact export/reload | GPU recommended; CPU supported; `use_flash=False` | Sample, single CSV, or preserved pre-split CSVs | Produces `tabdpt-dimer-context-v3` | **Candidate — clean-runtime evidence pending** |
| [`tabdpt_classifier_artifact_inference_colab.ipynb`](tabdpt_classifier_artifact_inference_colab.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/tabdpt-classifier-pipeline/blob/main/tutorials/tabdpt_classifier_artifact_inference_colab.ipynb) | `ARTIFACT-INFERENCE` | External artifact validation/reconstruction and new-data scoring | GPU recommended; CPU supported; `use_flash=False` | External unlabelled CSV required | Consumes externally supplied `tabdpt-dimer-context-v3` | **Candidate — clean-runtime evidence pending** |

A notebook MUST NOT be described as release-grade until the exact release revision has documented clean-runtime execution evidence. Static CI validation is necessary but is not execution evidence.

## Reproducible environment

Both notebooks clone an immutable repository revision and install [`requirements-colab.txt`](requirements-colab.txt), which pins the upstream TabDPT v1.2 reproducibility profile and the remaining first-order tutorial runtime dependencies used by this integration. Supported tutorial runtime is Python 3.11–3.13.

The notebooks print the effective Python and principal library versions at runtime. If a core package such as PyTorch is already imported before setup, the tutorial fails early and asks for a fresh runtime rather than hiding a restart boundary.

## Model provenance and portability

The repository fixes the model identity to:

- Hugging Face repository `Layer6/TabDPT`;
- revision `4462ffbd1d8dea25d4862d30beed4b70cd596ae5`;
- weight file `tabdpt1_2.safetensors`;
- SHA-256 `06680220fd66c4524051706b98c1c659a674d19d3a766cd0bb276505e99faccd`;
- upstream inference source commit `9cfb05e0a6bc380ae6c99c08adc8d50dacd4f246`.

The E2E notebook resolves and verifies the pinned checkpoint before model construction. Both notebooks explicitly use `use_flash=False` so the demonstrated path remains portable to common Tesla T4 (`sm_75`) Colab/Kaggle GPUs as well as newer accelerators. CPU execution is supported but may be slower.

## Learning and evidence boundaries

TabDPT is used here as an in-context learner. `TabDPTClassificationPipeline.fit()` fits repository preprocessing state and registers labelled support context; it does **not** gradient-train or fine-tune the TabDPT weights.

Public tutorial metrics are demonstration/sanity evidence only. TabDPT was pretrained on real-world tabular datasets, so overlap with public samples cannot be ruled out. The notebooks do not claim calibrated probabilities, universal decision thresholds, fairness, robustness, or production fitness from successful execution alone.

## Artifact trust and data governance

A DIMER TabDPT serving artifact includes `artifact.json` plus `training_context.parquet`; the immutable base checkpoint is referenced separately. The support context is part of the serving state and must receive the same confidentiality, licensing, retention, and disclosure controls as its source data.

The artifact-inference notebook intentionally accepts individual artifact files rather than ZIP/TAR archives. It validates model identity, artifact metadata, fitted preprocessing consistency, and the support-context SHA-256 before reconstruction. A matching manifest and digest establish internal consistency, not sender authenticity.

## Release gate

Before changing either registry entry to `release-grade`, record a clean supported-runtime execution for the exact PR/commit under review, including:

- notebook commit or PR head SHA;
- Python and principal runtime/library versions;
- accelerator/device;
- E2E default sample outcome and artifact reload-equivalence result;
- external-artifact notebook outcome using an artifact from a separate producing execution;
- confirmation that static validation and clean-runtime execution both passed.

See the repository [`README.md`](../README.md), [`MODEL_CARD.md`](../MODEL_CARD.md), [`TABULAR_CLASSIFICATION_DATASET_SPEC.md`](../TABULAR_CLASSIFICATION_DATASET_SPEC.md), and [`DIMER_CONTRACT.md`](../DIMER_CONTRACT.md) for the surrounding model, data, and serving contracts.
