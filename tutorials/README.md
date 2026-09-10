# TabDPT Classifier Tutorials

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/kurtvalcorza/tabdpt-classifier-pipeline)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Layer6%2FTabDPT-ffcc4d?style=flat)](https://huggingface.co/Layer6/TabDPT)
[![Upstream](https://img.shields.io/badge/Upstream-layer6ai--labs%2FTabDPT--inference-181717?style=flat&logo=github&logoColor=white)](https://github.com/layer6ai-labs/TabDPT-inference/tree/9cfb05e0a6bc380ae6c99c08adc8d50dacd4f246)
[![arXiv](https://img.shields.io/badge/arXiv-2410.18164-b31b1b.svg)](https://arxiv.org/abs/2410.18164)

These notebooks implement the DIMER tutorial contract for TabDPT v1.2 classification.

**Notebook specification:** DIMER Notebook Specification `1.0`

## Tutorial registry

| Notebook | Profile | Capability | Supported release runtime | BYOD | Artifact contract | Release status |
|---|---|---|---|---|---|---|
| [`tabdpt_classifier_colab.ipynb`](tabdpt_classifier_colab.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/tabdpt-classifier-pipeline/blob/main/tutorials/tabdpt_classifier_colab.ipynb) | `E2E` | In-context tabular classification, evaluation, new-data inference, artifact export/reload | **Google Colab**, Python 3.11–3.13; GPU recommended, CPU supported; `use_flash=False` | Sample, single CSV, or preserved pre-split CSVs | Produces `tabdpt-dimer-context-v3` | **Candidate — clean-runtime evidence pending** |
| [`tabdpt_classifier_artifact_inference_colab.ipynb`](tabdpt_classifier_artifact_inference_colab.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/tabdpt-classifier-pipeline/blob/main/tutorials/tabdpt_classifier_artifact_inference_colab.ipynb) | `ARTIFACT-INFERENCE` | External artifact validation/reconstruction and new-data scoring | **Google Colab**, Python 3.11–3.13; GPU recommended, CPU supported; `use_flash=False` | External unlabelled CSV required | Consumes externally supplied `tabdpt-dimer-context-v3` | **Candidate — clean-runtime evidence pending** |

Google Colab is the release-conformance environment for these notebooks. Other Jupyter environments may work, but are not part of the current release claim. A notebook MUST NOT be described as release-grade until the exact release revision has documented clean-runtime execution evidence. Static CI validation is necessary but is not execution evidence.

## Reproducible environment and code reachability

Both notebooks pin immutable repository code commit `cef1f0ae4af14c7c27364a1f1a9d8f573af3504e` and install [`requirements-colab.txt`](requirements-colab.txt), which pins the upstream TabDPT v1.2 reproducibility profile and remaining first-order tutorial runtime dependencies.

The pinned code commit is intentionally kept reachable by branch `anchors/notebook-spec-v1-code-20260910`. That branch is a retention anchor only: the immutable commit SHA, not the branch name, is the tutorial's code version. This prevents squash/rebase integration plus feature-branch deletion from orphaning the revision installed by the notebooks.

The notebooks print Python, principal library/framework versions, device/CUDA information, and material compile/FlashAttention/quantization/precision assumptions. If PyTorch is already imported before setup, the tutorial fails early rather than hiding a restart boundary.

## Model provenance and sources

The repository fixes the model identity to:

- Hugging Face repository [`Layer6/TabDPT`](https://huggingface.co/Layer6/TabDPT);
- revision `4462ffbd1d8dea25d4862d30beed4b70cd596ae5`;
- weight file `tabdpt1_2.safetensors`;
- SHA-256 `06680220fd66c4524051706b98c1c659a674d19d3a766cd0bb276505e99faccd`;
- upstream inference source commit [`9cfb05e0a6bc380ae6c99c08adc8d50dacd4f246`](https://github.com/layer6ai-labs/TabDPT-inference/tree/9cfb05e0a6bc380ae6c99c08adc8d50dacd4f246);
- paper: [*TabDPT: Scaling Tabular Foundation Models on Real Data*](https://arxiv.org/abs/2410.18164).

The E2E default sample is scikit-learn's copy of the [UCI Breast Cancer Wisconsin (Diagnostic) dataset](https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic), also documented by [`sklearn.datasets.load_breast_cancer`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html). UCI lists the dataset under **CC BY 4.0**. Sample metrics are tutorial/sanity evidence only, not benchmark evidence; TabDPT pretraining overlap cannot be ruled out.

See the repository [`README.md`](../README.md), [`MODEL_CARD.md`](../MODEL_CARD.md), [`TABULAR_CLASSIFICATION_DATASET_SPEC.md`](../TABULAR_CLASSIFICATION_DATASET_SPEC.md), and [`DIMER_CONTRACT.md`](../DIMER_CONTRACT.md) for surrounding model, data, and serving contracts.

## Production artifact/runtime parity

The tutorial does not define a second serving schema. Its v3 `runtimeConfig` uses the complete production `DimerRuntimeConfig` shape:

- `target_column`, `drop_columns`, `max_train_rows`, `validation_split`;
- `fine_tune` (required to be `false`);
- `n_ensembles`, `context_size`, `batch_size`, `temperature`, `seed`.

`runtimeConfig.drop_columns` preserves the requested runtime configuration and may include the target. The target is never a feature, so strict validation removes `runtimeConfig.target_column` before comparing the requested list with fitted `preprocessing.dropColumns`. Runtime/preprocessing seed values must match exactly.

The E2E fresh-boundary check deletes producer-side runtime-control variables and the resolved checkpoint path before reconstruction. Reload inference derives its settings only from the validated serialized `runtimeConfig`; the base checkpoint is reacquired through the repository's immutable model identity/checksum path.

## Learning, trust, and evidence boundaries

TabDPT is used here as an in-context learner. `TabDPTClassificationPipeline.fit()` fits repository preprocessing state and registers labelled support context; it does **not** gradient-train or fine-tune the TabDPT weights.

A DIMER TabDPT serving artifact includes `artifact.json` plus `training_context.parquet`; the immutable base checkpoint is referenced separately. The support context is part of serving state and must receive the same confidentiality, licensing, retention, and disclosure controls as its source data.

The artifact-inference notebook accepts individual artifact files rather than ZIP/TAR archives. It validates exact model identity, the complete production runtime schema, fitted preprocessing consistency, context size/path/digest, and unexpected files before reconstruction. Matching metadata and digests establish internal consistency, not sender authenticity.

## Release gate

Before changing either registry entry to `release-grade`, record a clean Google Colab execution for the exact PR/commit under review, including:

- notebook commit or PR head SHA;
- Python and principal runtime/library versions;
- accelerator/device;
- E2E default sample outcome and artifact reload-equivalence result;
- external-artifact notebook outcome using an artifact from a separate producing execution;
- confirmation that static validation and clean-runtime execution both passed.
