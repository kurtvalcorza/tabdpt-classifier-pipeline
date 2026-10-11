# Release verification

`tutorials/tabdpt_classifier_colab.ipynb` (`E2E`) and `tutorials/tabdpt_classifier_artifact_inference_colab.ipynb`
(`ARTIFACT-INFERENCE`) are **release candidates** until the exact notebook revisions have executed
top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell compilation, and
`tools/validate_release_assets.py` are necessary checks but are **not** runtime evidence under DIMER
Notebook Specification 2.2. This file is the durable release-gate record for both notebooks.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks, for each of the two notebooks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly the two tutorial notebooks, each named in `tutorials/README.md` with its profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.2`, mode `GUIDED`, `standalone: true` and
  `generated_from` (repository, generating commit, package module paths and SHA-256, carried-file digests,
  generator `build_notebook.py/3.0-tabular`);
- the standalone carrier and the isolated environment (ST1–ST6, PAR1–PAR3, RUN1, RUN10, ENV6): one carrier cell whose
  `CARRIED_FILES` / `CARRIED_BINARY` equal the repository's package (`src/tabdpt_classifier_pipeline/`), the notebook's
  stage runner (`tools/tutorial_stages.py` or `tools/tutorial_stages_artifact_inference.py`), the hash lock
  `tutorials/requirements-colab.lock.txt` (which must pin every `pyproject.toml` runtime pin, every entry hashed), the
  committed snapshot manifest, the licence and, for the companion, `examples/sample-artifact/`; `CARRIED_HASHES` match;
  the notebook is byte-identical to `tools/build_notebook.py` output for its template; no cell pip-installs into the
  notebook kernel and no text asks for a runtime restart; the install cell uses a pinned `uv` wheel, a managed CPython,
  `--require-hashes`, a lock-digest-keyed environment that is reused, `MPLBACKEND=Agg`, and drops
  `PYTHONPATH`/`PYTHONHOME`/`PYTHONSTARTUP`; the four Infrastructure cells are collapsed (`cellView: form`); every
  learner cell runs a stage;
- the profile-specific public-API calls in the carried stage runner — E2E: `from_pretrained(weights_dir=..., compile_model=False,
  use_flash=False, seed=SEED)`, `validate_inputs`, `majority_class_baseline`, the logistic-regression reference with its
  split-to-split range and the holdout error counts, `fit`, `evaluate`, `evaluation_report`, `predict_proba`, `predict`,
  `export_artifact_bundle`, `load_verified_artifact` in a fresh process with the no-refit and equivalence checks, the
  BYOD refusals for an absent or blank target; companion: the trusted-digest check before `validate_artifact_bundle`,
  `load_verified_artifact(..., model_weight_path=<verified checkpoint>, compile_model=False, use_flash=False)`, the
  fail-closed legacy-path and class-label checks, `validate_inputs(..., target_column=None, feature_columns=...)`,
  `predict_proba`, `predict`, a `not-measurable` `evaluation_report` — the exports per notebook, the learner-facing
  classification statements and the guided layer (how-to-use, task contract, roadmap, glossary, predictions,
  checkpoints, troubleshooting, conclusion), and the optional-input gates at their non-interactive defaults; no
  `assert` in the stage runners (verdicts are reported, contract checks raise with a message); forbidden patterns
  (credential-in-URL, any `git clone` / `github.com/kurtvalcorza` on the primary path, a mutable `revision='main'`,
  direct `tabdpt` / `huggingface_hub` / `sklearn` / `torch` use in the notebook's own cells, `trust_remote_code=True`,
  `pickle.load`, `torch.load(`, `extractall(`; in the companion also `load_breast_cancer(`, `export_artifact_bundle(`,
  `.fit(` in its stage runner);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter, single H1, required heading order, and immutable provenance.

CI also runs `tools/build_notebook.py --check` for both templates, `scripts/validate_repo.py`,
`scripts/validate_colab_tutorial.py` and the offline unit suite (`tests/`, including `test_notebook_parity.py`,
`test_companion_parity.py` and `test_notebook_review_fixes.py`, which execs the notebooks' own kernel cells with
stand-ins and runs the model-free stages; injected downloader, no weights, no model). `tools/build_sample_artifact.py
--check` reproduces the pinned sample artifact byte for byte. These are source/provenance and unit checks. They are
**not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab T4 GPU runtime (CPU also works); the kernel's Python does not matter — the stages run in an isolated CPython 3.12.12 environment | The runtime the tutorials are written for; a clean one-pass top-to-bottom run here is promotion evidence |
| Kaggle kernel | Kaggle T4 kernel | Reproducible clean-room executor of the same class; the notebook is pushed verbatim (no repository checkout is needed — the notebooks are standalone, and the companion's default path needs no supplied files) |
| Local harness (pre-flight only) | Workstation | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and not promotion evidence. `scripts/execute_notebook_release.py` was written for the previous repository-installing pair and does not apply to the /3 notebooks |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open the exact E2E notebook revision in a **fresh** Colab T4 runtime (or the Kaggle executor) with no repository
   checkout and a clean model cache;
3. choose **Run all** once, with every form field at its default (`USE_BYOD = False`, `RUN_ACTIVITY = False`); no
   restart may be needed — record `restarted: false`;
4. verify that Section 2 reports the generating revision recorded in `metadata.dimer.generated_from`, and that the
   isolated environment reports Python 3.12.12, `torch` 2.7.1, `tabdpt` 1.2.0, `numpy` 2.3.0, `pandas` 2.3.2,
   `scikit-learn` 1.7.0 and `cuda: True` on a T4; then re-run the Section 2 install cell once and check
   `environment_reused: True`;
5. verify every default-path stage completes:
   - `weights`: `fetched` names `tabdpt1_2.safetensors` (254,098,072 bytes) on a clean runtime and `verify_snapshot`
     passes;
   - `data` / `validate`: the breast-cancer sample (569 rows, CSV SHA-256 printed), the input manifest with one
     recorded rejection finding, the 455 / 114 stratified split;
   - `condition`: capacity (`modelFeatureCeiling`, `featureReductionActive`, imputer) and TabDPT's accuracy, log loss,
     ROC-AUC and holdout error count beside the majority baseline (0.6316, 42 errors) and the logistic-regression
     reference (0.9649, 4 errors; 0.9386–0.9912 over 20 seeded splits) — record TabDPT's printed numbers;
   - `report`: `outputs/tabdpt_classifier_evaluation_report.json` with verdict `sample-sanity`, both baselines and the
     interpretation;
   - `predict`: `outputs/tabdpt_classifier_predictions.csv`, `outputs/tabdpt_classifier_new_rows.csv` and
     `outputs/tabdpt_classifier_result.json` with the runtime identity and device;
   - `export` / `reload`: `outputs/artifact/{artifact.json,training_context.parquet}`, the printed artifact digest and
     `matches_pinned_sample_artifact` (record it; `True` confirms the companion's pinned sample), then the fresh-process
     reload with `preprocessing_restored_ is True`, identical labels and probabilities within `rtol=1e-5, atol=1e-6`;
6. in a **second** fresh runtime, run the exact companion notebook revision with every field at its default (no upload):
   verify the trusted digest of the pinned sample is `verified`, `validate_artifact_bundle` passes before
   reconstruction, `preprocessing_restored_ is True` and class labels match, the input manifest has one recorded
   rejection finding, and the predictions, the `not-measurable` evaluation report (`sample_kind: sample`) and the result
   JSON are written; then the REL12 journey: one compatible user input (the E2E run's `outputs/artifact/` with its
   printed digest in `EXPECTED_ARTIFACT_SHA256` and its `outputs/tabdpt_classifier_new_rows.csv` with
   `ID_COLUMNS = ['row_id']`) and one incompatible input (a digest with one changed character, or rows with a missing
   column), plus, in the E2E notebook, one compatible and one incompatible BYOD CSV (blank labels) through
   `BYOD_PATH` or the upload dialog;
7. verify the exports exist and the interpretation sections match the observed paths;
8. record the notebook Git blob ids, commit, runtime (platform, Python, PyTorch, tabdpt, device), `restarted`,
   model identifier and immutable revision, whether the model cache was clean, outcome, the printed metrics, the
   artifact digest, and any warning or applicable `SHOULD` deviation in the table below;
9. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of the notebook file (verify with
`git rev-parse <commit>:tutorials/<notebook>`). Wall times, when recorded, are the sum of per-cell
times reported by the executor and include installs and the model download; they are measurements
for the stated runtime, not general estimates.

### Manual clean-runtime evidence

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-11 | `45071534d021` (clean tree) — **previous repository-installing pair, spec 1.0** | Local host, Windows 11, Python 3.12.10, torch 2.7.1+cu128, tabdpt 1.2.0, numpy 2.3.0 (the `requirements-colab.txt` lock set of that revision), RTX 5070 Ti, `use_flash=False`; `scripts/execute_notebook_release.py --skip-bootstrap`, one fresh IPython kernel per notebook | E2E default sample path, then ARTIFACT-INFERENCE in a second kernel with the artifact and 8 fresh rows supplied externally | not recorded | **PASS** — E2E: sample accuracy 0.9912 / log_loss 0.0594 / roc_auc 0.9977 vs majority baseline 0.6316; artifact exported; no-refit reload with identical labels and equivalent probabilities. Companion: `preprocessing_restored_ = True`, 8 rows scored. **Not a Colab run, and not a run of the standalone carrier** — kept as evidence that the package paths the standalone notebooks carry executed on that revision |
| 2026-09-14 | `9c6cc36` / `9ec14a6daa7b` — **earlier standalone blob (generator /2, in-kernel pip install), superseded** | Kaggle T4 (`kurtvalcorza/dimer-nb2-tabdpt-classifier` v1) | Default sample path | 223.1 s | **PASSED** — 11/11 ok code cells executed cleanly, 4 files, 254 MB staged. History only (REL14): the runtime image, Python/torch/tabdpt versions, device, `restarted` flag and printed metrics were not recorded, and the blob is not the current one |
| 2026-10-11 | `ff8415b` / `cd16465668f7` of `tabdpt_classifier_colab.ipynb`, downloaded from GitHub at the PR head and blob-verified before the session; executed file `docs/execution-evidence/2026-10-11/tabdpt_classifier_colab_ff8415b_colab-cli-t4.ipynb`, sha256 `7e57b674bcecd595b364420a43dc038ef504d6b5e7d9041f060eafa8bdd818b8` (`…_run_summary.json` `7edcd8f3e524…`, `…_exec.log` `737350af7644…`) | Colab CLI 0.7.4 sequential execution, fresh Colab Tesla T4 (15,360 MiB, reported by cell 1), via the workspace `colab-cli-serial-test-suite`; isolated uv CPython 3.12.12 environment from the carried hash lock (55 packages; torch 2.7.1, tabdpt 1.2.0, numpy 2.3.0, pandas 2.3.2, scikit-learn 1.7.0). Not a browser Run all; the CLI records no execution counts, so order is evidenced by its `Executing cell k/N` log (1/11–11/11) | E2E default path: Breast Cancer sample, 455 support / 114 holdout rows, in-context conditioning only (no fine-tuning), device cuda; weight `tabdpt1_2.safetensors` downloaded from the Hub at revision `4462ffbd1d8d` (254,098,072 bytes, no stall) and digest-verified (`06680220…`) | 137.4 s session wall | **PASSED — one pass, no restart, 0 errors, 11/11 code cells.** Missing-target probe rejected; TabDPT holdout accuracy 0.9912, log loss 0.0594, ROC AUC 0.9977 (1 error in 114) — identical to the CPU reference; majority class 0.6316 (42 errors); standardised logistic regression 0.9649 / 0.0773 / 0.996 (4 errors); logistic accuracy over 20 seeded splits 0.9386–0.9912 (median 0.9737). `matches_pinned_sample_artifact: True` — the exported artifact's SHA-256 `b70144fd9a77ba69…` equals the pinned sample's. Fresh-process reload PASSED: preprocessing restored without refit, labels identical, max abs probability difference 0.0. Optional activity not run. Carried provenance names `1963393` (generation-time HEAD, not `ff8415b`). Status stays **Candidate** |
| 2026-10-11 | `ff8415b` / `08b97ab90028` of `tabdpt_classifier_artifact_inference_colab.ipynb`, downloaded from GitHub at the PR head and blob-verified before the session; executed file `docs/execution-evidence/2026-10-11/tabdpt_classifier_artifact_inference_colab_ff8415b_colab-cli-t4.ipynb`, sha256 `ec2c538c27aebc978c35214016241cfd01758ef57ac020c0d5be128543bc5be8` (`…_run_summary.json` `0176eb2694fe…`, `…_exec.log` `ad16caae6ec0…`) | Colab CLI 0.7.4 sequential execution, fresh Colab Tesla T4, via the workspace `colab-cli-serial-test-suite`; same 55-package isolated uv environment. Not a browser Run all; order from the CLI log (1/9–9/9) | Companion default path: the carried sample artifact (no release, no upload), base weight downloaded from the Hub and digest-verified | 134.4 s session wall | **PASSED — one pass, no restart, 0 errors, 9/9 code cells.** No upload in any output; trusted digest verified (`b70144fd9a77ba69…`, format `tabdpt-dimer-context-v3`); preprocessing restored without refit; missing-column probe rejected (`Feature schema mismatch; missing=['mean radius']`); the 8 predictions are identical to the E2E run's at 4 decimals (labels and both probabilities; 5 benign, 3 malignant). Optional activity not run. Carried provenance `1963393`. Status stays **Candidate** |

### Local lock-only drives (not clean-runtime evidence)

Builder pre-flight of the current blobs (generated at `1963393`; drives of the previous blobs `e779fee2` / `8985a6b5`, generated at `9cadc17` and differing only in that revision label, gave the same printed results): the notebooks' own code cells run in order in one
plain CPython namespace standing in for the kernel, in WSL Ubuntu x86_64 with the GPU hidden
(`CUDA_VISIBLE_DEVICES=-1`), so every stage ran on the CPU. The install cell built the isolated environment from
`tutorials/requirements-colab.lock.txt` (55 packages, `--require-hashes`, managed CPython 3.12.12, torch 2.7.1,
tabdpt 1.2.0) and, in the first drive, the `weights` stage downloaded the checkpoint from the Hub at the pinned revision and
verified it (later drives pre-staged the same file; the stage still checks its SHA-256).
Not a supported runtime, not hosted, and not promotion evidence.

| Date (UTC) | Notebook blob | Path exercised | Cells | Wall | Printed results |
|---|---|---|---|---|---|
| 2026-10-11 | E2E `cd16465668f7` | Default sample path | 11/11 | 486.6 s (env build 318 s; checkpoint pre-staged and digest-verified — an earlier drive fetched it from the Hub) | TabDPT accuracy 0.9912 / log loss 0.0594 / ROC-AUC 0.9977, 1 holdout error of 114; majority 0.6316 (42 errors); logistic regression 0.9649 (4 errors); `matches_pinned_sample_artifact: True` (artifact `b70144fd…5fc2`); fresh-process reload PASS, max probability difference 0.0 |
| 2026-10-11 | E2E `cd16465668f7` | Default sample path with `RUN_ACTIVITY = True` (Section 10) | 11/11 | 726.4 s (environment reused; activity 312 s on CPU) | canonical metrics as above; activity `n_ensembles` 2 → 8: accuracy 0.9912 → 0.9912 (1 error), log loss 0.0594 → 0.0514, ROC-AUC 0.9977 → 0.998; `canonical_outputs_unchanged: True` (7 files checked) |
| 2026-10-11 | E2E `cd16465668f7` | BYOD by `BYOD_PATH`: the sample table with 5 blank targets | 4 ok, then refused in the Section 4 cell | — | Section 4 stops: "the target column 'target' has 5 missing or blank value(s) out of 569 rows (file lines 5, 12, 52, 202, 402) … Nothing was written to outputs/."; `outputs/` empty |
| 2026-10-11 | ARTIFACT-INFERENCE `08b97ab90028` | Default path (pinned sample) | 9/9 | 45.6 s (`environment_reused: True` — the E2E drive's environment, same lock digest) | no upload dialog; `trusted_digest: verified`; 8 predictions identical to the E2E run's; report `not-measurable`, `sample_kind: sample` |

## Current status

**2026-10-11 — one-pass hosted runs of both current blobs are recorded** (the 2026-10-11 rows of the clean-runtime table): `tabdpt_classifier_colab.ipynb` (`cd16465668f7`) and `tabdpt_classifier_artifact_inference_colab.ipynb` (`08b97ab90028`) at `ff8415b`, each by Colab CLI 0.7.4 sequential execution on a fresh Colab Tesla T4 (not a browser Run all), no restart, 0 error outputs. Notes for the reviewer: (a) the T4 metrics are identical to the CPU reference (accuracy 0.9912, log loss 0.0594, ROC AUC 0.9977, 1/114 errors), and the exported artifact's SHA-256 equals the pinned sample's (`b70144fd…`); (b) the Hub checkpoint download completed on Colab (254,098,072 bytes, SHA-256 `06680220…` verified) — the stall seen in a local WSL drive is a local-only issue; (c) the companion's 8 predictions are identical to the E2E run's at 4 decimals (labels and both probabilities); (d) both carried revision labels read `1963393` (generation-time HEAD, not `ff8415b`) — follow-up; (e) the optional activity was not run; (f) the maintainer accepted the carried sample (no release; Apache-2.0; no weights; public breast-cancer rows plus imputer/scaler statistics; byte-identical to a real CPU export). Both notebooks stay **Candidate**; remaining gates: reviewer confirmation of these runs, the BYOD journeys and the optional activities.

Before 2026-10-11: no clean-runtime execution of the current standalone blobs had been recorded (the Kaggle row above is an earlier blob, the local drives are pre-flight only). Static validation (`tools/validate_release_assets.py`), nbformat validation, a
`compile()` sweep over every code cell, and the offline unit suite passed on the tutorial source at
the candidate revision, which is necessary but not sufficient. The registry status remains
**Candidate** until a reviewer confirms a recorded run against the notebook blobs under review and
an integrator promotes it; promotion is not performed by the builder. Facts a reviewer should weigh:
the real `hf_hub_download` fetch of `tabdpt1_2.safetensors` into a fresh `weights/tabdpt-1.2/` ran in the
local lock-only drive above, not yet on a hosted runtime;
`from_pretrained` loads no model, so the first real load of the checkpoint through the snapshot path happens inside
`fit`; the lock carries `huggingface-hub==0.36.2` whereas the 2026-09-11 run used 0.33.2; and the current standalone
carrier (isolated environment, stage subprocesses) has run end to end only in the local CPU drives above. The hosted
runs will be its first execution on a supported runtime and on a CUDA device (the stages use CUDA automatically when
the isolated environment sees one; `use_flash=False`).