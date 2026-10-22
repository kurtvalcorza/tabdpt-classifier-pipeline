# Release verification

`tutorials/tabdpt_classifier_colab.ipynb` (`E2E`, kind `TUTORIAL`) is a **release candidate** until the exact
notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell
compilation, and `tools/validate_release_assets.py` are necessary checks but are **not** runtime evidence under DIMER
Notebook Specification 3.0. This file is the durable release-gate record for the notebook.

Under NOTEBOOK_SPEC 3.0 (KND11) the former `ARTIFACT-INFERENCE` companion,
`tutorials/tabdpt_classifier_artifact_inference_colab.ipynb`, was folded into the E2E notebook's closing
external-artifact section (Section 10, §19.1) and deleted, together with its template, its stage runner, its parity
test and the script-built `examples/sample-artifact/` bundle (`tools/build_sample_artifact.py` ran no export stage, so
the bundle could not meet SART9). The companion's rows in the tables below are kept as history (REL14).

**Pinned sample artifact of Section 10 (SART6–SART9).** `SAMPLE_ARTIFACT_URL` is the release asset
`tabdpt_classifier_artifact.zip` of the tag `sample-artifact-breast-cancer-v1` (target `762eb5b`), pinned by
`SAMPLE_ARTIFACT_SHA256` in `tools/tutorial_stages.py`; 76,015 bytes. It packages `outputs/artifact/` of a recorded
hosted Colab T4 run of the default path of `tutorials/tabdpt_classifier_colab.ipynb` at `762eb5b` (fresh reload PASS),
packed **after the run** on the same VM by `…_pack.py` (`colab exec -f`), deterministically: the two members `artifact.json` and `training_context.parquet` at the ZIP root in sorted order,
each with the timestamp 1980-01-01 00:00:00, `ZIP_DEFLATED` and mode 0644 — the layout the migrated export stage
writes to `outputs/tabdpt_classifier_artifact.zip`, so a later re-pin from a run of this notebook needs no code
change. The member digests equal that run's printed `artifact_sha256` and `training_context_sha256`. The release
notes name the producing run. The asset is never replaced; a changed artifact gets a new tag and a new pin (SART7,
SART11).

**v1 does not meet SART9 and is superseded once v2 is pinned (owner ruling, 2026-10-22).** The v1 ZIP was assembled after
the run by a script, not written by the notebook's own default path, so it is not the export of that path that SART9
requires, even though its member digests equal the run's printed digests. From `spec3/sart9-inrun-zip` on, the export
stage's own output `outputs/tabdpt_classifier_artifact.zip` is the run's exported artifact: it is written in-run
(sorted members, 1980-01-01 00:00:00, mode 0644, `create_system` 3, atomic write) and its SHA-256 is recorded in
`outputs/tabdpt_classifier_result.json` under `artifact.zip_sha256`. The replacement, `sample-artifact-breast-cancer-v2`,
is that file uploaded unchanged from a recorded hosted run, published as an immutable release and re-pinned in a
separate commit. Until then the notebook keeps the v1 pin; the v1 tag stays in place and is not overwritten (SART7).

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly the one tutorial notebook, named in `tutorials/README.md` with its profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `3.0`, kind `TUTORIAL`, mode `GUIDED`, `standalone: true` and
  `generated_from` (repository, generating commit, package module paths and SHA-256, carried-file digests,
  generator `build_notebook.py/3.0-tabular`);
- the standalone carrier and the isolated environment (ST1–ST6, PAR1–PAR3, RUN1, RUN10, ENV6): one carrier cell whose
  `CARRIED_FILES` / `CARRIED_BINARY` equal the repository's package (`src/tabdpt_classifier_pipeline/`), the notebook's
  stage runner (`tools/tutorial_stages.py`), the hash lock
  `tutorials/requirements-colab.lock.txt` (which must pin every `pyproject.toml` runtime pin, every entry hashed), the
  committed snapshot manifest and the licence; `CARRIED_HASHES` match;
  the notebook is byte-identical to `tools/build_notebook.py` output for its template; no cell pip-installs into the
  notebook kernel and no text asks for a runtime restart; the install cell uses a pinned `uv` wheel, a managed CPython,
  `--require-hashes`, a lock-digest-keyed environment that is reused, `MPLBACKEND=Agg`, and drops
  `PYTHONPATH`/`PYTHONHOME`/`PYTHONSTARTUP`; the four Infrastructure cells are collapsed (`cellView: form`); every
  learner cell runs a stage;
- the profile-specific public-API calls in the carried stage runner — E2E: `from_pretrained(weights_dir=..., compile_model=False,
  use_flash=False, seed=SEED)`, `validate_inputs`, `majority_class_baseline`, the logistic-regression reference with its
  split-to-split range and the holdout error counts, `fit`, `evaluate`, `evaluation_report`, `predict_proba`, `predict`,
  `export_artifact_bundle`, `load_verified_artifact` in a fresh process with the no-refit and equivalence checks, the
  BYOD refusals for an absent or blank target, the deterministic export ZIP; the external-artifact section: the pinned
  URL and SHA-256, the digest check before extraction, the member-by-member extraction of exactly the two members,
  `validate_artifact_bundle`, the SART10 data-contract check,
  `load_verified_artifact(..., model_weight_path=<verified checkpoint>, compile_model=False, use_flash=False)`, the
  fail-closed legacy-path and class-label checks, `validate_inputs(..., feature_columns=...)` with a refused probe,
  and the outputs under `outputs/external_artifact/` — the exports, the learner-facing
  classification statements and the guided layer (how-to-use, task contract, roadmap, glossary, predictions,
  checkpoints, troubleshooting, conclusion), and the optional-input gates at their non-interactive defaults; no
  `assert` in the stage runners (verdicts are reported, contract checks raise with a message); forbidden patterns
  (credential-in-URL, any `git clone`, any `github.com/kurtvalcorza` reference except the exact pinned sample-artifact
  URL in the stage runner (whole-token match; `..`, `%2e`, `?` and `#` refused), a mutable `revision='main'`,
  direct `tabdpt` / `huggingface_hub` / `sklearn` / `torch` use in the notebook's own cells, `trust_remote_code=True`,
  `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter, single H1, required heading order, and immutable provenance.

CI also runs `tools/build_notebook.py --check`, `scripts/validate_repo.py`,
`scripts/validate_colab_tutorial.py` and the offline unit suite (`tests/`, including `test_notebook_parity.py`,
`test_notebook_review_fixes.py`, which execs the notebook's own kernel cells with stand-ins and runs the model-free
stages, and `test_external_artifact_section.py`: the pin, the digest refusal before extraction, the ZIP layout, the
SART10 contract, the section's dependency on the setup cells only, the output path and the ST1 exact-URL rule; injected
downloader, no weights, no model). These are source/provenance and unit checks. They are
**not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab T4 GPU runtime (CPU also works); the kernel's Python does not matter — the stages run in an isolated CPython 3.12.12 environment | The runtime the tutorials are written for; a clean one-pass top-to-bottom run here is promotion evidence |
| Kaggle kernel | Kaggle T4 kernel | Reproducible clean-room executor of the same class; the notebook is pushed verbatim (no repository checkout is needed — the notebook is standalone, and its external-artifact section downloads the pinned asset itself) |
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
   - `export` / `reload`: `outputs/artifact/{artifact.json,training_context.parquet}`, the printed artifact digest,
     `outputs/tabdpt_classifier_artifact.zip` with its `zip_sha256` (record it; Section 10 never uses it, and it need not equal the
     pinned `SAMPLE_ARTIFACT_SHA256`), then the fresh-process reload with
     `preprocessing_restored_ is True`, identical labels and probabilities within `rtol=1e-5, atol=1e-6`;
   - `external_artifact` / `external_reconstruct` / `external_predict` (Section 10): the provenance line before the
     download, `trusted_digest: verified (pinned SAMPLE_ARTIFACT_SHA256)`, the data contract (30 features, `target`,
     `benign` / `malignant`, 455 support rows), `preprocessing_restored_: True`, eight scored rows and
     `outputs/external_artifact/tabdpt_classifier_external_{predictions.csv,result.json}`;
6. the REL12 journey in Section 10: one compatible user input (`ARTIFACT_SOURCE = 'My own artifact ZIP'` with this
   run's `outputs/tabdpt_classifier_artifact.zip` and its printed digest in `EXPECTED_ZIP_SHA256`, and
   `outputs/tabdpt_classifier_new_rows.csv` as `EXTERNAL_ROWS_PATH`) and one incompatible input (a digest with one
   changed character, or rows with a missing column), plus one compatible and one incompatible BYOD CSV (blank labels)
   through `BYOD_PATH` or the upload dialog;
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
| 2026-10-11 | `ff8415b` / `cd16465668f7` of `tabdpt_classifier_colab.ipynb`, downloaded from GitHub at the PR head and blob-verified before the session; executed file `docs/execution-evidence/2026-10-11/tabdpt_classifier_colab_ff8415b_colab-cli-t4.ipynb`, sha256 `7e57b674bcecd595b364420a43dc038ef504d6b5e7d9041f060eafa8bdd818b8` (`…_run_summary.json` `7edcd8f3e524…`, `…_exec.log` `737350af7644…` (redacted 2026-10-22, now `125836fc59a0…`)) | Colab CLI 0.7.4 sequential execution, fresh Colab Tesla T4 (15,360 MiB, reported by cell 1), via the workspace `colab-cli-serial-test-suite`; isolated uv CPython 3.12.12 environment from the carried hash lock (55 packages; torch 2.7.1, tabdpt 1.2.0, numpy 2.3.0, pandas 2.3.2, scikit-learn 1.7.0). Not a browser Run all; the CLI records no execution counts, so order is evidenced by its `Executing cell k/N` log (1/11–11/11) | E2E default path: Breast Cancer sample, 455 support / 114 holdout rows, in-context conditioning only (no fine-tuning), device cuda; weight `tabdpt1_2.safetensors` downloaded from the Hub at revision `4462ffbd1d8d` (254,098,072 bytes, no stall) and digest-verified (`06680220…`) | 137.4 s session wall | **PASSED — one pass, no restart, 0 errors, 11/11 code cells.** Missing-target probe rejected; TabDPT holdout accuracy 0.9912, log loss 0.0594, ROC AUC 0.9977 (1 error in 114) — identical to the CPU reference; majority class 0.6316 (42 errors); standardised logistic regression 0.9649 / 0.0773 / 0.996 (4 errors); logistic accuracy over 20 seeded splits 0.9386–0.9912 (median 0.9737). `matches_pinned_sample_artifact: True` — the exported artifact's SHA-256 `b70144fd9a77ba69…` equals the pinned sample's. Fresh-process reload PASSED: preprocessing restored without refit, labels identical, max abs probability difference 0.0. Optional activity not run. Carried provenance names `1963393` (generation-time HEAD, not `ff8415b`). Status stays **Candidate** |
| 2026-10-11 | `ff8415b` / `08b97ab90028` of `tabdpt_classifier_artifact_inference_colab.ipynb`, downloaded from GitHub at the PR head and blob-verified before the session; executed file `docs/execution-evidence/2026-10-11/tabdpt_classifier_artifact_inference_colab_ff8415b_colab-cli-t4.ipynb`, sha256 `ec2c538c27aebc978c35214016241cfd01758ef57ac020c0d5be128543bc5be8` (`…_run_summary.json` `0176eb2694fe…`, `…_exec.log` `ad16caae6ec0…` (redacted 2026-10-22, now `82ae9633d36e…`)) | Colab CLI 0.7.4 sequential execution, fresh Colab Tesla T4, via the workspace `colab-cli-serial-test-suite`; same 55-package isolated uv environment. Not a browser Run all; order from the CLI log (1/9–9/9) | Companion default path: the carried sample artifact (no release, no upload), base weight downloaded from the Hub and digest-verified | 134.4 s session wall | **PASSED — one pass, no restart, 0 errors, 9/9 code cells.** No upload in any output; trusted digest verified (`b70144fd9a77ba69…`, format `tabdpt-dimer-context-v3`); preprocessing restored without refit; missing-column probe rejected (`Feature schema mismatch; missing=['mean radius']`); the 8 predictions are identical to the E2E run's at 4 decimals (labels and both probabilities; 5 benign, 3 malignant). Optional activity not run. Carried provenance `1963393`. Status stays **Candidate** |
| 2026-10-22 | `762eb5b` / `cd16465668f7` of `tabdpt_classifier_colab.ipynb` (main; input unchanged) — **the run that produced the pinned sample artifact of Section 10** | Colab CLI 0.7.4 `colab exec -f` on a fresh Colab Tesla T4 (15,360 MiB), session `suite-tabdpt-762eb5b-16f8`, via the workspace `colab-cli-serial-test-suite` (not a browser Run all) | Default sample path, 11/11 code cells, no errors | 121.7 s | **PASS** — TabDPT accuracy 0.9912 / log loss 0.0594 / ROC-AUC 0.9977 (1 holdout error of 114) vs majority 0.6316 and logistic regression 0.9649; export `artifact_sha256` `b70144fd9a77…`, `training_context_sha256` `9964a1377bd0…`; fresh reload PASS (preprocessing restored without refit, labels identical). `outputs/artifact/` was then packed on the VM by `…_pack.py` (sorted members at the ZIP root, 1980-01-01, `ZIP_DEFLATED`, mode 0644) into `tabdpt_classifier_artifact.zip` (76,015 bytes, SHA-256 `42aab6ce342b…`), published as release `sample-artifact-breast-cancer-v1`. Files in `docs/execution-evidence/2026-10-22-762eb5b/` (local paths redacted to `<local-run-dir>/` before the first commit; captured → committed SHA-256): executed notebook `b36d7f1f22e8…` → unchanged; `…_run_summary.json` `8b3e313a06a3…` → unchanged; `…_pack.py` `88cc4b6dbcce…` → unchanged; `…_exec.log` `4c2dbb3dc1bf…` → `d749e421db16…`; `…_suite.log` `b5b3386f9f87…` → `697e3027bf4f…` (2 and 4 path lines redacted) |

**First hosted run of the migrated notebook (2026-10-22, `b0a0173` / blob `e6ef7050`, Notebook Spec 3.0; the second pass of
§32.2).** Colab CLI 0.7.4 `colab exec -f` on a fresh Colab Tesla T4, default path of `tutorials/tabdpt_classifier_colab.ipynb` on
PR #29, started 2026-10-22T03:18:06Z, wall 134.5 s, exit 0, input unchanged; executed cell sources equal blob `e6ef7050`
(44 cells); verdict PASS. Evidence: `docs/execution-evidence/2026-10-22-b0a0173/`. The fresh reload restored the fitted
preprocessing without refit (labels identical). **Section 10 ran end to end on the default path:** it downloaded the release
asset of tag `sample-artifact-breast-cancer-v1`, verified SHA-256
`42aab6ce342b6bdf9870e1fe332a1371cdcb2e6d19ad6b6b484a39a1fc73bb19` before extraction, rebuilt the classifier from the files
alone and wrote `outputs/external_artifact/tabdpt_classifier_external_predictions.csv` and `..._external_result.json` (printed
in the executed notebook; the files stayed on the released VM). The run's own deterministic export,
`outputs/tabdpt_classifier_artifact.zip`, printed the same digest `42aab6ce…` as the pinned asset: the exporter reproduces the
published artifact byte for byte. Local paths were replaced by `<local-run-dir>/` before the first commit:

| File (under `execution-evidence/2026-10-22-b0a0173/`) | Captured | Committed |
|---|---|---|
| `tabdpt_classifier_colab_b0a0173_colab-cli-t4.ipynb` | `706be24e8f28b36a2250a524dd231b3ec627e1c9b68446a75764f7451124f411` | unchanged (no local path) |
| `tabdpt_classifier_colab_b0a0173_colab-cli-t4_run_summary.json` | `64d4c539d8335eda3b31bd09099b684707ece263e7797b05f8bea91c292174b8` | unchanged (no local path) |
| `tabdpt_classifier_colab_b0a0173_colab-cli-t4_exec.log` | `bf9248254ada3bb056758a57da1058a378a89c7f7365cf68a59809e1cf4ae220` | `4316671494d0bd19bc93f5af0cb9df1713f46e3772c90b1a432575d6028e2e66` |
| `tabdpt_classifier_colab_b0a0173_colab-cli-t4_suite.log` | `8b8629438e9474cdc4a94388c4e81e9c7c2d8bf771f439449cc535cbb1ef420e` | `e00f9e00e7142961a0e49acf13e307ff3267e6502460447afa5b672d29e6bfca` |

**In-run export identity run (2026-10-22, `6d47d75` / blob `69a07f6b`, PR #31; SART9).** Colab CLI 0.7.5 `colab exec -f` on a fresh Colab Tesla T4, default path of `tutorials/tabdpt_classifier_colab.ipynb` at the head of `spec3/sart9-inrun-zip`, started 2026-10-22T12:53:22Z, wall 172.2 s, exit 0, input unchanged, 14/14 code cells, no error; verdict PASS. Evidence: `docs/execution-evidence/2026-10-22-6d47d75/`. The export stage wrote `outputs/tabdpt_classifier_artifact.zip` in-run and recorded `artifact.zip_sha256` `42aab6ce342b6bdf9870e1fe332a1371cdcb2e6d19ad6b6b484a39a1fc73bb19` in the run's result file. That digest equals the release asset of `sample-artifact-breast-cancer-v1` byte for byte, as the export of the `b0a0173` run did. Section 10 downloaded the pinned asset and verified the same digest before extraction. Local paths were replaced by `<local-run-dir>/` before the first commit:

| File (under `execution-evidence/2026-10-22-6d47d75/`) | Captured | Committed |
|---|---|---|
| `tabdpt_classifier_colab_6d47d75_colab-cli-t4.ipynb` | `d7b0fad65d02fe09e01cc01e1927225f4384ff2d2cd047faf3fbe6541c8a120c` | unchanged (no local path) |
| `tabdpt_classifier_colab_6d47d75_colab-cli-t4_exec.log` | `f12abe89c2d1c86438a510bc598094e59363fb178c989d77affa628183250e22` | `cf78b9db927eb43a2d820df1ffdf8d463de02b6ee2245d561be534f3fb5ec131` |
| `tabdpt_classifier_colab_6d47d75_colab-cli-t4_run_summary.json` | `803b4739d4fc87ca57ac13e6ff6cbd17edede935a363520ba93899df133e69e6` | unchanged (no local path) |
| `tabdpt_classifier_colab_6d47d75_colab-cli-t4_suite.log` | `bced937a218eda4de583e1613ff2d7d82f1f9c7c68d9dd5cbdb4cf8b38ef5c16` | `93de3a459ac4422559e1f7a449023a899d1e449886c3b5730c4b5bfa26fb2ee5` |

**Optional-activity run (2026-10-22, `b37f496` / committed blob `69a07f6b`, `RUN_ACTIVITY = True`).** A run copy of committed blob `69a07f6b` with one override, `RUN_ACTIVITY = False` → `True` (run copy blob `0ac01f567735`, SHA-256 `24f36b06a0b35daf443b58c014b73b74e98e50dfca5ecb707798e7cb7da94e0f`; it differs from the committed blob only in that assignment literal and is committed beside the evidence as `…_activity_run_copy.ipynb`), run by Colab CLI 0.7.5 `colab exec -f` (a scripted run, not a browser Run all) on a fresh Colab Tesla T4 at `main` `b37f496`, started 2026-10-22T13:36:18Z, wall 138.6 s, exit 0, sequential, input (the run copy) unchanged, 14/14 code cells, no error; verdict PASS. Evidence: `docs/execution-evidence/2026-10-22-b37f496-activity/`. The Section 11 activity (`n_ensembles` 2 → 8) printed, on 114 holdout rows, accuracy 0.9912 → 0.9912, log loss 0.0594 → 0.0514, ROC AUC 0.9977 → 0.998, holdout errors 1 → 1, and `{'canonical_outputs_unchanged': True, 'files_checked': 8}`. Local paths were replaced by `<local-run-dir>` before the first commit:

| File (under `execution-evidence/2026-10-22-b37f496-activity/`) | Captured | Committed |
|---|---|---|
| `tabdpt_classifier_colab_b37f496_colab-cli-t4_activity.ipynb` | `37c3ecf38cf1aa111e1980b2db15b63406e47b13d2bd35e0163f093a5ba36ed6` | unchanged (no local path) |
| `tabdpt_classifier_colab_b37f496_colab-cli-t4_activity_exec.log` | `2759e825c99a3525c99e8a461636bf5b709a0187c80f2dbbfc683fff665350aa` | `ef557047639d5e9366b83acb53bbd8c535cd4a569cc3f16174057d197ecfca25` (2 replacements) |
| `tabdpt_classifier_colab_b37f496_colab-cli-t4_activity_run_summary.json` | `95ce928721106b4fbeeacfd9cfd62db0e6be53a57e338820225149639ce70fde` | unchanged (no local path) |
| `tabdpt_classifier_colab_b37f496_colab-cli-t4_activity_run_copy.ipynb` | `24f36b06a0b35daf443b58c014b73b74e98e50dfca5ecb707798e7cb7da94e0f` | unchanged (no local path) |

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

**Current status (2026-10-22): Candidate.** Recorded: the hosted Colab T4 run of blob `69a07f6b` at `6d47d75` (PR #31; Colab CLI, not a browser Run all) passed the default path end to end, including Section 10 with the v1 pin (see "In-run export identity run" above). The optional activity ran once on the same blob (see "Optional-activity run" above: a `RUN_ACTIVITY = True` run copy at `b37f496`, Colab CLI 0.7.5, fresh T4, PASS), which satisfies the owner's 2026-10-22 ruling that it runs before promotion. The earlier run of blob `e6ef7050` at `b0a0173` is history for the current blob (REL14). The notebook stays Candidate pending:

1. **REL12:** a BYOD run that accepts one representative input through the downstream stages and refuses one
   incompatible input;
2. **REL2:** a run in the hosted runtime's browser interface for the interactive journeys (BYOD upload, the optional
   activity toggle, the own-artifact ZIP).

**SART9 is met by identity (owner ruling, 2026-10-22).** The release asset of `sample-artifact-breast-cancer-v1` was packed on the VM after the `762eb5b` run from that run's exported `outputs/artifact/`. Since then the notebook's own export stage writes the deterministic ZIP in-run (sorted members, 1980-01-01 00:00:00, mode 0644, `create_system` 3, atomic write) and records its SHA-256 under `artifact.zip_sha256`. The in-run ZIP of the hosted runs at `b0a0173` and `6d47d75` is byte-identical to the v1 asset (`42aab6ce…`), so the pinned bytes are the export of the default path of a recorded run. The owner chose to keep the v1 pin rather than republish identical bytes under a new tag.

**SART7 deviation (2026-10-22).** `sample-artifact-breast-cancer-v1` is a mutable release: this repository does not enable immutable releases. The notebook pins the asset's SHA-256 (`SAMPLE_ARTIFACT_SHA256` = `42aab6ce342b6bdf9870e1fe332a1371cdcb2e6d19ad6b6b484a39a1fc73bb19`), and Section 10 refuses the download before extraction unless the digest matches, so a replaced asset cannot be used silently. The owner accepted this deviation on 2026-10-22.

**History, 2026-10-11 — one-pass hosted runs of both blobs current at that time (the 2.2-era pair) were recorded** (the 2026-10-11 rows of the clean-runtime table): `tabdpt_classifier_colab.ipynb` (`cd16465668f7`) and `tabdpt_classifier_artifact_inference_colab.ipynb` (`08b97ab90028`) at `ff8415b`, each by Colab CLI 0.7.4 sequential execution on a fresh Colab Tesla T4 (not a browser Run all), no restart, 0 error outputs. Notes for the reviewer: (a) the T4 metrics are identical to the CPU reference (accuracy 0.9912, log loss 0.0594, ROC AUC 0.9977, 1/114 errors), and the exported artifact's SHA-256 equals the pinned sample's (`b70144fd…`); (b) the Hub checkpoint download completed on Colab (254,098,072 bytes, SHA-256 `06680220…` verified) — the stall seen in a local WSL drive is a local-only issue; (c) the companion's 8 predictions are identical to the E2E run's at 4 decimals (labels and both probabilities); (d) both carried revision labels read `1963393` (generation-time HEAD, not `ff8415b`) — follow-up; (e) the optional activity was not run; (f) the maintainer accepted the carried sample (no release; Apache-2.0; no weights; public breast-cancer rows plus imputer/scaler statistics; byte-identical to a real CPU export). Both notebooks stay **Candidate**; remaining gates: reviewer confirmation of these runs, the BYOD journeys and the optional activities.

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

## Local-path redaction of earlier evidence (2026-10-22)

The files below were committed with local absolute paths in them. Each local run-directory prefix was replaced with `<local-run-dir>/`, the convention the evidence committed from 2026-10-22 on already uses. No other byte changed and line endings are unchanged. Git history still holds the original files. "Before" is the SHA-256 of the file as committed on `main` at `4851ac9`.

| File | Replacements | SHA-256 before | SHA-256 after |
| --- | --- | --- | --- |
| `docs/execution-evidence/2026-10-11/tabdpt_classifier_artifact_inference_colab_ff8415b_colab-cli-t4_exec.log` | 2 | `ad16caae6ec04ec200c6265369349601578c2e61d20e0a24c8e63385aeb9fc8e` | `82ae9633d36e5cbd8ef3e860833ba2ffa4f591bfd766bdc5440000e608f251b3` |
| `docs/execution-evidence/2026-10-11/tabdpt_classifier_colab_ff8415b_colab-cli-t4_exec.log` | 2 | `737350af7644fb7987599262831d4ebb0a56ff2268333eb10d61747433eb9864` | `125836fc59a0b287c313c2fdb0183c35760e67c582961f834b0dd4336dcafa17` |

The Recorded executions table above quotes the pre-redaction hash of 2 of these logs as the hash of the committed file. Those quotes are kept as recorded, and the post-redaction hash is noted beside each one.
