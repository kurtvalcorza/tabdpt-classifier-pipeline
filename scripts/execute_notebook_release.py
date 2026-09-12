#!/usr/bin/env python3
"""Execute the DIMER tutorial pair through real IPython kernels and record release evidence.

This is an *execution* check (NOTEBOOK_SPEC REL1/REL5/REL7), not the static validator. It runs
`tabdpt_classifier_colab.ipynb` (E2E) in one fresh kernel, then runs
`tabdpt_classifier_artifact_inference_colab.ipynb` (ARTIFACT-INFERENCE) in a *second* fresh kernel
whose artifact and new-data inputs come from the first run and a separately generated CSV — the
same external-artifact boundary an interactive Colab user crosses with the upload dialog.

Two disclosed substitutions make the Colab-shaped notebooks runnable outside Colab; both are
recorded in the evidence JSON so the record never claims a literal Colab run:

1. `/content/...` paths are rewritten to `<work>/content/...` (Colab's fixed workspace root does
   not exist on other hosts).
2. The bootstrap cell (`git clone` + `%pip install` of the lock set) is skipped when
   `--skip-bootstrap` is given; the executing interpreter must then already provide the lock set
   and this repository at the revision under test. `google.colab.files.upload()` is served by a
   small shim module that returns the files named in `DIMER_UPLOAD_FILES` (a `;`-separated path
   list, consumed one call at a time) instead of opening a browser dialog.

Everything else — every other cell, magic, assertion, and export — executes unmodified.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from nbclient import NotebookClient
from sklearn.datasets import load_breast_cancer

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "tutorials/tabdpt_classifier_colab.ipynb"
INFERENCE = ROOT / "tutorials/tabdpt_classifier_artifact_inference_colab.ipynb"

SHIM = textwrap.dedent(
    '''
    """Test-harness stand-in for google.colab.files: serves upload() from DIMER_UPLOAD_FILES."""
    import os
    from pathlib import Path

    _QUEUE = [p for p in os.environ.get("DIMER_UPLOAD_FILES", "").split(";") if p]


    def upload():
        if not _QUEUE:
            raise RuntimeError("harness upload queue is empty; set DIMER_UPLOAD_FILES")
        batch = _QUEUE.pop(0).split("|")
        return {Path(p).name: Path(p).read_bytes() for p in batch}
    '''
)


def _install_shim(work: Path) -> Path:
    site = work / "harness-site"
    (site / "google" / "colab").mkdir(parents=True, exist_ok=True)
    (site / "google" / "__init__.py").write_text("", encoding="utf-8")
    (site / "google" / "colab" / "__init__.py").write_text("from . import files\n", encoding="utf-8")
    (site / "google" / "colab" / "files.py").write_text(SHIM, encoding="utf-8")
    return site


def _prepare(source: Path, work: Path, skip_bootstrap: bool) -> nbformat.NotebookNode:
    nb = nbformat.read(source, as_version=4)
    content_root = (work / "content").as_posix()
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        if skip_bootstrap and idx == 1:
            # Keep the cell's pure-Python lines (imports, Python-floor guard); drop the clone,
            # lock-set install, and workspace reset that only make sense on a fresh Colab VM.
            kept = [l for l in cell.source.splitlines()
                    if not l.lstrip().startswith(("!", "%")) and "REPO_DIR" not in l]
            cell.source = "# bootstrap clone/install skipped by execute_notebook_release.py --skip-bootstrap\n" + "\n".join(kept) + "\n"
            continue
        cell.source = cell.source.replace('"/content/', f'"{content_root}/')
    return nb


def _execute(nb: nbformat.NotebookNode, executed: Path, cwd: Path, env: dict[str, str], timeout: int) -> None:
    os.environ.update(env)
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3", allow_errors=False,
                            resources={"metadata": {"path": str(cwd)}})
    client.execute()
    nbformat.write(nb, executed)


def make_genuinely_new_rows(path: Path) -> pd.DataFrame:
    """Rows near the sample's feature means but not present in the sample (checked)."""
    original = load_breast_cancer(as_frame=True).data.reset_index(drop=True).astype(float)
    means, scales = original.mean(axis=0), original.std(axis=0).replace(0.0, 1.0)
    fresh = pd.DataFrame([means + offset * scales for offset in np.linspace(-0.37, 0.37, 8)], columns=original.columns)
    original_rows = {tuple(row) for row in original.to_numpy(dtype=float)}
    for row in fresh.to_numpy(dtype=float):
        if tuple(row) in original_rows:
            raise RuntimeError("Release verifier generated a row already present in the producer sample")
    fresh.to_csv(path, index=False)
    return fresh


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", type=Path, default=Path("/content"), help="workspace root standing in for Colab's /content")
    parser.add_argument("--skip-bootstrap", action="store_true", help="skip the clone/pip bootstrap cell (interpreter already provides the lock set)")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--evidence", type=Path, default=None, help="where to write the evidence JSON (default <work>/release-evidence.json)")
    args = parser.parse_args()

    work = args.work.resolve()
    content = work / "content"
    content.mkdir(parents=True, exist_ok=True)
    site = _install_shim(work)
    env = {"PYTHONPATH": str(site) + os.pathsep + os.environ.get("PYTHONPATH", "")}

    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip()

    # 1. Producer (E2E) in a fresh kernel.
    main_executed = work / "tabdpt_classifier_colab.executed.ipynb"
    _execute(_prepare(MAIN, work, args.skip_bootstrap), main_executed, content, env, args.timeout)
    artifact_dir = content / "tabdpt-tutorial-output" / "artifact"
    manifest, context = artifact_dir / "artifact.json", artifact_dir / "training_context.parquet"
    if not (manifest.exists() and context.exists()):
        raise RuntimeError("E2E notebook did not produce artifact.json + training_context.parquet")
    metrics = json.loads((content / "tabdpt-tutorial-output" / "tabdpt_classification_metrics.json").read_text())

    # 2. Consumer (ARTIFACT-INFERENCE) in a second fresh kernel, fed from outside its execution.
    fresh_rows = work / "tabdpt_classifier_release_fresh_rows.csv"
    fresh = make_genuinely_new_rows(fresh_rows)
    env["DIMER_UPLOAD_FILES"] = f"{manifest}|{context};{fresh_rows}"
    inference_executed = work / "tabdpt_classifier_artifact_inference_colab.executed.ipynb"
    _execute(_prepare(INFERENCE, work, args.skip_bootstrap), inference_executed, content, env, args.timeout)

    predictions_path = content / "tabdpt-artifact-inference-output" / "tabdpt_classification_predictions.csv"
    if not predictions_path.exists():
        raise RuntimeError("Artifact-inference notebook did not write the predictions CSV")
    result = pd.read_csv(predictions_path)
    proba_cols = [c for c in result.columns if c.startswith("proba_")]
    if len(result) != len(fresh) or "prediction" not in result.columns or len(proba_cols) < 2:
        raise RuntimeError("Artifact-inference prediction output contract failed")
    if not np.allclose(result[proba_cols].sum(axis=1), 1.0, atol=1e-5):
        raise RuntimeError("Class probabilities do not sum to 1")
    if not all(result["prediction"].map(lambda l: f"proba_{l}" in proba_cols)):
        raise RuntimeError("Predicted labels are not among the exported probability columns")

    gpu = None
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        torch_version = torch.__version__
    except Exception:  # noqa: BLE001
        torch_version = None
    import importlib.metadata as md
    evidence = {
        "result": "PASS",
        "recordedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repositoryCommit": commit,
        "workingTreeDirty": bool(dirty),
        "notebooks": {"e2e": MAIN.name, "artifactInference": INFERENCE.name},
        "engine": "nbclient / IPython kernel, one fresh kernel per notebook",
        "environment": {"host": platform.node(), "os": platform.platform(), "python": platform.python_version(),
                        "torch": torch_version, "tabdpt": md.version("tabdpt"), "numpy": md.version("numpy"), "gpu": gpu},
        "substitutions": {"contentRoot": str(content), "bootstrapSkipped": args.skip_bootstrap,
                          "uploadShim": "google.colab.files.upload() served from DIMER_UPLOAD_FILES"},
        "notColab": "This is a clean local-kernel execution, not a Google Colab run; a Colab record is still required for a Colab claim.",
        "e2e": {"executedNotebook": str(main_executed), "metrics": metrics},
        "artifactInference": {"executedNotebook": str(inference_executed), "externalArtifact": str(manifest),
                              "freshRows": str(fresh_rows), "rowsScored": int(len(result)), "probabilityColumns": proba_cols},
    }
    evidence_path = args.evidence or (work / "release-evidence.json")
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
