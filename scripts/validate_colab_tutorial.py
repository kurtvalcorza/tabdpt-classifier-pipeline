#!/usr/bin/env python3
"""Statically validate DIMER tutorial notebooks against release-source invariants.

This script deliberately does not execute notebooks. Clean-runtime execution remains
an independent release gate under DIMER Notebook Specification 1.0.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUTORIALS_DIR = ROOT / "tutorials"
REQUIREMENTS = TUTORIALS_DIR / "requirements-colab.txt"

EXPECTED_PROFILES = {
    "tabdpt_classifier_colab.ipynb": "E2E",
    "tabdpt_classifier_artifact_inference_colab.ipynb": "ARTIFACT-INFERENCE",
}
PROFILE_MARKERS = {
    "E2E": (
        "By the end of this notebook you will be able to",
        "Prerequisites and runtime contract",
        "Verify the runtime and model provenance",
        "bring your own data",
        "majority-class",
        "Run inference on separate new records",
        "Export machine-readable outputs",
        "fresh reconstruction boundary",
        "Interpretation, limits, and next steps",
    ),
    "ARTIFACT-INFERENCE": (
        "externally supplied",
        "Prerequisites and trust boundary",
        "Supply and validate the external artifact",
        "Reconstruct the serving state",
        "genuinely new input",
        "Predict and export machine-readable results",
        "Interpretation and troubleshooting",
    ),
}
REQUIRED_CALLS = {
    "E2E": {
        "fit",
        "evaluate",
        "predict",
        "predict_proba",
        "export_preprocessing_state",
        "load_artifact",
    },
    "ARTIFACT-INFERENCE": {"load_artifact", "predict", "predict_proba"},
}
FORBIDDEN_ARTIFACT_CALLS = {"fit"}
PLACEHOLDER_PATTERN = re.compile(r"\b(?:TODO|TBD|FIXME)\b", re.IGNORECASE)
SHA_PATTERN = re.compile(r'REPO_REVISION\s*=\s*["\']([0-9a-f]{40})["\']')

# Disallow developer/author workstation paths like C:\\Users or /home/username.
# Permit Colab paths (/content/...) and web URLs.
ABSOLUTE_PATH_PATTERNS = [
    re.compile(r"[a-zA-Z]:[\\/]"),
    re.compile(r"/(?:Users|home|root)/"),
]


def clean_code_for_ast(code: str) -> str:
    """Comment out IPython magics and shell commands before Python AST parsing."""
    lines = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith(("%", "!")):
            lines.append(f"# {line}")
        else:
            lines.append(line)
    return "\n".join(lines)


def check_absolute_paths(code: str, filename: str, cell_idx: int) -> None:
    for line in code.splitlines():
        if "http://" in line or "https://" in line:
            continue
        for pattern in ABSOLUTE_PATH_PATTERNS:
            if pattern.search(line):
                raise AssertionError(
                    f"{filename} (cell {cell_idx}): developer-local filesystem path detected: "
                    f"{line.strip()}"
                )


def call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def require_use_flash_false(node: ast.Call, filename: str, cell_idx: int) -> None:
    value = node.func
    is_pipeline_constructor = isinstance(value, ast.Name) and value.id == "TabDPTClassificationPipeline"
    is_artifact_loader = (
        isinstance(value, ast.Attribute)
        and value.attr == "load_artifact"
        and isinstance(value.value, ast.Name)
        and value.value.id == "TabDPTClassificationPipeline"
    )
    if not (is_pipeline_constructor or is_artifact_loader):
        return
    has_use_flash_false = any(
        kw.arg == "use_flash"
        and isinstance(kw.value, ast.Constant)
        and kw.value.value is False
        for kw in node.keywords
    )
    if not has_use_flash_false:
        raise AssertionError(
            f"{filename} (cell {cell_idx}): pipeline construction/reload must pass "
            "`use_flash=False` explicitly for Tesla T4 portability."
        )


def validate_requirements() -> None:
    if not REQUIREMENTS.is_file():
        raise AssertionError(f"Missing tutorial requirements file: {REQUIREMENTS}")
    pins = []
    for raw in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        pins.append(line)
        if "==" not in line:
            raise AssertionError(
                f"{REQUIREMENTS.name}: tutorial dependency is not exactly pinned: {line}"
            )
    required = {"tabdpt[reproduce-results]==1.2.0", "pyarrow==25.0.1"}
    missing = sorted(required - set(pins))
    if missing:
        raise AssertionError(f"{REQUIREMENTS.name}: missing required pins: {missing}")


def validate_notebook(nb_path: Path) -> None:
    if not nb_path.exists():
        raise AssertionError(f"Missing notebook: {nb_path}")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    if nb.get("nbformat") != 4:
        raise AssertionError(f"{nb_path.name}: must use nbformat 4")

    expected_profile = EXPECTED_PROFILES[nb_path.name]
    dimer_meta = nb.get("metadata", {}).get("dimer", {})
    if dimer_meta.get("notebook_profile") != expected_profile:
        raise AssertionError(
            f"{nb_path.name}: metadata.dimer.notebook_profile must be {expected_profile!r}"
        )
    if dimer_meta.get("notebook_spec") != "1.0":
        raise AssertionError(f"{nb_path.name}: metadata.dimer.notebook_spec must be '1.0'")

    markdown = []
    code_text = []
    observed_calls: set[str] = set()
    for idx, cell in enumerate(nb.get("cells", [])):
        source = cell.get("source", "")
        text = "".join(source) if isinstance(source, list) else str(source)

        if PLACEHOLDER_PATTERN.search(text):
            raise AssertionError(f"{nb_path.name} (cell {idx}): unresolved placeholder marker")

        if cell.get("cell_type") == "markdown":
            markdown.append(text)
            continue
        if cell.get("cell_type") != "code":
            continue

        if cell.get("execution_count") is not None:
            raise AssertionError(f"{nb_path.name} (cell {idx}): execution_count must be cleared")
        if cell.get("outputs"):
            raise AssertionError(f"{nb_path.name} (cell {idx}): persisted outputs must be cleared")

        check_absolute_paths(text, nb_path.name, idx)
        code_text.append(text)
        clean_code = clean_code_for_ast(text)
        try:
            tree = ast.parse(clean_code, filename=f"{nb_path.name}:cell_{idx}")
        except SyntaxError as exc:
            raise AssertionError(
                f"Syntax error in {nb_path.name} (cell {idx}): {exc}"
            ) from exc

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = call_name(node)
                if name:
                    observed_calls.add(name)
                require_use_flash_false(node, nb_path.name, idx)

    all_markdown = "\n".join(markdown)
    all_code = "\n".join(code_text)

    if f"**Profile:** `{expected_profile}`" not in all_markdown:
        raise AssertionError(f"{nb_path.name}: visible normative profile declaration missing")
    for marker in PROFILE_MARKERS[expected_profile]:
        if marker.lower() not in all_markdown.lower():
            raise AssertionError(
                f"{nb_path.name}: required semantic marker missing for {expected_profile}: {marker!r}"
            )

    missing_calls = sorted(REQUIRED_CALLS[expected_profile] - observed_calls)
    if missing_calls:
        raise AssertionError(
            f"{nb_path.name}: required repository API calls not exercised: {missing_calls}"
        )

    match = SHA_PATTERN.search(all_code)
    if not match:
        raise AssertionError(f"{nb_path.name}: immutable 40-hex REPO_REVISION pin missing")
    if "requirements-colab.txt" not in all_code:
        raise AssertionError(f"{nb_path.name}: pinned tutorial requirements are not installed")
    if "checkout -q" not in all_code:
        raise AssertionError(f"{nb_path.name}: repository revision is not explicitly checked out")

    if expected_profile == "ARTIFACT-INFERENCE":
        forbidden = sorted(FORBIDDEN_ARTIFACT_CALLS & observed_calls)
        if forbidden:
            raise AssertionError(
                f"{nb_path.name}: artifact consumer must not self-create/refit an artifact: {forbidden}"
            )
        if "load_breast_cancer" in all_code:
            raise AssertionError(
                f"{nb_path.name}: artifact consumer must require external new input, not a built-in sample"
            )
        if "files.upload()" not in all_code:
            raise AssertionError(
                f"{nb_path.name}: external artifact/new-input upload path is missing"
            )

    print(f"[PASS] Validated {nb_path.name} as {expected_profile}")


def main() -> None:
    validate_requirements()
    notebooks = sorted(TUTORIALS_DIR.glob("*.ipynb"))
    expected_names = set(EXPECTED_PROFILES)
    actual_names = {path.name for path in notebooks}
    if actual_names != expected_names:
        raise AssertionError(
            f"Tutorial notebook inventory mismatch; expected={sorted(expected_names)}, "
            f"actual={sorted(actual_names)}"
        )
    for nb_path in notebooks:
        validate_notebook(nb_path)
    print(
        "Static tutorial validation passed. "
        "Clean-runtime execution remains a separate release requirement."
    )


if __name__ == "__main__":
    main()
