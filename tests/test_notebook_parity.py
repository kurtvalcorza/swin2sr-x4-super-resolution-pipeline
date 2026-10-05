"""NOTEBOOK_SPEC 2.2 parity tests (PAR1–PAR3) for the standalone tutorial notebook (isolated-environment carrier), and
the kernel-install boundary: the notebook carries the committed lock unchanged and pip-installs nothing into the kernel."""
# ruff: noqa: E501

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load("build_notebook")
TEMPLATE = _load("notebook_template").TEMPLATE
NOTEBOOK = ROOT / "tutorials" / TEMPLATE["notebook_name"]
LOCK = ROOT / TEMPLATE["carried"][TEMPLATE["lock"]]


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def _carrier(notebook: dict) -> tuple[dict, dict[str, str], dict[str, str]]:
    cells = [c for c in notebook["cells"] if c["cell_type"] == "code" and c.get("metadata", {}).get("dimer", {}).get("embedded_sources")]
    assert len(cells) == 1, "exactly one carrier cell is expected"
    values = {}
    for node in ast.parse(_source(cells[0])).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("CARRIED_FILES", "CARRIED_HASHES"):
            values[node.targets[0].id] = ast.literal_eval(node.value)
    return cells[0], values["CARRIED_FILES"], values["CARRIED_HASHES"]


def test_par1_carried_files_equal_repository_sources(notebook):
    _cell, files, _hashes = _carrier(notebook)
    assert list(files) == [*TEMPLATE["carried"], build.SOURCE_RECORD]
    for dest, source in TEMPLATE["carried"].items():
        assert files[dest] == (ROOT / source).read_text(encoding="utf-8"), f"carried {dest} drifted from {source}; regenerate the notebook"


def test_par1_carried_hashes_are_correct_and_recorded(notebook):
    cell, files, hashes = _carrier(notebook)
    assert set(hashes) == set(files)
    for dest, text in files.items():
        assert hashes[dest] == hashlib.sha256(text.encode("utf-8")).hexdigest(), dest
    assert cell["metadata"]["dimer"]["files"] == hashes
    assert notebook["metadata"]["dimer"]["generated_from"]["files"] == hashes


def test_par2_carried_lock_is_the_committed_lock_and_pins_pyproject(notebook):
    _cell, files, _hashes = _carrier(notebook)
    lock_text = LOCK.read_text(encoding="utf-8")
    assert files[TEMPLATE["lock"]] == lock_text
    build.check_lock(build._pins(ROOT), lock_text)  # every direct pin at its version, every entry hashed
    locked = build.lock_packages(lock_text)
    assert locked["torch"] == "2.14.0" and locked["transformers"] == "4.57.6"
    header = lock_text.splitlines()[1]
    assert "--generate-hashes" in header and "--only-binary :all:" in header and "x86_64-manylinux" in header


def test_par2_carried_manifest_is_the_committed_manifest(notebook):
    _cell, files, _hashes = _carrier(notebook)
    dest = "weights/swin2sr-x4-64/dimer-base-manifest.json"
    assert json.loads(files[dest]) == json.loads((ROOT / dest).read_text(encoding="utf-8"))
    meta = notebook["metadata"]["dimer"]
    assert meta["standalone"] is True and meta["requires_dimer_worker"] is False
    assert meta["notebook_spec"] == build.NOTEBOOK_SPEC == "2.2"
    assert (meta["notebook_profile"], meta["notebook_mode"]) == ("TASK-INFERENCE", "GUIDED")


def test_par3_generator_check_is_clean(notebook):
    recorded = notebook["metadata"]["dimer"]["generated_from"]["revision"]
    rendered = build.to_bytes(build.render(ROOT, TEMPLATE, recorded))
    assert NOTEBOOK.read_bytes().replace(b"\r\n", b"\n") == rendered, "notebook is stale; run python tools/build_notebook.py"


def test_outputs_are_cleared(notebook):
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None


def test_nothing_is_pip_installed_into_the_kernel(notebook):
    kernel_cells = [_source(c) for c in notebook["cells"] if c["cell_type"] == "code" and not c["metadata"].get("dimer", {}).get("embedded_sources")]
    for source in kernel_cells:
        assert "pip install" not in source and "'-m', 'pip'" not in source and "sys.executable" not in source
        assert not re.search(r"(?m)^\s*[!%]", source), "no shell escapes or magics"
    installs = [s for s in kernel_cells if "'pip', 'install'" in s]
    assert len(installs) == 1
    assert "[str(UV), 'pip', 'install', '--python', str(PYTHON), '--require-hashes', '--only-binary', ':all:'" in installs[0]
    assert "'venv', '--managed-python', '--python', '3.12.12'" in installs[0]
    assert "MPLBACKEND='Agg'" in installs[0] and "'HF_TOKEN'" in installs[0]


def test_no_repository_dependency_in_the_kernel(notebook):
    kernel = "\n".join(_source(c) for c in notebook["cells"] if c["cell_type"] == "code" and not c["metadata"].get("dimer", {}).get("embedded_sources"))
    assert "github.com" not in kernel and "git clone" not in kernel
    assert TEMPLATE["package"] not in kernel, "the kernel must not import the package; stages import the carried copy"
