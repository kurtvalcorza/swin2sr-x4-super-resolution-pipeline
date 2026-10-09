"""Regression tests for the 2026-10-05 notebook review findings (S2X-m1..m3).

They need only CI's lightweight dependencies: the generated notebook's own cell sources are executed with stand-ins
(a fake isolated interpreter). None of this is model evidence.
"""
# ruff: noqa: E501

from __future__ import annotations

import copy
import importlib.util
import json
import os
import shutil
import sys
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"s2x_fix_{name}", TOOLS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load("build_notebook")
TEMPLATE = _load("notebook_template").TEMPLATE
NOTEBOOK = ROOT / "tutorials" / TEMPLATE["notebook_name"]


def _markdown() -> str:
    return "\n".join(c["source"] for c in _cells() if c["cell_type"] == "markdown")


def test_s2x_m1_no_template_token_in_markdown_and_the_hosted_timing_is_quoted() -> None:
    import re

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    markdown = "\n".join(c["source"] for c in notebook["cells"] if c["cell_type"] == "markdown")
    assert not re.findall(r"\{[A-Z][A-Z0-9_]*\}", markdown)
    assert "has not been recorded yet" not in markdown
    assert "built the isolated environment in 86 s" in notebook["cells"][0]["source"]
    prerequisites = next(c["source"] for c in notebook["cells"] if c["source"].startswith("## Prerequisites"))
    assert "Tesla T4, 5 October 2026" in prerequisites and "evaluate 20.5 s" in prerequisites


def test_s2x_m1_validator_rejects_an_unformatted_markdown_token() -> None:
    validator = _load("validate_release_assets")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    validator._validate_notebook_structure(NOTEBOOK, notebook)  # the committed notebook passes
    broken = copy.deepcopy(notebook)
    broken["cells"][0]["source"] += "\n\nThis checkpoint, `{MODEL_ID}`, is a placeholder."
    with pytest.raises(validator.ValidationError, match=r"unformatted template tokens"):
        validator._validate_notebook_structure(NOTEBOOK, broken)


def test_s2x_m2_checkpoint_matches_the_recorded_run_and_species_lines_are_cautioned() -> None:
    md = _markdown()
    assert "about 1 dB" not in md and "far from 1 dB" not in md
    assert "+2.69 dB" in md and "2.09 to 3.35" in md
    assert "four crops per species, descriptive only" in md



# ---- S2X-m3: the uv-template re-run fixes (idempotent Section 1, environment reuse, run_stage guard) ------------------


def _cells() -> list[dict]:
    return build.render(ROOT, TEMPLATE, "test-revision")["cells"]


def _check_cell() -> str:
    return next(c["source"] for c in _cells() if c["cell_type"] == "code" and c["source"].startswith("# @title Infrastructure: check the runtime"))


def _run_check_cell(namespace: dict, source: str | None = None) -> dict:
    exec(source or _check_cell(), namespace)  # noqa: S102 - the notebook's own cell
    return namespace


@pytest.mark.skipif(sys.platform != "linux", reason="executes the Section 1 kernel cell, which shells out to nvidia-smi/df and a POSIX venv/bin/python (Linux runtimes only)")
def test_s2x_m3_section1_rerun_keeps_the_run_directory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(shutil, "disk_usage", lambda _path: types.SimpleNamespace(free=10**13))  # the disk check is not under test
    namespace = _run_check_cell({})
    first = namespace["ROOT"]
    assert first.is_dir() and first.parent == tmp_path / "outputs" / TEMPLATE["stem"]
    _run_check_cell(namespace)  # re-running Section 1 alone
    assert namespace["ROOT"] == first, "a Section 1 re-run must not strand the later cells in a new, empty run directory"
    _run_check_cell(namespace, _check_cell().replace("NEW_RUN_DIRECTORY = False", "NEW_RUN_DIRECTORY = True"))
    assert namespace["ROOT"] != first


@pytest.mark.skipif(sys.platform != "linux", reason="executes the Section 1 kernel cell, which shells out to nvidia-smi/df and a POSIX venv/bin/python (Linux runtimes only)")
def test_s2x_m3_environment_is_keyed_on_the_lock_not_the_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(shutil, "disk_usage", lambda _path: types.SimpleNamespace(free=10**13))  # the disk check is not under test
    a = _run_check_cell({})
    b = _run_check_cell({})
    assert a["ROOT"] != b["ROOT"] and a["ENV_ROOT"] == b["ENV_ROOT"]
    assert a["ROOT"].name not in str(a["ENV_ROOT"])


def _fake_ipython(monkeypatch) -> None:
    display = types.ModuleType("IPython.display")
    display.Image = display.display = lambda *a, **k: None
    package = types.ModuleType("IPython")
    package.display = display
    monkeypatch.setitem(sys.modules, "IPython", package)
    monkeypatch.setitem(sys.modules, "IPython.display", display)


@pytest.mark.skipif(sys.platform != "linux", reason="executes the Section 1 kernel cell, which shells out to nvidia-smi/df and a POSIX venv/bin/python (Linux runtimes only)")
def test_s2x_m3_install_cell_reuses_a_complete_environment_without_downloading(tmp_path: Path, monkeypatch) -> None:
    _fake_ipython(monkeypatch)
    install = next(c["source"] for c in _cells() if c["cell_type"] == "code" and c["source"].startswith("# @title Infrastructure: install the locked runtime"))
    env_root = tmp_path / "env"
    python = env_root / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\necho '" + json.dumps({"python": "3.12.12", "torch": "x", "transformers": "x", "cuda": False}) + "'\n", encoding="utf-8")
    python.chmod(0o755)
    lock_sha = "a" * 64
    spec = {"lock_sha256": lock_sha, "python": TEMPLATE["managed_python"], "uv": TEMPLATE["uv"]["version"]}
    (env_root / "ready.json").write_text(json.dumps(spec), encoding="utf-8")

    def no_network(*_a, **_k):
        raise AssertionError("a matching environment must be reused, not rebuilt")

    monkeypatch.setattr("urllib.request.urlopen", no_network)
    namespace = {"ENV_ROOT": env_root, "ROOT": tmp_path, "CARRIED_HASHES": {TEMPLATE["lock"]: lock_sha}, "NOTEBOOK_SOURCE": {"revision": "r"}, "SESSION_START": time.perf_counter(), "Path": Path, "GPU": None}
    exec("import hashlib, json, os, shutil, subprocess, time\n" + install, namespace)  # noqa: S102
    assert namespace["ENV_REUSED"] is True
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        assert name not in namespace["ENV"]
    assert namespace["ENV"]["MPLBACKEND"] == "Agg"
    namespace["CARRIED_HASHES"] = {TEMPLATE["lock"]: "b" * 64}  # a different lock is never reused
    with pytest.raises(AssertionError, match="must be reused"):
        exec("import hashlib, json, os, shutil, subprocess, time\n" + install, namespace)  # noqa: S102
    assert not (env_root / "ready.json").exists()


def test_s2x_m3_run_stage_names_the_cells_to_rerun_when_the_run_directory_is_empty(tmp_path: Path) -> None:
    install = next(c["source"] for c in _cells() if c["cell_type"] == "code" and c["source"].startswith("# @title Infrastructure: install the locked runtime"))
    namespace = {"ROOT": tmp_path / "empty", "WEIGHTS": tmp_path / "w", "PYTHON": Path(sys.executable), "ENV": dict(os.environ)}
    exec(install[install.index("def run_stage(") : install.index("def load_record(")], namespace)  # noqa: S102
    with pytest.raises(RuntimeError, match=r"run the three Infrastructure cells again in order \(Sections 1, 2 and 3\)"):
        namespace["run_stage"]("prepare")
    troubleshooting = next(c["source"] for c in _cells() if c["source"].startswith("## Troubleshooting"))
    assert "run the three Infrastructure cells again in order (Sections 1, 2 and 3)" in troubleshooting


def test_stage_processes_import_neither_ipython_nor_google_colab() -> None:
    """Stages run in the isolated environment, which has neither IPython nor google.colab: only kernel cells may use
    them (display in the install cell, the BYOD upload dialog). A carried module that imported either would fail on
    Colab; there is no worker and no google.colab stub to give a ModuleSpec."""
    import re

    carried = [ROOT / source for dest, source in TEMPLATE["carried"].items() if dest.endswith(".py")]
    assert any(path.name == "tutorial_stages.py" for path in carried)
    offenders = [str(path) for path in carried if re.search(r"^\s*(from|import)\s+(IPython|google)b", path.read_text(encoding="utf-8"), re.M)]
    assert not offenders, offenders
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert "sys.modules['google" not in json.dumps(nb) and 'sys.modules["google' not in json.dumps(nb)
