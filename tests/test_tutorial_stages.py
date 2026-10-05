"""The isolated-environment tutorial path (NOTEBOOK_SPEC 2.2 §25.13): the kernel's carrier and `run_stage` helper, and
the carried stage runner.

* Kernel side (numpy + Pillow only): the generated notebook's own carrier and `run_stage` code run with the current
  interpreter standing in for the isolated environment; an unpinned snapshot and an invalid BYOD input each stop the
  kernel with a RuntimeError that repeats the stage's own message.
* CPU pre-flight (torch + transformers): every stage runs in order, in process, against a tiny random-init snapshot
  and synthetic photographs standing in for the pinned ones; each stage rebuilds its inputs from files, so the
  hand-offs between stages are exercised. It proves the stage plumbing, not the model.
"""
# ruff: noqa: E501

from __future__ import annotations

import importlib.util
import io
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from conftest import TEST_REVISION, textured

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load("build_notebook")
stages = _load("tutorial_stages")
TEMPLATE = _load("notebook_template").TEMPLATE


def _kernel(tmp_path: Path) -> dict:
    notebook = build.render(ROOT, TEMPLATE, "test-revision")
    code = [c["source"] for c in notebook["cells"] if c["cell_type"] == "code"]
    carrier = next(s for s in code if s.startswith("# @title Infrastructure: write and verify the carried"))
    install = next(s for s in code if s.startswith("# @title Infrastructure: install the locked runtime"))
    run_root = tmp_path / "run"
    run_root.mkdir()
    namespace = {"ROOT": run_root, "WEIGHTS": tmp_path / "weights", "PYTHON": Path(sys.executable), "ENV": dict(os.environ)}
    exec("import hashlib\nimport json\nimport subprocess\n" + carrier, namespace)  # noqa: S102 - the notebook's own cell
    exec(install[install.index("def run_stage(") : install.index("def load_record(")], namespace)  # noqa: S102
    return namespace


def test_carrier_writes_and_verifies_every_carried_file(tmp_path):
    kernel = _kernel(tmp_path)
    for dest, source in TEMPLATE["carried"].items():
        assert (kernel["ROOT"] / dest).read_bytes() == (ROOT / source).read_text(encoding="utf-8").encode("utf-8"), dest
    assert kernel["NOTEBOOK_SOURCE"]["revision"] == "test-revision"


def test_invalid_byod_input_stops_the_kernel_with_the_validator_message(tmp_path):
    kernel = _kernel(tmp_path)
    textured((300, 40)).save(tmp_path / "too_big.png")
    with pytest.raises(RuntimeError, match=r"Stage 'byod' failed \(exit 2\): ValueError: .*MAX_INPUT_SIDE 256"):
        kernel["run_stage"]("byod", "--byod-path", tmp_path / "too_big.png", "--byod-mode", "lr")
    log = (kernel["ROOT"] / "logs" / "byod.log").read_text(encoding="utf-8")
    assert "STAGE FAILED (byod)" in log


@pytest.mark.skipif(json.loads((ROOT / "weights/swin2sr-x4-64/dimer-base-manifest.json").read_text())["revision"] != "unpinned", reason="snapshot pinned")
def test_unpinned_weights_stage_explains_itself(tmp_path):
    kernel = _kernel(tmp_path)
    with pytest.raises(RuntimeError, match="carries no immutable revision.*pin_snapshot"):
        kernel["run_stage"]("weights")


# ---------------------------------------------------------------------------------------------------------
# CPU pre-flight of every stage with a tiny model
# ---------------------------------------------------------------------------------------------------------


def _fake_photos() -> dict[str, bytes]:
    from swin2sr_x4_super_resolution_pipeline import EVAL_RECORDS, NEW_RECORDS

    out = {}
    for i, record in enumerate(EVAL_RECORDS + NEW_RECORDS):
        buffer = io.BytesIO()
        textured((400, 336), seed=i).save(buffer, format="JPEG", quality=90)
        out[record[0]] = buffer.getvalue()
    return out


def test_cpu_preflight_runs_every_stage_in_order(tmp_path, monkeypatch, pinned):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from conftest import write_tiny_snapshot
    from swin2sr_x4_super_resolution_pipeline import MANIFEST_NAME

    root, weights = tmp_path / "run", tmp_path / "weights"
    snap = write_tiny_snapshot(weights / stages.SNAPSHOT_KEY, revision=TEST_REVISION)
    (root / "weights" / stages.SNAPSHOT_KEY).mkdir(parents=True)
    shutil.copy(snap / MANIFEST_NAME, root / "weights" / stages.SNAPSHOT_KEY / MANIFEST_NAME)
    photos = _fake_photos()
    monkeypatch.setattr(stages, "load_photos", lambda run: photos)
    byod_dir = tmp_path / "byod"
    byod_dir.mkdir()
    textured((130, 98)).save(byod_dir / "mine.png")
    textured((64, 64), 3).save(byod_dir / "other.jpg")

    def run(stage: str, *extra: str) -> None:
        assert stages.main(["--root", str(root), "--weights", str(weights), "--stage", stage, *extra]) == 0, (root / "state" / f"{stage}.error.json").read_text() if (root / "state" / f"{stage}.error.json").exists() else stage

    for stage in ("weights", "prepare", "evaluate", "infer"):
        run(stage)
    run("activity", "--kernel", "box")
    run("activity", "--kernel", "bicubic", "--jpeg-quality", "30")
    run("byod", "--byod-path", str(byod_dir), "--byod-mode", "hr")
    run("byod", "--byod-path", str(byod_dir / "mine.png"), "--byod-mode", "lr")

    out = root / "outputs"
    stem = stages.STEM
    for name in (f"{stem}_input_manifest.json", f"{stem}_inputs.csv", f"{stem}_panels.png", f"{stem}_scores.csv", f"{stem}_evaluation_report.json", f"{stem}_new_panels.png", f"{stem}_predictions.csv", f"{stem}_result.json", f"{stem}_activity.json", f"{stem}_byod_hr/byod_result.json", f"{stem}_byod_lr/byod_result.json"):
        assert (out / name).is_file(), name
    assert len(list((out / f"{stem}_sample_outputs").glob("*.png"))) == 24
    report = json.loads((out / f"{stem}_evaluation_report.json").read_text())
    assert report["border_px"] == 4 and report["model"]["n"] == 24 and set(report["baselines"]) == {"bicubic", "nearest"}
    result = json.loads((out / f"{stem}_result.json").read_text())
    assert result["model"]["revision"] == TEST_REVISION and result["provenance"]["safetensors_only"] is True
    assert [p["id"] for p in result["new_inputs"]][-1] == "new-synthetic-scene"
    assert all(p["verdict"].startswith("not-measurable") for p in result["new_inputs"])
    byod_hr = json.loads((out / f"{stem}_byod_hr/byod_result.json").read_text())
    assert byod_hr["preprocessing"]["mine"]["pixels_removed"] == {"right": 2, "bottom": 2}
    assert byod_hr["comparison"]["n_images"] == 2
    byod_lr = json.loads((out / f"{stem}_byod_lr/byod_result.json").read_text())
    assert byod_lr["verdict"].startswith("not-measurable") and byod_lr["inputs"][0]["output_size"] == [520, 392]
    manifest = json.loads((out / f"{stem}_input_manifest.json").read_text())
    assert all("rejected" in p for p in manifest["probes"]) and len(manifest["probes"]) == 5


def test_a_stage_refuses_to_run_out_of_order(tmp_path):
    root, weights = tmp_path / "run", tmp_path / "weights"
    assert stages.main(["--root", str(root), "--weights", str(weights), "--stage", "infer"]) == 2
    error = json.loads((root / "state" / "infer.error.json").read_text())
    assert "run the notebook from the top" in error["message"]
