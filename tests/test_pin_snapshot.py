"""tools/pin_snapshot.py against a fake Hub: no network, no weights."""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "89abcdef0123456789abcdef0123456789abcdef"
MANIFEST = "weights/swin2sr-x4-64/dimer-base-manifest.json"
MODULE = "src/swin2sr_x4_super_resolution_pipeline/pipeline.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("pin_snapshot", ROOT / "tools" / "pin_snapshot.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pin_snapshot = _load_tool()
UPSTREAM = {name: (ROOT / "weights/swin2sr-x4-64" / name).read_bytes() for name in ("README.md", "config.json", "preprocessor_config.json")}
FILES = {**UPSTREAM, "model.safetensors": b"\x00" * 64}
BIN = b"\x01" * 32


def _copy_repo(tmp_path: Path) -> Path:
    for relative in ("tools/notebook_template.py", "README.md", "MODEL_CARD.md", "STATUS.md", "docs/WEIGHTS.md"):
        if (ROOT / relative).exists():
            (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / relative, tmp_path / relative)
    shutil.copytree(ROOT / "src", tmp_path / "src")
    shutil.copytree(ROOT / "weights", tmp_path / "weights", ignore=shutil.ignore_patterns("*.safetensors", "inat-birds"))
    _unpin(tmp_path)
    return tmp_path


def _unpin(root: Path) -> None:
    """Return the copy to the pre-pin state the tool runs on: no revision and no weight or reference digest."""
    path = root / MANIFEST
    manifest = json.loads(path.read_text())
    manifest["revision"] = "unpinned"
    for entry in manifest["files"]:
        if entry["path"].endswith(".safetensors"):
            entry["sha256"] = None
    for entry in manifest.get("referenceFiles", []):
        entry["sha256"] = None
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    module = root / MODULE
    module.write_text(re.sub(r'^MODEL_REVISION = "[^"]*"$', 'MODEL_REVISION = "unpinned"', module.read_text(), count=1, flags=re.M))


def _fake_hub(files: dict[str, bytes], lfs_override: dict[str, str] | None = None):
    lfs_override = lfs_override or {}
    listed = {**files, "pytorch_model.bin": BIN}

    def model_info(model_id, revision, files_metadata):
        assert files_metadata is True
        siblings = []
        for name, data in listed.items():
            lfs = None
            if name.endswith((".safetensors", ".bin")):
                lfs = SimpleNamespace(sha256=lfs_override.get(name, hashlib.sha256(data).hexdigest()), size=len(data))
            siblings.append(SimpleNamespace(rfilename=name, lfs=lfs, size=len(data)))
        return SimpleNamespace(sha=COMMIT, siblings=siblings)

    calls = []

    def download(model_id, filename, revision, local_dir):
        calls.append((filename, revision))
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(files[filename])

    return model_info, download, calls


def test_pin_writes_commit_digests_reference_and_module_revision(tmp_path):
    root = _copy_repo(tmp_path)
    model_info, download, calls = _fake_hub(FILES)
    assert pin_snapshot.pin(root, model_info=model_info, download=download) == 0
    assert {revision for _name, revision in calls} == {COMMIT}
    manifest = json.loads((root / MANIFEST).read_text())
    assert manifest["revision"] == COMMIT and "pinStatus" not in manifest
    assert {f["path"]: f["sha256"] for f in manifest["files"]} == {name: hashlib.sha256(data).hexdigest() for name, data in FILES.items()}
    assert manifest["referenceFiles"][0]["sha256"] == hashlib.sha256(BIN).hexdigest()
    assert f'MODEL_REVISION = "{COMMIT}"' in (root / MODULE).read_text()


def test_pin_refuses_a_weight_digest_that_disagrees_with_the_hub(tmp_path):
    root = _copy_repo(tmp_path)
    before, module_before = (root / MANIFEST).read_text(), (root / MODULE).read_text()
    model_info, download, _calls = _fake_hub(FILES, lfs_override={"model.safetensors": "f" * 64})
    assert pin_snapshot.pin(root, model_info=model_info, download=download) == 1
    assert (root / MANIFEST).read_text() == before and (root / MODULE).read_text() == module_before


def test_pin_refuses_a_text_file_that_changed_since_it_was_recorded(tmp_path):
    root = _copy_repo(tmp_path)
    before = (root / MANIFEST).read_text()
    changed = {**FILES, "config.json": FILES["config.json"].replace(b'"upscale": 4', b'"upscale": 2')}
    model_info, download, _calls = _fake_hub(changed)
    assert pin_snapshot.pin(root, model_info=model_info, download=download) == 1
    assert (root / MANIFEST).read_text() == before


def test_pin_refuses_when_a_manifest_file_is_absent_upstream(tmp_path):
    root = _copy_repo(tmp_path)
    partial = {name: data for name, data in FILES.items() if name != "README.md"}
    model_info, download, calls = _fake_hub(partial)
    assert pin_snapshot.pin(root, model_info=model_info, download=download) == 1
    assert calls == []


def test_dry_run_writes_nothing(tmp_path):
    root = _copy_repo(tmp_path)
    before = (root / MANIFEST).read_text()
    model_info, download, _calls = _fake_hub(FILES)
    assert pin_snapshot.pin(root, dry_run=True, model_info=model_info, download=download) == 0
    assert (root / MANIFEST).read_text() == before
