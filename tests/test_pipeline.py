"""Offline contract tests for pipeline.py (numpy + Pillow only; no model library, no weights)."""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

import swin2sr_x4_super_resolution_pipeline.pipeline as pl
from swin2sr_x4_super_resolution_pipeline import (
    MANIFEST_NAME,
    MAX_INPUT_SIDE,
    MIN_INPUT_SIDE,
    MODEL_ID,
    UPSCALE,
    UnpinnedSnapshotError,
    describe_input,
    stage_missing_files,
    validate_image,
    validate_inputs,
    verify_snapshot,
    window_padding,
)

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "weights" / "swin2sr-x4-64"


def test_identity_constants():
    assert MODEL_ID == "caidas/swin2SR-classical-sr-x4-64"
    assert UPSCALE == 4 and pl.WINDOW == 8
    assert (MIN_INPUT_SIDE, MAX_INPUT_SIDE) == (8, 256)
    assert pl.MODEL_REVISION == pl.UNPINNED or len(pl.MODEL_REVISION) == 40


def test_committed_manifest_matches_committed_files_and_upstream_sizes():
    manifest = json.loads((SNAPSHOT / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["modelId"] == MODEL_ID and manifest["revision"] == pl.MODEL_REVISION
    entries = {e["path"]: e for e in manifest["files"]}
    assert set(entries) == {"README.md", "config.json", "model.safetensors", "preprocessor_config.json"}
    assert entries["model.safetensors"]["bytes"] == 49_051_724  # Hub-reported size of the SafeTensors weights
    assert manifest["totalBytes"] == sum(e["bytes"] for e in entries.values())
    for name in ("README.md", "config.json", "preprocessor_config.json"):
        data = (SNAPSHOT / name).read_bytes()
        assert len(data) == entries[name]["bytes"] and hashlib.sha256(data).hexdigest() == entries[name]["sha256"], name
    config = json.loads((SNAPSHOT / "config.json").read_text(encoding="utf-8"))
    assert config["upscale"] == UPSCALE and config["window_size"] == pl.WINDOW and config["upsampler"] == "pixelshuffle"
    assert json.loads((SNAPSHOT / "preprocessor_config.json").read_text(encoding="utf-8"))["pad_size"] == pl.WINDOW
    refs = {e["path"] for e in manifest.get("referenceFiles", [])}
    assert refs == {"pytorch_model.bin"}  # recorded for provenance; never staged


@pytest.mark.skipif(pl.MODEL_REVISION != pl.UNPINNED, reason="snapshot already pinned")
def test_unpinned_snapshot_is_refused_everywhere(tmp_path):
    shutil.copytree(SNAPSHOT, tmp_path / "snap")
    with pytest.raises(UnpinnedSnapshotError, match="pin_snapshot"):
        verify_snapshot(tmp_path / "snap")
    with pytest.raises(UnpinnedSnapshotError, match="not pinned yet"):
        stage_missing_files(tmp_path / "snap", allow_download=True, downloader=lambda *_: pytest.fail("must not download"))


def _manifest(root: Path, revision: str, files: dict[str, bytes], *, digest: str | None = "auto") -> None:
    entries = []
    for name, data in files.items():
        (root / name).write_bytes(data)
        entries.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest() if digest == "auto" else digest})
    (root / MANIFEST_NAME).write_text(json.dumps({"modelId": MODEL_ID, "revision": revision, "files": entries, "totalBytes": 0}), encoding="utf-8")


def test_verify_and_stage_with_a_pinned_manifest(tmp_path, pinned):
    _manifest(tmp_path, pinned, {"config.json": b"{}", "model.safetensors": b"\0" * 8})
    assert verify_snapshot(tmp_path)["files"] == ["config.json", "model.safetensors"]
    (tmp_path / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)
    fetched = []
    assert stage_missing_files(tmp_path, allow_download=True, downloader=lambda rel, root: (fetched.append(rel), (root / rel).write_bytes(b"\0" * 8))) == ["model.safetensors"]
    assert fetched == ["model.safetensors"]
    verify_snapshot(tmp_path)


def test_verify_refuses_tampered_bytes_foreign_revision_and_missing_digest(tmp_path, pinned):
    _manifest(tmp_path, pinned, {"config.json": b"{}", "model.safetensors": b"\0" * 8})
    (tmp_path / "model.safetensors").write_bytes(b"\1" * 8)
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(tmp_path)
    _manifest(tmp_path, "f" * 40, {"config.json": b"{}", "model.safetensors": b"\0" * 8})
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(tmp_path)
    _manifest(tmp_path, pinned, {"config.json": b"{}", "model.safetensors": b"\0" * 8}, digest=None)
    with pytest.raises(UnpinnedSnapshotError, match="without a SHA-256"):
        verify_snapshot(tmp_path)


def test_verify_refuses_a_snapshot_without_safetensors(tmp_path, pinned):
    _manifest(tmp_path, pinned, {"config.json": b"{}", "pytorch_model.bin": b"\0" * 8})
    with pytest.raises(ValueError, match="SafeTensors"):
        verify_snapshot(tmp_path)


@pytest.mark.parametrize(("size", "expected"), [((80, 80), (8, 8)), ((100, 76), (4, 4)), ((8, 8), (8, 8)), ((63, 9), (1, 7))])
def test_window_padding_matches_the_processor_rule(size, expected):
    assert window_padding(*size) == expected


def test_validate_image_names_the_rule_and_the_fix():
    with pytest.raises(TypeError, match="PIL.Image.open"):
        validate_image("x.png")
    with pytest.raises(ValueError, match="MIN_INPUT_SIDE"):
        validate_image(Image.new("RGB", (7, 40)))
    with pytest.raises(ValueError, match="never resizes silently"):
        validate_image(Image.new("RGB", (MAX_INPUT_SIDE + 1, 40)))
    assert validate_image(Image.new("L", (MAX_INPUT_SIDE, MIN_INPUT_SIDE))).mode == "RGB"


def test_describe_input_reports_conversion_padding_and_tiling():
    record = describe_input(Image.new("RGBA", (100, 76)), name="a")
    assert record["converted_to_rgb"] and record["alpha_discarded"]
    assert record["output_size"] == [400, 304]
    assert record["padding"] == {"right": 4, "bottom": 4, "mode": "symmetric", "removed_after_upscale": True}
    assert record["tiling"] == "none"


def test_validate_inputs_manifest_and_refusals():
    manifest = validate_inputs([Image.new("RGB", (16, 16)), Image.new("RGB", (24, 8))], names=["a", "b"])
    assert manifest["verdict"] == "accepted" and manifest["n_images"] == 2
    with pytest.raises(ValueError, match="unique"):
        validate_inputs([Image.new("RGB", (16, 16))] * 2, names=["a", "a"])
    with pytest.raises(ValueError, match="at least one"):
        validate_inputs([])
