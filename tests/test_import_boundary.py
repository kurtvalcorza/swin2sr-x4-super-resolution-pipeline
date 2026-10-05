"""Import-boundary contract: rejected requests never import model libraries; a valid snapshot still reaches them."""

from __future__ import annotations

import hashlib
import json

import pytest
from PIL import Image

from swin2sr_x4_super_resolution_pipeline import (
    MANIFEST_NAME,
    MODEL_ID,
    Swin2SRX4Pipeline,
    UnpinnedSnapshotError,
    prepare_byod,
    validate_inputs,
    validate_pairs,
)

_CONFIG = b'{"model_type": "swin2sr"}'


def _snapshot(root, revision, tamper=False):
    entries = []
    for name, data in (("config.json", _CONFIG), ("model.safetensors", b"\0" * 8)):
        (root / name).write_bytes(data)
        entries.append({"path": name, "bytes": len(data), "sha256": "0" * 64 if tamper else hashlib.sha256(data).hexdigest()})
    (root / MANIFEST_NAME).write_text(json.dumps({"modelId": MODEL_ID, "revision": revision, "files": entries}), encoding="utf-8")


def test_missing_snapshot_is_refused_before_model_imports(tmp_path, forbid_model_imports):
    with pytest.raises(FileNotFoundError, match="no snapshot manifest"):
        Swin2SRX4Pipeline.from_pretrained(device="cpu", weights_dir=tmp_path)


def test_unpinned_snapshot_is_refused_before_model_imports(tmp_path, forbid_model_imports):
    _snapshot(tmp_path, "unpinned")
    with pytest.raises(UnpinnedSnapshotError):
        Swin2SRX4Pipeline.from_pretrained(device="cpu", weights_dir=tmp_path)


def test_tampered_snapshot_is_refused_before_model_imports(tmp_path, pinned, forbid_model_imports):
    _snapshot(tmp_path, pinned, tamper=True)
    with pytest.raises(ValueError, match="sha256"):
        Swin2SRX4Pipeline.from_pretrained(device="cpu", weights_dir=tmp_path)


def test_valid_snapshot_reaches_model_import(tmp_path, pinned, forbid_model_imports):
    _snapshot(tmp_path, pinned)
    with pytest.raises(AssertionError, match="model dependency imported before rejection"):
        Swin2SRX4Pipeline.from_pretrained(device="cpu", weights_dir=tmp_path)


def test_validation_and_byod_refusals_import_no_model_library(tmp_path, forbid_model_imports):
    with pytest.raises(ValueError):
        validate_inputs(Image.new("RGB", (300, 300)))
    with pytest.raises(ValueError):
        validate_pairs([{"id": "a", "hr": Image.new("RGB", (64, 64)), "lr": Image.new("RGB", (32, 32))}])
    Image.new("RGB", (300, 30)).save(tmp_path / "x.png")
    with pytest.raises(ValueError):
        prepare_byod(tmp_path / "x.png", "lr")
