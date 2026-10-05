"""Shared fixtures: an import guard for refusal paths, synthetic pairs, and a tiny random-init Swin2SR x4 snapshot
(the real architecture class with a toy configuration) behind a manifest with a test-only revision."""

# ruff: noqa: E501

from __future__ import annotations

import builtins
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
TEST_REVISION = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture
def forbid_model_imports(monkeypatch):
    """Rejected requests must stop before importing or initializing model libraries."""
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.partition(".")[0] in {"torch", "transformers", "safetensors"}:
            raise AssertionError(f"model dependency imported before rejection: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def textured(size: tuple[int, int], seed: int = 0) -> Image.Image:
    """A seeded RGB image with smooth and sharp structure."""
    width, height = size
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    base = np.stack([128 + 100 * np.sin(xx / 7.0 + seed), 128 + 100 * np.cos(yy / 5.0), (xx * yy) % 256], axis=-1)
    return Image.fromarray(np.clip(base + rng.normal(0, 8, base.shape), 0, 255).astype(np.uint8))


def synthetic_pairs(n: int = 4, hr: int = 64) -> list[dict]:
    from swin2sr_x4_super_resolution_pipeline import degrade

    out = []
    for i in range(n):
        image = textured((hr, hr), seed=i)
        out.append({"id": f"img-{i}", "hr": image, "lr": degrade(image), "category": "a" if i % 2 else "b"})
    return out


def write_tiny_snapshot(root: Path, revision: str = TEST_REVISION) -> Path:
    """Save a tiny random-init Swin2SR x4 model + processor as a manifest-verified snapshot (needs torch + transformers)."""
    import torch
    from transformers import Swin2SRConfig, Swin2SRForImageSuperResolution, Swin2SRImageProcessor

    from swin2sr_x4_super_resolution_pipeline import MANIFEST_NAME, MODEL_ID

    torch.manual_seed(0)
    config = Swin2SRConfig(image_size=16, embed_dim=12, depths=[1], num_heads=[1], window_size=8, upscale=4, upsampler="pixelshuffle", mlp_ratio=2.0)
    root.mkdir(parents=True, exist_ok=True)
    Swin2SRForImageSuperResolution(config).save_pretrained(root, safe_serialization=True)
    Swin2SRImageProcessor(do_rescale=True, do_pad=True, pad_size=8).save_pretrained(root)
    (root / "README.md").write_text("tiny random-init test snapshot\n", encoding="utf-8")
    files = []
    for name in ("README.md", "config.json", "model.safetensors", "preprocessor_config.json"):
        data = (root / name).read_bytes()
        files.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    manifest = {"format": "dimer_hf_snapshot", "formatVersion": 1, "modelKey": "swin2sr-x4-64", "modelId": MODEL_ID, "revision": revision, "files": files, "totalBytes": sum(f["bytes"] for f in files)}
    (root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root


@pytest.fixture
def pinned(monkeypatch):
    """Pretend the package is pinned at TEST_REVISION (the real constant stays 'unpinned' until tools/pin_snapshot.py runs)."""
    import swin2sr_x4_super_resolution_pipeline as package
    import swin2sr_x4_super_resolution_pipeline.pipeline as pipeline

    monkeypatch.setattr(pipeline, "MODEL_REVISION", TEST_REVISION)
    monkeypatch.setattr(package, "MODEL_REVISION", TEST_REVISION)
    return TEST_REVISION
