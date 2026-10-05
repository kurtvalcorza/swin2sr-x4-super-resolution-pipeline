"""Model-backed checks on CPU with a tiny random-init Swin2SR x4 (the real Transformers classes, a toy config).

Skipped when torch or transformers is not installed. No real weights are needed: these tests check the loading,
padding, cropping and output contract, not reconstruction quality.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from conftest import synthetic_pairs, textured, write_tiny_snapshot  # noqa: E402
from swin2sr_x4_super_resolution_pipeline import (  # noqa: E402
    PARAMETER_COUNT,
    Swin2SRX4Pipeline,
    sr_metrics,
    window_padding,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tiny(tmp_path, pinned):
    return Swin2SRX4Pipeline.from_pretrained(device="cpu", weights_dir=write_tiny_snapshot(tmp_path / "snap"))


def test_parameter_count_of_the_pinned_architecture():
    config = transformers.Swin2SRConfig.from_json_file(str(ROOT / "weights" / "swin2sr-x4-64" / "config.json"))
    with torch.device("meta"):
        model = transformers.Swin2SRForImageSuperResolution(config)
    assert sum(p.numel() for p in model.parameters()) == PARAMETER_COUNT


@pytest.mark.parametrize("size", [(8, 8), (80, 80), (100, 76), (37, 21)])
def test_upscale_contract(tiny, size):
    result = tiny.upscale(textured(size), name="x")
    assert result["image"].shape == (size[1] * 4, size[0] * 4, 3) and result["image"].dtype == np.uint8
    assert (result["input"]["padding"]["right"], result["input"]["padding"]["bottom"]) == window_padding(*size)


def test_processor_padding_equals_the_reported_padding(tmp_path, pinned):
    snap = write_tiny_snapshot(tmp_path / "snap")
    processor = transformers.Swin2SRImageProcessor.from_pretrained(str(snap))
    for size in [(80, 80), (100, 76), (8, 8)]:
        pixel_values = processor(images=textured(size), return_tensors="pt")["pixel_values"]
        right, bottom = window_padding(*size)
        assert tuple(pixel_values.shape[-2:]) == (size[1] + bottom, size[0] + right)


def test_upscale_is_deterministic_and_rgb_converted(tiny):
    image = textured((40, 32)).convert("RGBA")
    first, second = tiny.upscale(image)["image"], tiny.upscale(image)["image"]
    assert np.array_equal(first, second)


def test_evaluation_runs_end_to_end(tiny):
    records = synthetic_pairs(3)
    outputs = [tiny.upscale(r["lr"])["image"] for r in records]
    metrics = sr_metrics(outputs, records)
    assert metrics["n"] == 3 and np.isfinite(metrics["psnr_y"])


def test_from_pretrained_loads_only_safetensors(tmp_path, pinned):
    snap = write_tiny_snapshot(tmp_path / "snap")
    manifest = json.loads((snap / "dimer-base-manifest.json").read_text())
    assert "model.safetensors" in {f["path"] for f in manifest["files"]}
    assert not list(snap.glob("*.bin"))
    Swin2SRX4Pipeline.from_pretrained(device="cpu", weights_dir=snap)
