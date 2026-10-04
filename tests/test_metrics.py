"""Metric definitions: Y-channel PSNR/SSIM with the 4 px border crop, RGB PSNR, baselines, the paired bootstrap."""

from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import synthetic_pairs, textured
from swin2sr_x4_super_resolution_pipeline import (
    BORDER,
    bicubic_baseline,
    compare,
    interpolate,
    luma,
    nearest_baseline,
    paired_bootstrap,
    psnr_rgb,
    psnr_y,
    sr_metrics,
    ssim_y,
)


def test_border_is_the_scale_factor():
    assert BORDER == 4


def test_luma_bt601_range():
    assert luma(np.zeros((1, 1, 3), np.uint8))[0, 0] == pytest.approx(16.0)
    assert luma(np.full((1, 1, 3), 255, np.uint8))[0, 0] == pytest.approx(235.0)


def test_identical_images_score_perfectly():
    a = np.asarray(textured((40, 40)))
    assert psnr_y(a, a) == math.inf and psnr_rgb(a, a) == math.inf
    assert ssim_y(a, a) == pytest.approx(1.0)


def test_known_psnr_value_and_border_crop():
    a = np.zeros((32, 32, 3), np.uint8)
    b = a.copy()
    b[:] = 10
    assert psnr_rgb(a, b) == pytest.approx(10 * math.log10(255**2 / 100))
    edge = a.copy()
    edge[:BORDER] = 255  # damage only the top border rows
    assert psnr_y(edge, a) == math.inf  # cropped away
    assert psnr_y(edge, a, border=0) < 30


def test_metric_input_checks():
    a = np.zeros((16, 16, 3), np.uint8)
    with pytest.raises(ValueError, match="shape mismatch"):
        psnr_y(a, np.zeros((16, 17, 3), np.uint8))
    with pytest.raises(TypeError, match="uint8"):
        psnr_y(a.astype(np.float32), a.astype(np.float32))
    with pytest.raises(ValueError, match="11 px"):
        ssim_y(a, a)  # 16 - 2*4 = 8 < 11


def test_baselines_and_comparison_shapes():
    records = synthetic_pairs(4)
    assert interpolate(records[0]["lr"], "bicubic").shape == (64, 64, 3)
    with pytest.raises(ValueError):
        interpolate(records[0]["lr"], "lanczos")
    bicubic, nearest = bicubic_baseline(records), nearest_baseline(records)
    assert bicubic["psnr_y"] > nearest["psnr_y"]  # bicubic beats pixel replication on smooth content
    perfect = sr_metrics([np.asarray(r["hr"]) for r in records], records)  # an oracle "model"
    table = compare(perfect, bicubic, nearest)
    assert table["means"]["psnr_y"]["model"] == math.inf
    assert set(perfect["per_category"]) == {"a", "b"}
    assert [r["id"] for r in perfect["per_record"]] == [r["id"] for r in records]


def test_paired_bootstrap_is_seeded_and_brackets_the_mean():
    a, b = [30.0, 31.0, 29.5, 32.0, 30.5], [29.0, 29.5, 29.0, 30.0, 29.5]
    one, two = paired_bootstrap(a, b, seed=0), paired_bootstrap(a, b, seed=0)
    assert one == two
    assert one["ci95"][0] <= one["mean_difference"] <= one["ci95"][1]
    assert one["wins"] == 5
    with pytest.raises(ValueError):
        paired_bootstrap([1.0], [1.0])
