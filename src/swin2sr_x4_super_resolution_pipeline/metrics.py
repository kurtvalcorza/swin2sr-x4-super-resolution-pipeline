"""Super-resolution measures and two non-neural baselines, in numpy.

The principal measures follow the classical-SR evaluation convention the Swin2SR authors use: PSNR and SSIM on the
BT.601 luma (Y) channel, with ``BORDER = UPSCALE`` (4) pixels cropped from every edge of both images before scoring,
because the outermost pixels of an upscaled image depend on the padding rule rather than on the model. RGB PSNR
without a crop is reported beside them as a secondary measure. SSIM uses an 11-tap Gaussian window of sigma 1.5.

The baselines upscale the same low-resolution input 4x with bicubic or nearest-neighbour interpolation (Pillow) and
are scored by the same function, so every number in a comparison shares one definition.
"""
# ruff: noqa: E501  -- definition strings are kept on one line each

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from PIL import Image

from .pipeline import UPSCALE

BORDER = UPSCALE  # pixels cropped from each edge before the Y-channel measures
METRIC_DEFINITIONS = {
    "psnr_y": f"peak signal-to-noise ratio in dB on the BT.601 luma channel (16..235 scale, 255 peak) after cropping {BORDER} px from every edge; higher is better, inf for identical images",
    "ssim_y": f"structural similarity on the BT.601 luma channel after cropping {BORDER} px from every edge (11-tap Gaussian window, sigma 1.5); in -1..1, higher is better",
    "psnr_rgb": "peak signal-to-noise ratio in dB over the three RGB channels, no border crop; secondary measure",
}
_GAUSS = np.exp(-((np.arange(11) - 5) ** 2) / (2 * 1.5**2))
_WINDOW = np.outer(_GAUSS, _GAUSS) / np.outer(_GAUSS, _GAUSS).sum()
INTERPOLATIONS = {"bicubic": Image.Resampling.BICUBIC, "nearest": Image.Resampling.NEAREST}


def _check_pair(pred: Any, ref: Any) -> tuple[np.ndarray, np.ndarray]:
    pred, ref = np.asarray(pred), np.asarray(ref)
    if pred.shape != ref.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs ref {ref.shape}")
    if pred.dtype != np.uint8 or ref.dtype != np.uint8:
        raise TypeError("metrics expect uint8 arrays")
    if pred.ndim != 3 or pred.shape[2] != 3:
        raise ValueError(f"metrics expect RGB arrays of shape (H, W, 3), got {pred.shape}")
    return pred, ref


def luma(rgb: np.ndarray) -> np.ndarray:
    """BT.601 luma (16..235) of a uint8 RGB array, as float64 — the Y channel of the classical-SR convention."""
    arr = np.asarray(rgb, dtype=np.float64)
    return 16.0 + (65.481 * arr[..., 0] + 128.553 * arr[..., 1] + 24.966 * arr[..., 2]) / 255.0


def _crop(arr: np.ndarray, border: int) -> np.ndarray:
    if border <= 0:
        return arr
    if min(arr.shape[:2]) <= 2 * border:
        raise ValueError(f"image {arr.shape[:2]} is too small for a {border} px border crop")
    return arr[border:-border, border:-border]


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0.0 else float(10.0 * np.log10(255.0**2 / mse))


def psnr_rgb(pred: Any, ref: Any) -> float:
    """PSNR in dB over all RGB channels, no crop (see METRIC_DEFINITIONS)."""
    pred, ref = _check_pair(pred, ref)
    return _psnr(pred, ref)


def psnr_y(pred: Any, ref: Any, *, border: int = BORDER) -> float:
    """PSNR in dB on the luma channel after a `border` px crop (see METRIC_DEFINITIONS)."""
    pred, ref = _check_pair(pred, ref)
    return _psnr(_crop(luma(pred), border), _crop(luma(ref), border))


def _filter(z: np.ndarray) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view

    return np.einsum("ijkl,kl->ij", sliding_window_view(z, (11, 11)), _WINDOW)


def ssim_y(pred: Any, ref: Any, *, border: int = BORDER) -> float:
    """SSIM on the luma channel after a `border` px crop (see METRIC_DEFINITIONS)."""
    pred, ref = _check_pair(pred, ref)
    x, y = _crop(luma(pred), border), _crop(luma(ref), border)
    if min(x.shape) < 11:
        raise ValueError(f"ssim needs at least 11 px per side after the {border} px crop, got {x.shape}")
    mu_x, mu_y = _filter(x), _filter(y)
    sxx, syy, sxy = _filter(x * x) - mu_x**2, _filter(y * y) - mu_y**2, _filter(x * y) - mu_x * mu_y
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sxx + syy + c2))
    return float(s.mean())


def score_pair(pred: Any, ref: Any) -> dict[str, float]:
    """All three measures for one reconstruction."""
    return {"psnr_y": psnr_y(pred, ref), "ssim_y": ssim_y(pred, ref), "psnr_rgb": psnr_rgb(pred, ref)}


def _mean(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(rows)}
    for key in METRIC_DEFINITIONS:
        values = [r[key] for r in rows if np.isfinite(r[key])]
        out[key] = round(float(np.mean(values)), 4) if values else float("inf")
    return out


def sr_metrics(outputs: Sequence[np.ndarray], records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Score one uint8 RGB reconstruction per record against `record['hr']`: per-record rows (keeping each record's
    id and category), means overall and per category, and the metric definitions."""
    if len(outputs) != len(records) or not outputs:
        raise ValueError("outputs and records must be non-empty and the same length")
    rows = []
    for output, record in zip(outputs, records, strict=True):
        ref = np.asarray(record["hr"].convert("RGB"))
        rows.append({"id": record["id"], "category": record.get("category"), **{k: round(v, 4) for k, v in score_pair(output, ref).items()}})
    categories = sorted({r["category"] for r in rows if r["category"] is not None})
    return {
        **_mean(rows),
        "per_category": {c: _mean([r for r in rows if r["category"] == c]) for c in categories},
        "per_record": rows,
        "border_px": BORDER,
        "definitions": dict(METRIC_DEFINITIONS),
    }


def interpolate(lr: Image.Image, method: str) -> np.ndarray:
    """The low-resolution input upscaled 4x by `method` ('bicubic' or 'nearest') as a uint8 RGB array."""
    if method not in INTERPOLATIONS:
        raise ValueError(f"method must be one of {sorted(INTERPOLATIONS)}")
    rgb = lr.convert("RGB")
    return np.asarray(rgb.resize((rgb.width * UPSCALE, rgb.height * UPSCALE), INTERPOLATIONS[method]), dtype=np.uint8)


def bicubic_baseline(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The classical reference: each low-resolution input upscaled 4x with bicubic interpolation."""
    return {**sr_metrics([interpolate(r["lr"], "bicubic") for r in records], records), "baseline": "bicubic 4x interpolation of the LR input (Pillow)"}


def nearest_baseline(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The floor: each low-resolution input upscaled 4x by pixel replication."""
    return {**sr_metrics([interpolate(r["lr"], "nearest") for r in records], records), "baseline": "nearest-neighbour 4x interpolation of the LR input (Pillow)"}


def paired_bootstrap(a: Sequence[float], b: Sequence[float], *, n_resamples: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Mean of the per-image differences ``a - b`` and a percentile 95 % interval from a seeded bootstrap over images.
    It describes the spread over *these* images only; it is not a population confidence interval."""
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("paired_bootstrap needs two equal-length sequences of at least two values")
    diffs = [float(x) - float(y) for x, y in zip(a, b, strict=True)]
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples))
    return {
        "mean_difference": round(sum(diffs) / n, 4),
        "ci95": [round(means[int(0.025 * n_resamples)], 4), round(means[int(0.975 * n_resamples) - 1], 4)],
        "n_images": n,
        "n_resamples": n_resamples,
        "seed": seed,
        "wins": sum(d > 0 for d in diffs),
        "method": "percentile bootstrap over images (resampling the per-image differences with replacement)",
    }


def compare(model: Mapping[str, Any], bicubic: Mapping[str, Any], nearest: Mapping[str, Any], *, seed: int = 0) -> dict[str, Any]:
    """The comparison table: the three means per measure, and the model's paired difference to bicubic."""
    table = {key: {"nearest": nearest[key], "bicubic": bicubic[key], "model": model[key]} for key in METRIC_DEFINITIONS}
    model_rows = {r["id"]: r for r in model["per_record"]}
    bicubic_rows = {r["id"]: r for r in bicubic["per_record"]}
    ids = list(model_rows)
    return {
        "means": table,
        "model_minus_bicubic": {
            key: paired_bootstrap([model_rows[i][key] for i in ids], [bicubic_rows[i][key] for i in ids], seed=seed) for key in ("psnr_y", "ssim_y")
        },
        "n_images": len(ids),
    }
