"""4x single-image super-resolution with the ``caidas/swin2SR-classical-sr-x4-64`` checkpoint.

The pipeline loads weights only from a digest-verified local snapshot (``weights/<MODEL_KEY>/``) through the
Transformers ``Swin2SRForImageSuperResolution`` and ``Swin2SRImageProcessor`` classes with
``trust_remote_code=False`` and ``local_files_only=True``. Only ``model.safetensors`` is loaded; the upstream
``pytorch_model.bin`` (a pickle checkpoint) is never staged or deserialised.

Until ``tools/pin_snapshot.py`` has recorded an immutable revision and every file's SHA-256 in the manifest,
``MODEL_REVISION`` is ``"unpinned"`` and the package refuses to stage, verify or load weights: an unpinned
snapshot is never trusted and there is no fallback to a mutable branch.
"""

from __future__ import annotations

# ruff: noqa: E501  -- refusal messages name the condition and the corrective action on one line
import hashlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

MODEL_ID = "caidas/swin2SR-classical-sr-x4-64"
MODEL_REVISION = "unpinned"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "swin2sr-x4-64"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"
UNPINNED = "unpinned"
PIN_COMMAND = "python tools/pin_snapshot.py"
WEIGHT_FILE = "model.safetensors"

UPSCALE = 4  # config.json "upscale": 4
WINDOW = 8  # config.json "window_size": 8; preprocessor_config.json "pad_size": 8
PARAMETER_COUNT = 12_239_283  # counted from a model built from the pinned config.json (tests check it)
# Input ceilings. Swin2SR runs windowed attention over every input pixel (patch_size 1), so time and activation
# memory grow with the input area. Measured for this architecture on a 4-vCPU CPU host (float32, one forward pass):
# 80 px 3.2 s, 128 px 4.6 s, 256 px 28.7 s, 512 px 184.1 s. The 256 px ceiling (1024 px output) keeps one call under
# about a minute on a 2-vCPU hosted CPU runtime; see MODEL_CARD.md, Runtime.
MIN_INPUT_SIDE = 8
MAX_INPUT_SIDE = 256
PAD_MODE = "symmetric"  # Swin2SRImageProcessor.pad: bottom/right, numpy "symmetric"
TILING = "none"  # the whole image runs in one forward pass; the ceiling is enforced instead of tiling


class UnpinnedSnapshotError(RuntimeError):
    """The package or a manifest carries no immutable revision or no per-file digest."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        return json.load(fh)


def _require_pinned(manifest: dict[str, Any]) -> None:
    """Refuse an unpinned package constant, an unpinned manifest or a manifest entry without a digest."""
    if MODEL_REVISION == UNPINNED or manifest.get("revision") in (None, UNPINNED):
        raise UnpinnedSnapshotError(
            f"{MODEL_ID} is not pinned yet (MODEL_REVISION={MODEL_REVISION!r}, manifest revision={manifest.get('revision')!r}): "
            f"run `{PIN_COMMAND}` on a machine that can reach huggingface.co, commit the manifest, then regenerate the notebook"
        )
    unhashed = [entry["path"] for entry in manifest.get("files", []) if not entry.get("sha256")]
    if unhashed:
        raise UnpinnedSnapshotError(f"manifest lists files without a SHA-256: {unhashed}; run `{PIN_COMMAND}`")


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local snapshot against its DIMER manifest; raise naming the first mismatch."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = _read_manifest(root)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    _require_pinned(manifest)
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest["files"]:
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    if not any(entry["path"] == WEIGHT_FILE for entry in manifest["files"]):
        raise ValueError(f"manifest does not list {WEIGHT_FILE}; only SafeTensors weights are loaded")
    return {
        "path": str(root),
        "model_id": manifest["modelId"],
        "revision": manifest["revision"],
        "files": [entry["path"] for entry in manifest["files"]],
        "total_bytes": manifest.get("totalBytes"),
    }


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but git-ignores the
    weights). Returns the relative paths fetched; `verify_snapshot` still has to run afterwards."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = _read_manifest(root)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest names {manifest.get('modelId')}, package pins {MODEL_ID}; refusing to stage")
    _require_pinned(manifest)
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest names {MODEL_ID}@{manifest.get('revision')}, package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage")
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(f"snapshot at {root} is missing {missing}; pass allow_download=True to fetch them at {MODEL_REVISION}")
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


# ---------------------------------------------------------------------------------------------------------
# Input contract
# ---------------------------------------------------------------------------------------------------------


def window_padding(width: int, height: int) -> tuple[int, int]:
    """Pixels the image processor adds on the right and at the bottom: it always pads up to the next multiple of
    the 8 px window *above* the current size, so a side that is already a multiple of 8 still gains 8 px."""
    return (width // WINDOW + 1) * WINDOW - width, (height // WINDOW + 1) * WINDOW - height


def validate_image(image: Any, *, name: str = "image") -> Image.Image:
    """Type- and size-check one low-resolution input and return it as RGB. The message names the failed rule and the fix."""
    if not isinstance(image, Image.Image):
        raise TypeError(f"{name}: expected a PIL.Image.Image, got {type(image).__name__}; open the file with PIL.Image.open first")
    width, height = image.size
    if min(width, height) < MIN_INPUT_SIDE:
        raise ValueError(f"{name}: shorter side {min(width, height)} px < MIN_INPUT_SIDE {MIN_INPUT_SIDE}; supply an image of at least {MIN_INPUT_SIDE} x {MIN_INPUT_SIDE} px")
    if max(width, height) > MAX_INPUT_SIDE:
        raise ValueError(
            f"{name}: longer side {max(width, height)} px > MAX_INPUT_SIDE {MAX_INPUT_SIDE} (the output would be {max(width, height) * UPSCALE} px); "
            f"crop or downscale the input to at most {MAX_INPUT_SIDE} px first — the pipeline never resizes silently"
        )
    return image.convert("RGB")


def describe_input(image: Image.Image, *, name: str = "image") -> dict[str, Any]:
    """The per-input record of the validation stage: what is accepted, what preprocessing will change, what comes out."""
    rgb = validate_image(image, name=name)
    pad_right, pad_bottom = window_padding(rgb.width, rgb.height)
    return {
        "id": name,
        "mode": image.mode,
        "converted_to_rgb": image.mode != "RGB",
        "alpha_discarded": "A" in image.getbands(),
        "size": [rgb.width, rgb.height],
        "output_size": [rgb.width * UPSCALE, rgb.height * UPSCALE],
        "padding": {"right": pad_right, "bottom": pad_bottom, "mode": PAD_MODE, "removed_after_upscale": True},
        "tiling": TILING,
    }


INPUT_SCHEMA: dict[str, Any] = {
    "input": "one PIL.Image.Image per call (any mode; converted to RGB, alpha discarded)",
    "image_side_px": [MIN_INPUT_SIDE, MAX_INPUT_SIDE],
    "scale": UPSCALE,
    "output": f"uint8 RGB array of shape ({UPSCALE}H, {UPSCALE}W, 3)",
    "preprocessing": (
        f"rescale to [0, 1]; pad bottom/right with {PAD_MODE} reflection up to the next multiple of the {WINDOW} px "
        f"attention window above the input size; the padding is cropped off at output scale ({UPSCALE}x)"
    ),
    "tiling": "none: the whole image runs in one forward pass, so the side ceiling is enforced instead",
}


def validate_inputs(images: Any, *, names: Sequence[str] | None = None) -> dict[str, Any]:
    """Validation stage: the input manifest (schema, per-input observations, verdict). Raises on the first invalid
    input with the same message `upscale` would raise."""
    batch = [images] if isinstance(images, Image.Image) else images
    if not isinstance(batch, Sequence) or isinstance(batch, str | bytes):
        raise TypeError("images must be a PIL.Image.Image or a sequence of them")
    if len(batch) < 1:
        raise ValueError("at least one image is required")
    if names is not None and len(names) != len(batch):
        raise ValueError("names must have one entry per image")
    if names is not None and len(set(names)) != len(names):
        raise ValueError("names must be unique; duplicate ids would make outputs ambiguous")
    inputs = [describe_input(image, name=names[i] if names else f"image-{i}") for i, image in enumerate(batch)]
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": inputs,
        "n_images": len(inputs),
        "verdict": "accepted",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


# ---------------------------------------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------------------------------------


@dataclass
class Swin2SRX4Pipeline:
    """4x single-image super-resolution over the pinned Swin2SR classical-SR checkpoint."""

    _runner: Callable[[Image.Image], np.ndarray]
    device: str
    _model: Any = field(default=None, repr=False)
    weight_sha256: str | None = None

    @classmethod
    def from_pretrained(cls, device: str | None = None, weights_dir: str | Path | None = None, allow_download: bool = False) -> Swin2SRX4Pipeline:
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        if not (root / MANIFEST_NAME).is_file():
            raise FileNotFoundError(f"no snapshot manifest at {root}; stage {MODEL_ID} under weights/{MODEL_KEY} first")
        stage_missing_files(root, allow_download=allow_download)
        verify_snapshot(root)  # refuse invalid snapshots before importing model libraries
        entries = _read_manifest(root)["files"]
        weight_sha256 = next(e["sha256"] for e in entries if e["path"] == WEIGHT_FILE)

        import torch
        from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor

        resolved = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        kwargs = {"local_files_only": True, "trust_remote_code": False}
        processor = Swin2SRImageProcessor.from_pretrained(str(root), **kwargs)
        model = Swin2SRForImageSuperResolution.from_pretrained(str(root), use_safetensors=True, **kwargs)
        return cls.from_components(model, processor, resolved, weight_sha256=weight_sha256)

    @classmethod
    def from_components(cls, model: Any, processor: Any, device: str = "cpu", *, weight_sha256: str | None = None) -> Swin2SRX4Pipeline:
        """Wrap an already-built model and processor (the loader above, or a test's tiny random-init model)."""
        import torch

        model = model.to(device).eval()
        for param in model.parameters():
            param.requires_grad_(False)
        if int(model.config.upscale) != UPSCALE:
            raise ValueError(f"model upscale {model.config.upscale} != {UPSCALE}")

        def runner(image: Image.Image) -> np.ndarray:
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.inference_mode():
                reconstruction = model(**inputs).reconstruction
            out = reconstruction[0, :, : image.height * UPSCALE, : image.width * UPSCALE]  # crop the window padding
            out = out.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
            return out.permute(1, 2, 0).cpu().numpy()

        return cls(runner, device, model, weight_sha256)

    def upscale(self, image: Image.Image, *, name: str = "image") -> dict[str, Any]:
        """Return the 4x-upscaled RGB image as a uint8 array of shape (4H, 4W, 3) with the input record."""
        record = describe_input(image, name=name)
        rgb = image.convert("RGB")
        expected = (rgb.height * UPSCALE, rgb.width * UPSCALE, 3)
        started = time.perf_counter()
        result = np.asarray(self._runner(rgb))
        seconds = time.perf_counter() - started
        if result.shape != expected or result.dtype != np.uint8:
            raise RuntimeError(f"backend returned {result.shape} {result.dtype}, expected {expected} uint8")
        return {
            "image": result,
            "scale": UPSCALE,
            "input": record,
            "input_size": tuple(record["size"]),
            "output_size": tuple(record["output_size"]),
            "seconds": round(seconds, 4),
            "device": self.device,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
