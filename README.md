# Swin2SR ×4 super-resolution pipeline

DIMER task-inference wrapper for **Swin2SR classical-sr-x4-64** (`caidas/swin2SR-classical-sr-x4-64`), loaded only from a digest-verified local SafeTensors snapshot. The pipeline upscales one RGB image by exactly 4× and returns a uint8 array with a record of the padding it applied; the evaluation helpers score it against bicubic and nearest-neighbour interpolation with Y-channel PSNR and SSIM. It is the classical (bicubic-degradation) variant, not a compression-artefact or real-world restorer, and nothing is fine-tuned.

> **The snapshot is pinned** to commit `c69ef3e2d2ac5777ff4a9f2f5afcd86d20604b7e`. `model.safetensors` (49,051,724 bytes) has SHA-256 `e4e0680f28b663d62a64a3e1884336d0bac5b215da7903c618f41f58f4cc0ff4`, matching the Hub's LFS record; every file is checked by size and SHA-256 before any load. It was pinned on 2026-10-05 by running `python tools/pin_snapshot.py --dry-run` in a Google Colab runtime (the Hub is unreachable from the build environment); the manifest and `MODEL_REVISION` were written from its output exactly as the tool writes them. See `docs/WEIGHTS.md`.

## Upstream alignment

- Model: `caidas/swin2SR-classical-sr-x4-64` (Conde et al., arXiv:2209.11345; code https://github.com/mv-lab/swin2sr)
- Revision: `c69ef3e2d2ac5777ff4a9f2f5afcd86d20604b7e`
- Upstream weight licence: Apache-2.0
- Upstream task: single-image super-resolution, ×4, classical (bicubic) degradation
- Weights used: `model.safetensors` only; the upstream `pytorch_model.bin` (pickle) is never downloaded or loaded

## Quick start

```python
from PIL import Image
from swin2sr_x4_super_resolution_pipeline import Swin2SRX4Pipeline, degrade, psnr_y, interpolate

pipe = Swin2SRX4Pipeline.from_pretrained(allow_download=True)   # after pinning: stages + verifies weights/swin2sr-x4-64
small = Image.open("small.png")                                   # sides 8..256 px, one image per call
result = pipe.upscale(small)
Image.fromarray(result["image"]).save("large.png")                # uint8 (4H, 4W, 3)
print(result["input"]["padding"], result["output_size"])

# score against a sharp original whose sides are multiples of 4: shrink it, upscale, compare with bicubic
reference = Image.open("sharp.png").convert("RGB")
lr = degrade(reference)
print(psnr_y(pipe.upscale(lr)["image"], reference), psnr_y(interpolate(lr, "bicubic"), reference))
```

Install into a Python 3.12 environment with `pip install -e ".[dev]"` (the runtime pins are exact), or install the pins yourself and use `pip install -e . --no-deps`. Run `pytest` for the offline suite: no weights and no network are needed, and the tiny-model tests run when `torch` and `transformers` are installed.

## Package layout

```
src/swin2sr_x4_super_resolution_pipeline/
  pipeline.py   # identity, snapshot verification/staging, input validation (padding, ceilings), Swin2SRX4Pipeline
  metrics.py    # psnr_y, ssim_y (BT.601 luma, 4 px border crop), psnr_rgb, baselines, paired bootstrap
  samples.py    # 27 pinned CC0 photographs, x4 bicubic pairing, synthetic scene, pair validation, BYOD loaders
tools/
  build_notebook.py         # generates the standalone notebook (build_notebook.py/3.0)
  notebook_template.py      # learner-facing prose and cells
  tutorial_stages.py        # the stage runner the notebook carries and runs in its isolated environment
  pin_snapshot.py           # records the immutable revision and digests
  validate_release_assets.py
weights/swin2sr-x4-64/
  dimer-base-manifest.json  # modelId, pinned revision, per-file bytes + SHA-256
  config.json, preprocessor_config.json, README.md   # upstream text files, byte-identical to the Hub's main
  model.safetensors         # git-ignored, 49,051,724 bytes, downloaded after pinning
tutorials/
  swin2sr_x4_super_resolution_colab.ipynb
  requirements-colab.lock.txt   # hash-locked, wheels only, for the notebook's isolated environment
```

## Contract

- **Input:** one `PIL.Image.Image` (any mode; converted to RGB, alpha discarded), sides `MIN_INPUT_SIDE = 8` to `MAX_INPUT_SIDE = 256` px. Larger or smaller inputs are refused with a message naming the rule and the fix; nothing is resized silently and there is no tiling.
- **Preprocessing:** rescale to `[0, 1]`; pad right and bottom with symmetric reflection to the next multiple of 8 px above the input size (an 80 px side gains 8 px); the padding is cropped off at 4×. `describe_input` / `validate_inputs` report it per image.
- **Output:** `upscale(...)["image"]`, uint8 `(4H, 4W, 3)`, plus `input`, `input_size`, `output_size`, `seconds`, `device`, `model_id`, `model_revision`.
- **Scored pairs:** `{id, hr, lr, category?}` with `hr` sides multiples of 4 within 32..1024 px and `lr` exactly `hr / 4` (`validate_pairs`). `degrade(hr)` is the stated degradation: Pillow bicubic downsampling by 4.
- **Metrics:** `psnr_y` and `ssim_y` on the BT.601 luma channel after a 4 px border crop (the classical-SR convention), `psnr_rgb` without a crop; `bicubic_baseline`, `nearest_baseline`; `compare` adds the paired model-minus-bicubic difference with a seeded bootstrap interval and win count.
- **BYOD:** `prepare_byod(path, mode)` reads one image, a folder or a zip (in memory; traversal, symlinks, non-images and duplicates refused). `mode="hr"`: the image is the reference, cropped to a multiple of 4 (reported), shrunk 4× and scored. `mode="lr"`: the image is the input, upscaled only.

## Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/swin2sr-x4-super-resolution-pipeline/blob/main/tutorials/swin2sr_x4_super_resolution_colab.ipynb)

`tutorials/swin2sr_x4_super_resolution_colab.ipynb` is a `TASK-INFERENCE` notebook in `GUIDED` mode under DIMER Notebook Specification 2.2, and it is **standalone**: it carries the package, the stage runner, the hash-locked requirements, the manifest and the licence, verified against their SHA-256, so it runs without this repository. **Nothing is installed into the notebook kernel.** A pinned `uv` wheel (size and SHA-256 checked) builds an isolated CPython 3.12.12 environment from `tutorials/requirements-colab.lock.txt` with `--require-hashes --only-binary :all:`, and each stage runs there as its own process with Hugging Face tokens removed and `MPLBACKEND=Agg`, so the hosted runtime's packages are never replaced and no restart is needed (RUN10, ENV6). It runs on Linux x86_64 only; a T4 GPU is recommended and the CPU also works.

The default path stages and verifies the snapshot, fetches 27 digest-pinned CC0 iNaturalist photographs, builds 24 bicubic ×4 pairs (320 px reference, 80 px input) and validates them with refusal probes, scores the model beside bicubic and nearest-neighbour interpolation, upscales three native 96 px crops and a seeded synthetic scene that have no reference, and writes the input manifest, scores, panels, output PNGs, predictions and a provenance record under `outputs/`. An optional activity changes the degradation (box, bilinear or nearest kernel, or added JPEG), and an optional BYOD branch runs your images in `hr` or `lr` mode via an upload or a `BYOD_PATH` field. See `tutorials/README.md`.

## Release status

**Candidate** — initial development. The snapshot is pinned, and a Google Colab T4 `Run all` with the real weights completed on 2026-10-05 (Y-channel PSNR 30.84 dB against 28.15 dB for bicubic on 24 images). The hosted BYOD journey is still open. A CPU pre-flight of the whole notebook against a tiny random-initialised snapshot is recorded in `docs/release-verification.md`; it is not release evidence. See `STATUS.md` for the remaining steps.

## Documentation

- `MODEL_CARD.md` — MODEL_CARD_SPEC 1.2 card: provenance, contract, metrics, limits, verification records.
- `docs/WEIGHTS.md` — weight provenance, pin procedure and hosting notes.
- `docs/release-verification.md` — coverage, procedure and recorded executions.
- `STATUS.md` — release status.

## Licensing

This repository's code is Apache-2.0 (see `LICENSE`). The upstream weights are Apache-2.0; see `docs/WEIGHTS.md` and `MODEL_CARD.md`. The tutorial photographs are CC0 1.0 and are fetched at run time, not distributed here.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
