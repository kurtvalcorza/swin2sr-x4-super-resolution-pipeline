---
license: apache-2.0
model_card_spec: "1.2"
pipeline_tag: image-to-image
task: "Others - Image Super-Resolution"
base_model: caidas/swin2SR-classical-sr-x4-64
date_published: "2022-09"
date_published_source: "Month of the upstream release of the Swin2SR classical-SR weights with the paper (arXiv:2209.11345, submitted September 2022; code and weights at https://github.com/mv-lab/swin2sr). The creation date of the Hugging Face repository that hosts the converted checkpoint was not verified for this card."
---

# Swin2SR classical-sr-x4-64 — Image Super-Resolution ×4 (Task Inference)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-caidas%2Fswin2SR--classical--sr--x4--64-ffcc4d?style=flat)](https://huggingface.co/caidas/swin2SR-classical-sr-x4-64)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-mv--lab%2Fswin2sr-181717?style=flat&logo=github&logoColor=white)](https://github.com/mv-lab/swin2sr)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2209.11345-b31b1b.svg)](https://arxiv.org/abs/2209.11345)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

> [!IMPORTANT]
> **The snapshot is not yet pinned.** `MODEL_REVISION` is `"unpinned"` and the manifest records no SHA-256 for `model.safetensors`, so the package refuses to stage, verify or load the weights until `python tools/pin_snapshot.py` has recorded an immutable commit and every file's digest. No result in this card was produced with the real weights.

---

## Interactive Colab Tutorials

- **Task-inference Guided Notebook**:
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/swin2sr-x4-super-resolution-pipeline/blob/main/tutorials/swin2sr_x4_super_resolution_colab.ipynb) [`swin2sr_x4_super_resolution_colab.ipynb`](https://github.com/kurtvalcorza/swin2sr-x4-super-resolution-pipeline/blob/main/tutorials/swin2sr_x4_super_resolution_colab.ipynb)
  *4× super-resolution of 24 CC0 bird-photo crops shrunk by bicubic downsampling, scored with Y-channel PSNR and SSIM beside bicubic and nearest-neighbour interpolation, reference-free inference on three native photo crops and a seeded synthetic scene, an optional degradation activity, and Bring Your Own Data in a scoring (`hr`) or upscale-only (`lr`) mode. The notebook is standalone and runs every stage in an isolated hash-locked environment.*

---

#### Description

This repository packages `caidas/swin2SR-classical-sr-x4-64`, the Transformers-format release of the Swin2SR classical 4× super-resolution model of Conde et al. (arXiv:2209.11345). Swin2SR is a SwinV2 image transformer. The snapshot `config.json` declares `Swin2SRForImageSuperResolution` with `patch_size` 1, so every input pixel is a token. It has `embed_dim` 180, six residual Swin groups of six layers (`depths` and `num_heads` all 6), `window_size` 8, `mlp_ratio` 2.0, a `1conv` residual connection and a `pixelshuffle` upsampler with `upscale` 4. A model built from that configuration has 12,239,283 parameters (`PARAMETER_COUNT`).

At inference the model maps one RGB image, rescaled to `[0, 1]` and padded to the next multiple of 8 px, to a 4× larger RGB reconstruction in one forward pass. Nothing is trained, fine-tuned or conditioned in this repository.

The repository adds packaging and evaluation code around the upstream weights:

- `verify_snapshot` and `stage_missing_files` check the manifest identity and every file's byte size and SHA-256, and fetch only absent manifest-listed files at the pinned revision.
- `Swin2SRX4Pipeline.from_pretrained` loads only `model.safetensors` from the verified directory with `local_files_only=True` and `trust_remote_code=False`.
- `upscale` validates the input, reports the padding, crops it off at 4× and returns a uint8 array.
- `metrics.py` computes Y-channel PSNR and SSIM with a 4 px border crop, RGB PSNR, bicubic and nearest-neighbour baselines, and a paired bootstrap.
- `samples.py` pins 27 CC0 photographs, builds 4× bicubic pairs, generates a seeded synthetic scene, and loads Bring-Your-Own-Data images in two modes.
- `tools/pin_snapshot.py` records the immutable revision and digests when run where the Hugging Face Hub is reachable.

#### Intended Use and Limitations

The uses below are the ones the package was built to support; everything else is out of scope or prohibited.

###### Primary Intended Uses

The task is 4× single-image super-resolution. The input is one RGB still (`PIL.Image.Image`, any mode, converted to RGB) with sides of 8 to 256 px. The output is a uint8 array of shape `(4H, 4W, 3)` with the input record (`size`, `output_size`, `padding`, `tiling`).

Envisioned applications are enlarging small, clean photographs or thumbnails for display, preparing low-resolution archival or catalogue images for web layouts, and teaching how super-resolution is evaluated against interpolation baselines. The checkpoint is the classical variant, trained on bicubic-downsampled images. Its intended input is therefore a clean image that was downscaled, not a compressed, noisy or blurred one.

The pipeline is meant to be embedded as an inference component in a reader's own application or used as a reference baseline when comparing other upscalers. The evaluation helpers let a reader score it on their own high-resolution originals with the same metric convention.

###### Primary Intended Users

Intended users are machine-learning engineers, imaging researchers, educators and application developers. The envisioned settings are research prototypes, teaching, and self-hosted applications that integrate a fixed-factor upscaler.

A user is expected to understand four things. Super-resolution predicts plausible detail; it does not recover true detail. PSNR and SSIM against a reference measure closeness to that reference, not perceived quality. The model's advantage over bicubic depends on the input having been made by bicubic downscaling. CPU cost grows with input area, which is why inputs are capped at 256 px.

Users who need other scale factors, denoising, JPEG-artefact removal, face restoration or very large images are expected to know that none of these is provided here.

###### Out-of-scope use cases

1. **Capability boundary — scale:** only 4× upscaling. The 2× classical checkpoint is packaged in the public sibling repository `kurtvalcorza/swin2sr-super-resolution-pipeline`. No 3× and no arbitrary scale factors.
2. **Capability boundary — degradation:** no denoising, deblurring, compression-artefact removal or blind restoration. The upstream `compressed-sr` and `realworld-sr` Swin2SR checkpoints are not packaged here. No fine-tuning or adaptation is exposed.
3. **Capability boundary — other modalities:** no face-specific restoration, no text or document restoration, no video and no temporal consistency.
4. **Input boundary — size:** `upscale` raises `ValueError` for a side below `MIN_INPUT_SIDE = 8` px or above `MAX_INPUT_SIDE = 256` px, and `TypeError` for anything that is not a `PIL.Image.Image`. There is no tiling and no silent resizing.
5. **Input boundary — scored pairs:** a reference must have sides that are multiples of 4 within 32 to 1024 px, and its input must be exactly a quarter of its size. In BYOD `hr` mode a reference is cropped on the right and bottom to a multiple of 4 px and the removed pixels are reported.
6. **Input boundary — distribution:** inputs that were not produced by downscaling a sharp image — screenshots, scanned text, medical or scientific images, heavily compressed social-media photos — are outside the training degradation. The pipeline does not detect them and the output on them is undefined.
7. **Decision boundary:** not for forensic enhancement of faces, licence plates or documents presented as recovered evidence, and not for any diagnostic or measurement use where invented pixels could change a decision. No output may feed an autonomous decision without human review.

---

#### Factors

###### Groups

This pipeline is not human-centric. It regresses pixel values and neither classifies nor identifies people. Photographs given to it may still contain faces and bodies.

The upstream paper names DIV2K and Flickr2K as the classical-SR training sets. These are general photo collections that neither the upstream authors nor this repository have audited for skin tone, age, gender or other group balance. Any difference in reconstruction fidelity across such groups is unknown, not known to be absent.

The tutorial sample contains only bird photographs, so it carries no group information at all. An operator who upscales images of people is responsible for a fairness audit on their own data. The audit scores `psnr_y` and `ssim_y` per group on held-out high-resolution originals and compares the strata before the output is relied on; BYOD `hr` mode produces the per-image scores.

###### Instrumentation

The upstream training pairs are synthetic: high-resolution photographs downscaled by bicubic interpolation. The instrument that produced the model's inputs is therefore a resampling kernel, not a camera, and the model learned to invert that kernel. The upstream data used MATLAB-style bicubic resizing; this repository's pairing uses Pillow's bicubic filter, which is similar but not identical.

The tutorial references are centred 320 px crops of iNaturalist photographs served at "medium" size. They carry each camera's own blur, sharpening and JPEG compression before the 4× bicubic downsampling.

At deployment the inputs come from whatever produced the user's small image — a phone camera, a thumbnail generator, a scanner or a video frame. Each differs from bicubic downscaling in blur, noise, sharpening and compression. Those differences reach the model as an unmodelled degradation and can appear as ringing, enlarged artefacts or invented texture. The pipeline validates type and size only; it cannot detect the capture chain. It pads each input on the right and bottom with symmetric reflection (`preprocessor_config.json`: `pad_size` 8) and crops the padding off the output.

###### Environment

**Operating environment.** The package targets Python 3.12 with `torch==2.14.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `huggingface-hub==0.36.2`, `numpy==2.5.3` and `pillow==11.3.0`. Inference runs in float32. CUDA is used automatically when visible; otherwise the CPU is used. The tutorial notebook needs a Linux x86_64 runtime because its hash-locked environment is built from manylinux wheels.

The compute cost of this architecture was measured on a 4-vCPU Linux host in float32 with randomly initialised weights, which have the same cost as trained weights. One forward pass took 3.2 s for an 80 px input, 4.6 s for 128 px, 28.7 s for 256 px and 184.1 s for 512 px. The 512 px measurement is why `MAX_INPUT_SIDE` is 256. No GPU timing has been measured.

**Data environment.** The model assumes a sharp photograph that was downscaled by exactly 4× with bicubic interpolation. Sharper-than-expected inputs can be over-sharpened. Blurrier inputs are not deblurred. Compressed inputs can have their block artefacts enlarged rather than removed. The tutorial's optional activity re-scores the sample under a different kernel or added JPEG compression so a user can observe this effect.

---

#### Metrics

###### Performance Measures

The evaluation path (`sr_metrics`, `bicubic_baseline`, `nearest_baseline`, `compare` in `metrics.py`) reports three measures per image, then means overall and per `category`:

- `psnr_y` — peak signal-to-noise ratio in dB on the BT.601 luma channel after cropping `BORDER = 4` px from every edge. It captures pixel-wise reconstruction error.
- `ssim_y` — structural similarity on the same cropped luma channel with an 11-tap Gaussian window of sigma 1.5. It captures local agreement of brightness, contrast and structure.
- `psnr_rgb` — PSNR over the three RGB channels without a crop, as a secondary check that colour is not degraded.

`compare` adds `model_minus_bicubic`: the mean per-image difference, a seeded percentile bootstrap interval (2,000 resamples), and the number of images the model wins.

These measures were chosen because Y-channel PSNR and SSIM with a scale-sized border crop are the convention under which classical-SR results, including the Swin2SR paper's, are reported. The bicubic baseline is the reference any super-resolution model must beat, and nearest-neighbour is the floor. Reading only PSNR hides structural errors and favours smooth outputs over textured ones. Reading only SSIM hides absolute brightness and colour error. Neither measures perceived quality, and no perceptual or no-reference metric is computed.

On an input with no reference (the tutorial's new inputs, BYOD `lr` mode) nothing is scored and the result is labelled `not-measurable`; a caller who wants a score must supply the high-resolution original.

No value of these measures has been produced with the real weights, because the snapshot is not yet pinned. The upstream paper's Set5, Set14 and Urban100 results are not reproduced or claimed here.

###### Decision thresholds

No decision threshold is applied. `upscale` clamps the reconstruction to `[0, 1]`, scales it to 0–255 and rounds to uint8. That clamp-and-round is the only value-level rule in the code path, and it is not a threshold on any score.

The metrics return numbers without judging them, and the pipeline ships no acceptance rule. The input ceilings (`MIN_INPUT_SIDE = 8`, `MAX_INPUT_SIDE = 256`) are operational limits on compute, not quality thresholds.

A deployment that wants an acceptance rule — for example a minimum `psnr_y` gain over bicubic on its own originals before a model change ships — must set it against its own data and owns revisiting it. It should weigh the cost of invented detail that misleads a viewer (accepting a bad enlargement) against the cost of rejecting usable enlargements, and set the rule stricter where an output could be mistaken for evidence.

###### Approaches to uncertainty and variability

The tutorial estimates each measure on one fixed set of 24 image pairs in one deterministic pass. It reports the mean over images, and for the model-minus-bicubic difference a percentile bootstrap interval over images (2,000 resamples, seed 0) with the win count. The interval describes image-to-image spread within this sample only. It is not a population confidence interval and does not cover other image sources or degradations. No result has been produced with the real weights yet.

The sample construction is deterministic: the photographs are pinned by SHA-256, the crops are centred, the bicubic pairing has no randomness and the synthetic scene uses seed `SYNTHETIC_SEED = 20260926`. Inference has no sampling and no dropout. Remaining run-to-run variability comes from floating-point kernel selection on different CPUs and GPUs and from the uint8 rounding at the output. A fixed input on fixed hardware is repeatable, but bitwise identity across machines is not guaranteed.

The model emits no confidence map and no per-pixel uncertainty. A caller who needs an uncertainty estimate must supply high-resolution references and compute the measures over many images or bootstrap resamples, as `paired_bootstrap` does.

---

#### Ethical considerations and biases

No external ethics board, red-team or population-specific review has examined this repository or, as far as this repository can establish, the upstream checkpoint. Nothing below implies such a review.

###### Data

The snapshot README does not disclose training data. The Swin2SR paper reports training the classical-SR models on DIV2K and Flickr2K, public photo collections whose per-image consent and licence status the paper does not enumerate. Whether personal data such as faces or private property is present is unknown, not ruled out.

This repository distributes code, tests, documentation, the snapshot manifest and the three small upstream text files (`README.md`, `config.json`, `preprocessor_config.json`). It does not distribute the 49,051,724-byte `model.safetensors`, which is downloaded at the pinned revision into the git-ignored `weights/swin2sr-x4-64/`. It never downloads the upstream `pytorch_model.bin` (49,198,181 bytes, a pickle file). It ships no images: the 27 tutorial photographs are CC0 iNaturalist uploads fetched at run time by photo id and verified by size and SHA-256, and each record keeps its observation page.

The operator must audit the images they submit for personal, proprietary or otherwise restricted content and for consent to enhancement; the pipeline performs no such check. The tutorial warns users not to upload confidential data and keeps BYOD files inside the runtime.

###### Human Life

This pipeline is not intended for decisions in health, safety, criminal justice, employment, credit or housing. It has not been validated or certified for any of them by this repository, the upstream authors or any regulator.

Foreseeable but unintended sensitive uses include enlarging medical images before a reading, enhancing surveillance stills for identification, and restoring documents for legal proceedings. Such a use would be admissible only under four conditions. The original low-resolution image is retained and shown alongside the output. Human reviewers treat added detail as synthetic. Domain experts validate the method on the deployment's own data. The domain's regulatory or evidentiary requirements are met.

###### Mitigations

- **Supply-chain integrity:** `stage_missing_files`, `verify_snapshot` and `from_pretrained` raise `UnpinnedSnapshotError` while `MODEL_REVISION` is `"unpinned"`, while the manifest revision is unpinned, or while any listed file lacks a SHA-256. Once pinned, `verify_snapshot` checks every listed file's byte size and SHA-256 before any model library is imported, and refuses a manifest that does not list `model.safetensors`. `from_pretrained` loads with `local_files_only=True`, `trust_remote_code=False` and `use_safetensors=True`. `tools/pin_snapshot.py` refuses to write when a downloaded file disagrees with the Hub's LFS SHA-256 or with a digest already recorded. Tests cover each refusal and assert that refusals happen before `torch` or `transformers` is imported.
- **Input integrity:** `validate_image` rejects non-PIL inputs and sides outside 8 to 256 px with a message naming the rule and the fix. `validate_pairs` rejects duplicate ids, references whose sides are not multiples of 4 or are outside 32 to 1024 px, and inputs that are not exactly a quarter of their reference. `upscale` raises if the backend returns anything other than a `(4H, 4W, 3)` uint8 array. Every input record reports RGB conversion, alpha removal and padding.
- **BYOD and archive safety:** `read_byod_files` reads zips in memory without extracting. It refuses absolute paths, `..` traversal, symbolic links, non-image files and duplicate names. It caps a run at 64 files and 200 MB, and each image at 4096 × 4096 decoded pixels.
- **Reproducibility:** `pyproject.toml` pins every runtime dependency with `==`. The notebook installs the hash-locked, wheels-only `tutorials/requirements-colab.lock.txt` with `--require-hashes` into an isolated environment. Photographs are pinned by SHA-256, and the synthetic scene and the bootstrap are seeded. Every result carries `model_id` and `model_revision`, and the notebook writes a provenance record.
- **Refusals:** no pickle weights, no remote code, no download without `allow_download=True`, no unpinned snapshot, no silent resizing and no tiling of oversized inputs.
- **Statistical mitigations:** none applies; the model is a dense regressor with no classes to balance.

###### Risks and harms

- **Invented detail:** the model synthesises texture that was never in the source. A viewer, the data subject or a third party bears the harm when that detail is read as real, such as a face, a digit or a lesion. This is likely on any input with fine structure, and the harm ranges from cosmetic to a wrong identification.
- **Degradation mismatch:** inputs that are compressed, noisy or sharpened rather than bicubically downscaled can produce halos and enlarged artefacts. The operator bears the harm, and it is likely for web-sourced images.
- **Overconfidence outside the training distribution:** the output looks equally crisp whether or not the input matches the training degradation, and no signal warns the user.
- **Automation bias:** a crisp output invites more trust than a blurry input, so reviewers may stop questioning it.
- **Privacy exposure and data leakage:** images of people or private spaces are processed without any content check. Uploading them to a hosted runtime the operator is not authorised to use exposes the data subject.
- **Bias amplification:** any under-representation in the upstream training photographs is reproduced as lower fidelity for those inputs, undetected because no per-group evaluation exists.
- **Misread evidence:** sample scores on 24 bird crops can be mistaken for a benchmark result or for performance on other image types.
- **Resource exhaustion:** a 256 px input took 28.7 s on a 4-vCPU CPU host and memory grows with area, so a request stream at the ceiling can still exhaust a shared host.

###### Use cases

The following uses are prohibited even where the model would work:

- enhancing images for covert surveillance, biometric identification or demographic profiling, and social scoring;
- presenting upscaled output as authentic evidence or as a faithful record of a document, face or scene;
- any use that discriminates unlawfully in employment, housing, credit, insurance, education or healthcare access;
- deceptive or non-consensual uses, including enhancing intimate or private images or restoring content whose subject has not consented;
- any use that violates the Apache-2.0 licence of the weights, the terms of the deployment that runs the pipeline, or the data-protection obligations attached to the processed images.

## Immutable provenance

- Model: `caidas/swin2SR-classical-sr-x4-64`.
- Revision: not yet pinned (`MODEL_REVISION = "unpinned"`). `python tools/pin_snapshot.py` resolves an immutable commit and records it with every digest.
- Snapshot manifest: `weights/swin2sr-x4-64/dimer-base-manifest.json`, 4 files, `totalBytes` 49,053,290.
- `model.safetensors`: 49,051,724 bytes as reported by the Hub; SHA-256 not yet recorded.
- `config.json`: 772 bytes, SHA-256 `827237812fb18548d2e66dd7c67bb62a9b0c1a88cd79968639b18089f0d6ac2f`.
- `preprocessor_config.json`: 152 bytes, SHA-256 `cbc36266fcc93d5bc1e9ca69bcc648ae9d268918ad14cd3507216740f129cc4d`.
- `README.md`: 642 bytes, SHA-256 `670af06ed69a933293554ee8d19a993912a59044a34d886477f5273dcf926b22`.
- The three text-file digests were computed from the bytes the Hub served for its `main` branch on 2026-10-04; the pin tool refuses to pin a commit at which any of them differs.
- Reference only, never staged or loaded: `pytorch_model.bin`, 49,198,181 bytes, a pickle checkpoint.
- Weight format: SafeTensors only.
- Tutorial sample: 27 iNaturalist photographs (CC0 1.0; six bird species), each pinned by photo id, byte size and SHA-256 in `samples.py` and fetched from `https://inaturalist-open-data.s3.amazonaws.com/photos/<id>/medium.<ext>`. Twenty-four become pairs (centred 320 px reference, 80 px input by Pillow bicubic 4× downsampling); three become native 96 px reference-free inputs.

## Input/output contract

- `Swin2SRX4Pipeline.from_pretrained(device=None, weights_dir=None, allow_download=False)` stages missing manifest files only with `allow_download=True`, verifies the snapshot, then loads; `device` defaults to `cuda:0` when visible, else `cpu`.
- `upscale(image, *, name="image") -> dict` returns `image` (uint8 `(4H, 4W, 3)` RGB), `scale` (4), `input` (the `describe_input` record), `input_size`, `output_size`, `seconds`, `device`, `model_id`, `model_revision`.
- `describe_input(image, *, name)` and `validate_inputs(images, *, names=None)` return the input manifest: mode, RGB conversion, alpha removal, size, output size, `padding` (`right`, `bottom`, `mode` `symmetric`) and `tiling` `none`.
- Ceilings: `MIN_INPUT_SIDE = 8`, `MAX_INPUT_SIDE = 256`, `UPSCALE = 4`, `WINDOW = 8`.
- Metrics: `psnr_y`, `ssim_y`, `psnr_rgb`, `sr_metrics(outputs, records)`, `bicubic_baseline(records)`, `nearest_baseline(records)`, `compare(model, bicubic, nearest, *, seed=0)`, `paired_bootstrap(a, b, *, n_resamples=2000, seed=0)`, `METRIC_DEFINITIONS`, `BORDER = 4`.
- Data: records `{id, hr, lr, category?}`; `degrade(hr, *, kernel="bicubic", jpeg_quality=None)`; `validate_pairs(records)`; `build_sample_pairs(files)`; `build_new_inputs(files)`; `synthetic_scene(*, seed=20260926)`; `fetch_photos(*, cache_dir=None, fetcher=None)`; `prepare_byod(path, mode)` with `mode` `hr` or `lr`; `read_byod_files(path)`.

## Runtime and verification records

Static checks — unit tests, the notebook parity check and `tools/validate_release_assets.py` — run in CI and are not execution evidence. The records below are the executions performed so far.

- **Date:** 2026-10-04
- **Subject:** `tutorials/swin2sr_x4_super_resolution_colab.ipynb` regenerated into a scratch copy from the repository sources with two test substitutions: `MODEL_REVISION` set to a test value, and the manifest replaced by one describing a tiny random-initialised Swin2SR ×4 snapshot (12-dim embedding, one stage) staged beforehand.
- **Runtime:** Linux x86_64, 4 vCPU, no GPU; kernel CPython 3.12.12; isolated environment built by the notebook with CPython 3.12.12, `torch 2.14.0+cu130`, `transformers 4.57.6`.
- **Procedure:** fresh kernel, all cells in order through `nbclient`. One default run, and three runs with form fields set in executed copies: activity with `box` and with `bicubic` + JPEG 30; BYOD `hr` on a folder of three odd-sized images; BYOD `lr` on a zip of two RGBA images; BYOD `lr` on a 300 × 200 image.
- **Observed result:** the default run completed every cell in 74.8 s, including building the isolated environment from the lock. All 27 photographs were fetched from the iNaturalist bucket and verified. Twenty-four 320/80 px pairs passed validation and all five refusal probes were rejected. Evaluation, reference-free inference and the provenance record completed and every listed output was written. The activity and both positive BYOD runs completed; BYOD `hr` reported the pixels cropped to reach a multiple of 4. The 300 × 200 `lr` input stopped the BYOD cell with the stage's `MAX_INPUT_SIDE 256` message.
- **Caveats:** the tiny model's outputs are meaningless, so its scores are not reported as model performance. The real weights were not used because the snapshot is not yet pinned. No GPU and no hosted Colab or Kaggle runtime was used. The upload dialog was not exercised.

- **Date:** 2026-10-04
- **Subject:** forward-pass timing of the architecture declared in `weights/swin2sr-x4-64/config.json` (a short timing script, not the notebook).
- **Runtime:** Linux x86_64, 4 vCPU (Intel Xeon, 2.80 GHz), no GPU; CPython 3.12.12, `torch 2.14.0+cu130`, `transformers 4.57.6`; 4 torch threads.
- **Procedure:** `Swin2SRForImageSuperResolution` built from the pinned `config.json` with random weights (`torch.manual_seed(0)`), float32, one `inference_mode` forward pass per size on a random input padded as the processor pads it; wall time per pass.
- **Observed result:** 12,239,283 parameters; 80 px 3.18 s, 96 px 2.22 s, 128 px 4.59 s, 256 px 28.71 s, 512 px 184.14 s.
- **Caveats:** one pass per size, no warm-up; the 80 px pass was the first and includes start-up cost, which is why it is slower than 96 px. Random weights have the same compute as trained weights but say nothing about output quality. The `MAX_INPUT_SIDE = 256` ceiling was set from these numbers.

## References

- Conde, M. V., Choi, U.-J., Burchi, M., & Timofte, R. (2023). Swin2SR: SwinV2 transformer for compressed image super-resolution and restoration. In *Computer Vision – ECCV 2022 Workshops* (pp. 669–687). https://doi.org/10.1007/978-3-031-25063-7_42 (preprint: https://arxiv.org/abs/2209.11345)
- Liu, Z., et al. (2022). Swin Transformer V2: Scaling up capacity and resolution. *CVPR 2022*. https://doi.org/10.1109/CVPR52688.2022.01170
- Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: From error visibility to structural similarity. *IEEE Transactions on Image Processing, 13*(4), 600–612. https://doi.org/10.1109/TIP.2003.819861
- Upstream code: https://github.com/mv-lab/swin2sr
- Upstream card: https://huggingface.co/caidas/swin2SR-classical-sr-x4-64
- Transformers `Swin2SR` documentation: https://huggingface.co/docs/transformers/model_doc/swin2sr
