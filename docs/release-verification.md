# Release verification

The tutorial notebook `tutorials/swin2sr_x4_super_resolution_colab.ipynb` is promoted to Release-grade only by a recorded clean-runtime **Run all** of the exact committed notebook blob (NOTEBOOK_SPEC 2.2 §24). Static and unit checks are necessary but are not execution evidence (REL8).

## Coverage: automatic versus manual (REL9)

| Check | Where | Automatic? |
|---|---|---|
| Lint, unit tests, tiny-model CPU tests of the loader, padding, metrics, every stage in order and both BYOD modes | CI (`.github/workflows/ci.yml`) | yes |
| Notebook parity with the repository (carried files, hashes, lock, generator output), cleared outputs, no kernel `pip install` | CI (`tests/test_notebook_parity.py`, `python tools/build_notebook.py --check`) | yes |
| Static release-asset validation (card, registry, status, guided layer, infrastructure titles, forbidden patterns) | CI (`python tools/validate_release_assets.py`) | yes |
| Real weights: staging, digest verification, reconstruction quality | manual — CI cannot reach the Hugging Face Hub (snapshot pinned 2026-10-05) | no |
| Clean hosted Run all, one pass, no restart; BYOD positive and negative (REL12) | manual — Colab T4 | no |

## Procedure for the hosted run

1. Pin the snapshot (`python tools/pin_snapshot.py`), regenerate the notebook, commit, and note the notebook blob (`git rev-parse HEAD:tutorials/swin2sr_x4_super_resolution_colab.ipynb`).
2. Open the notebook from GitHub in a fresh Colab runtime with a T4 GPU. Select **Run all** with no field edited. Do not restart the runtime.
3. Record: date, commit, blob, runtime (GPU, Python, `torch`, `transformers`), total wall time, per-section outcome, the Section 5 comparison (`psnr_y`, `ssim_y`, `psnr_rgb` for nearest, bicubic and model; the model-minus-bicubic difference with its interval and win count), and confirmation that no clone, worker, credential, upload or restart was needed.
4. BYOD (REL12): in a copy, set `USE_BYOD = True` and `BYOD_PATH` to (a) a folder of sharp images with `BYOD_MODE = 'hr'`, (b) a small image with `BYOD_MODE = 'lr'`, and (c) an image larger than 256 px with `BYOD_MODE = 'lr'`, which must be rejected with the size message. Record each outcome.

## Conformance report (NOTEBOOK_SPEC 2.2 §30)

```
Notebook profile: TASK-INFERENCE
Pedagogical mode: GUIDED
Notebook spec: 2.2

Standalone contract:
- repository clone/runtime DIMER-source fetch required: no
- external DIMER workers required: no
- user credential required on default path: no
- automatic sample data/input: yes
- automatic sample artifact: N/A (no artifact is produced or consumed)
- BYOD required by capability: yes
- BYOD implemented: yes (hr and lr modes, upload or path)
- Run all clean runtime: not yet run (blocked until the snapshot is pinned)

Stages:
- validation: notebook-local; exercised on CPU with a tiny model
- fine-tuning/adaptation: N/A (task inference)
- evaluation: notebook-local; exercised on CPU with a tiny model
- inference: notebook-local; exercised on CPU with a tiny model
- artifact fresh reload: N/A
- BYOD positive/negative validation: exercised on CPU with a tiny model; not yet in a hosted runtime

Execution evidence:
- hosted runtime: none
```

## Recorded executions

### 2026-10-05 — Google Colab T4, default `Run all` with the real weights

- **Date:** 2026-10-05
- **Subject:** `tutorials/swin2sr_x4_super_resolution_colab.ipynb`, blob `caa2969c6695de39501ede9183b0de61639de1fc` (branch `ccr-24656dfc-ax1ln2` at `bec3d3e`; carried files of that commit, labelled with its parent `a112444`). Every source cell of the executed copy is byte-identical to that blob. The executed copy is `docs/execution-evidence/2026-10-05/swin2sr_x4_super_resolution_colab_caa2969_colab-t4.ipynb`.
- **Runtime:** Google Colab, Tesla T4 (15,360 MiB), kernel CPython 3.13.15. Stage environment built by the notebook: CPython 3.12.12 managed by `uv`, the 45-package lock, `torch 2.14.0+cu130` with CUDA, `transformers 4.57.6`, `numpy 2.5.3`, `pillow 11.3.0`; built in 86 s.
- **Procedure:** `Run all` with no field edited (`RUN_ACTIVITY = False`, `USE_BYOD = False`).
- **Observed result:** all 9 code cells completed in order with no error and no restart.
  - Section 3: `model.safetensors` fetched at `c69ef3e2d2ac5777ff4a9f2f5afcd86d20604b7e`; all four files verified by size and SHA-256 (2.1 s).
  - Section 4: 27 photographs fetched and verified (2,547,912 bytes); 24 pairs of 320 px references and 80 px inputs, 4 per species; padding reported per input; all five refusal probes rejected (13.7 s).
  - Section 5, on CUDA (0.23 s per image): Y-channel PSNR `nearest` 26.72 dB, `bicubic` 28.15 dB, model 30.84 dB; Y-channel SSIM 0.7364, 0.7879, 0.8616; RGB PSNR 25.34, 26.78, 29.45 dB. Model minus bicubic: +2.69 dB PSNR-Y (95 % bootstrap 2.09 to 3.35), +0.0737 SSIM-Y (0.0581 to 0.0932); the model won on 24 of 24 images. Per species, the model's PSNR-Y ranged from 27.88 dB (white-throated sparrow) to 33.47 dB (dark-eyed junco), each above its bicubic value (20.5 s).
  - Section 6: four reference-free inputs upscaled ×4 (three 96 px photographs to 384 px, the synthetic scene 100 × 76 to 400 × 304) with per-image provenance; every listed output written (6.9 s).
- **Caveats:** the runtime was not completely fresh. Execution counts start at 2 and the run directory sits inside a clone of this repository, so the notebook ran in the same Colab session that had just run the pin dry run. The notebook writes and verifies its own carried code and builds its own isolated environment, and it downloaded the weights fresh, so the outcome does not depend on that session state; still, it is not a clean-state run in the strict sense of REL2. The optional activity and the BYOD branch were not run (REL12 open). One pass, one runtime; the sample is 24 CC0 photographs of six species, not a benchmark. Transformers printed a harmless deprecation notice about `pad_size`.

### 2026-10-04 — CPU pre-flight with a tiny random-initialised snapshot (not release evidence)

- **Date:** 2026-10-04
- **Subject:** the notebook regenerated into a scratch copy from the sources of commit `72257df` (`git archive`), with two test substitutions — `MODEL_REVISION` set to a test value, and the manifest replaced by one describing a tiny random-initialised Swin2SR ×4 snapshot (12-dim embedding, one stage, the real Transformers classes) staged under `weights/` beforehand. Everything else, including the lock and the stage runner, was the committed text.
- **Runtime:** Linux x86_64, 4 vCPU, no GPU; kernel CPython 3.12.12 driven by `nbclient`; the notebook built its own isolated environment (CPython 3.12.12, 45 locked packages, `torch 2.14.0+cu130`, `transformers 4.57.6`).
- **Procedure and observed result:**

| Run | Fields set in the executed copy | Outcome |
|---|---|---|
| 1 | none (default path) | all 9 code cells ok, 100.9 s including the environment build; 27 photographs fetched from the iNaturalist bucket and verified; 24 pairs (320/80 px) validated; 5/5 refusal probes rejected; evaluate, infer and the result record completed; every listed output written |
| 2 | `RUN_ACTIVITY=True`, `ACTIVITY_KERNEL='box'`, `USE_BYOD=True`, `BYOD_MODE='hr'`, `BYOD_PATH=<folder of 3 odd-sized PNGs>` | ok, 200.3 s; BYOD reported the pixels cropped to reach a multiple of 4 and wrote scores, panels and `byod_result.json` |
| 3 | `RUN_ACTIVITY=True`, `ACTIVITY_KERNEL='bicubic'`, `ACTIVITY_JPEG_QUALITY=30`, `USE_BYOD=True`, `BYOD_MODE='lr'`, `BYOD_PATH=<zip of 2 RGBA PNGs in a sub-folder>` | ok, 179.0 s; RGB conversion and alpha removal reported; outputs labelled `not-measurable` |
| 4 | `USE_BYOD=True`, `BYOD_MODE='lr'`, `BYOD_PATH=<300 × 200 PNG>` | the BYOD cell stopped as designed with `RuntimeError: Stage 'byod' failed (exit 2): ValueError: … longer side 300 px > MAX_INPUT_SIDE 256 …`; every earlier cell ok |

- **Caveats:** wall times include building the isolated environment on a shared host and vary with its load. The tiny model's outputs are meaningless, so no score from these runs is model performance. The real weights were not used (the snapshot was not pinned at the time, and the Hub is unreachable from that host). The upload dialog, a GPU and a hosted runtime were not exercised. The unmodified notebook's Section 3 refusal for the unpinned snapshot is covered by `tests/test_tutorial_stages.py::test_unpinned_weights_stage_explains_itself`.
