# Release verification

The tutorial notebook `tutorials/swin2sr_x4_super_resolution_colab.ipynb` is promoted to Release-grade only by a recorded clean-runtime **Run all** of the exact committed notebook blob (NOTEBOOK_SPEC 2.2 §24). Static and unit checks are necessary but are not execution evidence (REL8).

## Coverage: automatic versus manual (REL9)

| Check | Where | Automatic? |
|---|---|---|
| Lint, unit tests, tiny-model CPU tests of the loader, padding, metrics, every stage in order and both BYOD modes | CI (`.github/workflows/ci.yml`) | yes |
| Notebook parity with the repository (carried files, hashes, lock, generator output), cleared outputs, no kernel `pip install` | CI (`tests/test_notebook_parity.py`, `python tools/build_notebook.py --check`) | yes |
| Static release-asset validation (card, registry, status, guided layer, infrastructure titles, forbidden patterns) | CI (`python tools/validate_release_assets.py`) | yes |
| Real weights: staging, digest verification, reconstruction quality | manual — CI cannot reach the Hugging Face Hub, and the snapshot is not yet pinned | no |
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

### 2026-10-04 — CPU pre-flight with a tiny random-initialised snapshot (not release evidence)

- **Date:** 2026-10-04
- **Subject:** the notebook regenerated into a scratch copy from the sources of the commit that added this record, with two test substitutions — `MODEL_REVISION` set to a test value, and the manifest replaced by one describing a tiny random-initialised Swin2SR ×4 snapshot (12-dim embedding, one stage, the real Transformers classes) staged under `weights/` beforehand. Everything else, including the lock and the stage runner, was the committed text.
- **Runtime:** Linux x86_64, 4 vCPU, no GPU; kernel CPython 3.12.12 driven by `nbclient`; the notebook built its own isolated environment (CPython 3.12.12, 45 locked packages, `torch 2.14.0+cu130`, `transformers 4.57.6`).
- **Procedure and observed result:**

| Run | Fields set in the executed copy | Outcome |
|---|---|---|
| 1 | none (default path) | all 9 code cells ok, 74.8 s including the environment build; 27 photographs fetched from the iNaturalist bucket and verified; 24 pairs (320/80 px) validated; 5/5 refusal probes rejected; evaluate, infer and the result record completed; every listed output written |
| 2 | `RUN_ACTIVITY=True`, `ACTIVITY_KERNEL='box'`, `USE_BYOD=True`, `BYOD_MODE='hr'`, `BYOD_PATH=<folder of 3 odd-sized PNGs>` | ok, 86.5 s; BYOD reported the pixels cropped to reach a multiple of 4 and wrote scores, panels and `byod_result.json` |
| 3 | `RUN_ACTIVITY=True`, `ACTIVITY_KERNEL='bicubic'`, `ACTIVITY_JPEG_QUALITY=30`, `USE_BYOD=True`, `BYOD_MODE='lr'`, `BYOD_PATH=<zip of 2 RGBA PNGs in a sub-folder>` | ok, 99.1 s; RGB conversion and alpha removal reported; outputs labelled `not-measurable` |
| 4 | `USE_BYOD=True`, `BYOD_MODE='lr'`, `BYOD_PATH=<300 × 200 PNG>` | the BYOD cell stopped as designed with `RuntimeError: Stage 'byod' failed (exit 2): ValueError: … longer side 300 px > MAX_INPUT_SIDE 256 …`; every earlier cell ok |

- **Caveats:** the tiny model's outputs are meaningless, so no score from these runs is model performance. The real weights were not used (the snapshot is not yet pinned and the Hub is unreachable from that host). The upload dialog, a GPU and a hosted runtime were not exercised. Before the run that substituted the test values, the unmodified notebook's Section 3 refusal for the unpinned snapshot is covered by `tests/test_tutorial_stages.py::test_unpinned_weights_stage_explains_itself`.
