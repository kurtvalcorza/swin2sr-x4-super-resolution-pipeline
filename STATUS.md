# Release status

Current status: **Candidate** — initial development. The `TASK-INFERENCE` tutorial notebook `tutorials/swin2sr_x4_super_resolution_colab.ipynb` (NOTEBOOK_SPEC 2.2, `GUIDED`, standalone, isolated hash-locked uv environment) completed `Run all` on a Google Colab T4 on 2026-10-05 with the real weights (blob `caa2969`): Y-channel PSNR 30.84 dB against 28.15 dB for bicubic, the model ahead on 24 of 24 images. That runtime had just run the pin dry run, so it was not a strictly fresh session (recorded in `docs/release-verification.md`). The `caidas/swin2SR-classical-sr-x4-64` snapshot is pinned to commit `c69ef3e2d2ac5777ff4a9f2f5afcd86d20604b7e`, with every file's size and SHA-256 recorded; it was pinned on 2026-10-05 by running `python tools/pin_snapshot.py --dry-run` in a Google Colab runtime (the Hub is unreachable from the build environment); the manifest and `MODEL_REVISION` were written from its output exactly as the tool writes them.

What exists and is checked: the package, the offline unit suite and the tiny-model CPU tests, the stage runner, the generated notebook and its parity checks, the hash-locked requirements, `MODEL_CARD.md` (MODEL_CARD_SPEC 1.2), the static validator (`tools/validate_release_assets.py`) and the CI workflow. A CPU run of the full notebook against a tiny random-initialised snapshot (with a test revision substituted) is recorded in `docs/release-verification.md` as pre-flight evidence only.

Remaining before Release-grade:

1. Exercise the BYOD branch in a fresh hosted runtime (REL12), together with a clean-state default `Run all`: one compatible `hr` input, one compatible `lr` input, one rejected input.
