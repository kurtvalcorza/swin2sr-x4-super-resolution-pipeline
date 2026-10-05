# Release status

Current status: **Candidate** — initial development. The `TASK-INFERENCE` tutorial notebook `tutorials/swin2sr_x4_super_resolution_colab.ipynb` (NOTEBOOK_SPEC 2.2, `GUIDED`, standalone, isolated hash-locked uv environment) has **no hosted execution evidence** yet. The `caidas/swin2SR-classical-sr-x4-64` snapshot is pinned to commit `c69ef3e2d2ac5777ff4a9f2f5afcd86d20604b7e`, with every file's size and SHA-256 recorded; it was pinned on 2026-10-05 by running `python tools/pin_snapshot.py --dry-run` in a Google Colab runtime (the Hub is unreachable from the build environment); the manifest and `MODEL_REVISION` were written from its output exactly as the tool writes them.

What exists and is checked: the package, the offline unit suite and the tiny-model CPU tests, the stage runner, the generated notebook and its parity checks, the hash-locked requirements, `MODEL_CARD.md` (MODEL_CARD_SPEC 1.2), the static validator (`tools/validate_release_assets.py`) and the CI workflow. A CPU run of the full notebook against a tiny random-initialised snapshot (with a test revision substituted) is recorded in `docs/release-verification.md` as pre-flight evidence only.

Remaining before Release-grade:

1. Run the regenerated notebook top-to-bottom with **Run all** in a clean Colab (T4) runtime, in one pass with no restart, and record it in `docs/release-verification.md` (blob, commit, runtime, outcome, metrics).
2. Exercise the BYOD branch in that runtime (REL12): one compatible `hr` input, one compatible `lr` input, one rejected input.
