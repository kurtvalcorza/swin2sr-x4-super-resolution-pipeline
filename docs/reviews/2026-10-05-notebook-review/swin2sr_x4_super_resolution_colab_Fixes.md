# Swin2SR x4 super-resolution notebook — review fixes

**Review:** `swin2sr_x4_super_resolution_colab_Review.md` (5 October 2026, S2X-m1..m3, no Majors).
**Fixed in:** the generator (`tools/build_notebook.py`, `tools/notebook_template.py`) and `tools/validate_release_assets.py`; the notebook was regenerated and `--check` passes.
**Readiness:** **Verification pending** until the hosted gates below are recorded. `STATUS.md` and every release label are unchanged.

## Findings

| ID | Status | Change | Cells / files | Evidence |
|---|---|---|---|---|
| S2X-m1 | Fixed | The unformatted `` `{MODEL_ID}` `` in the opening paragraph is replaced by "the checkpoint this notebook pins (named in Section 3)" (the identifier is already rendered in Section 3; no identifier is added to repository text). The Run-all paragraph and the Prerequisites quote the recorded T4 run (5 October 2026, revision `a112444`, 86 s environment; weights 2.1 s, prepare 13.7 s, evaluate 20.5 s at 0.23 s per image, infer 6.9 s) beside the labelled CPU figures; the 7 GB disk check is labelled as including a margin over the 5.5 GB measured. The validator now rejects any `{UPPER_CASE}` token in markdown cells. | `intro`, `run_all`, Prerequisites, How to use; `_validate_notebook_structure` | `test_s2x_m1_no_template_token_in_markdown_and_the_hosted_timing_is_quoted`, `test_s2x_m1_validator_rejects_an_unformatted_markdown_token` |
| S2X-m2 | Fixed | The Section 5 checkpoint is built on the recorded shape (+2.69 dB PSNR-Y, interval 2.09 to 3.35, 24/24 wins) and keeps "trust your run"; the sample answer adds that a win on every image is a statement about these 24 crops. "What to notice" marks the species lines as four crops each, descriptive only. | Section 5 notes | `test_s2x_m2_checkpoint_matches_the_recorded_run_and_species_lines_are_cautioned` |
| S2X-m3 | Fixed (stronger than the acceptance check) | The uv-template change piloted in the MediaPipe notebook: Section 1 keeps this session's run directory when re-run (`NEW_RUN_DIRECTORY` for a fresh one); the environment directory is keyed on the lock digest, managed Python and `uv` version and reused through a `ready.json` marker; `run_stage` names Sections 1–3 when the run directory or interpreter is missing; Troubleshooting row; `PYTHONSTARTUP` dropped. | `CHECK_CELL`, `INSTALL_CELL`, `run_stage`; validator | `test_s2x_m3_*` (4 tests) |

## User-visible changes

- Section 1 has a new form field, `NEW_RUN_DIRECTORY` (off); re-runs keep the run directory; the environment is reused (`environment_reused`) and lives at `<tmp>/swin2sr_x4_super_resolution_env_<lock key>`.
- The validator fails on any `{UPPER_CASE}` token left in a markdown cell.

## Verification (offline, not clean-runtime evidence)

- `build_notebook.py --check`: OK. `validate_release_assets.py`: PASS. `ruff check src tests tools`: clean.
- `pytest` with CI's lightweight dependencies and no torch: 63 passed / 4 skipped before, **70 passed / 4 skipped** after.
- No model stage ran (the Hugging Face Hub is unreachable here). Numbers in the prose are quoted from the recorded Colab T4 run in `docs/release-verification.md`.

## Remaining gates

1. A hosted one-pass **Run all** of the regenerated notebook in a fresh T4 runtime (no restart), then a re-run of the export/infer cell.
2. The REL12 BYOD journey on a hosted runtime (`hr` folder, small `lr` image, oversized `lr` image refused), including the upload dialog.
