# Swin2SR x4 Super-Resolution Task-Inference Notebook — Review

**Verdict: Needs revision** (no Major findings; three Minors, one of which breaches a MUST)  
**Review date:** 5 October 2026  
**Repository:** `kurtvalcorza/swin2sr-x4-super-resolution-pipeline`  
**Notebook:** `tutorials/swin2sr_x4_super_resolution_colab.ipynb`  
**Reviewed commit:** `46e6444` (`main`, the merge of PR #1)  
**Notebook Git blob:** `caa2969c6695de39501ede9183b0de61639de1fc`, generated from `a112444`. This is the blob executed in the recorded Colab T4 run of 2026-10-05. At the reviewed commit `tools/build_notebook.py --check` and `tools/validate_release_assets.py` both exit 0.  
**Finding prefix:** `S2X`  
**Framework:** Notebook Review Framework v1. **Requirements baseline:** NOTEBOOK_SPEC 2.2, `ml-worker` `origin/main` at `b9fdd1f`.

## Executive assessment

This is the cleanest of the four notebooks built this round. It is a standalone, generator-built `TASK-INFERENCE` / `GUIDED` notebook.

- **Runtime.** It installs nothing into the kernel. A pinned `uv` builds managed CPython 3.12.12 from a hash lock of 45 packages.
- **Weights.** It loads only `model.safetensors` from an immutable Hub commit, never the pickle, with `trust_remote_code=False`.
- **Sample.** It fetches 27 CC0 iNaturalist photographs, each digest-pinned, and builds 24 pairs by 4x bicubic downsampling.
- **Validation.** Pairs are validated before any model loads, and the padding to the 8 px window is reported.
- **Evaluation.** It scores the model against bicubic and nearest-neighbour with the literature's Y-channel convention: BT.601 `16 + (65.481R + 128.553G + 24.966B)/255`, a 4 px border crop and an 11-tap Gaussian SSIM. It adds a paired, seeded bootstrap and a win count.
- **Outputs.** Reference-free inputs get outputs with no score. The optional degradation activity writes its own record.
- **BYOD.** The branch is careful about crop reporting, refusals and never resizing silently.
- **Guided layer.** It is complete.

This review independently re-fetched the pinned photographs and recomputed both baselines. Every number matched the Colab record.

| Measure (Colab T4, blob `caa2969`, 2026-10-05) | Value | This review (torch-free recomputation) |
|---|---|---|
| Code cells | 9/9 in order; environment 86 s, stages ≈ 43 s (execution counts start at 2: not a fresh session) | — |
| Sample digest | `db74c01e3c872fff…` | **identical** |
| Nearest Y-PSNR / Y-SSIM / RGB-PSNR | 26.7177 / 0.7364 / 25.3368 | **identical** |
| Bicubic Y-PSNR / Y-SSIM / RGB-PSNR | 28.1487 / 0.7879 / 26.7813 | **identical** |
| Model Y-PSNR / Y-SSIM / RGB-PSNR | 30.8417 / 0.8616 / 29.446 | not computed (needs torch) |
| Model − bicubic, Y-PSNR | +2.693 dB, 95 % CI [2.092, 3.351], 24/24 wins | — |

Three Minors stand between this notebook and `Ready for intended use`:

- **S2X-m1:** the opening cell shows a raw template placeholder (`` `{MODEL_ID}` ``) and says that no hosted duration has been recorded.
- **S2X-m2:** the Section 5 checkpoint is built around a hypothetical 1 dB gap, while the documented run shows 2.7 dB. Four-image species means are printed without a caution.
- **S2X-m3:** the shared behaviour on a partial re-run, and the environment rebuild on every run.

Two gates remain, and both are already listed in `STATUS.md`:

- The recorded hosted run was not clean-state. It started at execution count 2, inside a repository clone, right after the pin dry run.
- REL12 BYOD has not been run on a hosted runtime.

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `TASK-INFERENCE` / `GUIDED` |
| Declared spec | DIMER Notebook Specification **2.2**, standalone |
| Intended audience | Stated: learners who run hosted notebook cells and read short Python; no SR background assumed |
| Supported runtime | Linux x86_64; a GPU is used if present, CPU supported; float32 |
| Promised outcomes | <ul><li>verified snapshot at the pinned commit</li><li>27 pinned photographs and 24 validated pairs, with five refusals and reported padding</li><li>model versus bicubic versus nearest-neighbour, on Y-PSNR, Y-SSIM and RGB-PSNR</li><li>paired bootstrap and win count</li><li>reference-free inference on 4 inputs</li><li>predictions CSV, PNGs with digests, and a result record</li><li>optional degradation activity</li><li>BYOD in `hr` and `lr` modes</li></ul> |
| Generator | `tools/build_notebook.py` (`build_notebook.py/3.0`) + `tools/notebook_template.py`; generating revision `a112444` |
| Release status | `Candidate` (`STATUS.md`): Colab run recorded, not clean-state; REL12 hosted BYOD pending |

### Evidence actually obtained

- **Source inspection.**
  - All 35 cells (9 code; cell 9 is the 9-file carrier).
  - `tools/tutorial_stages.py`: evaluate, infer, activity and byod.
  - `samples.py`: `fetch_photos`, `degrade`, `build_sample_pairs`, `validate_pairs` and `prepare_byod`.
  - `metrics.py`: the luma formula, the border crop, the SSIM window, `interpolate` and `paired_bootstrap`.
  - `STATUS.md` and `docs/release-verification.md`.
  - The x2 sibling's release record, to check the activity's cross-reference ("the x2 sibling scored below bicubic on JPEG-compressed inputs"). It does: 25.92 dB frozen against 26.31 dB for bicubic on JPEG-40.
- **Documented execution evidence.** The Colab T4 run of the reviewed blob (`docs/execution-evidence/2026-10-05/…_caa2969_colab-t4.ipynb`). The default path ran; the activity and BYOD did not.
- **Direct execution (this review), without torch.**
  - Environment: Python 3.12 with `numpy` 2.5.3 and `pillow` 11.3.0.
  - P3: `fetch_photos` from `inaturalist-open-data.s3.amazonaws.com`. All 27 files passed the size and SHA-256 checks.
  - P3: `build_sample_pairs` and `validate_pairs`, then the bicubic and nearest-neighbour baselines with the package's own metrics, compared with the Colab record.
  - P3: the bicubic baseline under the activity's four degradations.
  - P4: `prepare_byod` in both modes on 11 constructed inputs, including the empty folder that a cancelled upload leaves.
  - Scripts and results are in `swin2sr_x4_super_resolution_colab_Review_Probes.zip`.
- **Not executed here:** the model. The Hub is unreachable from this container and the locked torch is the multi-GB CUDA build. Model numbers come from the Colab record.
- **Learner observation:** none.

## 2. Separate judgments

- **Technical correctness: very good.**
  - The metric conventions match the classical-SR literature, and the baseline values reproduce exactly from the pinned photographs.
  - The data pipeline is digest-locked end to end. The safetensors-only load and `trust_remote_code=False` are the right trust posture.
- **Promise fulfilment:** every promised stage ran on Colab and wrote its outputs. The gaps are prose ones (S2X-m1, S2X-m2).
- **Learner experience: strong.**
  - Strengths: why the notebook makes its own pairs; the meaning of the interval; synthetic detail is not recovered detail; the degradation activity.
  - Friction: a visible `{MODEL_ID}` placeholder in the very first concept paragraph, and a checkpoint anchored to 1 dB.
- **Spec conformance:**
  - Unresolved MUSTs: SRC3 (S2X-m1: a template placeholder and a stale statement in a released canonical notebook) and UX12.
  - SHOULD deviations: UX4 (S2X-m2); SRC2, UX10 and GDL13 (S2X-m3).
  - REL1 and REL10 clean-state evidence and REL12 hosted BYOD are outstanding (gates).

## 3. Promise and objective tracing

| Claim / objective | Implementation | Observable result | Learner interpretation | Status |
|---|---|---|---|---|
| One-pass Run all, no restart | cells 6, 9, 12 | 9/9 in order, `setup_seconds` 86 | Section 2 prose | Met (not a clean session; gate) |
| Verified snapshot at the pinned commit; no pickle | cell 15 / `weights` | 4 files verified at `c69ef3e2…` | trust boundary stated | Met |
| 24 pairs, validation, padding report, refusals | cell 18 / `prepare` | 24 × (320 → 80), digest `db74c01e…` (P3 identical); 5 refusals naming rules; padding 8/8 | the checkpoint explains why pairs are self-made | Met |
| Model vs two baselines; bootstrap; wins | cell 21 / `evaluate` | 30.84 vs 28.15 vs 26.72 dB; +2.69 [2.09, 3.35]; 24/24 | the checkpoint supposes about 1 dB | Met (S2X-m2) |
| Reference-free inference and exports | cell 24 / `infer` | 4 outputs with padding; predictions CSV; `result.json`; listing | "added detail is synthetic" | Met |
| Opening description of the model | cell 0 | "This checkpoint, `{MODEL_ID}`, is the *classical* …" | — | **Not met** (S2X-m1) |
| Optional degradation activity | cell 28 / `activity` | not run on Colab; separate `activity.json` | sample answer supported by the x2 record | Met in source |
| BYOD `hr` / `lr` | cell 31 / `byod` | P4: 2 accepted (with conversions); 9 refused naming file and rule, including an empty upload folder | contract and privacy stated first | Met locally (reader); hosted REL12 pending |
| Runtime statements | cells 0, 4 | "Its hosted duration has not been recorded yet"; CPU estimates only | — | **Not met** (S2X-m1) |

All seven learning objectives are observable learner actions, and each is backed by a prediction or checkpoint.

## 4. Journeys

| Journey | Basis | Result |
|---|---|---|
| **First-time learner** | Source inspection, all 35 cells | The guided layer is complete and accurate. The learner meets a literal `` `{MODEL_ID}` `` in the opening paragraph (S2X-m1), and a checkpoint that supposes a 1 dB gap when their run will most likely show about 2.7 dB (S2X-m2). |
| **Clean default** | Documented (Colab T4) + direct (P3: data and baselines) | 9/9 in order. That session had already run the pin dry run: execution counts start at 2 and the working directory is inside a repository clone, so it is not clean-state evidence (`STATUS.md` says so). P3 reproduced the sample digest and both baselines exactly from a fresh fetch of the pinned photographs. |
| **Active learning** | Source + P3 | The activity rebuilds the same references with a different degradation and writes `activity.json`; the canonical outputs are untouched. P3's bicubic baselines under the activity settings (Y-PSNR): <ul><li>bilinear 27.56 dB</li><li>box 28.16 dB</li><li>nearest 26.53 dB</li><li>bicubic + JPEG 30: 26.09 dB</li></ul> The model column needs torch and was not run. |
| **Reuse and recovery** | Direct (P4) + source | See the two parts below. |

**Reuse and recovery, BYOD (P4).**

- Accepted:
  - a 322 × 320 `hr` image (cropped to a multiple of 4, with the cropped pixels reported in the record);
  - a 100 × 90 RGBA `lr` image.
- Refused, each naming the file, the rule and the fix:
  - `hr` 30 px and 1100 px;
  - `lr` 300 px and 6 px;
  - text named `.png`;
  - a zip containing `README.txt`;
  - a `../` zip member;
  - duplicate names in subfolders;
  - an empty folder, which is what a cancelled upload leaves.

**Reuse and recovery, re-runs (source).** Cell 6 has the same `uuid4` run-directory pattern as the MediaPipe notebook, where re-running Section 1 alone was shown to strand the later cells (S2X-m3).

## 5. Findings

### Minor

#### S2X-m1 — The opening cell shows a raw `{MODEL_ID}` placeholder and says no hosted duration has been recorded

- **Cell/section:** cell 0 (the model paragraph and the **Run all** paragraph); cell 4 (*Time on a CPU*). Generator: `tools/notebook_template.py` line 89, a plain string carrying `` `{MODEL_ID}` `` that is not formatted.
- **Observed issue:**
  - The first concept paragraph reads "This checkpoint, `{MODEL_ID}`, is the *classical* 4x super-resolution variant…". The placeholder is never substituted. Line 276 uses the same token inside a formatted string, and that one renders correctly in cell 14.
  - Cell 0 ends "Its hosted duration has not been recorded yet."
  - Cell 4 gives only CPU timings measured with random weights. The Colab T4 run of this blob is recorded: environment 86 s; weights 2.1 s, prepare 13.7, evaluate 20.5 (0.23 s per image), infer 6.9.
- **Consequence:**
  - A visible template artefact sits in the first explanation of the model.
  - The learner is told that no hosted timing exists, and plans by CPU estimates.
- **Evidence:**
  - Source: `grep -c "{MODEL_ID}" tutorials/swin2sr_x4_super_resolution_colab.ipynb` gives 2. One is cell 0's prose; the other is inside the carried code, where it is a legitimate f-string.
  - Documented: the Colab outputs.
- **Recommended correction:**
  - Format the template string, or write the model id literally.
  - Quote the T4 timings, with runtime, date and revision, beside the labelled CPU figures.
  - Add a validator check that rejects `{…}` tokens in markdown cells.
- **Acceptance check:** no markdown cell contains `{MODEL_ID}` or another `{UPPER_CASE}` token, and the Run all paragraph and Prerequisites quote a hosted timing that matches `docs/release-verification.md`.
- **Spec:** SRC3, UX12.

#### S2X-m2 — The principal-result checkpoint is built around a 1 dB gap; species means of n = 4 are printed without a caution

- **Cell/section:** Section 5 (cell 22 "What to notice" and the checkpoint).
- **Observed issue:**
  - The checkpoint asks the learner to "suppose the model wins by about 1 dB on average", and the answer ends "If your run shows a gap far from 1 dB, trust your run".
  - The documented default run shows **+2.69 dB**, interval [2.09, 3.35], with 24/24 wins. That is "several", in the prompt's own terms.
  - The stage prints six species lines (n = 4 each, model against bicubic Y-PSNR) with no interval. "What to notice" introduces them as "one line per species follows", with no warning that four images cannot rank species.
- **Consequence:**
  - The guidance is anchored on a gap the default run does not produce.
  - Learners are invited to read species differences, from 27.88 to 33.47 dB, that four crops cannot support.
- **Evidence:** documented (Colab outputs); source.
- **Recommended correction:**
  - Phrase the checkpoint around the recorded shape: a gain of a few dB, an interval well clear of zero, and a model win on every image. Keep "trust your run".
  - Add one sentence that the species lines are four crops each, descriptive only.
- **Acceptance check:** the Section 5 guidance is consistent with the recorded default run, and the species lines carry an n = 4 caution in the prose or the printed record.
- **Spec:** UX4, EVAL15.

#### S2X-m3 — Re-running the Section 1 cell alone strands later cells; every Run all rebuilds the environment

- **Cell/section:** cell 6; cells 9 and 12; Troubleshooting.
- **Observed issue:**
  - **Partial re-run.** `ROOT` and `ENV_ROOT` come from a fresh `uuid4`. Re-running only cell 6 points `ROOT` at an empty directory while `PYTHON` names the old environment. The next learner cell fails with `can't open file '<new ROOT>/tutorial_stages.py'` and "see the log above".
  - **Environment rebuild.** Every Run all builds a new environment (5.5 GB in the local CPU run) with its own `uv` cache.
- **Consequence:** a re-check after switching to a GPU runtime yields an unguided error, and repeated runs cost minutes and disk.
- **Evidence:** inferred from source. The code path is identical to the MediaPipe notebook's, where this review executed it (`mediapipe-face-landmarker-pipeline` review, P5).
- **Recommended correction:**
  - Have `run_stage` check that the carried runner and `PYTHON` exist, and name the cells to re-run.
  - Add a Troubleshooting row.
  - Key `ENV_ROOT` on the lock digest.
- **Acceptance check:** re-running cell 6 and then cell 18 stops with a message naming Sections 2–3, and a second Run all reuses the environment.
- **Spec:** SRC2, UX10, GDL13.

### Suggestions

- **S2X-S1** — Every model stage prints ``self.pad_size` attribute is deprecated and will be removed in v5``, from `transformers` 4.57.6. Add a Troubleshooting row saying that it is harmless, as the NAFNet notebook does for its upstream warning.
- **S2X-S2** — Section 6 calls the 100 × 76 synthetic scene's size "odd". Both sides are even but not multiples of 8. Say "not a multiple of 8".
- **S2X-S3** — Record the activity's bicubic baseline beside the model for each degradation (P3 has the values). Box downsampling gives a slightly *higher* bicubic baseline than the default (28.16 against 28.15 dB), which is itself worth a sentence.
- **S2X-S4** — Re-run the default path in a genuinely fresh session (execution count 1, outside any clone) together with the REL12 BYOD journey, as `STATUS.md` plans, so the release record has clean-state evidence.

## 6. Readiness

**Needs revision.** There are no Majors.

- S2X-m1 breaches SRC3 (a placeholder and a stale statement in a released notebook). It and S2X-m2 are small template edits; S2X-m3 is a shared infrastructure fix.

Remaining gates after the fixes:

- one **clean-state** hosted Run all of the regenerated blob;
- the REL12 BYOD journey, with one compatible `hr` input, one compatible `lr` input and one rejected input, including the upload dialog;
- both recorded in `docs/release-verification.md`.

## 7. Verified versus inferred

- **Verified by direct execution (torch-free):**
  - all 27 pinned photographs, fetched and digest-checked;
  - the sample digest;
  - both baselines on every metric, identical to the Colab record;
  - the activity-degradation baselines;
  - the BYOD acceptance and refusal matrix.
- **Verified from documented evidence:** every model number (Colab T4, reviewed blob); the x2 sibling's JPEG result, which the activity cites.
- **Inferred from source:**
  - the S2X-m3 re-run behaviour (executed on the MediaPipe notebook's identical code);
  - the upload branch's handling of multiple files, which writes all of them to one folder;
  - the empty-upload outcome, reproduced through `prepare_byod` on an empty folder rather than through the dialog.
- **Most likely to be wrong:** S2X-m2's weight. The "suppose 1 dB" checkpoint is explicitly hypothetical and ends with "trust your run", so a maintainer could keep it. It is rated a finding because the documented default lands well outside it.
