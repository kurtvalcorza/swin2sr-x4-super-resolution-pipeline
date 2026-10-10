"""Per-repository template for tools/build_notebook.py /3 (NOTEBOOK_SPEC 2.2 §4 standalone, §25.13 isolated environment).

The generator writes the infrastructure cells (runtime check, carrier, isolated install + stage runner, snapshot
staging) from repository files; this template holds the learner-facing prose, the list of carried files and the
learner cells. Every learner cell calls ``run_stage(...)``: the carried ``tutorial_stages.py`` (``tools/`` in the
repository) runs one stage per process in an isolated, hash-locked environment, so nothing is installed into the
notebook kernel.

This template configures a TASK-INFERENCE workflow: the pinned Swin2SR classical x4 checkpoint is staged and
digest-verified, 24 pinned CC0 photographs are turned into 4x bicubic low-/high-resolution pairs and validated, the
model and two interpolation baselines are scored with Y-channel PSNR and SSIM, three reference-free new inputs and a
seeded synthetic scene are upscaled and exported with provenance, and two optional branches follow: a change-one-thing
activity on the degradation, and Bring Your Own Data in an 'hr' (score) or 'lr' (upscale only) mode.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

REPO = "swin2sr-x4-super-resolution-pipeline"
NOTEBOOK = "swin2sr_x4_super_resolution_colab.ipynb"

BADGES = [
    ("GitHub", "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white", f"https://github.com/kurtvalcorza/{REPO}"),
    ("Open In Colab", "https://colab.research.google.com/assets/colab-badge.svg", f"https://colab.research.google.com/github/kurtvalcorza/{REPO}/blob/main/tutorials/{NOTEBOOK}"),
    (
        "Hugging Face",
        "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-caidas%2Fswin2SR--classical--sr--x4--64-ffcc4d?style=flat",
        "https://huggingface.co/caidas/swin2SR-classical-sr-x4-64",
    ),
    ("Upstream", "https://img.shields.io/badge/Upstream-mv--lab%2Fswin2sr-181717?style=flat&logo=github&logoColor=white", "https://github.com/mv-lab/swin2sr"),
    ("License", "https://img.shields.io/badge/License-Apache--2.0-blue.svg", "https://www.apache.org/licenses/LICENSE-2.0"),
]

TEMPLATE = {
    "package": "swin2sr_x4_super_resolution_pipeline",
    "repo_name": REPO,
    "stem": "swin2sr_x4_super_resolution",
    "notebook_name": NOTEBOOK,
    "profile": "TASK-INFERENCE",
    "mode": "GUIDED",
    "run_all": (
        "Selecting **Run all** in a fresh Linux runtime — a Colab or Kaggle T4 GPU is recommended, a CPU-only runtime also works "
        "but is slower — builds an isolated Python environment from the carried hash-locked requirements (torch, transformers, "
        "safetensors, huggingface-hub, numpy, pillow and their dependencies) without touching the notebook kernel's own packages, "
        "then runs each stage below in its own process: it stages and digest-verifies the pinned 49 MB Swin2SR x4 snapshot from the "
        "Hub, fetches 27 CC0 iNaturalist photographs as digest-verified JPEGs (2.5 MB, no credential), turns 24 of them into "
        "320 px references with 80 px inputs by 4x bicubic downsampling and validates every pair, upscales the 24 inputs and scores "
        "the model beside bicubic and nearest-neighbour interpolation with Y-channel PSNR and SSIM (4 px border crop) and a seeded "
        "bootstrap over images, upscales three native photo crops and a seeded synthetic scene that have no reference, and writes "
        "the panels, per-image scores, predictions and a provenance record under `outputs/`. The default path needs no repository "
        "clone, no DIMER worker or service, no credential, no upload dialog, no configuration edit and no runtime restart "
        "(NOTEBOOK_SPEC 2.2 §5). The recorded hosted run (Google Colab, Tesla T4, 5 October 2026, default fields, notebook generated "
        "from revision `a112444`) built the isolated environment in 86 s and spent about 43 s in the model stages; a second **Run all** "
        "in the same runtime reuses that environment."
    ),
    "byod": (
        "After the default path completes, Section 8 lets you run the same validation, inference and output contract on your own "
        "images: set `USE_BYOD = True`, choose `BYOD_MODE` — `'hr'` when your image is the sharp original (it is cropped to a "
        "multiple of 4 px if needed, downsampled 4x by bicubic interpolation, upscaled and scored against itself and the two "
        "baselines) or `'lr'` when your image is the small input (it is upscaled only; nothing can be scored) — and either set "
        "`BYOD_PATH` to an image, folder or zip already in the runtime or leave it empty to get an upload dialog. The schema, the "
        "size limits and the privacy guidance are stated in the Prerequisites and in Section 8; uploaded files stay inside this "
        "runtime. BYOD is optional and never part of the default path."
    ),
    "weights_key": "swin2sr-x4-64",
    # Carried byte for byte (UTF-8 text, LF newlines) into the run directory and verified against CARRIED_HASHES.
    "carried": {
        "src/swin2sr_x4_super_resolution_pipeline/__init__.py": "src/swin2sr_x4_super_resolution_pipeline/__init__.py",
        "src/swin2sr_x4_super_resolution_pipeline/pipeline.py": "src/swin2sr_x4_super_resolution_pipeline/pipeline.py",
        "src/swin2sr_x4_super_resolution_pipeline/metrics.py": "src/swin2sr_x4_super_resolution_pipeline/metrics.py",
        "src/swin2sr_x4_super_resolution_pipeline/samples.py": "src/swin2sr_x4_super_resolution_pipeline/samples.py",
        "tutorial_stages.py": "tools/tutorial_stages.py",
        "requirements.txt": "tutorials/requirements-colab.lock.txt",
        "weights/swin2sr-x4-64/dimer-base-manifest.json": "weights/swin2sr-x4-64/dimer-base-manifest.json",
        "LICENSE": "LICENSE",
    },
    "stage_runner": "tutorial_stages.py",
    "lock": "requirements.txt",
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "disk_gib": {"weights": 0.1, "environment": 7},
    "runtime_modules": ["torch", "transformers", "numpy", "PIL"],
    "title": "Swin2SR x4 — DIMER Guided Notebook: 4x image super-resolution (task inference, standalone)",
    "badges": BADGES,
    "capability": "4x single-image super-resolution with a 12.2 M-parameter Swin2SR transformer (classical, bicubic degradation), scored with Y-channel PSNR and SSIM against bicubic and nearest-neighbour interpolation, plus reference-free inference on new images",
    "intro": (
        "Swin2SR (Conde et al., 2023) is a SwinV2 transformer (Liu et al., 2022) for image restoration. The checkpoint this "
        "notebook pins (named in Section 3) is the *classical* 4x super-resolution variant: it was trained to undo one known degradation — a sharp "
        "photograph shrunk 4x with bicubic interpolation — so it receives an `H × W` RGB image and returns a `4H × 4W` image in a "
        "single forward pass. Every input pixel is a token (`patch_size` 1), self-attention runs inside 8 × 8 windows over six "
        "residual groups, and a pixel-shuffle head rearranges channels into the 16 × larger output.\n\n"
        "Two properties of the task shape this notebook. **Super-resolution has a ground truth only when you made the input "
        "yourself**: the notebook takes sharp photographs, shrinks them 4x, and asks the model to recover them, so the original is "
        "the reference. **The detail it adds is a prediction, not a recovery**: on an image that was never downscaled from a "
        "sharper one there is nothing to compare against, so Section 6 reports outputs without a score. Nothing is trained here: "
        "this is a pretrained model used as published, which is why the profile is task inference."
    ),
    "learning_objectives": (
        "by the end of this notebook you will be able to —\n\n"
        "1. **Explain** the input → output contract of 4x super-resolution and why the window padding is added and cropped off (Sections 4–5).\n"
        "2. **Diagnose** an invalid low-/high-resolution pair from a validation refusal before any model runs (Section 4).\n"
        "3. **Compute and interpret** Y-channel PSNR and SSIM with a border crop, and **compare** the model with bicubic and nearest-neighbour interpolation on identical inputs (Section 5).\n"
        "4. **Distinguish** a mean difference from its spread across images using a paired bootstrap (Section 5).\n"
        "5. **Apply** the model to new images that have no reference and **identify** what can and cannot be claimed about them (Section 6).\n"
        "6. **Predict**, run and **explain** how a mismatched degradation changes the model's advantage over bicubic, in an optional activity (Section 7).\n"
        "7. **Write** an evidence-based conclusion that names the baseline, the spread and the limits of 24 photographs (Interpretation and conclusion)."
    ),
    "exclusions": (
        "x2 or x3 upscaling (the x2 checkpoint has its own repository), fine-tuning, compressed-input or real-world restoration "
        "(other upstream Swin2SR checkpoints that are not packaged here), face or text restoration, video, tiled inference of images "
        "larger than 256 px, perceptual or no-reference quality metrics, and any claim that PSNR or SSIM measures how good an image "
        "looks to a person. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported **Linux x86_64** runtime (Google Colab, Kaggle, or a Linux Jupyter kernel). A CUDA GPU such as a T4 is used automatically when present; without one every stage runs on the CPU in float32. The kernel's own Python version does not matter: the notebook installs nothing into it and runs every stage with CPython 3.12.12 in an isolated environment built from {n_locked} hash-locked packages (torch 2.14.0, whose Linux wheel is the CUDA 13.0 build and also runs on a CPU). The Section 1 check asks for about 7 GB of free disk for that environment (it occupied 5.5 GB in a local CPU run; the rest is a margin) and 0.1 GB for the weights and photographs.",
        "- **Time on the documented runtime:** in the recorded hosted run (Google Colab, Tesla T4, 5 October 2026, default fields, notebook generated from revision `a112444`; see `docs/release-verification.md`) the environment was built in 86 s and the stages took: weights 2.1 s, prepare 13.7 s, evaluate 20.5 s (0.23 s per image on CUDA), infer 6.9 s.",
        "- **Time on a CPU:** measured for this architecture on a 4-vCPU host in float32, one forward pass took 3.2 s for an 80 px input, 4.6 s for 128 px and 28.7 s for 256 px (the largest accepted input). The 24 sample inputs are 80 px, so Section 5 is estimated (not measured) at one to two minutes on such a host, and longer on a 2-vCPU hosted runtime. These are measurements of the model's compute with random weights on that host, not of a hosted run of this notebook.",
        "- **Knowledge:** you can run notebook cells and read short Python. No prior experience with super-resolution is assumed; the glossary below defines PSNR, SSIM, the Y channel and the other terms.",
        "- **Weights:** only `model.safetensors` is loaded; the upstream repository's `pytorch_model.bin` (a pickle file) is never downloaded or unpickled, and no Hub-hosted code is executed — the model classes come from `transformers` on PyPI with `trust_remote_code=False`. The weights are released under Apache-2.0.",
        "- **Pin status:** the model is pinned by an immutable Hugging Face commit and per-file SHA-256 digests recorded in the carried manifest. If the manifest still says `\"revision\": \"unpinned\"`, the notebook was generated before the maintainer recorded that commit; Section 3 then stops with a message instead of downloading unverified weights (see **Troubleshooting**).",
        "- **Data contract:** an input is one RGB image (any mode; converted to RGB, alpha discarded) with sides of 8..256 px; the output is `4H × 4W × 3` uint8. A scored pair is `{{id, hr, lr}}` where `hr` has sides that are multiples of 4 within 32..1024 px and `lr` is exactly `hr` shrunk 4x. Nothing is resized silently: inputs outside the limits are refused with a message naming the rule.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — photographs of identifiable people, licence plates, medical or client images are exactly that. The default path uploads nothing, and BYOD files stay in this runtime; nothing is sent to any service other than the downloads named here.",
        "- **External access (data):** besides the Hub, the default path fetches 27 pinned photographs (about 2.5 MB) from the public iNaturalist open-data bucket `inaturalist-open-data.s3.amazonaws.com` over HTTPS, digest-verified before decoding; every photo is CC0 and its observation page is recorded.",
    ],
    "guided": {
        "opening": [
            (
                "## How to use this notebook\n\n"
                "**Who this notebook is for.** Learners who can open a hosted notebook (Google Colab or Jupyter), run cells in order and "
                "read short Python, and who want to see how an image super-resolution model is checked against simple baselines. No "
                "prior experience with super-resolution is assumed: each term is explained where it is first needed, and the glossary "
                "below collects them. A GPU is recommended but not required; the **Prerequisites** give the details.\n\n"
                "**Running it.** In Colab, optionally choose *Runtime → Change runtime type → T4 GPU*, then *Runtime → Run all*. The "
                "default path needs no edit, no upload, no account, no token and no runtime restart. Sections 1–3 build an isolated "
                "environment from hash-locked packages before any model runs (about 90 s in the recorded T4 run; a later Run all in the "
                "same runtime reuses the environment); read ahead while they finish, or run the notebook one cell at a time with "
                "*Shift + Enter*. Re-running the Section 1 cell on its own is safe: it keeps this session's run directory, so the cells "
                "after it keep working.\n\n"
                "**Where the code runs.** The notebook kernel installs nothing and imports no model library. Each learner cell calls "
                "`run_stage('…')`, which runs one stage of the carried stage runner in its own process with the isolated environment's "
                "Python, streams what it prints, and stops the notebook with the stage's own error message if it fails. Stages hand "
                "results to each other only through files in the run directory: the verified snapshot, the photo cache and JSON records.\n\n"
                "**Two kinds of cell.** *Learner cells* (Sections 4–8) are the machine-learning workflow; each runs one stage and prints "
                "compact dictionaries for you to read. *Infrastructure cells* (Sections 1–3: the runtime check, the carried code, the "
                "isolated install and the pinned-weight staging) are collapsed and titled **Infrastructure**. You may run them without "
                "studying their implementation: they exist for reproducibility and provenance, not as prerequisite machine-learning "
                "knowledge. Open one with *Show code* if you are curious.\n\n"
                "**Form controls.** Two optional cells start with fields that Colab renders as a form: `RUN_ACTIVITY`, `ACTIVITY_KERNEL` "
                "and `ACTIVITY_JPEG_QUALITY` (Section 7), and `USE_BYOD`, `BYOD_MODE` and `BYOD_PATH` (Section 8). Leave them at their "
                "defaults for the first run: both branches are off, and the notes and sample answers describe the default path. The Section 1 "
                "infrastructure cell has one more, `NEW_RUN_DIRECTORY`, off by default.\n\n"
                "**Section tags.** Each numbered heading carries one tag. **[Concept]** — what the model does and why. **[Evaluation "
                "practice]** — how the evidence is produced and how to read it. **[Engineering]** — reproducibility, provenance and "
                "packaging.\n\n"
                "**Predict, then check.** Before each principal result a **Predict before running** prompt asks you to commit to an "
                "expectation; after it, **What to notice** describes normal output and a collapsed **Check your reasoning** answer "
                "follows each checkpoint. Write your own answer first, then open it. Exact numbers vary with the runtime and library "
                "versions, so the notes describe the shape of a normal result rather than fixed values."
            ),
            (
                "## The task: Input → Model/System → Output\n\n"
                "| Stage | Input | Model / system | Output |\n"
                "|---|---|---|---|\n"
                "| **Pairing (sample and BYOD `hr` mode)** | a sharp image (the reference) | crop to a multiple of 4 px, shrink 4x with bicubic interpolation | an input `lr` exactly ¼ the size of its reference `hr` |\n"
                "| **Super-resolution** | one RGB image, 8..256 px per side | rescale to [0, 1] → pad to the next 8 px window → Swin2SR (six residual Swin groups, pixel-shuffle head) → crop the padding → clamp and round | one `4H × 4W` uint8 RGB image |\n"
                "| **Evaluation** | the model's output, bicubic and nearest-neighbour enlargements of the same input, and the reference | Y-channel PSNR and SSIM after a 4 px border crop; a paired bootstrap over images | per-image scores, means, and the model-minus-bicubic difference with its spread |\n\n"
                "An image with no sharper original (Section 6, BYOD `lr` mode) goes through the second row only: it gets an output and "
                "no score.\n\n"
                "## Roadmap\n\n"
                "| Section | Tag | What happens | What you read |\n"
                "|---|---|---|---|\n"
                "| 1. Check the runtime | [Engineering] | Linux and disk checked, GPU detected if present; a fresh run directory | the accelerator |\n"
                "| 2. Carry the code, install the runtime | [Engineering] | carried files verified; an isolated hash-locked environment | versions |\n"
                "| 3. Pin, stage and verify | [Engineering] | the snapshot downloaded at its pinned commit and digest-checked | file list |\n"
                "| 4. Sample pairs and validation | [Evaluation practice] | 24 photographs → 24 pairs; refusal probes | pair sizes, padding, refusals |\n"
                "| 5. Run the task and evaluate | [Evaluation practice] | model vs bicubic vs nearest on identical inputs | the principal result |\n"
                "| 6. New-image inference and outputs | [Concept] | three photo crops and a synthetic scene with no reference | output sizes, files |\n"
                "| Interpretation and conclusion | [Evaluation practice] | limits and an evidence-based conclusion | your conclusion |\n"
                "| 7. Optional activity | [Concept] | change the degradation (off by default) | your comparison |\n"
                "| 8. Bring Your Own Data | [Evaluation practice] | your images in `hr` or `lr` mode (off by default) | your results |\n"
                "| Troubleshooting | [Engineering] | common hosted-runtime failures | when something fails |\n\n"
                "**Fast path.** Short on time? Run all, then read Section 5 and the conclusion: they carry the principal result. The "
                "canonical path ends with Section 6; Sections 7 and 8 change nothing unless you switch them on."
            ),
            (
                "<details>\n"
                "<summary><strong>Glossary</strong> — open when a term is unfamiliar</summary>\n\n"
                "| Term | Meaning in this notebook |\n"
                "|---|---|\n"
                "| **Super-resolution (SR)** | Producing a larger image from a smaller one by predicting the missing detail. |\n"
                "| **Scale factor (x4)** | Each side of the output is 4 times the input's: an 80 px input becomes 320 px, 16 times as many pixels. |\n"
                "| **Classical SR** | SR trained on inputs made by bicubic downscaling of clean photographs, as opposed to inputs with noise, blur or compression. |\n"
                "| **LR / HR** | Low-resolution input / high-resolution reference. Here `lr` is `hr` shrunk 4x. |\n"
                "| **Degradation** | The process that made the small image from the large one; here bicubic downsampling. A model trained for one degradation can fail on another. |\n"
                "| **Bicubic interpolation** | A classical enlargement that blends 4 × 4 neighbouring pixels with a cubic kernel. The baseline any SR model must beat. |\n"
                "| **Nearest-neighbour interpolation** | Enlargement by copying each pixel into a 4 × 4 block: the floor. |\n"
                "| **Window / padding** | Swin2SR attends inside 8 × 8 pixel windows, so the input is padded on the right and bottom up to the next multiple of 8; the padding is cut off the output. |\n"
                "| **Y channel (luma)** | The brightness component of an image (BT.601 formula). SR papers score on Y because the eye is most sensitive to brightness detail. |\n"
                "| **Border crop** | Removing 4 px (the scale factor) from every edge before scoring, because edge pixels depend on the padding rule. |\n"
                "| **PSNR** | Peak signal-to-noise ratio in dB: `10·log10(255² / mean squared error)`. Higher means pixel values closer to the reference; +1 dB is a clear change, but PSNR favours smooth images over textured ones. |\n"
                "| **SSIM** | Structural similarity: compares local brightness, contrast and structure in 11 × 11 Gaussian windows; 1 means identical. |\n"
                "| **Paired comparison** | Model and baseline are scored on the same images, and the per-image differences are analysed. |\n"
                "| **Bootstrap interval** | Resample the 24 per-image differences with replacement many times (seeded) and take the middle 95 % of the resampled means: how much the mean could move with a different draw of similar images. |\n"
                "| **Reference-free** | An input with no sharper original; its output can be inspected but not scored. |\n"
                "| **Digest (SHA-256)** | A fingerprint of a file's bytes; a single changed byte changes it. |\n"
                "| **Hash-locked environment** | A separate Python environment built from a requirements file that pins every package to one version and SHA-256; the installer refuses anything else. |\n"
                "| **Stage** | One step of the workflow run as its own process by `run_stage`; it reads the files earlier stages wrote and writes its own. |\n"
                "| **BYOD** | Bring Your Own Data: an optional branch that runs the same contract on your images. |\n\n"
                "</details>"
            ),
        ],
    },
    "setup": [
        {
            "cell": "check",
            "md": (
                "## 1. Check the runtime · [Engineering]\n\n"
                "> **Infrastructure.** The code cells in Sections 1–3 are collapsed. You may run them without studying their "
                "implementation; they exist for reproducibility and provenance. The learning activities start in Section 4.\n\n"
                "**Input:** a fresh hosted runtime. **System:** checks that it is Linux x86_64 with enough free disk, detects a GPU if "
                "one is attached, and creates a new run directory. **Output:** the accelerator and the directories this run will use. "
                "Each run writes to a new directory under `outputs/{stem}/`, so an earlier export cannot be mistaken for a current "
                "result. The verified snapshot and photographs are kept in `weights/` and reused by a later run."
            ),
            "after": (
                "**Expected result:** one dictionary naming the GPU (for example `Tesla T4, 15360 MiB`) or saying that the stages will "
                "run on the CPU, the kernel's Python version, the run directory, the weights directory, the isolated environment's "
                "directory and the free disk. If the cell stops with a platform or disk message, see **Troubleshooting**."
            ),
        },
        {
            "cell": "carrier",
            "md": (
                "## 2. Carry the code and install the locked runtime · [Engineering]\n\n"
                "> **Infrastructure.** The next two code cells are collapsed. The first **is** the repository's code, carried so that "
                "this notebook works on its own; the second builds the environment every stage runs in.\n\n"
                "The first cell holds, as text, the files the workflow needs: the package's four modules under "
                "`src/swin2sr_x4_super_resolution_pipeline/` (identity constants, snapshot verification and staging, input validation, "
                "the pipeline class, the metrics and baselines, the pinned sample and the BYOD loaders), the stage runner "
                "`tutorial_stages.py`, the hash-locked `requirements.txt` ({n_locked} packages), the snapshot manifest and the licence. "
                "It writes each file into the run directory and checks its SHA-256 against `CARRIED_HASHES`, stopping on any mismatch. "
                "The text is the repository's files byte for byte; the repository's parity test (`tests/test_notebook_parity.py`) fails "
                "whenever the two diverge, so what runs here is what the repository tests. Nothing in this cell runs a model."
            ),
            "after": (
                "**Expected result:** `carried_files`, `verified: True`, and the repository revision the notebook was generated from.\n\n"
                "The next cell installs nothing into this notebook's kernel. It downloads one pinned file — the `uv` installer wheel, "
                "refused unless its size and SHA-256 match — creates a separate virtual environment with its own CPython 3.12.12, and "
                "installs `requirements.txt` into it with `--require-hashes --only-binary :all:`: every package must be the locked "
                "version, a prebuilt wheel, and match a locked digest (the lock contains no source distribution, so nothing is built). "
                "Hugging Face tokens are removed from the environment the stages see, and `MPLBACKEND=Agg` is set there so any figure "
                "is written to a file that this kernel displays. The hosted runtime's own packages are never replaced, which is why no "
                "restart is needed. The cell also defines `run_stage`, `load_record` and `show_image`, the three helpers the learner "
                "cells use."
            ),
        },
        {
            "cell": "install",
            "md": (
                "**Infrastructure: the isolated environment.** Installation messages from `uv` are normal and can take a few minutes. "
                "A failed download or a hash mismatch stops the cell; never remove a pin or a hash to get past one."
            ),
            "after": (
                "**Expected result:** one dictionary with the generating revision, the isolated environment's Python (3.12.12), the "
                "`torch`, `transformers`, `numpy` and `PIL` versions, whether CUDA is usable (`'cuda': True` on a GPU runtime, `False` "
                "on a CPU runtime — both are supported), the number of locked packages and the setup time."
            ),
        },
        {
            "cell": "weights",
            "md": (
                "## 3. Pin, stage and verify the model · [Engineering]\n\n"
                "> **Infrastructure.** The next code cell is collapsed. It downloads about 49 MB of pinned weights and checks every "
                "file's size and SHA-256; you may run it without studying its implementation.\n\n"
                "The model identity is carried twice — `MODEL_ID`/`MODEL_REVISION` in the carried `pipeline.py` and the snapshot "
                "manifest (paths, byte sizes, SHA-256) — and the `weights` stage first checks that they agree. It installs the carried "
                "manifest into `weights/`, fetches exactly the files that are absent from the Hugging Face Hub **at the pinned commit** "
                "(never `main`), and re-hashes every file, raising on the first size or digest mismatch. Four files are staged: "
                "`README.md`, `config.json`, `preprocessor_config.json` and `model.safetensors` of `{MODEL_ID}`. There is no fallback "
                "to a different download, and no remote model code is executed. Every later stage verifies the snapshot again before "
                "loading it."
            ),
            "after": (
                "**What to notice:** the model id, revision, licence and file count; a `fetched` list of the files downloaded on this "
                "run (empty on a rerun, because staging only fetches files that are absent); and the list of verified files. A size or "
                "SHA-256 mismatch stops the cell with a `ValueError` naming the file, and an unpinned notebook stops with a message "
                "saying so — see **Troubleshooting**, and never edit a manifest to get past either."
            ),
        },
    ],
    "cells": [
        {
            "md": (
                "## 4. Sample photographs, x4 pairs and validation · [Evaluation practice]\n\n"
                "From here on, every code cell runs one stage of the carried runner with `run_stage`; its printed dictionaries appear "
                "under the cell. This cell runs the `prepare` stage. The sample is 27 research-grade iNaturalist photographs of six "
                "common North American birds, every one CC0, fetched by photo id from the open-data bucket and refused on any "
                "byte-size or SHA-256 mismatch (`fetch_photos`). Twenty-four of them — four per species — become **pairs**: the "
                "centred 320 px square of each photo is the reference `hr`, and `degrade` shrinks it 4x with bicubic interpolation "
                "to the 80 px input `lr`. That is the degradation this checkpoint was trained to invert, so this is the model's home "
                "ground. The other three photos are cut to native 96 px crops that were never shrunk by this code: Section 6 uses "
                "them, with a seeded synthetic scene, as new inputs that have no reference.\n\n"
                "**Validation** checks every pair before any model is loaded: unique ids, reference sides that are multiples of 4, "
                "an input exactly a quarter of its reference, and the size limits. It also writes, per input, what preprocessing "
                "will change — the conversion to RGB, the right and bottom **padding** up to the next 8 px window, and that no tiling "
                "is used. A **refusal probe** is a deliberately broken pair used to show that the check works.\n\n"
                "**Predict before running:** the first input is 80 × 80 px, already a multiple of the 8 px window. How many pixels "
                "of padding do you expect it to receive, and which rule will each of the five refusal probes break?\n\n"
                "**Expected result:** 24 pairs (four per species), one reference size `(320, 320)` and one input size `(80, 80)`, a "
                "first-input record showing 8 px of padding on the right and bottom (80 is already a multiple of 8, and the processor "
                "always pads up to the next window), the four new inputs with their padding, and five refusal probes — each "
                "`rejected` with a message that names the broken pair and the rule it broke."
            ),
            "code": "run_stage('prepare')",
        },
        {
            "md": (
                "**What to notice:** every probe line says `rejected`, and its message tells you what to fix — the same messages BYOD "
                "gives for a bad image. `outputs/{stem}_input_manifest.json` records the schema, every input's padding and the probe "
                "results; `outputs/{stem}_inputs.csv` lists every input with its source.\n\n"
                "**Checkpoint:** why does the notebook make its own inputs by shrinking sharp photographs, instead of scoring the model "
                "on small images found on the web?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "A score needs a reference: the image the output *should* have been. Shrinking a sharp photograph yourself gives you "
                "both halves of the pair, and you know exactly which degradation produced the input. A small image from the web has no "
                "sharper original, and its degradation — camera blur, sharpening, JPEG compression, an unknown resize filter — is "
                "unknown. The price of this design is that the sample tests the model only on the degradation it was trained for; "
                "Section 7 lets you see what happens when that assumption breaks. The pairing uses Pillow's bicubic filter, which is "
                "close to, but not identical with, the MATLAB-style bicubic used to make the upstream training data.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 5. Run the task and evaluate against two baselines · [Evaluation practice]\n\n"
                "**Question tested:** on inputs made by the degradation it was trained for, does the model reconstruct the references "
                "more closely than plain interpolation of the same inputs — and by how much, image by image?\n\n"
                "The `evaluate` stage loads the verified snapshot and calls `pipe.upscale` on each of the 24 inputs: the processor "
                "rescales the pixels to [0, 1] and pads them, the model runs once, and the padding is cropped off at 4x before the "
                "values are clamped and rounded to 8-bit. The same inputs are enlarged with **bicubic** interpolation (the classical "
                "baseline) and **nearest-neighbour** interpolation (the floor). All three are scored against the reference the same "
                "way: **PSNR** and **SSIM** on the luma (Y) channel after cropping 4 px from every edge — the convention of the "
                "classical-SR literature (Conde et al., 2023) — with RGB PSNR beside them. PSNR measures pixel-wise error (Wang & "
                "Bovik, 2009); SSIM compares local structure (Wang et al., 2004). Because model and bicubic are scored on identical "
                "images, the stage also reports the mean of the 24 per-image differences, how many images the model wins, and a "
                "seeded bootstrap interval for that mean (Efron, 1979).\n\n"
                "**Predict before running:** will the model beat bicubic on Y-PSNR — by a fraction of a decibel, about one, or several? "
                "Will it win on every one of the 24 images? And where will nearest-neighbour land?"
            ),
            "code": (
                "run_stage('evaluate')\n"
                "show_image('{stem}_panels.png', 'Input (80 px, enlarged for display) | bicubic x4 | Swin2SR x4 | reference (320 px)')"
            ),
        },
        {
            "md": (
                "**What to notice:** the `psnr_y`, `ssim_y` and `psnr_rgb` rows each list nearest, bicubic and model means; the "
                "`model_minus_bicubic` rows give the mean per-image difference, its 95 % bootstrap interval, and how many of the 24 "
                "images the model won; one line per species follows — **four crops per species, descriptive only**: four images cannot rank "
                "species, so do not read a difference between species lines as a property of the birds. In the panels, compare feather edges and the background: bicubic "
                "is smooth and soft, the model's output has crisper edges. The per-image numbers are in `outputs/{stem}_scores.csv`, "
                "every output image in `outputs/{stem}_sample_outputs/`, and the full record in `outputs/{stem}_evaluation_report.json`.\n\n"
                "**Checkpoint:** suppose, as in the recorded T4 run, the model wins by a few dB on average (there +2.69 dB PSNR-Y, 95 % "
                "bootstrap interval 2.09 to 3.35) and on every one of the 24 images. What does that interval tell you, and what does "
                "it *not* tell you?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "The interval says that, among images like these 24, the average advantage is unlikely to be zero: resampling the "
                "per-image differences rarely produces a mean near zero. It does not say the model wins on every image (read `wins`), "
                "it does not extend to other kinds of images (all 24 are centred crops of bird photographs from one website, shrunk by "
                "one filter), and it says nothing about how the output looks to a person — PSNR rewards getting average brightness "
                "right and penalises plausible texture placed slightly wrong, so a sharper-looking image can score lower. Two further "
                "limits: the comparison is one deterministic pass, so the interval reflects image-to-image variation only; and because "
                "the photos are public, overlap with the model's training data cannot be ruled out, although the upstream training "
                "sets are general photo collections rather than iNaturalist. A win on every image here is a statement about these 24 bicubic "
                "crops, not a guarantee on yours. If your run shows a different gap, trust your run.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 6. New-image inference and exported outputs · [Concept]\n\n"
                "**Question tested:** what does the model produce on inputs that were *not* made by shrinking a sharper image, and what "
                "can be said about those outputs?\n\n"
                "The `infer` stage upscales four reference-free inputs: three native 96 px crops of photographs outside the evaluation "
                "set, and a seeded synthetic scene of 100 × 76 px (gradient, sharp rectangle, ring, fine stripes, mild noise) whose odd "
                "size shows 4 px of padding on each padded side. It writes each output as a PNG with its SHA-256, a predictions CSV "
                "(input size, output size, padding, file, digest — one row per input id), a comparison panel, and "
                "`outputs/{stem}_result.json`: the model id and revision, the weight digest, device and precision, the runtime "
                "versions, the sample digest, the evaluation summary and every prediction. Finally it lists every file under "
                "`outputs/`.\n\n"
                "**Predict before running:** on the native photo crops, will the model's output look as convincing as in Section 5? "
                "What will the synthetic scene's stripes look like?"
            ),
            "code": (
                "run_stage('infer')\n"
                "show_image('{stem}_new_panels.png', 'New inputs (no reference): input | bicubic x4 | Swin2SR x4')"
            ),
        },
        {
            "md": (
                "**What to notice:** each new input prints its size, its 4x output size and its padding, and its `verdict` in the "
                "result record is `not-measurable`: there is no reference to score against. The native crops carry the camera's own "
                "blur, sharpening and JPEG compression, which is not the degradation the model learned to invert, so its added detail "
                "can look different from Section 5. The synthetic stripes are close to the finest pattern 96 px can hold; watch for "
                "invented or bent lines. This is the end of the canonical path; everything it produced is under `outputs/` in the run "
                "directory.\n\n"
                "**Checkpoint:** a colleague says the output of a native crop \"shows the bird's real feathers in 4x more detail\". "
                "How would you correct them?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "The extra pixels are the model's prediction of what a sharp original *might* have looked like, learned from training "
                "photographs; they are not information recovered from the scene. Without a reference there is no evidence that any "
                "particular added feather edge is real. A correct description is \"a 4x enlargement whose added detail is "
                "synthetic\". That distinction matters most when an output could be used as evidence — a face, a licence plate, a "
                "medical image — which is why the repository rules those uses out.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## Interpretation and limits · [Evaluation practice]\n\n"
                "The question this notebook can answer is narrow: on 24 centred crops of CC0 bird photographs, shrunk 4x by Pillow's "
                "bicubic filter, does the pinned Swin2SR x4 checkpoint reconstruct the references more closely than bicubic or "
                "nearest-neighbour interpolation of the same inputs? Your run answers it in Section 5, with the per-image differences "
                "and a bootstrap interval; read the size and direction of the gap there rather than from this text.\n\n"
                "The numbers are tutorial sample evidence, not a benchmark. PSNR and SSIM measure closeness to one reference, not "
                "perceived quality; 24 images from one source and one degradation give an interval over those images only; the "
                "upstream paper's Set5, Set14 or Urban100 results are not reproduced here and must not be quoted as this notebook's. "
                "On real small images (Section 6) the model's added detail cannot be scored at all.\n\n"
                "Successful execution proves that the recorded repository revision's package and stage runner, carried in this "
                "standalone notebook and run in an isolated hash-locked environment, can stage and digest-verify the pinned safetensors "
                "snapshot, fetch and validate digest-pinned photographs, run 4x super-resolution with reported padding, score it against "
                "two baselines with a stated metric convention, run reference-free inference, and emit the shown machine-readable "
                "outputs — without the repository being reachable. It does **not** establish benchmark superiority, production "
                "fitness, or image quality beyond the checks shown.\n\n"
                "## Conclude with evidence · [Evaluation practice]\n\n"
                "Complete this in your own words, using the numbers your run printed:\n\n"
                "> On [24 bicubic-downsampled 320 px bird-photo crops / your BYOD `hr` images], the Swin2SR classical x4 checkpoint "
                "reached a mean Y-PSNR of [model] dB against [bicubic] dB for bicubic and [nearest] dB for nearest-neighbour "
                "interpolation (SSIM [model] vs [bicubic]). The model won on [wins] of [n] images; the mean per-image difference to "
                "bicubic was [difference] dB with a 95 % bootstrap interval of [low, high]. The most important uncertainty or failure "
                "mode is [for example: one degradation and one image source; PSNR's blindness to perceived sharpness; invented detail "
                "on native crops]. These numbers do not show [behaviour on real low-resolution captures / perceived quality / other "
                "image types]. Next I would [specific next experiment].\n\n"
                "<details>\n<summary>Check your reasoning: what makes a conclusion strong? (open after writing yours)</summary>\n\n"
                "A strong conclusion names the data (24 crops, one source, one degradation), reports the model beside **both** "
                "baselines, gives the paired difference with its interval and the win count rather than the mean alone, and states that "
                "PSNR and SSIM measure closeness to a reference, not perceived quality. It keeps the reference-free outputs of Section 6 "
                "out of the score, and does not generalise to other degradations — Section 7 is where that assumption is tested. It "
                "ends with a specific next step: more and more varied images, another degradation, or a perceptual comparison by "
                "people. A weak conclusion says only that \"the model makes images sharper\".\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 7. Optional activity: change one thing — the degradation · [Concept]\n\n"
                "**Predict → Change one thing → Run → Observe → Explain.** This activity is off by default and changes nothing the "
                "canonical path produced: with `RUN_ACTIVITY = False` the next cell only prints how to switch it on. It runs the "
                "`activity` stage, which rebuilds the same 24 references, makes their inputs with a **different** degradation, and "
                "scores the model and bicubic again exactly as in Section 5. It downloads nothing new.\n\n"
                "**Change one thing:** set `ACTIVITY_KERNEL` to `'bilinear'`, `'box'` or `'nearest'` (the downsampling filter), or keep "
                "`'bicubic'` and set `ACTIVITY_JPEG_QUALITY` to a value such as 30 to add JPEG compression after downsampling "
                "(0 means none). Then set `RUN_ACTIVITY = True` and run the cell. The references, the model and the metric stay fixed; "
                "only the way the input was made changes.\n\n"
                "**Predict before running:** with box-filter or JPEG-compressed inputs, will the model's advantage over bicubic grow, "
                "shrink, or reverse? Write your prediction down."
            ),
            "code": (
                'RUN_ACTIVITY = False  # @param {{type:"boolean"}}\n'
                "ACTIVITY_KERNEL = 'bilinear'  # @param [\"bicubic\", \"bilinear\", \"box\", \"nearest\"]\n"
                'ACTIVITY_JPEG_QUALITY = 0  # @param {{type:"integer"}}\n\n'
                "if RUN_ACTIVITY:\n"
                "    run_stage('activity', '--kernel', ACTIVITY_KERNEL, '--jpeg-quality', ACTIVITY_JPEG_QUALITY)\n"
                "else:\n"
                "    print({{'activity': 'skipped (optional)', 'to_run': 'set RUN_ACTIVITY = True, choose ACTIVITY_KERNEL or ACTIVITY_JPEG_QUALITY, then run this cell'}})"
            ),
        },
        {
            "md": (
                "**Observe:** two rows (`psnr_y`, `ssim_y`), each with the model and bicubic under the default bicubic-made inputs and "
                "under your degradation, then the model-minus-bicubic difference for both with the new bootstrap interval. The full "
                "record is `outputs/{stem}_activity.json`.\n\n"
                "**Explain:** did the result match your prediction? Is the model's advantage a property of the model alone, or of the "
                "model *and* the degradation it was trained for?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "A classical-SR model learns to invert one specific degradation. Change the filter and the input carries a different "
                "blur or aliasing pattern than the model expects; add JPEG and it carries 8 × 8 block artefacts that the model was never "
                "shown, which it often sharpens instead of removing. The usual result is that the advantage over bicubic shrinks, and "
                "with JPEG it can reverse — the x2 sibling of this checkpoint scored below bicubic on JPEG-compressed inputs. "
                "\"Usually\" is the honest word: 24 images are a small sample, and a result against that direction is an observation "
                "about this run. Only the degradation changed, so any difference is caused by it. The lesson carries to real data: "
                "before trusting an SR model's scores, ask how your small images were made.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 8. Bring Your Own Data (optional) · [Evaluation practice]\n\n"
                "This branch is off by default (`USE_BYOD = False`). It runs the same validation, inference and output contract as the "
                "sample path on your images, in one of two modes:\n\n"
                "| `BYOD_MODE` | Your image is | What happens | Score |\n"
                "|---|---|---|---|\n"
                "| `'hr'` | the sharp original, 32..1024 px per side | cropped on the right/bottom to a multiple of 4 px if needed (the removed pixels are reported), shrunk 4x by bicubic, validated, upscaled | Y-PSNR, SSIM and RGB PSNR against your original, beside bicubic and nearest-neighbour |\n"
                "| `'lr'` | the small input, 8..256 px per side | validated (converted to RGB, alpha discarded, padding reported), upscaled 4x | none — there is no reference |\n\n"
                "**Accepted files:** one image, a folder, or a zip of up to 64 images (PNG, JPEG, BMP, TIFF or WebP; at most 4096 × 4096 "
                "decoded pixels each; 200 MB in total). Each file name becomes the output id, so names must be distinct. Zips are read "
                "in memory, never extracted: absolute paths, `..` traversal and symbolic links are refused. **Nothing is resized "
                "silently** — an image outside the limits is refused with a message that names the file, the rule and the fix, before "
                "any model is loaded.\n\n"
                "**Where your data goes:** uploaded files are written under this run's directory in the runtime and read only by the "
                "`byod` stage; nothing is sent anywhere. Do not upload confidential or restricted data unless you are authorized to "
                "process it in this hosted runtime.\n\n"
                "**How to run it:** set `USE_BYOD = True` and choose `BYOD_MODE`. Either set `BYOD_PATH` to a file, folder or zip that is "
                "already in the runtime (for example one you copied in from mounted storage), or leave it empty to get an upload dialog "
                "(Google Colab only). Then run the cell."
            ),
            "code": (
                'USE_BYOD = False  # @param {{type:"boolean"}}\n'
                "BYOD_MODE = 'hr'  # @param [\"hr\", \"lr\"]\n"
                "BYOD_PATH = ''  # @param {{type:\"string\"}}\n\n"
                "if USE_BYOD:\n"
                "    if BYOD_PATH:\n"
                "        byod_path = Path(BYOD_PATH)\n"
                "    else:\n"
                "        from google.colab import files\n"
                "        uploaded = files.upload()\n"
                "        byod_path = ROOT / 'byod' / uuid.uuid4().hex[:8]\n"
                "        byod_path.mkdir(parents=True)\n"
                "        for file_name, payload in uploaded.items():\n"
                "            (byod_path / Path(file_name).name).write_bytes(payload)\n"
                "    run_stage('byod', '--byod-path', byod_path.resolve(), '--byod-mode', BYOD_MODE)\n"
                "    show_image('{stem}_byod_' + BYOD_MODE + '/panels.png', 'Your images: input | bicubic x4 | Swin2SR x4' + (' | your original' if BYOD_MODE == 'hr' else ''))\n"
                "else:\n"
                "    print({{'byod': 'skipped (optional)', 'to_run': \"set USE_BYOD = True, BYOD_MODE = 'hr' or 'lr', and optionally BYOD_PATH, then run this cell\"}})"
            ),
        },
        {
            "md": (
                "**What to notice:** one line per image with its input size, output size, padding and any preprocessing (for `'hr'`, "
                "whether pixels were cropped to reach a multiple of 4); in `'hr'` mode, per-image scores and — with two or more images — "
                "the same comparison table as Section 5. Results go to `outputs/{stem}_byod_<mode>/`: the 4x PNGs with their SHA-256, "
                "a panel, and `byod_result.json` with the input manifest, preprocessing, scores and provenance. An invalid image stops "
                "the cell with `RuntimeError: Stage 'byod' failed …` followed by the validator's own message, for example "
                "`longer side 300 px > MAX_INPUT_SIDE 256 … crop or downscale the input`.\n\n"
                "Remember that `'hr'` mode scores your images under a degradation the notebook made, so it tells you how the model "
                "handles your *content*, not how it handles your camera's real small images."
            ),
        },
        {
            "md": (
                "## Troubleshooting · [Engineering]\n\n"
                "| Symptom | Likely cause | What to do |\n"
                "|---|---|---|\n"
                "| Section 1 stops with `This notebook needs a Linux x86_64 runtime` | a local Windows or macOS kernel, or an ARM machine | "
                "Use Google Colab, Kaggle, or a Linux x86_64 machine: the locked environment is built for manylinux x86_64 wheels. |\n"
                "| Section 1 prints `none: the stages will run on the CPU` | no GPU is attached | Nothing to fix: the notebook works on the "
                "CPU. For speed, choose *Runtime → Change runtime type → T4 GPU* and run all again from the top. |\n"
                "| Section 1 stops with `Not enough free disk` | the check asks for about 7 GB for the isolated environment (5.5 GB measured, plus a margin) | Start a fresh runtime with more "
                "free disk; an environment built from the same lock earlier in this runtime is reused and needs no more. |\n"
                "| `The run directory … has no carried files, or the isolated environment is gone: run the three Infrastructure cells again in order (Sections 1, 2 and 3)` | Section 1 was run with `NEW_RUN_DIRECTORY` ticked (a fresh, empty run directory), or the runtime's temporary directory was cleared | Run Sections 1, 2 and 3 again in order, then the cell you wanted, or choose *Runtime → Run all*. Re-running the Section 1 cell on its own with the default setting keeps the run directory and needs nothing else. |\n"
                "| `Carried file integrity failure` in Section 2 | a carried file was edited in the notebook | Do not edit the "
                "infrastructure cells; open a fresh copy of the notebook from the repository. |\n"
                "| `uv 0.12.15 wheel size/hash mismatch`, or a `URLError` / timeout while downloading it | a network failure or an "
                "unexpected response from PyPI | Re-run the Section 2 install cell. Never replace the pinned URL or digest. |\n"
                "| `CalledProcessError` from `uv venv` or `uv pip install` (a hash mismatch, `Failed to download`, HTTP 5xx) | a transient "
                "PyPI or network failure | Re-run the Section 2 install cell: `uv` reuses what it already downloaded. If a hash mismatch "
                "repeats, stop and report it — never remove `--require-hashes`, a pin or a hash to get past it. |\n"
                "| `RuntimeError: Stage '…' failed (exit 2): …` | the stage raised an error; the text after the colon is the stage's own "
                "message, and its full log (with the traceback) is printed above it and kept in the run directory's `logs/` | Find the "
                "message in the rows below. A stage reads only files, so after fixing the cause you can re-run that cell and the cells "
                "after it. |\n"
                "| Section 3: `this notebook revision carries no immutable revision` | the notebook was generated before the maintainer "
                "pinned the snapshot (`MODEL_REVISION = \"unpinned\"`) | Use a newer copy of the notebook from the repository. A maintainer "
                "runs `python tools/pin_snapshot.py` where huggingface.co is reachable, commits the manifest and regenerates the notebook. "
                "Never edit the manifest or the revision by hand. |\n"
                "| A download error (timeout, HTTP 429/5xx) in Section 3 | a transient Hugging Face Hub failure | Re-run the Section 3 cell: "
                "staging only fetches the files that are still absent. |\n"
                "| `ValueError: ...: size ... != manifest ...` or `sha256 ... != manifest ...` | a partial or corrupted download | Delete "
                "the named file under `weights/` and re-run the Section 3 cell. Never edit a manifest to get past a mismatch. |\n"
                "| A photograph fetch fails in Section 4 (`URLError`, or `... bytes, pinned ...` / `sha256 ... != pinned ...`) | a network "
                "failure or an unexpected response from the iNaturalist bucket | Re-run the Section 4 cell; photographs already verified "
                "are kept in `weights/inat-birds/` and reused. A repeated digest mismatch means the served file changed: do not use it. |\n"
                "| `… is missing: run the stage that writes it before …` | a learner cell was run before an earlier stage | Run the "
                "notebook from the top, or re-run the earlier cells in order. |\n"
                "| `CUDA out of memory` | another program holds GPU memory | Re-run the cell: each stage is its own process, so the failed "
                "stage's memory was released. Inputs are capped at 256 px, so the model alone needs well under 1 GB. |\n"
                "| `ModuleNotFoundError: No module named 'google.colab'` with `USE_BYOD = True` | the upload dialog needs Google Colab | Set "
                "`BYOD_PATH` to a file, folder or zip already in the runtime. |\n"
                "| `BYOD path not found` | `BYOD_PATH` points nowhere | Check the path with the file browser; relative paths are resolved "
                "from the notebook's working directory. |\n"
                "| `longer side … > MAX_INPUT_SIDE 256` (`'lr'` mode) or `longer side … > 1024 px` (`'hr'` mode) | the image is above the size "
                "ceiling | Crop or downscale it yourself; the notebook never resizes silently. |\n"
                "| `too small to score in 'hr' mode` or `shorter side … < MIN_INPUT_SIDE 8` | the image is below the size floor | Use `'lr'` "
                "mode for a small input, or a larger original. |\n"
                "| `not image files: [...]`, `not a decodable image`, `duplicate file names` or `has an absolute or parent-relative path` | "
                "the folder or zip holds other files, a corrupt image, two files with one name, or unsafe paths | Keep only distinct, "
                "decodable images in the folder or zip, with plain relative names. |"
            ),
        },
    ],
    "closing": (
        "## Transfer\n\n"
        "**Transfer:** switch on BYOD (Section 8) in `'hr'` mode with three or more sharp photographs of a different kind — text, "
        "architecture, satellite tiles, microscope images — and write the conclusion above for them. Before running, predict whether "
        "the model's advantage over bicubic will be larger or smaller than on the bird crops, and why. Then try one of your own small "
        "images in `'lr'` mode and describe its output without claiming that the added detail is real.\n\n"
        "## References\n\n"
        "- Conde, M. V., Choi, U.-J., Burchi, M., & Timofte, R. (2023). Swin2SR: SwinV2 transformer for compressed image super-resolution and restoration. In *Computer Vision – ECCV 2022 Workshops* (pp. 669–687). Springer. https://doi.org/10.1007/978-3-031-25063-7_42\n"
        "- Efron, B. (1979). Bootstrap methods: Another look at the jackknife. *The Annals of Statistics, 7*(1), 1–26. https://doi.org/10.1214/aos/1176344552\n"
        "- Liu, Z., Hu, H., Lin, Y., Yao, Z., Xie, Z., Wei, Y., Ning, J., Cao, Y., Zhang, Z., Dong, L., Wei, F., & Guo, B. (2022). Swin Transformer V2: Scaling up capacity and resolution. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 12009–12019). https://doi.org/10.1109/CVPR52688.2022.01170\n"
        "- Wang, Z., & Bovik, A. C. (2009). Mean squared error: Love it or leave it? A new look at signal fidelity measures. *IEEE Signal Processing Magazine, 26*(1), 98–117. https://doi.org/10.1109/MSP.2008.930649\n"
        "- Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: From error visibility to structural similarity. *IEEE Transactions on Image Processing, 13*(4), 600–612. https://doi.org/10.1109/TIP.2003.819861\n"
        "- Hugging Face model repository: https://huggingface.co/caidas/swin2SR-classical-sr-x4-64 (revision `{MODEL_REVISION}` as carried in the manifest)\n"
        "- Upstream code: https://github.com/mv-lab/swin2sr\n"
        "- Repository README: https://github.com/kurtvalcorza/swin2sr-x4-super-resolution-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/swin2sr-x4-super-resolution-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weights notes: https://github.com/kurtvalcorza/swin2sr-x4-super-resolution-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- uv (the installer that builds the isolated environment): https://docs.astral.sh/uv/\n"
        "- DIMER Notebook Specification 2.2 and Model Card Specification 1.2 (in the ml-worker repository)\n"
    ),
}
