"""Stage runner for the standalone Swin2SR x4 tutorial (NOTEBOOK_SPEC 2.2 §25.13 isolated-environment pattern).

The tutorial notebook carries this file verbatim (as ``tutorial_stages.py`` in its run directory, beside the carried
package under ``src/``) and runs every stage with the interpreter of an isolated, hash-locked environment::

    python -u tutorial_stages.py --root RUN_DIR --weights WEIGHTS_DIR --stage prepare

Nothing is installed into the notebook kernel. Each stage is a separate process, so a stage starts from files only:
the verified snapshot under ``--weights``, the digest-pinned photographs cached beside it, and the JSON records of
earlier stages. Learner-facing exports go to ``RUN_DIR/outputs``; hand-off state goes to ``RUN_DIR/state``. On
failure a stage writes ``RUN_DIR/state/<stage>.error.json`` with the exception type and message, which the notebook
re-raises in the kernel.

Stages: weights → prepare → evaluate → infer, plus the optional ``activity`` and ``byod``.
"""
# ruff: noqa: E501  -- the printed dictionaries are the learner-facing output; they are kept on one line each
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

STEM = "swin2sr_x4_super_resolution"
SNAPSHOT_KEY = "swin2sr-x4-64"
SAMPLE_CACHE = "inat-birds"
BOOTSTRAP_SEED = 0
PANEL_IDS = ("song_sparrow-00", "dark_eyed_junco-01", "american_goldfinch-02")


# --------------------------------------------------------------------------------------------------
# run context and small helpers
# --------------------------------------------------------------------------------------------------


class Run:
    """Paths of one run: carried sources and state under ``root``, the snapshot and photo cache under ``weights``."""

    def __init__(self, root: Path, weights: Path, options: argparse.Namespace) -> None:
        self.root = root
        self.weights = weights
        self.options = options
        self.out = root / "outputs"
        self.state = root / "state"
        self.out.mkdir(parents=True, exist_ok=True)
        self.state.mkdir(parents=True, exist_ok=True)

    @property
    def snapshot(self) -> Path:
        return self.weights / SNAPSHOT_KEY

    def write_state(self, name: str, value: Any) -> Path:
        path = self.state / name
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")
        return path

    def read_state(self, name: str, needed_by: str) -> Any:
        path = self.state / name
        if not path.is_file():
            raise RuntimeError(f"{name} is missing: run the stage that writes it before '{needed_by}' (run the notebook from the top)")
        return json.loads(path.read_text(encoding="utf-8"))

    def write_output(self, name: str, value: Any) -> Path:
        path = self.out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")
        return path


def runtime_versions() -> dict[str, Any]:
    import numpy
    import PIL
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": numpy.__version__,
        "pillow": PIL.__version__,
        "cuda": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor() or platform.machine(),
    }


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_png(array: Any, path: Path) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)
    return path


def panel(rows: list[list[tuple[str, Any]]], path: Path, tile: int = 320) -> Path:
    """A labelled grid: one row per image, one column per (label, uint8 array or PIL image)."""
    import numpy as np
    from PIL import Image, ImageDraw

    label_h = 18
    columns = max(len(r) for r in rows)
    sheet = Image.new("RGB", (tile * columns, (tile + label_h) * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    for r, row in enumerate(rows):
        for c, (label, image) in enumerate(row):
            im = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image))
            scale = tile / max(im.size)  # keep the aspect ratio; nearest-neighbour so pixels stay visible
            im = im.convert("RGB").resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.Resampling.NEAREST)
            x, y = c * tile, r * (tile + label_h)
            sheet.paste(im, (x + (tile - im.width) // 2, y + label_h + (tile - im.height) // 2))
            draw.text((x + 4, y + 3), label, fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return path


# --------------------------------------------------------------------------------------------------
# model and data factories (the CPU pre-flight test replaces load_pipeline with a tiny random-init model)
# --------------------------------------------------------------------------------------------------


def load_pipeline(run: Run) -> Any:
    from swin2sr_x4_super_resolution_pipeline import Swin2SRX4Pipeline

    return Swin2SRX4Pipeline.from_pretrained(weights_dir=run.snapshot, allow_download=False)


def load_photos(run: Run) -> dict[str, bytes]:
    from swin2sr_x4_super_resolution_pipeline import fetch_photos

    return fetch_photos(cache_dir=run.weights / SAMPLE_CACHE)


def load_pairs(run: Run, stage: str, *, kernel: str = "bicubic", jpeg_quality: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rebuild the evaluation pairs recorded by `prepare` and refuse if the default pairs changed."""
    from swin2sr_x4_super_resolution_pipeline import build_sample_pairs, dataset_digest

    data = run.read_state("data.json", stage)
    files = load_photos(run)
    if (kernel, jpeg_quality) == ("bicubic", None):
        records = build_sample_pairs(files)
        digest = dataset_digest(records)
        if digest != data["dataset_digest"]:
            raise RuntimeError(f"the sample pairs changed since 'prepare' (digest {digest[:16]}… != {data['dataset_digest'][:16]}…); re-run from Section 4")
    else:
        records = build_sample_pairs(files, kernel=kernel, jpeg_quality=jpeg_quality)
    return records, data


def upscale_all(pipe: Any, records: list[dict[str, Any]]) -> tuple[list[Any], list[float]]:
    outputs, seconds = [], []
    for record in records:
        result = pipe.upscale(record["lr"], name=record["id"])
        outputs.append(result["image"])
        seconds.append(result["seconds"])
    return outputs, seconds


def write_scores_csv(path: Path, model: dict[str, Any], bicubic: dict[str, Any], nearest: dict[str, Any]) -> Path:
    rows = {r["id"]: r for r in model["per_record"]}
    b = {r["id"]: r for r in bicubic["per_record"]}
    n = {r["id"]: r for r in nearest["per_record"]}
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "category", "model_psnr_y", "bicubic_psnr_y", "nearest_psnr_y", "model_ssim_y", "bicubic_ssim_y", "nearest_ssim_y", "model_psnr_rgb", "bicubic_psnr_rgb"])
        for rid, r in rows.items():
            writer.writerow([rid, r["category"] or "", r["psnr_y"], b[rid]["psnr_y"], n[rid]["psnr_y"], r["ssim_y"], b[rid]["ssim_y"], n[rid]["ssim_y"], r["psnr_rgb"], b[rid]["psnr_rgb"]])
    return path


# --------------------------------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------------------------------


def stage_weights(run: Run) -> None:
    """Section 3: install the carried manifest, fetch the absent files at the pinned revision, verify every file."""
    from swin2sr_x4_super_resolution_pipeline import (
        MANIFEST_NAME,
        MODEL_ID,
        MODEL_LICENSE,
        MODEL_REVISION,
        PIN_COMMAND,
        UNPINNED,
        stage_missing_files,
        verify_snapshot,
    )

    carried = run.root / "weights" / SNAPSHOT_KEY / MANIFEST_NAME
    manifest = json.loads(carried.read_text(encoding="utf-8"))
    if (manifest["modelId"], manifest["revision"]) != (MODEL_ID, MODEL_REVISION):
        raise RuntimeError("the carried manifest does not name the identity pinned by the package; regenerate the notebook")
    if MODEL_REVISION == UNPINNED:
        raise RuntimeError(
            f"this notebook revision carries no immutable revision for {MODEL_ID} (MODEL_REVISION is {UNPINNED!r}), so it refuses to download unverified weights. "
            f"A maintainer must run `{PIN_COMMAND}` and regenerate the notebook; see Troubleshooting"
        )
    target = run.snapshot
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(carried, target / MANIFEST_NAME)
    print({"model_id": MODEL_ID, "revision": MODEL_REVISION, "license": MODEL_LICENSE, "files": len(manifest["files"]), "total_bytes": manifest["totalBytes"]}, flush=True)
    fetched = stage_missing_files(target, allow_download=True)
    print({"weights_dir": str(target), "fetched": fetched}, flush=True)
    verified = verify_snapshot(target)
    print({"verified_files": verified["files"], "revision": verified["revision"]}, flush=True)
    run.write_output("weights.json", {"model_id": MODEL_ID, "revision": MODEL_REVISION, "license": MODEL_LICENSE, "files": verified["files"], "fetched": fetched, "total_bytes": verified["total_bytes"]})


def stage_prepare(run: Run) -> None:
    """Section 4: fetch the pinned photographs, build the x4 pairs, validate them, and probe the refusals."""
    from PIL import Image

    from swin2sr_x4_super_resolution_pipeline import (
        CORPUS_BYTES,
        HR_CROP,
        INPUT_SCHEMA,
        LR_SIDE,
        MAX_INPUT_SIDE,
        build_new_inputs,
        build_sample_pairs,
        validate_inputs,
        validate_pairs,
        write_pairs_csv,
    )

    t0 = time.perf_counter()
    files = load_photos(run)
    records = build_sample_pairs(files)
    print({"photos": len(files), "bytes": sum(len(v) for v in files.values()), "pinned_bytes": CORPUS_BYTES, "seconds": round(time.perf_counter() - t0, 1)})
    report = validate_pairs(records)
    print({"pairs": report["n_records"], "per_species": report["category_counts"], "hr_sizes": report["hr_sizes"], "lr_sizes": report["lr_sizes"], "digest": report["digest"][:16] + "..."})
    manifest = validate_inputs([r["lr"] for r in records], names=[r["id"] for r in records])
    first = manifest["inputs"][0]
    print({"first_input": first})
    print({"schema": INPUT_SCHEMA})
    new_inputs = build_new_inputs(files)
    new_manifest = validate_inputs([r["lr"] for r in new_inputs], names=[r["id"] for r in new_inputs])
    for item in new_manifest["inputs"]:
        print({"new_input": item["id"], "size": item["size"], "output_size": item["output_size"], "padding": item["padding"]})

    good = records[0]
    probes = {
        "lr is not hr/4": [{**good, "lr": good["hr"].resize((HR_CROP // 2, HR_CROP // 2))}],
        "hr not divisible by 4": [{**good, "hr": good["hr"].crop((0, 0, HR_CROP - 2, HR_CROP)), "lr": good["lr"]}],
        "pair above the size ceiling": [{"id": "too-large", "hr": Image.new("RGB", ((MAX_INPUT_SIDE + 8) * 4, 64)), "lr": Image.new("RGB", (MAX_INPUT_SIDE + 8, 16))}],
        "duplicate id": [good, good],
        "lr not an image": [{**good, "lr": "photo.jpg"}],
    }
    probe_results = []
    for name, probe in probes.items():
        try:
            validate_pairs(probe)
            probe_results.append({"probe": name, "verdict": "accepted"})
        except (TypeError, ValueError) as exc:
            probe_results.append({"probe": name, "rejected": str(exc)[:160]})
        print(probe_results[-1])
    if any(p.get("verdict") == "accepted" for p in probe_results):
        raise AssertionError(f"a refusal probe was accepted: {probe_results}")

    csv_path = write_pairs_csv(records + new_inputs, run.out / f"{STEM}_inputs.csv")
    input_manifest = {
        "sample": {"pairs": manifest, "new_inputs": new_manifest},
        "pairing": f"hr = centred {HR_CROP} px crop of the served photo; lr = hr downsampled 4x with Pillow bicubic ({LR_SIDE} px)",
        "dataset_digest": report["digest"],
        "probes": probe_results,
    }
    run.write_output(f"{STEM}_input_manifest.json", input_manifest)
    print({"input_manifest": f"outputs/{STEM}_input_manifest.json", "inputs_csv": str(csv_path.relative_to(run.root))})
    run.write_state("data.json", {"dataset_digest": report["digest"], "ids": [r["id"] for r in records], "new_ids": [r["id"] for r in new_inputs]})


def stage_evaluate(run: Run) -> None:
    """Section 5: upscale every sample input, score the model and the two baselines on the Y channel, write panels."""
    from swin2sr_x4_super_resolution_pipeline import (
        BORDER,
        METRIC_DEFINITIONS,
        bicubic_baseline,
        compare,
        interpolate,
        nearest_baseline,
        sr_metrics,
    )

    records, data = load_pairs(run, "evaluate")
    pipe = load_pipeline(run)
    t0 = time.perf_counter()
    outputs, seconds = upscale_all(pipe, records)
    model = sr_metrics(outputs, records)
    bicubic, nearest = bicubic_baseline(records), nearest_baseline(records)
    comparison = compare(model, bicubic, nearest, seed=BOOTSTRAP_SEED)
    print({"device": pipe.device, "images": len(records), "model_seconds_total": round(sum(seconds), 1), "seconds_per_image": round(sum(seconds) / len(seconds), 2), "border_px": BORDER})
    for key, row in comparison["means"].items():
        print({key: row})
    for key, row in comparison["model_minus_bicubic"].items():
        print({f"model_minus_bicubic_{key}": {k: row[k] for k in ("mean_difference", "ci95", "wins", "n_images")}})
    for category, row in model["per_category"].items():
        print({"species": category, "model_psnr_y": row["psnr_y"], "bicubic_psnr_y": bicubic["per_category"][category]["psnr_y"], "n": row["n"]})

    by_id = {r["id"]: (r, o) for r, o in zip(records, outputs, strict=True)}
    rows = []
    for rid in PANEL_IDS:
        record, output = by_id[rid]
        rows.append([(f"{rid} input 80px", record["lr"]), ("bicubic x4", interpolate(record["lr"], "bicubic")), ("Swin2SR x4", output), ("reference 320px", record["hr"])])
    panel_path = panel(rows, run.out / f"{STEM}_panels.png")
    scores_csv = write_scores_csv(run.out / f"{STEM}_scores.csv", model, bicubic, nearest)
    for record, output in zip(records, outputs, strict=True):
        save_png(output, run.out / f"{STEM}_sample_outputs" / f"{record['id']}_x4.png")
    report = {
        "task": "4x single-image super-resolution (classical SR, bicubic degradation)",
        "estimation": f"{len(records)} images, one deterministic pass; means over images; paired percentile bootstrap over images (seed {BOOTSTRAP_SEED}) for the model-minus-bicubic difference",
        "evidence_level": "tutorial sample evidence on 24 CC0 photographs, not a benchmark",
        "definitions": METRIC_DEFINITIONS,
        "border_px": BORDER,
        "dataset_digest": data["dataset_digest"],
        "model": model,
        "baselines": {"bicubic": bicubic, "nearest": nearest},
        "comparison": comparison,
        "device": pipe.device,
        "seconds_per_image": seconds,
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
    }
    run.write_output(f"{STEM}_evaluation_report.json", report)
    run.write_state("evaluate.json", {"comparison": comparison, "model_means": {k: model[k] for k in METRIC_DEFINITIONS}, "seconds_per_image": round(sum(seconds) / len(seconds), 3)})
    print({"panels": str(panel_path.relative_to(run.root)), "scores_csv": str(scores_csv.relative_to(run.root)), "report": f"outputs/{STEM}_evaluation_report.json"})


def stage_infer(run: Run) -> None:
    """Section 6: upscale the reference-free new inputs, export them, and write the result/provenance record."""
    from swin2sr_x4_super_resolution_pipeline import (
        CORPUS_BASE_URL,
        CORPUS_LICENSE,
        MODEL_ID,
        MODEL_LICENSE,
        MODEL_REVISION,
        PARAMETER_COUNT,
        build_new_inputs,
        interpolate,
    )

    data = run.read_state("data.json", "infer")
    evaluated = run.read_state("evaluate.json", "infer")
    new_inputs = build_new_inputs(load_photos(run))
    if [r["id"] for r in new_inputs] != data["new_ids"]:
        raise RuntimeError("the new inputs changed since 'prepare'; re-run from Section 4")
    pipe = load_pipeline(run)
    predictions = []
    rows = []
    for record in new_inputs:
        result = pipe.upscale(record["lr"], name=record["id"])
        path = save_png(result["image"], run.out / f"{STEM}_new_outputs" / f"{record['id']}_x4.png")
        predictions.append({"id": record["id"], "kind": record["kind"], "input": result["input"], "output_file": str(path.relative_to(run.root)), "output_sha256": file_sha256(path), "seconds": result["seconds"], "verdict": "not-measurable (no high-resolution reference)"})
        print({k: predictions[-1][k] for k in ("id", "seconds", "output_file")} | {"input_size": result["input_size"], "output_size": result["output_size"], "padding": result["input"]["padding"]})
        rows.append([(f"{record['id'][4:30]} input", record["lr"]), ("bicubic x4", interpolate(record["lr"], "bicubic")), ("Swin2SR x4", result["image"])])
    panel_path = panel(rows, run.out / f"{STEM}_new_panels.png")
    with open(run.out / f"{STEM}_predictions.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "kind", "input_width", "input_height", "output_width", "output_height", "pad_right", "pad_bottom", "output_file", "output_sha256"])
        for p in predictions:
            writer.writerow([p["id"], p["kind"], *p["input"]["size"], *p["input"]["output_size"], p["input"]["padding"]["right"], p["input"]["padding"]["bottom"], p["output_file"], p["output_sha256"]])

    source_path = run.root / "source.json"
    notebook_source = json.loads(source_path.read_text(encoding="utf-8")) if source_path.is_file() else None
    weights = json.loads((run.out / "weights.json").read_text(encoding="utf-8")) if (run.out / "weights.json").is_file() else None
    result_payload = {
        "notebook_source": notebook_source,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "license": MODEL_LICENSE, "parameters": PARAMETER_COUNT, "weight_sha256": pipe.weight_sha256, "device": pipe.device, "precision": "float32"},
        "provenance": {"snapshot": weights, "safetensors_only": True, "remote_code_executed": False, "data_base_url": CORPUS_BASE_URL, "data_license": CORPUS_LICENSE, "dataset_digest": data["dataset_digest"]},
        "runtime": {**runtime_versions(), "environment": "isolated hash-locked environment (one process per stage)"},
        "evaluation": evaluated["comparison"],
        "new_inputs": predictions,
        "outputs_dir": "outputs/",
    }
    run.write_output(f"{STEM}_result.json", result_payload)
    print({"new_panels": str(panel_path.relative_to(run.root)), "result": f"outputs/{STEM}_result.json"})
    print("outputs/:")
    for path in sorted(run.out.rglob("*")):
        if path.is_file() and "_sample_outputs" not in path.parts[-2]:
            print(f"  - {path.relative_to(run.root).as_posix()} ({path.stat().st_size / 1024:.1f} KB)")
    print(f"  - outputs/{STEM}_sample_outputs/ ({len(list((run.out / f'{STEM}_sample_outputs').glob('*.png')))} PNG files)")


def stage_activity(run: Run) -> None:
    """Section 7 (optional): change one thing — the downsampling kernel (or add JPEG) — and re-score model and bicubic."""
    from swin2sr_x4_super_resolution_pipeline import bicubic_baseline, compare, nearest_baseline, sr_metrics

    opts = run.options
    quality = opts.jpeg_quality or None
    records, _data = load_pairs(run, "activity", kernel=opts.kernel, jpeg_quality=quality)
    default = run.read_state("evaluate.json", "activity")["comparison"]["means"]
    pipe = load_pipeline(run)
    outputs, _seconds = upscale_all(pipe, records)
    changed = compare(sr_metrics(outputs, records), bicubic_baseline(records), nearest_baseline(records), seed=BOOTSTRAP_SEED)
    label = opts.kernel + (f" + JPEG {quality}" if quality else "")
    rows = {}
    for key in ("psnr_y", "ssim_y"):
        rows[key] = {"bicubic-degraded input (default)": {k: default[key][k] for k in ("model", "bicubic")}, f"{label} input": {k: changed["means"][key][k] for k in ("model", "bicubic")}}
        print({key: rows[key]})
    print({"model_minus_bicubic_psnr_y": {"default": round(default["psnr_y"]["model"] - default["psnr_y"]["bicubic"], 3), label: changed["model_minus_bicubic"]["psnr_y"]["mean_difference"], "ci95": changed["model_minus_bicubic"]["psnr_y"]["ci95"]}})
    run.write_output(f"{STEM}_activity.json", {"degradation": label, "comparison": rows, "detail": changed})


def stage_byod(run: Run) -> None:
    """Section 8 (optional): validate the reader's images, upscale them, and score them in 'hr' mode."""
    from swin2sr_x4_super_resolution_pipeline import (
        MODEL_ID,
        MODEL_REVISION,
        bicubic_baseline,
        compare,
        interpolate,
        nearest_baseline,
        prepare_byod,
        sr_metrics,
        validate_inputs,
    )

    opts = run.options
    prepared = prepare_byod(Path(opts.byod_path), opts.byod_mode)  # raises before any model import on invalid input
    records = prepared["records"]
    manifest = validate_inputs([r["lr"] for r in records], names=[r["id"] for r in records])
    for record, item in zip(records, manifest["inputs"], strict=True):
        print({"id": record["id"], "source": record["source"], "input_size": item["size"], "output_size": item["output_size"], "padding": item["padding"], "preprocessing": record["preprocessing"]})
    pipe = load_pipeline(run)
    out_dir = run.out / f"{STEM}_byod_{opts.byod_mode}"
    shutil.rmtree(out_dir, ignore_errors=True)
    outputs, seconds = upscale_all(pipe, records)
    files = []
    for record, output in zip(records, outputs, strict=True):
        path = save_png(output, out_dir / f"{record['id']}_x4.png")
        files.append({"id": record["id"], "source": record["source"], "output_file": str(path.relative_to(run.root)), "output_sha256": file_sha256(path)})
    payload: dict[str, Any] = {"mode": opts.byod_mode, "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "device": pipe.device, "input_digest": prepared["digest"], "inputs": manifest["inputs"], "preprocessing": {r["id"]: r["preprocessing"] for r in records}, "outputs": files, "seconds_per_image": seconds, "data_movement": "files stayed inside this runtime"}
    if opts.byod_mode == "hr":
        model = sr_metrics(outputs, records)
        comparison = compare(model, bicubic_baseline(records), nearest_baseline(records), seed=BOOTSTRAP_SEED) if len(records) >= 2 else None
        payload.update({"model": model, "comparison": comparison, "evidence_level": "your images under a synthetic bicubic degradation; not evidence about real low-resolution captures"})
        for row in model["per_record"]:
            print({"id": row["id"], "psnr_y": row["psnr_y"], "ssim_y": row["ssim_y"]})
        if comparison:
            for key, row in comparison["means"].items():
                print({key: row})
        rows = [[("input", r["lr"]), ("bicubic x4", interpolate(r["lr"], "bicubic")), ("Swin2SR x4", o), ("your original", r["hr"])] for r, o in list(zip(records, outputs, strict=True))[:4]]
    else:
        payload.update({"verdict": "not-measurable (lr mode: no reference)"})
        rows = [[("your input", r["lr"]), ("bicubic x4", interpolate(r["lr"], "bicubic")), ("Swin2SR x4", o)] for r, o in list(zip(records, outputs, strict=True))[:4]]
    panel(rows, out_dir / "panels.png")
    run.write_output(f"{STEM}_byod_{opts.byod_mode}/byod_result.json", payload)
    print({"byod_outputs": str(out_dir.relative_to(run.root)), "images": len(records), "result": f"{out_dir.relative_to(run.root)}/byod_result.json"})


STAGES = {
    "weights": stage_weights,
    "prepare": stage_prepare,
    "evaluate": stage_evaluate,
    "infer": stage_infer,
    "activity": stage_activity,
    "byod": stage_byod,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="run directory holding the carried sources")
    parser.add_argument("--weights", type=Path, required=True, help="directory holding the snapshot and the photo cache")
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--kernel", default="bilinear", choices=("bicubic", "bilinear", "box", "nearest"), help="activity: downsampling kernel")
    parser.add_argument("--jpeg-quality", type=int, default=0, help="activity: JPEG quality applied after downsampling (0 = none)")
    parser.add_argument("--byod-path", default="", help="byod: an image, a directory or a zip")
    parser.add_argument("--byod-mode", default="hr", choices=("hr", "lr"), help="byod: hr = score against your original; lr = upscale only")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    options = parse_args(argv)
    root = options.root.resolve()
    carried_src = root / "src"
    if carried_src.is_dir() and str(carried_src) not in sys.path:
        sys.path.insert(0, str(carried_src))
    run = Run(root, options.weights.resolve(), options)
    error_file = run.state / f"{options.stage}.error.json"
    error_file.unlink(missing_ok=True)
    started = time.perf_counter()
    try:
        STAGES[options.stage](run)
    except Exception as exc:  # the notebook re-raises this message in the kernel
        traceback.print_exc()
        message = str(exc) or repr(exc)
        error_file.write_text(json.dumps({"stage": options.stage, "type": type(exc).__name__, "message": message}), encoding="utf-8")
        print(f"STAGE FAILED ({options.stage}): {type(exc).__name__}: {message}", flush=True)
        return 2
    print({"stage": options.stage, "status": "ok", "seconds": round(time.perf_counter() - started, 1)}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
