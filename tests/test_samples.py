"""Sample pins, the x4 pairing, the record contract and the BYOD loaders (offline; the photo fetcher is faked)."""

# ruff: noqa: E501

from __future__ import annotations

import io
import os
import zipfile

import pytest
from PIL import Image

from conftest import synthetic_pairs, textured
from swin2sr_x4_super_resolution_pipeline import (
    CORPUS_BYTES,
    EVAL_RECORDS,
    HR_CROP,
    LR_SIDE,
    MAX_HR_SIDE,
    NEW_RECORDS,
    SPECIES,
    build_new_inputs,
    build_sample_pairs,
    dataset_digest,
    degrade,
    fetch_photos,
    image_digest,
    prepare_byod,
    read_byod_files,
    synthetic_scene,
    validate_pairs,
)

SYNTHETIC_DIGEST = image_digest(synthetic_scene())


def test_sample_pins():
    assert len(EVAL_RECORDS) == 24 and len(NEW_RECORDS) == 3
    assert {r[1] for r in EVAL_RECORDS} == set(SPECIES)
    assert all(sum(r[1] == s for r in EVAL_RECORDS) == 4 for s in SPECIES)
    ids = [r[0] for r in EVAL_RECORDS + NEW_RECORDS]
    photos = [r[2] for r in EVAL_RECORDS + NEW_RECORDS]
    assert len(set(ids)) == 27 and len(set(photos)) == 27  # the new inputs are distinct photographs
    assert all(len(r[6]) == 64 for r in EVAL_RECORDS + NEW_RECORDS)
    assert CORPUS_BYTES == 2_547_912
    assert (HR_CROP, LR_SIDE) == (320, 80)


def _jpeg(seed: int, size=(400, 336)) -> bytes:
    buffer = io.BytesIO()
    textured(size, seed).save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def test_fetch_refuses_size_and_digest_mismatch(tmp_path):
    with pytest.raises(ValueError, match="pinned"):
        fetch_photos(cache_dir=tmp_path, fetcher=lambda url: b"not the photo")
    first = EVAL_RECORDS[0]
    with pytest.raises(ValueError, match="sha256"):
        fetch_photos(cache_dir=tmp_path, fetcher=lambda url: b"\0" * first[5])


def test_build_pairs_and_new_inputs_from_bytes():
    files = {r[0]: _jpeg(i) for i, r in enumerate(EVAL_RECORDS + NEW_RECORDS)}
    pairs = build_sample_pairs(files)
    report = validate_pairs(pairs)
    assert report["n_records"] == 24 and report["hr_sizes"] == [(320, 320)] and report["lr_sizes"] == [(80, 80)]
    assert dataset_digest(build_sample_pairs(files)) == report["digest"]  # deterministic
    new = build_new_inputs(files)
    assert [r["id"] for r in new][-1] == "new-synthetic-scene" and new[0]["lr"].size == (96, 96)
    assert "hr" not in new[0]


def test_synthetic_scene_is_seeded():
    assert image_digest(synthetic_scene()) == SYNTHETIC_DIGEST
    assert image_digest(synthetic_scene(seed=1)) != SYNTHETIC_DIGEST
    assert synthetic_scene().size == (100, 76)


def test_degrade_contract():
    hr = textured((64, 48))
    assert degrade(hr).size == (16, 12)
    assert degrade(hr, kernel="box").size == (16, 12)
    assert degrade(hr, jpeg_quality=30).size == (16, 12)
    with pytest.raises(ValueError, match="multiples of 4"):
        degrade(textured((62, 48)))
    with pytest.raises(ValueError, match="kernel"):
        degrade(hr, kernel="lanczos")


def test_validate_pairs_refusals():
    good = synthetic_pairs(2)
    assert validate_pairs(good)["n_records"] == 2
    cases = {
        "downscaled by exactly 4": [{**good[0], "lr": good[0]["hr"].resize((32, 32))}],
        "multiples of 4": [{**good[0], "hr": good[0]["hr"].crop((0, 0, 62, 64))}],
        "duplicate id": [good[0], good[0]],
        "missing 'lr'": [{"id": "x", "hr": good[0]["hr"]}],
        "id must match": [{**good[0], "id": "bad id!"}],
        f"within 32..{MAX_HR_SIDE}": [{"id": "big", "hr": Image.new("RGB", (MAX_HR_SIDE + 4, 64)), "lr": Image.new("RGB", (MAX_HR_SIDE // 4 + 1, 16))}],
    }
    for message, records in cases.items():
        with pytest.raises(ValueError, match=message):
            validate_pairs(records)


def test_byod_hr_mode_crops_to_a_multiple_of_four_and_reports_it(tmp_path):
    textured((101, 66)).save(tmp_path / "odd.png")
    textured((64, 64), 1).convert("RGBA").save(tmp_path / "even.png")
    prepared = prepare_byod(tmp_path, "hr")
    by_id = {r["id"]: r for r in prepared["records"]}
    assert by_id["odd"]["hr"].size == (100, 64) and by_id["odd"]["lr"].size == (25, 16)
    assert by_id["odd"]["preprocessing"]["pixels_removed"] == {"right": 1, "bottom": 2}
    assert by_id["even"]["preprocessing"]["cropped_to_multiple_of_4"] is False
    assert by_id["even"]["preprocessing"]["alpha_discarded"] is True


def test_byod_lr_mode_from_a_zip(tmp_path):
    archive = tmp_path / "in.zip"
    with zipfile.ZipFile(archive, "w") as z:
        buffer = io.BytesIO()
        textured((40, 30)).save(buffer, "PNG")
        z.writestr("folder/small.png", buffer.getvalue())
    prepared = prepare_byod(archive, "lr")
    assert prepared["records"][0]["id"] == "small" and "hr" not in prepared["records"][0]


@pytest.mark.parametrize(
    ("mode", "size", "message"),
    [("lr", (300, 40), "MAX_INPUT_SIDE 256"), ("hr", (1100, 64), "longer side 1100 px > 1024"), ("hr", (20, 64), "too small to score"), ("xx", (40, 40), "mode must be one of")],
)
def test_byod_size_and_mode_refusals(tmp_path, mode, size, message):
    textured(size).save(tmp_path / "a.png")
    with pytest.raises(ValueError, match=message):
        prepare_byod(tmp_path / "a.png", mode)


def test_byod_file_refusals(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        read_byod_files(tmp_path / "missing")
    (tmp_path / "notes.txt").write_text("x")
    with pytest.raises(ValueError, match="not image files"):
        read_byod_files(tmp_path)
    (tmp_path / "notes.txt").unlink()
    (tmp_path / "broken.png").write_bytes(b"not a png")
    with pytest.raises(ValueError, match="not a decodable image"):
        prepare_byod(tmp_path, "lr")
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as z:
        z.writestr("../escape.png", b"x")
    with pytest.raises(ValueError, match="parent-relative"):
        read_byod_files(evil)
    dup = tmp_path / "dup.zip"
    with zipfile.ZipFile(dup, "w") as z:
        z.writestr("a/x.png", b"1")
        z.writestr("b/x.png", b"2")
    with pytest.raises(ValueError, match="duplicate file names"):
        read_byod_files(dup)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="no symlink support")
def test_byod_refuses_symlinks(tmp_path):
    folder = tmp_path / "in"
    folder.mkdir()
    textured((40, 40)).save(tmp_path / "real.png")
    (folder / "link.png").symlink_to(tmp_path / "real.png")
    with pytest.raises(ValueError, match="symbolic links"):
        read_byod_files(folder)
