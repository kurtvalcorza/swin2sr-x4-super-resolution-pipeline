"""Sample inputs and the data contract: digest-pinned CC0 photographs paired by 4x bicubic downsampling, a seeded
synthetic scene, structural validation of paired records, and the Bring-Your-Own-Data loaders.

**Paired records** are ``{id, hr, lr, category?}``: ``hr`` is the high-resolution reference (RGB, both sides
multiples of ``UPSCALE``) and ``lr`` is exactly ``hr`` downsampled by ``UPSCALE`` — the input the model is asked to
reverse. The default sample is 24 research-grade iNaturalist photographs of six North American bird species (four
per species, every photo CC0 1.0, one photo per observer per species), pinned by photo id, byte size and SHA-256 and
fetched from the public iNaturalist open-data bucket. Each photo's centred ``HR_CROP`` (320 px) square is the
reference; its input is that crop downsampled 4x with Pillow's bicubic filter (``degrade``) to 80 px — the same
kind of degradation the classical-SR checkpoint was trained to invert (the upstream training used MATLAB-style
bicubic downsampling, which differs slightly from Pillow's).

**New-image inputs** have no reference: three further pinned photographs, each cut to a native 96 px centre crop
that was never downsampled by this code, and a seeded synthetic scene (``synthetic_scene``) whose odd size shows the
window padding.

**BYOD** accepts one image, a directory or a zip of images in one of two modes: ``hr`` (your image is the reference;
it is cropped to a multiple of 4 if needed, downsampled 4x and scored) or ``lr`` (your image is the input; it is
upscaled only, with no score). Every rejection names the file, the rule and the fix.
"""
# ruff: noqa: E501  -- record and pin literals are kept on single lines

from __future__ import annotations

import csv
import hashlib
import io
import re
import urllib.request
import zipfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from .pipeline import MAX_INPUT_SIDE, MODEL_ID, UPSCALE, validate_image

CORPUS_NAME = "iNaturalist CC0 bird photographs (six species)"
CORPUS_BASE_URL = "https://inaturalist-open-data.s3.amazonaws.com/photos/"
CORPUS_LICENSE = "CC0 1.0 (each photo's licence as published by its observer on iNaturalist; the observation page is kept per record)"
DEFAULT_CACHE_DIR = Path("weights") / "inat-birds"
HR_CROP = 320  # px; every pinned photo's shorter side is at least 328 px
LR_SIDE = HR_CROP // UPSCALE  # 80 px
NEW_INPUT_CROP = 96  # px; native crops used as reference-free new inputs (output 384 px)
SYNTHETIC_SEED = 20260926
SYNTHETIC_SIZE = (100, 76)  # (width, height): deliberately not multiples of the 8 px window
KERNELS = {"bicubic": Image.Resampling.BICUBIC, "bilinear": Image.Resampling.BILINEAR, "box": Image.Resampling.BOX, "nearest": Image.Resampling.NEAREST}
DEFAULT_KERNEL = "bicubic"
MIN_HR_SIDE = 32  # after the 4 px border crop on each side SSIM still has a 24 px window field
MAX_HR_SIDE = MAX_INPUT_SIDE * UPSCALE  # 1024: the reference whose input sits at the 256 px input ceiling
MAX_RECORDS = 200
MAX_BYOD_FILES = 64
MAX_BYOD_BYTES = 200 * 1024 * 1024  # expanded bytes accepted from one directory or zip
MAX_IMAGE_PIXELS = 4096 * 4096  # decoded pixels per BYOD file, checked before full decoding
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
BYOD_MODES = ("hr", "lr")
_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
SPECIES: dict[str, tuple[str, str]] = {
    "song_sparrow": ("Melospiza melodia", "Song Sparrow"),
    "chipping_sparrow": ("Spizella passerina", "Chipping Sparrow"),
    "white_throated_sparrow": ("Zonotrichia albicollis", "White-throated Sparrow"),
    "dark_eyed_junco": ("Junco hyemalis", "Dark-eyed Junco"),
    "house_finch": ("Haemorhous mexicanus", "House Finch"),
    "american_goldfinch": ("Spinus tristis", "American Goldfinch"),
}
# (id, species key, iNat photo id, iNat observation id, observer login, bytes, sha256 of the served
#  <photo id>/medium.<ext>, ext). The digests pin the served bytes.
EVAL_RECORDS = (
    ('song_sparrow-00', 'song_sparrow', 129376982, 79016324, 'andywilson', 43427, '7a9d9304a82f202e992655ec5f65477cd3d7c1dce03aa89a214c2daa38f9d61d', 'jpg'),
    ('song_sparrow-01', 'song_sparrow', 480991086, 267636534, 'lyneisfilm', 162073, '11f77ff277dd2703c1000f2c787136ff0c3ca7ffad7017056892fe789d65efec', 'jpg'),
    ('song_sparrow-02', 'song_sparrow', 546060381, 302980489, 'swpollinators', 27899, '4e70b9519c6e5f7384a4495b91b45465f2b1599f86491d8f9f9b12635a4046f6', 'jpg'),
    ('song_sparrow-03', 'song_sparrow', 308625896, 177450028, 'radrat', 70961, '1211da4fdb24ae85ef0c6c3e2d03542c430457856aec661fe8f5f2de0027eee5', 'jpeg'),
    ('chipping_sparrow-00', 'chipping_sparrow', 198992636, 117809422, 'k-simpkins', 62563, 'cf9f3b0c1863808e21af596b2e609b047ddbc28cb2ed076625e2546425ff0adf', 'jpg'),
    ('chipping_sparrow-01', 'chipping_sparrow', 248210057, 144599194, 'w_mark_c', 193389, '497d0a0fef81c326bcc87b5d1eb97fe987d559b8f2b71bb60fe422ae34dce150', 'jpg'),
    ('chipping_sparrow-02', 'chipping_sparrow', 156350853, 94266719, 'ellyne', 142332, '6a60ebac34476a372b4790a87d823432cdda8f72590930b5d18ae166cf4c7ba2', 'jpeg'),
    ('chipping_sparrow-03', 'chipping_sparrow', 16128796, 11327134, 'reuvenm', 70532, '5ad36c9cdd6c92e225a1b8ab3c04d2f65bc4e971d0243958f8ac090b0996115c', 'jpeg'),
    ('white_throated_sparrow-00', 'white_throated_sparrow', 339621218, 193338380, 'rawcomposition', 31152, 'd1c08bfaca721bf0873437455b4cc010c6860d08b4777136c775007b0b9d07b6', 'jpg'),
    ('white_throated_sparrow-01', 'white_throated_sparrow', 166821399, 99992799, 'dziakj1', 125954, 'c595a41fbc8948b0d918b59117340dc320e2ba80d29e92bb9dacaaed5b404852', 'jpeg'),
    ('white_throated_sparrow-02', 'white_throated_sparrow', 469820434, 261505977, 'joy4birds', 111767, '7ce091492c73c68395667bb45578dfff11457511b0958d1d0d320d1df3e55ceb', 'jpg'),
    ('white_throated_sparrow-03', 'white_throated_sparrow', 260413022, 150969515, 'andywilson', 45958, 'fb7c533c92239775da6a5b353ff398f6bdd8f65c13445fc1dd72986af5a46466', 'jpeg'),
    ('dark_eyed_junco-00', 'dark_eyed_junco', 172110799, 102901486, 'schylerbrown', 182973, '185209c7a1111fc626a068e136ec3cfcdff3174d15f1c60af209d7b32fe7bef9', 'jpeg'),
    ('dark_eyed_junco-01', 'dark_eyed_junco', 46691943, 29901256, 'haida_gwaii', 46823, 'a92dca21e6e58c375fc313f0d1da2c86c11408beb0df8fd96fc60ae035ae80d5', 'jpg'),
    ('dark_eyed_junco-02', 'dark_eyed_junco', 707222551, 386266764, 'ben142', 289608, '7bf320edf4d34a4848f3e9d175cfafa66cc5bab6a1a8f97c41c670263de3ff89', 'jpg'),
    ('dark_eyed_junco-03', 'dark_eyed_junco', 346777340, 196961623, 'zacharyfoster', 46712, 'ea8f9f0eebb4d2f86193c705344a8fab09bf634bc2217a0874ee57c4f0f5b4ab', 'jpg'),
    ('house_finch-00', 'house_finch', 697940852, 381438133, 'ben142', 196735, 'a89f8e0263fdabb404b462acaa592f5dd2ac88ee4615da444470de4a1fae82d5', 'jpg'),
    ('house_finch-01', 'house_finch', 176982307, 105476125, 'vicki936', 22211, 'c377fb361df0324c7a856d9344968886ece3b94bd67188c9325b8d2d284d3a2f', 'jpeg'),
    ('house_finch-02', 'house_finch', 117990649, 72375345, 'kristen163', 75945, 'eeafad0dd2e91ecfe45c9d1f27dd0392a01bd81099549c60fd2e36a4b4342a9f', 'jpeg'),
    ('house_finch-03', 'house_finch', 389479656, 220010434, 'aster-asti', 82128, 'a8848197b4e7890e07538d492480c4275b75d04e10c1ae95aee91aaafe3319c5', 'jpg'),
    ('american_goldfinch-00', 'american_goldfinch', 84579952, 53187208, 'glennberry', 59673, '72d36079e592e0a83c2f774f9073bfd4cc81253452c925d1673217ddd4b52a36', 'jpeg'),
    ('american_goldfinch-01', 'american_goldfinch', 12533322, 9255418, 'braincellsgone', 55661, '6344e0125e74791f43ac6e07e5e1b9fbfce6d19bc62b6bb5d83b3caff9f7bcbc', 'jpg'),
    ('american_goldfinch-02', 'american_goldfinch', 131102823, 80016788, 'radrat', 98990, '232a944f7e3351d4916a12ef2f6d598e7b007caaa95a9b064a14c1ba3af2a6ff', 'jpeg'),
    ('american_goldfinch-03', 'american_goldfinch', 175048222, 104466897, 'eug302', 44231, '11c723482cc75fcc3a723ac1c0818a68e2fcf74c4ca684cf60195bf0d33f274e', 'jpg'),
)
NEW_RECORDS = (
    ('song_sparrow-04', 'song_sparrow', 494793016, 275349085, 'k-simpkins', 58410, '255538cf450197257e86ed3d41dc69fb78e594434e9cb338c6314288c6cff26e', 'jpg'),
    ('dark_eyed_junco-04', 'dark_eyed_junco', 8793471, 6892999, 'truthseqr', 45302, 'f8241eab39797c4e097a1432b13658466287d61ed515448457907c17e60cf5e2', 'jpeg'),
    ('american_goldfinch-04', 'american_goldfinch', 68849595, 43390778, 'mefisher', 154503, '66f07bc59bb3fdedd65a4537ebabd0cafd457826b8bf4bb633181f584a3edfd1', 'jpg'),
)
CORPUS_BYTES = sum(r[5] for r in EVAL_RECORDS + NEW_RECORDS)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def photo_url(photo_id: int, ext: str = "jpg") -> str:
    """The served object for a pinned photo (the bucket key is case-sensitive)."""
    if ext.lower() not in ("jpg", "jpeg", "png"):
        raise ValueError(f"unsupported photo extension {ext!r}")
    return f"{CORPUS_BASE_URL}{photo_id}/medium.{ext}"


def observation_url(observation_id: int) -> str:
    return f"https://www.inaturalist.org/observations/{observation_id}"


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "swin2sr-x4-super-resolution-pipeline"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - pinned https URL
        return response.read()


def fetch_photos(*, cache_dir: str | Path | None = None, fetcher: Callable[[str], bytes] | None = None) -> dict[str, bytes]:
    """Every pinned photo (bytes keyed by record id) from the cache or the open-data bucket; a cached or fetched file
    is refused on any byte-size or SHA-256 mismatch."""
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    fetch = fetcher or _download
    out = {}
    for rid, _label, photo_id, _obs, _user, size, digest, ext in EVAL_RECORDS + NEW_RECORDS:
        local = cache / f"{photo_id}.{ext}"
        data = local.read_bytes() if local.is_file() else b""
        if len(data) != size or _sha256_bytes(data) != digest:
            data = fetch(photo_url(photo_id, ext))
            if len(data) != size:
                raise ValueError(f"{rid}: photo {photo_id} is {len(data)} bytes, pinned {size}; re-run to fetch again")
            got = _sha256_bytes(data)
            if got != digest:
                raise ValueError(f"{rid}: photo {photo_id} sha256 {got} != pinned {digest}; the served file changed, do not use it")
            local.write_bytes(data)
        out[rid] = data
    return out


def centre_crop(image: Image.Image, size: int) -> Image.Image:
    """The centred `size` x `size` crop of an image as RGB (raises when the image is smaller)."""
    if image.width < size or image.height < size:
        raise ValueError(f"image {image.size} is smaller than the {size} px crop")
    left, top = (image.width - size) // 2, (image.height - size) // 2
    return image.convert("RGB").crop((left, top, left + size, top + size))


def degrade(hr: Image.Image, *, kernel: str = DEFAULT_KERNEL, jpeg_quality: int | None = None) -> Image.Image:
    """The low-resolution input for a reference: downsample by exactly UPSCALE with `kernel` (bicubic by default,
    the degradation the checkpoint was trained on), optionally JPEG-compressed at `jpeg_quality`. Deterministic."""
    if not isinstance(hr, Image.Image):
        raise TypeError("hr must be a PIL image")
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {sorted(KERNELS)}, got {kernel!r}")
    if hr.width % UPSCALE or hr.height % UPSCALE:
        raise ValueError(f"hr sides must be multiples of {UPSCALE}: {hr.size}")
    small = hr.convert("RGB").resize((hr.width // UPSCALE, hr.height // UPSCALE), KERNELS[kernel])
    if jpeg_quality is None:
        return small
    if not 1 <= int(jpeg_quality) <= 95:
        raise ValueError("jpeg_quality must be in 1..95")
    buffer = io.BytesIO()
    small.save(buffer, format="JPEG", quality=int(jpeg_quality))
    return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")


def _decode(data: bytes, where: str) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(data))
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise ValueError(f"{where}: {image.width} x {image.height} px exceeds {MAX_IMAGE_PIXELS:,} pixels; downscale it before use")
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"{where}: not a decodable image ({type(exc).__name__}); supply PNG, JPEG, BMP, TIFF or WebP") from exc
    return image


def build_sample_pairs(files: Mapping[str, bytes], *, kernel: str = DEFAULT_KERNEL, jpeg_quality: int | None = None) -> list[dict[str, Any]]:
    """The 24 evaluation records: centred HR_CROP reference, `degrade`d input, species as `category`, provenance."""
    out = []
    for rid, label, photo_id, obs_id, user, _size, _digest, _ext in EVAL_RECORDS:
        if rid not in files:
            raise ValueError(f"photo set is missing {rid}")
        hr = centre_crop(_decode(files[rid], rid), HR_CROP)
        out.append(
            {
                "id": rid,
                "hr": hr,
                "lr": degrade(hr, kernel=kernel, jpeg_quality=jpeg_quality),
                "category": label,
                "species": SPECIES[label][1],
                "inat_photo_id": photo_id,
                "inat_observation_url": observation_url(obs_id),
                "observer": user,
            }
        )
    return out


def build_new_inputs(files: Mapping[str, bytes], *, seed: int = SYNTHETIC_SEED) -> list[dict[str, Any]]:
    """Reference-free new inputs: three native NEW_INPUT_CROP crops of photos outside the evaluation set and the
    seeded synthetic scene."""
    out = []
    for rid, label, photo_id, obs_id, _user, _size, _digest, _ext in NEW_RECORDS:
        out.append({"id": f"new-{rid}", "lr": centre_crop(_decode(files[rid], rid), NEW_INPUT_CROP), "category": label, "inat_photo_id": photo_id, "inat_observation_url": observation_url(obs_id), "kind": "native photo crop (never downsampled here)"})
    out.append({"id": "new-synthetic-scene", "lr": synthetic_scene(seed=seed), "category": None, "kind": f"synthetic scene, seed {seed}"})
    return out


def synthetic_scene(*, seed: int = SYNTHETIC_SEED, size: tuple[int, int] = SYNTHETIC_SIZE) -> Image.Image:
    """A deterministic RGB test scene: a smooth gradient, a sharp-edged rectangle, a ring, fine stripes and seeded
    low-amplitude noise. Same seed and size, same pixels."""
    width, height = size
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    img = np.zeros((height, width, 3))
    img[..., 0] = 40 + 160 * xx / max(width - 1, 1)
    img[..., 1] = 60 + 120 * yy / max(height - 1, 1)
    img[..., 2] = 150
    img[height // 5 : height // 2, width // 8 : width // 2] = (230, 220, 60)
    r = np.hypot(xx - 0.72 * width, yy - 0.62 * height)
    img[(r > 0.12 * min(size)) & (r < 0.2 * min(size))] = (20, 30, 200)
    stripes = (np.sin(xx * 1.6) > 0) & (yy > 0.8 * height)
    img[stripes] = (250, 250, 250)
    img += rng.normal(0.0, 4.0, img.shape)
    return Image.fromarray(np.clip(img, 0, 255).round().astype(np.uint8))


# ---------------------------------------------------------------------------------------------------------
# Record contract
# ---------------------------------------------------------------------------------------------------------


def _check_record(record: Any, index: int) -> dict[str, Any]:
    where = f"records[{index}]"
    if not isinstance(record, Mapping):
        raise ValueError(f"{where} must be a mapping with id/hr/lr")
    for key in ("id", "hr", "lr"):
        if key not in record:
            raise ValueError(f"{where} is missing {key!r}")
    rid = record["id"]
    if not isinstance(rid, str) or not _ID_RE.match(rid):
        raise ValueError(f"{where}: id must match {_ID_RE.pattern}")
    hr, lr = record["hr"], record["lr"]
    for name, image in (("hr", hr), ("lr", lr)):
        if not isinstance(image, Image.Image):
            raise ValueError(f"{where}.{name} ({rid}) must be a PIL.Image.Image, got {type(image).__name__}")
    if min(hr.size) < MIN_HR_SIDE or max(hr.size) > MAX_HR_SIDE:
        raise ValueError(f"{where} ({rid}): hr sides must be within {MIN_HR_SIDE}..{MAX_HR_SIDE} px, got {hr.size}")
    if hr.width % UPSCALE or hr.height % UPSCALE:
        raise ValueError(f"{where} ({rid}): hr sides must be multiples of {UPSCALE} px, got {hr.size}; crop the reference first")
    if (lr.width * UPSCALE, lr.height * UPSCALE) != hr.size:
        raise ValueError(f"{where} ({rid}): lr {lr.size} is not hr {hr.size} downscaled by exactly {UPSCALE}")
    validate_image(lr, name=f"{where}.lr ({rid})")
    item = {"id": rid, "hr": hr.convert("RGB"), "lr": lr.convert("RGB")}
    for key in ("category", "species", "observer", "inat_photo_id", "inat_observation_url", "source", "preprocessing"):
        if record.get(key) is not None:
            item[key] = record[key]
    return item


def validate_pairs(records: Sequence[Mapping[str, Any]], *, max_records: int = MAX_RECORDS) -> dict[str, Any]:
    """Structural validation of paired records before any model import: ids, ×4 divisibility, exact pairing, ceilings."""
    if isinstance(records, Mapping) or not isinstance(records, Sequence) or isinstance(records, str | bytes):
        raise ValueError("records must be a list of {id, hr, lr} mappings")
    if not 1 <= len(records) <= max_records:
        raise ValueError(f"{len(records)} records; 1..{max_records} are accepted")
    checked, ids, categories = [], set(), {}
    for index, record in enumerate(records):
        item = _check_record(record, index)
        if item["id"] in ids:
            raise ValueError(f"duplicate id {item['id']!r}; ids must be unique so every output maps to one input")
        ids.add(item["id"])
        if item.get("category"):
            categories[item["category"]] = categories.get(item["category"], 0) + 1
        checked.append(item)
    return {
        "records": checked,
        "n_records": len(checked),
        "category_counts": dict(sorted(categories.items())),
        "hr_sizes": sorted({tuple(r["hr"].size) for r in checked}),
        "lr_sizes": sorted({tuple(r["lr"].size) for r in checked}),
        "digest": dataset_digest(checked),
        "model_id": MODEL_ID,
    }


def image_digest(image: Image.Image) -> str:
    """SHA-256 of the decoded RGB pixels (size-prefixed)."""
    rgb = image.convert("RGB")
    return _sha256_bytes(f"{rgb.width}x{rgb.height}:".encode() + rgb.tobytes())


def dataset_digest(records: Sequence[Mapping[str, Any]]) -> str:
    """Order-independent SHA-256 over (id, hr pixel digest, lr pixel digest); an LR-only record contributes '-' for hr."""
    parts = sorted(f"{r['id']}:{image_digest(r['hr']) if r.get('hr') is not None else '-'}:{image_digest(r['lr'])}" for r in records)
    return _sha256_bytes("\n".join(parts).encode("utf-8"))


def write_pairs_csv(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """A summary table (id, category, hr size, lr size, provenance), one row per record."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "category", "hr_width", "hr_height", "lr_width", "lr_height", "source"])
        for r in records:
            hr = r.get("hr")
            writer.writerow([r["id"], r.get("category") or "", hr.width if hr else "", hr.height if hr else "", r["lr"].width, r["lr"].height, r.get("inat_observation_url") or r.get("source") or r.get("kind", "")])
    return out


# ---------------------------------------------------------------------------------------------------------
# Bring Your Own Data
# ---------------------------------------------------------------------------------------------------------


def _safe_member(name: str) -> str:
    """Refuse absolute paths, drive letters and `..` traversal in an archive member name; return its base name."""
    posix = PurePosixPath(name.replace("\\", "/"))
    if posix.is_absolute() or ".." in posix.parts or re.match(r"^[A-Za-z]:", name):
        raise ValueError(f"zip member {name!r} has an absolute or parent-relative path; rebuild the zip with plain relative names")
    return posix.name


def read_byod_files(path: str | Path) -> list[tuple[str, bytes]]:
    """(name, bytes) for every image in one file, a directory (recursively) or a zip. Nothing is extracted to disk;
    symlinks, traversal paths, oversize archives, duplicate names and non-image files are refused with the reason."""
    source = Path(path)
    if not source.exists():
        raise ValueError(f"BYOD path not found: {source}; upload the file or set BYOD_PATH to an existing file or directory")
    items: list[tuple[str, bytes]] = []
    if source.is_dir():
        for file in sorted(source.rglob("*")):
            if file.is_symlink():
                raise ValueError(f"{file}: symbolic links are not accepted; copy the image itself")
            if file.is_file() and not file.name.startswith("."):
                items.append((file.name, file.read_bytes()))
    elif zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            infos = [i for i in archive.infolist() if not i.is_dir()]
            if sum(i.file_size for i in infos) > MAX_BYOD_BYTES:
                raise ValueError(f"{source.name}: expands to more than {MAX_BYOD_BYTES:,} bytes; split it into smaller zips")
            for info in infos:
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError(f"{source.name}: member {info.filename!r} is a symbolic link; symlinks are not accepted")
                name = _safe_member(info.filename)
                if name.startswith(".") or info.filename.startswith("__MACOSX/"):
                    continue
                items.append((name, archive.read(info)))
    elif source.is_file():
        items.append((source.name, source.read_bytes()))
    else:
        raise ValueError(f"{source} is neither a file, a directory nor a zip")
    unknown = [name for name, _ in items if not name.lower().endswith(IMAGE_SUFFIXES)]
    if unknown:
        raise ValueError(f"not image files: {unknown[:5]}; BYOD accepts {', '.join(IMAGE_SUFFIXES)} only")
    names = [name for name, _ in items]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"duplicate file names {duplicates[:5]}; every image needs a distinct name because the name becomes its output id")
    if not items:
        raise ValueError(f"no image found in {source}")
    if len(items) > MAX_BYOD_FILES:
        raise ValueError(f"{len(items)} images; at most {MAX_BYOD_FILES} per run are accepted")
    if sum(len(data) for _, data in items) > MAX_BYOD_BYTES:
        raise ValueError(f"more than {MAX_BYOD_BYTES:,} bytes of images; use fewer or smaller files")
    return items


def _byod_id(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]", "_", Path(name).stem)[:64] or "image"


def prepare_byod(path: str | Path, mode: str) -> dict[str, Any]:
    """Validate BYOD images in `mode` 'hr' (reference → crop to a multiple of 4, downsample 4x, score later) or 'lr'
    (input → upscale only). Every change made to an image is recorded under `preprocessing`; invalid images raise."""
    if mode not in BYOD_MODES:
        raise ValueError(f"BYOD mode must be one of {BYOD_MODES}, got {mode!r}: 'hr' = I have the sharp original, 'lr' = upscale my small image")
    records = []
    for name, data in read_byod_files(path):
        image = _decode(data, name)
        rid = _byod_id(name)
        if mode == "lr":
            validate_image(image, name=name)
            records.append({"id": rid, "lr": image.convert("RGB"), "source": name, "preprocessing": {"converted_to_rgb": image.mode != "RGB", "alpha_discarded": "A" in image.getbands()}})
            continue
        width, height = image.size
        if min(width, height) < MIN_HR_SIDE:
            raise ValueError(f"{name}: shorter side {min(width, height)} px < {MIN_HR_SIDE} px, too small to score in 'hr' mode; use 'lr' mode to upscale it instead")
        if max(width, height) > MAX_HR_SIDE:
            raise ValueError(f"{name}: longer side {max(width, height)} px > {MAX_HR_SIDE} px; in 'hr' mode the 4x-downsampled input must stay within {MAX_INPUT_SIDE} px, so crop the image first")
        cropped = (width - width % UPSCALE, height - height % UPSCALE)
        hr = image.convert("RGB").crop((0, 0, *cropped))
        records.append(
            {
                "id": rid,
                "hr": hr,
                "lr": degrade(hr),
                "source": name,
                "preprocessing": {
                    "converted_to_rgb": image.mode != "RGB",
                    "alpha_discarded": "A" in image.getbands(),
                    "cropped_to_multiple_of_4": [width % UPSCALE, height % UPSCALE] != [0, 0],
                    "pixels_removed": {"right": width % UPSCALE, "bottom": height % UPSCALE},
                    "degradation": f"{DEFAULT_KERNEL} downsampling by {UPSCALE} (Pillow)",
                },
            }
        )
    ids = [r["id"] for r in records]
    clashes = sorted({i for i in ids if ids.count(i) > 1})
    if clashes:
        raise ValueError(f"file names map to the same id {clashes}; rename the files so their stems differ")
    if mode == "hr":
        validate_pairs(records, max_records=MAX_BYOD_FILES)
    return {"mode": mode, "records": records, "n_images": len(records), "digest": dataset_digest(records)}
