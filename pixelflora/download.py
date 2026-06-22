"""Media downloader: polite, concurrent, resumable, deduplicating.

Fetches the actual image bytes referenced by each ImageRef, verifies they decode
as images, records true dimensions + sha256 + size, dedupes identical bytes, and
writes files to ``<out>/images/``. Updates each ImageRef in place. Bytes are only
ever fetched here — harvesting and filtering stay cheap and re-runnable.

Resume: pass ``prior`` (image_id -> a previous run's per-image manifest record) and
any image already fetched ok whose file is still on disk is reused untouched. Only
new or previously failed images are (re)fetched, and prior checksums seed the dedup
set, so retrying failures never re-downloads or duplicates data we already hold.

Target: pass ``target_per_class`` to fetch each class only until that many unique
images have actually landed. Duplicates and failed fetches do not count, so the
downloader keeps pulling from the harvested buffer until the target is met.
"""
from __future__ import annotations

import hashlib
import io
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from .http import PoliteClient
from .schema import ImageRef, OccurrenceRecord

_EXT = {"JPEG": "jpg", "PNG": "png", "GIF": "gif", "WEBP": "webp", "TIFF": "tiff"}


def _species_folder(label: str | None) -> str:
    """Filesystem-friendly per-species subfolder name, e.g. 'Lupinus_sericeus'."""
    return "_".join((label or "unspecified").split())


def download_images(
    records: list[OccurrenceRecord], out_dir: str, client: PoliteClient,
    *, workers: int = 4, min_pixels: int = 0, max_dimension: int = 0,
    prior: dict | None = None, target_per_class: int | None = None,
) -> dict:
    img_dir = Path(out_dir) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    seen_hashes: dict[str, str] = {}
    stats = {"ok": 0, "failed": 0, "duplicate": 0, "too_small": 0, "reused": 0}

    # Resume pass: an image a prior run already fetched (ok/duplicate) whose file is
    # still on disk is reused as-is — never re-downloaded — and its checksum seeds
    # the dedup set so a retried image identical to it is still caught. Everything
    # else (new, previously failed, previously too-small) is grouped by class, in
    # harvest order, as a candidate to (re)fetch.
    pending: "OrderedDict[str, list[tuple[OccurrenceRecord, ImageRef]]]" = OrderedDict()
    have_ok: dict[str, int] = {}
    for rec in records:
        label = rec.label or rec.scientific_name or "unspecified"
        for img in rec.images:
            p = prior.get(img.image_id) if prior else None
            if (p and p.get("download_status") in ("ok", "duplicate")
                    and p.get("local_path") and Path(p["local_path"]).exists()):
                img.download_status = p["download_status"]
                img.local_path = p["local_path"]
                img.sha256 = p.get("sha256")
                img.width, img.height = p.get("width"), p.get("height")
                img.image_format, img.file_size = p.get("image_format"), p.get("file_size")
                if img.sha256:
                    seen_hashes.setdefault(img.sha256, img.local_path)
                stats["reused"] += 1
                if img.download_status == "ok":
                    have_ok[label] = have_ok.get(label, 0) + 1
                continue
            pending.setdefault(label, []).append((rec, img))

    def fetch(job):
        rec, img = job
        try:
            data = client.get_bytes(img.source_url)
            im = Image.open(io.BytesIO(data))
            im.verify()
            im = Image.open(io.BytesIO(data))  # reopen after verify
            w, h = im.size
            fmt = (im.format or "JPEG").upper()
        except Exception:
            img.download_status = "failed"
            return "failed"
        if min_pixels and min(w, h) < min_pixels:
            img.download_status = "too_small"
            return "too_small"
        # downscale so the longest edge <= max_dimension, preserving aspect ratio
        # (downscale only — never upscale; re-encode the capped image)
        if max_dimension and max(w, h) > max_dimension:
            save_fmt = fmt if fmt in ("PNG", "WEBP") else "JPEG"
            if save_fmt == "JPEG" and im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            im.thumbnail((max_dimension, max_dimension), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format=save_fmt, **({"quality": 90} if save_fmt == "JPEG" else {}))
            data = buf.getvalue()
            w, h, fmt = im.width, im.height, save_fmt
        sha = hashlib.sha256(data).hexdigest()
        img.width, img.height, img.image_format = w, h, fmt.lower()
        img.sha256, img.file_size = sha, len(data)
        if sha in seen_hashes:
            img.local_path = seen_hashes[sha]
            img.download_status = "duplicate"
            return "duplicate"
        ext = _EXT.get(fmt, "jpg")
        species_dir = img_dir / _species_folder(rec.label)
        species_dir.mkdir(parents=True, exist_ok=True)
        path = species_dir / f"{rec.source}_{rec.occurrence_id}_{img.image_id.replace(':', '-')}.{ext}"
        if not path.exists():
            path.write_bytes(data)
        img.local_path = str(path)
        img.download_status = "ok"
        seen_hashes[sha] = str(path)
        return "ok"

    # Fetch per class, stopping once the class reaches target_per_class unique images.
    # Duplicates and failed fetches don't count toward the target, so the next buffered
    # candidate is pulled to top it up. With no target, every candidate is fetched.
    # Each batch is sized to the remaining need, so we fetch about target + losses, no
    # more. (PoliteClient still rate-limits per host.)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for label, cands in pending.items():
            idx = 0
            while idx < len(cands) and (
                    target_per_class is None or have_ok.get(label, 0) < target_per_class):
                if target_per_class is None:
                    batch = cands[idx:]
                else:
                    need = target_per_class - have_ok.get(label, 0)
                    batch = cands[idx:idx + max(need, 1)]
                idx += len(batch)
                for result in ex.map(fetch, batch):
                    stats[result] = stats.get(result, 0) + 1
                    if result == "ok":
                        have_ok[label] = have_ok.get(label, 0) + 1
    return stats
