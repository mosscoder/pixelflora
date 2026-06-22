"""Media downloader: polite, concurrent, resumable, deduplicating.

Fetches the actual image bytes referenced by each ImageRef, verifies they decode
as images, records true dimensions + sha256 + size, dedupes identical bytes, and
writes files to ``<out>/images/``. Updates each ImageRef in place. Bytes are only
ever fetched here — harvesting and filtering stay cheap and re-runnable.
"""
from __future__ import annotations

import hashlib
import io
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
    max_images: int | None = None,
) -> dict:
    img_dir = Path(out_dir) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # flatten to a work list, honoring the global image cap
    jobs: list[tuple[OccurrenceRecord, ImageRef]] = []
    for rec in records:
        for img in rec.images:
            jobs.append((rec, img))
            if max_images is not None and len(jobs) >= max_images:
                break
        if max_images is not None and len(jobs) >= max_images:
            break

    seen_hashes: dict[str, str] = {}
    stats = {"ok": 0, "failed": 0, "duplicate": 0, "too_small": 0}

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

    # modest concurrency; PoliteClient still rate-limits per host
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for result in ex.map(fetch, jobs):
            stats[result] = stats.get(result, 0) + 1
    return stats
