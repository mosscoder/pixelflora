"""End-to-end orchestration: resolve -> harvest -> filter -> download ->
assemble -> split -> attribution -> (optional) publish. Writes a manifest and
provenance artifacts at each step so any pull is inspectable and reproducible.
"""
from __future__ import annotations

import json
import shutil
from collections import Counter, OrderedDict
from pathlib import Path

from . import attribution
from .config import Config
from .dataset import assemble_configs
from .download import download_images
from .filters import apply_filters
from .http import PoliteClient
from .publish import publish, resolve_repo_id
from .request import RequestSpec
from .schema import OccurrenceRecord
from .sources import get_source


def _log(msg: str) -> None:
    print(f"[pixelflora] {msg}", flush=True)


def _cap_images_per_occurrence(rec: OccurrenceRecord, spec: RequestSpec) -> None:
    n = spec.media.images_per_occurrence
    if isinstance(n, int) and n > 0:
        rec.images = rec.images[:n]


def _cap_per_species(records: list[OccurrenceRecord], max_images: int) -> list[OccurrenceRecord]:
    """Trim each class (label) down to at most ``max_images`` images, so a
    multi-species request stays roughly balanced across classes."""
    by_label: "OrderedDict[str, list[OccurrenceRecord]]" = OrderedDict()
    for r in records:
        by_label.setdefault(r.label, []).append(r)
    out: list[OccurrenceRecord] = []
    for recs in by_label.values():
        count = 0
        for r in recs:
            if count >= max_images:
                break
            r.images = r.images[: max_images - count]
            count += len(r.images)
            if r.images:
                out.append(r)
    return out


def run(request_path: str, config: Config | None = None) -> dict:
    config = config or Config.load()
    spec = RequestSpec.from_toml(request_path)
    out = Path(spec.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    client = PoliteClient(
        user_agent=config.user_agent, timeout_s=config.timeout_s,
        max_retries=config.max_retries, rate_limit_s=config.rate_limit_s,
        cache_dir=config.cache_dir,
    )
    # Photo files live on iNaturalist's bulk open-data CDN/S3, a different host than the
    # rate-limited API, so they get their own concurrent, un-throttled client. The API
    # harvest above stays polite at rate_limit_s; only static file fetches are sped up.
    media_client = PoliteClient(
        user_agent=config.user_agent, timeout_s=config.timeout_s,
        max_retries=config.max_retries, rate_limit_s=config.download_rate_limit_s,
        cache_dir=None,
    )

    # 1. resolve + 2. harvest — per species (a request may list many), per source
    all_records: list[OccurrenceRecord] = []
    taxa = []
    budget = spec.media.buffer  # per species: harvest a candidate buffer above the target (max_images)
    for sp in spec.species:
        label = RequestSpec.label_of(sp)
        for source_name in spec.sources:
            src = get_source(source_name, config, client)
            t = src.resolve_taxon(sp.genus, sp.species, taxon_key=sp.taxon_key)
            taxa.append(t)
            _log(f"{source_name}: resolved {t.scientific_name} "
                 f"(taxon_key={t.taxon_key}) for class '{label}'")
            got = 0
            for rec in src.harvest(spec, t):
                rec.label = label
                _cap_images_per_occurrence(rec, spec)
                all_records.append(rec)
                got += len(rec.images)
                if got >= budget:
                    break
    _log(f"harvested {len(all_records)} occurrences across "
         f"{len({RequestSpec.label_of(s) for s in spec.species})} class(es)")
    _write_manifest(out / "manifest.raw.jsonl", all_records)

    # 3. filter, then cap to the per-class candidate buffer. The downloader trims to the
    #    target (max_images) after fetching, topping up past duplicates and failures.
    kept, reasons = apply_filters(all_records, spec.filters)
    kept = _cap_per_species(kept, spec.media.buffer)
    per_class = Counter(r.label for r in kept for _ in r.images)
    _log(f"filter+cap: kept {len(kept)} candidate occurrences; per-class candidates={dict(per_class)}; "
         f"rejections={dict(reasons)}")
    _write_manifest(out / "manifest.filtered.jsonl", kept)
    (out / "rejections.json").write_text(json.dumps(dict(reasons), indent=2))

    # 4. download bytes (per-class cap already applied; no global cap here).
    #    Resume from any prior run's image manifest: reuse files already fetched ok,
    #    retry only what failed, and carry dedup across runs.
    prior = _load_prior_images(out / "manifest.images.jsonl")
    if prior:
        _log(f"resume: {len(prior)} images in prior manifest; reusing successful ones, retrying the rest")
    stats = download_images(
        kept, str(out), media_client, workers=config.download_workers,
        min_pixels=spec.media.min_pixels, max_dimension=spec.media.max_dimension,
        prior=prior, target_per_class=spec.media.max_images,
    )
    downloaded_per_class = Counter(
        r.label for r in kept for img in r.images if img.download_status == "ok")
    _log(f"download: {stats}; per-class kept={dict(downloaded_per_class)}")
    _write_manifest(out / "manifest.images.jsonl", kept)

    # 5. assemble + 6. split: one configuration per species, each split on its own
    configs = assemble_configs(kept, spec.split)
    config_sizes = {name: {s: ds.num_rows for s, ds in dd.items()}
                    for name, dd in configs.items()}
    _log(f"assemble: {len(configs)} configuration(s): {config_sizes}")
    for name, dd in configs.items():
        cfg_dir = out / "dataset" / name
        if cfg_dir.exists():           # re-runs: clear stale shards before re-saving
            shutil.rmtree(cfg_dir)
        dd.save_to_disk(str(cfg_dir))
    _write_metadata_csv(out / "metadata.csv", configs)

    # 7. attribution / provenance artifacts
    accessed = attribution.today()
    lic_report = attribution.license_report(kept)
    (out / "license_report.json").write_text(json.dumps(lic_report, indent=2))
    (out / "bibliography.bib").write_text(attribution.bibliography(kept, accessed))
    if spec.dataset.name:
        name = spec.dataset.name
    elif spec.is_multispecies:
        name = f"Botanical dataset ({len(config_sizes)} species)"
    else:
        name = RequestSpec.label_of(spec.species[0])
    (out / "CITATION.cff").write_text(
        attribution.citation_cff(name, accessed, lic_report["total_images"], spec.sources))
    (out / "README.md").write_text(attribution.dataset_card(
        name=name, description=spec.dataset.description, taxa=taxa, records=kept,
        split_strategy=spec.split.strategy, config_sizes=config_sizes, lic_report=lic_report,
        accessed=accessed, sources=spec.sources))

    # 8. publish (private by default; dry run unless opted in): one config per species
    pub = publish(configs, str(out), spec, config)
    _log(f"publish: {pub}")

    summary = {
        "classes": dict(downloaded_per_class),
        "configs": config_sizes,
        "taxa": [t.model_dump() for t in taxa],
        "sources": spec.sources,
        "occurrences_kept": len(kept),
        "download_stats": stats,
        "license_report": lic_report,
        "repo_id": resolve_repo_id(spec, config),
        "publish": pub,
        "output_dir": str(out),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    _log(f"done -> {out}")
    return summary


def _load_prior_images(path: Path) -> dict | None:
    """Map image_id -> a previous run's per-image manifest record, so a re-run can
    reuse already-downloaded files and retry only the failures. Returns None if no
    prior manifest exists (a first run)."""
    if not path.exists():
        return None
    prior: dict = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        for img in rec.get("images", []):
            iid = img.get("image_id")
            if iid:
                prior[iid] = img
    return prior or None


def _write_manifest(path: Path, records: list[OccurrenceRecord]) -> None:
    with path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r.model_dump(exclude={"raw"}), default=str) + "\n")


def _write_metadata_csv(path: Path, configs) -> None:
    """One flat table across all configurations, with config and split columns."""
    import csv
    rows, fieldnames = [], None
    for cfg_name, dd in configs.items():
        for split_name, ds in dd.items():
            cols = [c for c in ds.column_names if c != "image"]
            fieldnames = ["config", "split"] + cols
            for r in ds.remove_columns(["image"]):
                rows.append({"config": cfg_name, "split": split_name, **r})
    if fieldnames:
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
