"""Provenance & attribution artifacts.

Because the assembled dataset is PRIVATE training data for a model (not a public
image release), these artifacts serve citation/provenance for the model and any
peer-reviewed publication — not redistribution clearance. We emit:

  * license_report.json   — exact per-image license breakdown
  * bibliography.bib       — one BibTeX entry per contributing dataset (+ GBIF/iNat)
  * CITATION.cff           — machine-readable citation for the assembled dataset
  * README.md (card)       — a private "data provenance statement" with the field
                             schema, source/contributor/license breakdown, the
                             split design, and how to cite the sources.
"""
from __future__ import annotations

import datetime as _dt
from collections import Counter

from . import licenses as lic
from .schema import OccurrenceRecord


def license_report(records: list[OccurrenceRecord]) -> dict:
    per_image = Counter()
    redistributable = 0
    total = 0
    for r in records:
        for img in r.images:
            if img.download_status not in ("ok", "duplicate"):
                continue
            total += 1
            per_image[img.license] += 1
            if lic.is_redistributable(img.license):
                redistributable += 1
    return {
        "total_images": total,
        "by_license": dict(per_image),
        "redistributable_images": redistributable,
        "non_redistributable_images": total - redistributable,
        "note": ("Dataset is PRIVATE training data for a model; non-redistributable "
                 "licenses are retained and used internally, not republished."),
    }


def _contributors(records) -> Counter:
    c = Counter()
    for r in records:
        for img in r.images:
            if img.download_status in ("ok", "duplicate"):
                c[img.creator or r.recorded_by or "Unknown"] += 1
    return c


def _datasets(records) -> dict[str, dict]:
    """Group contributing sources for citation."""
    out: dict[str, dict] = {}
    for r in records:
        key = r.dataset_key or r.dataset_title or (r.publisher or r.source)
        d = out.setdefault(key, {"title": r.dataset_title or r.publisher or r.source,
                                 "doi": r.dataset_doi, "publisher": r.publisher,
                                 "source": r.source, "count": 0})
        d["count"] += sum(1 for i in r.images if i.download_status in ("ok", "duplicate"))
    return out


def bibliography(records, accessed: str) -> str:
    entries = []
    for i, (key, d) in enumerate(sorted(_datasets(records).items())):
        cite_key = f"{d['source']}_{i}"
        doi = f"  doi = {{{d['doi']}}},\n" if d.get("doi") else ""
        entries.append(
            f"@misc{{{cite_key},\n"
            f"  title = {{{d['title']}}},\n"
            f"  author = {{{d.get('publisher') or d['source']}}},\n"
            f"  howpublished = {{Accessed via {d['source']} on {accessed}}},\n"
            f"{doi}"
            f"  year = {{{accessed[:4]}}}\n}}"
        )
    return "\n\n".join(entries) + "\n"


def citation_cff(name: str, accessed: str, n_images: int, sources: list[str]) -> str:
    return (
        "cff-version: 1.2.0\n"
        f"title: {name}\n"
        "message: >-\n"
        "  This private dataset aggregates third-party biodiversity media. Cite the\n"
        "  underlying source datasets (see bibliography.bib) when publishing.\n"
        "type: dataset\n"
        f"date-released: {accessed}\n"
        "keywords:\n  - botany\n  - biodiversity\n  - image-classification\n"
        f"abstract: >-\n  {n_images} attributed botanical images harvested from "
        f"{', '.join(sources)} for model training.\n"
    )


def class_counts(records) -> Counter:
    c = Counter()
    for r in records:
        c[r.label or r.scientific_name] += sum(
            1 for i in r.images if i.download_status in ("ok", "duplicate"))
    return c


def dataset_card(*, name, description, taxa, records, split_strategy, config_sizes,
                 lic_report, accessed, sources) -> str:
    contributors = _contributors(records)
    datasets = _datasets(records)
    classes = class_counts(records)
    lic_rows = "\n".join(
        f"| {k} | {v} | {'yes' if lic.is_redistributable(k) else 'no'} |"
        for k, v in sorted(lic_report["by_license"].items(), key=lambda x: -x[1])
    )
    # taxon line: one species inline, many -> a Classes table below
    taxa_by_label = {}
    for t in taxa:
        taxa_by_label.setdefault(t.scientific_name, t)
    if len(classes) == 1:
        t0 = taxa[0] if taxa else None
        taxon_line = (f"- **Taxon:** {next(iter(classes))} "
                      f"(`{t0.taxon_key if t0 else ''}`, {t0.rank if t0 else 'species'})")
    else:
        taxon_line = f"- **Classes:** {len(classes)} species (see below)"
    class_rows = "\n".join(f"| {k} | {v} |" for k, v in classes.most_common())
    config_rows = "\n".join(
        f"| `{cfg}` | " + ", ".join(f"{s} {n}" for s, n in sorted(sizes.items())) + " |"
        for cfg, sizes in config_sizes.items()
    )
    ds_rows = "\n".join(
        f"| {d['title']} | {d['source']} | {d.get('doi') or '—'} | {d['count']} |"
        for d in sorted(datasets.values(), key=lambda x: -x["count"])
    )
    top_contrib = "\n".join(f"- {who} ({n})" for who, n in contributors.most_common(15))
    return f"""---
tags:
  - botany
  - biodiversity
  - image-classification
pretty_name: {name}
viewer: false
---

# {name}

> **Private training data — not for redistribution.** The shipped product is a
> model; these images are aggregated third-party media used internally under
> their original licenses. {description or ''}

{taxon_line}
- **Sources:** {', '.join(sources)}
- **Images:** {lic_report['total_images']}
- **Accessed:** {accessed}
- **Split strategy:** `{split_strategy}`

## Classes (species)

| class (label) | images |
|---|---|
{class_rows}

## Configurations (one per species)

Each species is a separate configuration with its own train and test split,
loaded with `load_dataset("<repo_id>", "<configuration>")`. New species can be
added later as new configurations without changing the ones already published.

| configuration | splits (images) |
|---|---|
{config_rows}

## License breakdown (per image)

| license | images | publicly redistributable |
|---|---|---|
{lic_rows}

{lic_report['note']}

## Contributing datasets / sources

| dataset | source | DOI | images |
|---|---|---|---|
{ds_rows}

When publishing, cite these sources (see `bibliography.bib`) and iNaturalist
(www.inaturalist.org), accessed {accessed}.

## Top contributors (photographer / collector)

{top_contrib}

## Field schema

Every row is one image with denormalized provenance, spatial, temporal, and
botanical-trait metadata (organization, author, dataset+DOI, collection date,
basis of record = field vs. museum, coordinates + uncertainty, phenology, etc.).
See `metadata.csv` for the full per-image table.
"""


def today() -> str:
    return _dt.date.today().isoformat()
