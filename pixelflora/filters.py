"""Predicate engine applying a request's filters to normalized records.

Occurrence-level filters drop whole records; image-level filters (license,
min pixels) prune images within a surviving record. Returns the kept records
(with pruned image lists) plus a tally of rejection reasons for the report.
"""
from __future__ import annotations

from collections import Counter

from .request import FiltersSpec
from .schema import OccurrenceRecord


def apply_filters(
    records: list[OccurrenceRecord], f: FiltersSpec
) -> tuple[list[OccurrenceRecord], Counter]:
    kept: list[OccurrenceRecord] = []
    reasons: Counter = Counter()

    keep_lic = {x.upper() for x in f.license}
    drop_lic = {x.upper() for x in f.exclude_license}
    basis = {x.upper() for x in f.basis_of_record}
    repro = {x.lower() for x in f.reproductive_condition}

    for r in records:
        # ---- occurrence-level ----
        if f.has_coordinates is True and not r.has_coordinates:
            reasons["no_coordinates"] += 1
            continue
        if f.require_research_grade and (r.quality_grade or "").lower() != "research":
            reasons["not_research_grade"] += 1
            continue
        if basis and (r.basis_of_record or "").upper() not in basis:
            reasons["basis_of_record"] += 1
            continue
        if f.exclude_captive and (r.establishment_means or "").lower() == "cultivated":
            reasons["captive"] += 1
            continue
        if f.year_range and r.year is not None and not (f.year_range[0] <= r.year <= f.year_range[1]):
            reasons["year_range"] += 1
            continue
        if (f.max_coordinate_uncertainty_m is not None and r.coordinate_uncertainty_m is not None
                and r.coordinate_uncertainty_m > f.max_coordinate_uncertainty_m):
            reasons["coordinate_uncertainty"] += 1
            continue
        if repro and (r.reproductive_condition or "").lower() not in repro:
            reasons["reproductive_condition"] += 1
            continue

        # ---- image-level ----
        imgs = []
        for img in r.images:
            if keep_lic and img.license.upper() not in keep_lic:
                reasons["license_not_allowed"] += 1
                continue
            if drop_lic and img.license.upper() in drop_lic:
                reasons["license_excluded"] += 1
                continue
            imgs.append(img)
        if not imgs:
            reasons["no_images_after_image_filters"] += 1
            continue
        r.images = imgs
        kept.append(r)

    return kept, reasons
