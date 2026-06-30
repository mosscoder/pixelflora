"""iNaturalist source — pixelflora's primary (and currently only) provider.

Talks to the iNaturalist API v1. Pushes as much filtering server-side as the API
supports (quality grade, geo, date range, photo license, captive exclusion, plant
phenology) and captures the iNat-native signals that matter for a clean training
set:

  * original/large/medium image selection (the API returns a thumbnail URL)
  * plant phenology annotation (controlled attribute 12) -> reproductive_condition
  * captive/cultivated flag                              -> establishment_means
  * coordinate obscuring (geoprivacy)                    -> coordinates_obscured
  * public_positional_accuracy                           -> coordinate_uncertainty_m
  * identification agreement / disagreement counts
"""
from __future__ import annotations

import re
from collections.abc import Iterator

from .. import licenses as lic
from ..schema import ImageRef, OccurrenceRecord, TaxonRef
from .base import Source

API = "https://api.inaturalist.org/v1"

# iNat "Plant Phenology" controlled term (attribute 12)
_PHENOLOGY = {13: "flowering", 14: "fruiting", 15: "budding", 21: "no evidence of flowering"}
_PHENOLOGY_REV = {v: k for k, v in _PHENOLOGY.items()}

# canonical license tag -> iNat photo_license code (for source-side filtering)
_TAG_TO_INAT = {
    lic.CC0: "cc0", lic.CC_BY: "cc-by", lic.CC_BY_SA: "cc-by-sa",
    lic.CC_BY_NC: "cc-by-nc", lic.CC_BY_NC_SA: "cc-by-nc-sa",
    lic.CC_BY_ND: "cc-by-nd", lic.CC_BY_NC_ND: "cc-by-nc-nd",
}
_SIZES = ("original", "large", "medium", "small", "square")
_SIZE_RE = re.compile(r"/(square|small|medium|large|original)\.")


def water_fill(avail: dict[int, int], total: int) -> dict[int, int]:
    """Allocate ``total`` candidates across keys as evenly as possible, bounded by each
    key's availability — the "equal per bucket, subject to supply" rule used for
    month-balanced sampling.

    Keys with enough supply are equalized to a common ceiling ``L`` chosen so
    ``Σ min(avail, L) = total``; scarcer keys contribute everything they have and their
    shortfall redistributes to the richer keys. If total supply <= ``total``, take
    everything. Returns integer quotas summing to ``min(total, Σ avail)``.
    """
    keys = [k for k, a in avail.items() if a > 0]
    if not keys or total <= 0:
        return {}
    supply = sum(avail[k] for k in keys)
    if supply <= total:
        return {k: avail[k] for k in keys}
    lo, hi = 0.0, float(max(avail[k] for k in keys))
    for _ in range(64):                       # binary-search the common ceiling L
        mid = (lo + hi) / 2
        if sum(min(avail[k], mid) for k in keys) < total:
            lo = mid
        else:
            hi = mid
    quota = {k: int(min(avail[k], hi)) for k in keys}
    deficit = total - sum(quota.values())     # integer remainder left by the floors
    headroom = sorted((k for k in keys if quota[k] < avail[k]), key=lambda k: -avail[k])
    j = 0
    while deficit > 0 and headroom:           # hand the remainder to keys above the ceiling
        k = headroom[j % len(headroom)]
        if quota[k] < avail[k]:
            quota[k] += 1
            deficit -= 1
        j += 1
    return quota


class INaturalistSource(Source):
    name = "inaturalist"

    # ---- taxonomy -------------------------------------------------------
    def resolve_taxon(self, genus, species, *, taxon_key=None) -> TaxonRef:
        if taxon_key:
            data = self.client.get_json(f"{API}/taxa/{taxon_key}").get("results", [{}])[0]
            tid = str(taxon_key)
        else:
            data = self.client.get_json(
                f"{API}/taxa", {"q": f"{genus} {species}", "rank": "species", "per_page": 1}
            ).get("results", [{}])[0]
            tid = str(data.get("id", ""))
        return TaxonRef(
            source=self.name, taxon_key=tid, scientific_name=data.get("name"),
            rank=data.get("rank"), genus=genus, species=species,
            kingdom=data.get("iconic_taxon_name"), accepted_name=data.get("name"),
            match_confidence=None,
        )

    # ---- harvest --------------------------------------------------------
    def _base_params(self, spec, taxon) -> dict:
        f = spec.filters
        p = {"taxon_id": taxon.taxon_key, "photos": "true"}
        if f.has_coordinates:
            p["geo"] = "true"
        if f.require_research_grade:
            p["quality_grade"] = "research"
        if f.year_range:
            p["d1"], p["d2"] = f"{f.year_range[0]}-01-01", f"{f.year_range[1]}-12-31"
        if getattr(f, "exclude_captive", False):
            p["captive"] = "false"
        if f.license:
            codes = [c for c in (_TAG_TO_INAT.get(x.upper()) for x in f.license) if c]
            if codes:
                p["photo_license"] = ",".join(codes)
        # plant phenology filter, pushed server-side when a single value is requested
        if len(f.reproductive_condition) == 1:
            vid = _PHENOLOGY_REV.get(f.reproductive_condition[0].lower())
            if vid:
                p["term_id"], p["term_value_id"] = 12, vid
        return p

    def estimate_count(self, spec, taxon) -> int | None:
        try:
            return self.client.get_json(
                f"{API}/observations", {**self._base_params(spec, taxon), "per_page": 0}
            ).get("total_results")
        except Exception:
            return None

    def harvest(self, spec, taxon) -> Iterator[OccurrenceRecord]:
        """Yield candidate occurrences. ``media.sampling`` selects the strategy:
        'recent' (newest-first by observation id) or 'month_balanced' (uniform across
        observed month-of-year, water-filled to availability)."""
        if spec.media.sampling == "month_balanced":
            yield from self._harvest_month_balanced(spec, taxon)
        else:
            yield from self._page(spec, taxon, spec.media.buffer)

    def _page(self, spec, taxon, cap, *, month=None) -> Iterator[OccurrenceRecord]:
        """Page newest-first (id desc) up to ``cap`` image-bearing records, optionally
        restricted to a single observed ``month`` (1-12). The candidate buffer (cap)
        sits above the target so the downloader can top up past dups/failures."""
        per_page, id_below, emitted = 200, None, 0
        size = spec.media.prefer_size if spec.media.prefer_size in _SIZES else "original"
        while emitted < cap:
            params = {**self._base_params(spec, taxon),
                      "per_page": per_page, "order_by": "id", "order": "desc"}
            if month is not None:
                params["month"] = month
            if id_below:
                params["id_below"] = id_below
            data = self.client.get_json(f"{API}/observations", params)
            results = data.get("results", [])
            if not results:
                break
            for obs in results:
                rec = self._normalize(obs, size=size)
                if rec.images:
                    emitted += 1
                    yield rec
                    if emitted >= cap:
                        break
            id_below = results[-1]["id"]
            if len(results) < per_page:
                break

    def _month_histogram(self, spec, taxon) -> dict[int, int]:
        """Per-month-of-year availability (all years) under the request's filters — the
        same filters _base_params pushes server-side, so it matches the harvestable pool."""
        params = {**self._base_params(spec, taxon),
                  "date_field": "observed", "interval": "month_of_year"}
        data = self.client.get_json(f"{API}/observations/histogram", params)
        res = (data.get("results") or {}).get("month_of_year") or {}
        return {int(k): int(v) for k, v in res.items() if int(v) > 0}

    def _harvest_month_balanced(self, spec, taxon) -> Iterator[OccurrenceRecord]:
        """Equalize candidates across month-of-year. Probe per-month availability, water-fill
        a per-month quota up to the candidate buffer, page each month newest-first, then
        ROUND-ROBIN the months so any prefix the downloader keeps (target_per_class) stays
        balanced. Falls back to newest-first if the histogram probe yields nothing."""
        try:
            avail = self._month_histogram(spec, taxon)
        except Exception:
            avail = {}
        allow = set(spec.media.months) if spec.media.months else set(range(1, 13))
        avail = {m: c for m, c in avail.items() if m in allow}
        if not avail:                                  # probe failed / empty -> graceful fallback
            yield from self._page(spec, taxon, spec.media.buffer)
            return
        quota = water_fill(avail, spec.media.buffer)
        by_month = {m: list(self._page(spec, taxon, q, month=m)) for m, q in sorted(quota.items())}
        months = [m for m in sorted(by_month) if by_month[m]]
        i = 0
        while any(i < len(by_month[m]) for m in months):
            for m in months:
                if i < len(by_month[m]):
                    yield by_month[m][i]
            i += 1

    # ---- normalization --------------------------------------------------
    def _normalize(self, obs: dict, *, size: str) -> OccurrenceRecord:
        oid = str(obs["id"])
        taxon = obs.get("taxon") or {}
        user = obs.get("user") or {}
        creator = user.get("name") or user.get("login")

        # phenology from annotations
        repro = None
        for a in obs.get("annotations", []):
            if a.get("controlled_attribute_id") == 12:
                repro = _PHENOLOGY.get(a.get("controlled_value_id")) or repro

        # coordinates (+ obscuring)
        lat = lng = None
        geo = obs.get("geojson") or {}
        if geo.get("type") == "Point":
            lng, lat = geo["coordinates"][0], geo["coordinates"][1]
        geoprivacy = obs.get("geoprivacy") or obs.get("taxon_geoprivacy")
        obscured = geoprivacy in ("obscured", "private")

        images = []
        for ph in obs.get("photos", []):
            url = ph.get("url")
            if not url:
                continue
            src_url = _SIZE_RE.sub(f"/{size}.", url, count=1)
            tag = lic.normalize(ph.get("license_code"))
            dims = ph.get("original_dimensions") or {}
            images.append(ImageRef(
                image_id=str(ph.get("id")), source_url=src_url,
                license=tag, license_url=lic.license_url(tag),
                rights_holder=creator, creator=creator,
                attribution=_credit(ph.get("attribution"), creator, tag),
                width=dims.get("width"), height=dims.get("height"),
            ))

        od = obs.get("observed_on_details") or {}
        captive = obs.get("captive")
        return OccurrenceRecord(
            source=self.name, occurrence_id=oid, uuid=obs.get("uuid"),
            source_record_url=f"https://www.inaturalist.org/observations/{oid}",
            basis_of_record="HUMAN_OBSERVATION",
            scientific_name=taxon.get("name"),
            common_name=taxon.get("preferred_common_name"),
            taxon_rank=taxon.get("rank"), taxon_key=str(taxon.get("id", "")),
            iconic_taxon=taxon.get("iconic_taxon_name"),
            publisher="iNaturalist", dataset_title="iNaturalist",
            recorded_by=creator, observer_login=user.get("login"),
            occurrence_license=lic.normalize(obs.get("license_code")),
            decimal_latitude=lat, decimal_longitude=lng,
            coordinate_uncertainty_m=obs.get("public_positional_accuracy")
                or obs.get("positional_accuracy"),
            coordinates_obscured=obscured, geoprivacy=geoprivacy,
            locality=obs.get("place_guess"),
            event_date=obs.get("observed_on"),
            year=od.get("year"), month=od.get("month"), day=od.get("day"),
            reproductive_condition=repro,
            establishment_means=("cultivated" if captive else "wild") if captive is not None else None,
            quality_grade=obs.get("quality_grade"),
            num_identification_agreements=obs.get("num_identification_agreements"),
            num_identification_disagreements=obs.get("num_identification_disagreements"),
            images=images,
        ).fill_derived()


def _credit(source_attribution: str | None, creator: str | None, tag: str) -> str:
    """Use iNaturalist's own credit line, but ensure the creator's name is present
    (iNat's CC0 string is just 'no rights reserved', dropping the name)."""
    att = (source_attribution or "").strip()
    if creator and creator not in att:
        return f"{creator} — {att}" if att else f"{creator} ({tag})"
    return att or f"({tag})"
