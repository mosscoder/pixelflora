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
        per_page, id_below, emitted = 200, None, 0
        cap = spec.media.buffer  # harvest a buffer above the target so dups/failures can be topped up
        size = spec.media.prefer_size if spec.media.prefer_size in _SIZES else "original"
        while emitted < cap:
            params = {**self._base_params(spec, taxon),
                      "per_page": per_page, "order_by": "id", "order": "desc"}
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
            id_below = results[-1]["id"]
            if len(results) < per_page:
                break

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
