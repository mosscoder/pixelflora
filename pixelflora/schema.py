"""Canonical normalized records — the source-agnostic interlingua.

Every source (GBIF, iNaturalist, ...) maps its native payload into these models,
so everything downstream (filter / download / assemble / split / publish) is
written once and never needs to know which provider the data came from.

An ``OccurrenceRecord`` is one organism observed at a place/time and carries one
or more ``ImageRef`` media items. The Hugging Face dataset is *image-level*: at
assembly time each record is "exploded" to one flat row per downloaded image
(occurrence + taxon + spatial + temporal fields denormalized onto each image),
which is what enables filtering and stratified splitting by any metadata column.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from . import licenses as lic


class TaxonRef(BaseModel):
    """Resolved taxonomy for the requested species."""
    model_config = ConfigDict(extra="ignore")
    source: str
    taxon_key: Optional[str] = None          # source-native id (GBIF usageKey / iNat taxon id)
    scientific_name: Optional[str] = None
    rank: Optional[str] = None
    kingdom: Optional[str] = None
    family: Optional[str] = None
    genus: Optional[str] = None
    species: Optional[str] = None
    accepted_name: Optional[str] = None
    match_confidence: Optional[float] = None


class ImageRef(BaseModel):
    """A single media item (image) plus its per-image rights metadata."""
    model_config = ConfigDict(extra="ignore")
    image_id: str
    source_url: str                          # original/largest-size URL to fetch
    license: str = lic.UNKNOWN               # canonical tag (see licenses.py)
    license_url: Optional[str] = None
    rights_holder: Optional[str] = None
    creator: Optional[str] = None            # photographer / author
    attribution: Optional[str] = None        # preformatted credit line
    width: Optional[int] = None
    height: Optional[int] = None
    image_format: Optional[str] = None
    # filled in by the downloader:
    local_path: Optional[str] = None
    sha256: Optional[str] = None
    file_size: Optional[int] = None
    download_status: str = "pending"         # pending | ok | failed | duplicate


class OccurrenceRecord(BaseModel):
    """One observation/specimen with its full provenance, location, and traits."""
    model_config = ConfigDict(extra="ignore")

    # identity / provenance
    source: str                              # "gbif" | "inaturalist"
    occurrence_id: str
    source_record_url: Optional[str] = None
    basis_of_record: Optional[str] = None    # HUMAN_OBSERVATION (field) | PRESERVED_SPECIMEN (museum) | ...

    # taxonomy
    label: Optional[str] = None              # requested class label, "Genus species"
    scientific_name: Optional[str] = None
    taxon_rank: Optional[str] = None
    taxon_key: Optional[str] = None
    kingdom: Optional[str] = None
    family: Optional[str] = None
    genus: Optional[str] = None
    species: Optional[str] = None

    # attribution: organization / author / dataset (all if applicable)
    publisher: Optional[str] = None          # publishing organization
    dataset_key: Optional[str] = None
    dataset_title: Optional[str] = None
    dataset_doi: Optional[str] = None
    recorded_by: Optional[str] = None        # collector / observer (author)
    identified_by: Optional[str] = None
    institution_code: Optional[str] = None   # specimen IDs / type status group
    collection_code: Optional[str] = None
    catalog_number: Optional[str] = None
    type_status: Optional[str] = None
    occurrence_license: Optional[str] = None # occurrence-level license (distinct from per-image)

    # spatial (precision + quality group)
    decimal_latitude: Optional[float] = None
    decimal_longitude: Optional[float] = None
    coordinate_uncertainty_m: Optional[float] = None
    geodetic_datum: Optional[str] = None
    country: Optional[str] = None
    country_code: Optional[str] = None
    state_province: Optional[str] = None
    locality: Optional[str] = None
    elevation: Optional[float] = None

    # temporal
    event_date: Optional[str] = None         # ISO date string of collection/observation
    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None
    day_of_year: Optional[int] = None        # derived; useful for phenology

    # botanical traits / phenology group
    reproductive_condition: Optional[str] = None  # flowering / fruiting / budding ...
    life_stage: Optional[str] = None
    sex: Optional[str] = None
    habitat: Optional[str] = None
    establishment_means: Optional[str] = None     # wild / cultivated (recorded, not filtered by default)
    occurrence_status: Optional[str] = None

    # quality
    quality_grade: Optional[str] = None      # iNat research / needs_id / casual
    issues: list[str] = Field(default_factory=list)

    # iNaturalist-specific
    uuid: Optional[str] = None
    common_name: Optional[str] = None
    iconic_taxon: Optional[str] = None       # e.g. "Plantae"
    observer_login: Optional[str] = None
    coordinates_obscured: Optional[bool] = None   # iNat randomizes coords for sensitive taxa
    geoprivacy: Optional[str] = None
    num_identification_agreements: Optional[int] = None
    num_identification_disagreements: Optional[int] = None

    images: list[ImageRef] = Field(default_factory=list)
    raw: Optional[dict[str, Any]] = None     # original payload, kept in the manifest for full provenance

    @property
    def has_coordinates(self) -> bool:
        return self.decimal_latitude is not None and self.decimal_longitude is not None

    def fill_derived(self) -> "OccurrenceRecord":
        """Populate year/month/day/day_of_year from event_date when possible."""
        if self.event_date:
            try:
                d = _dt.date.fromisoformat(self.event_date[:10])
                self.year = self.year or d.year
                self.month = self.month or d.month
                self.day = self.day or d.day
                self.day_of_year = self.day_of_year or d.timetuple().tm_yday
            except (ValueError, TypeError):
                pass
        return self


# The fixed flat column set of the assembled image-level dataset, curated to the
# fields iNaturalist actually populates. Keeping this explicit (rather than
# inferring) guarantees a stable Hugging Face Features schema across pulls.
OCCURRENCE_COLUMNS = [
    "label",
    "source", "occurrence_id", "uuid", "source_record_url", "basis_of_record",
    "scientific_name", "common_name", "taxon_rank", "taxon_key", "iconic_taxon",
    "recorded_by", "observer_login", "occurrence_license",
    "decimal_latitude", "decimal_longitude", "coordinate_uncertainty_m",
    "coordinates_obscured", "geoprivacy", "locality",
    "event_date", "year", "month", "day", "day_of_year",
    "reproductive_condition", "establishment_means",
    "quality_grade", "num_identification_agreements", "num_identification_disagreements",
]
IMAGE_COLUMNS = [
    "image_id", "image_source_url", "license", "license_url", "rights_holder",
    "creator", "attribution", "image_width", "image_height", "image_format",
    "sha256", "file_size", "local_path",
]


def explode_to_rows(occ: OccurrenceRecord, *, only_downloaded: bool = True) -> list[dict]:
    """Flatten an occurrence into one row per image for the dataset."""
    base = {c: getattr(occ, c) for c in OCCURRENCE_COLUMNS}
    rows = []
    for img in occ.images:
        if only_downloaded and img.download_status != "ok":
            continue
        row = dict(base)
        row.update({
            "image_id": img.image_id,
            "image_source_url": img.source_url,
            "license": img.license,
            "license_url": img.license_url,
            "rights_holder": img.rights_holder,
            "creator": img.creator,
            "attribution": img.attribution,
            "image_width": img.width,
            "image_height": img.height,
            "image_format": img.image_format,
            "sha256": img.sha256,
            "file_size": img.file_size,
            "local_path": img.local_path,
        })
        rows.append(row)
    return rows
