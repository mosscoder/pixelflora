"""Parse and validate a request TOML into a typed ``RequestSpec``.

A request is the single unit of work: which species, from which source(s), with
what filters, how many images, how to split, and where to publish (private by
default). See ``dev/lupinus_sericeus/request.toml`` for a worked example.
"""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SpeciesSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    genus: str
    species: str
    infraspecific: Optional[str] = None
    taxon_key: Optional[str] = None  # override taxon resolution if you already know it


class FiltersSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # license filtering is OPT-IN now (datasets are private). If empty, keep all licenses.
    license: list[str] = Field(default_factory=list)            # canonical tags to KEEP
    exclude_license: list[str] = Field(default_factory=list)    # canonical tags to DROP
    basis_of_record: list[str] = Field(default_factory=list)
    year_range: Optional[list[int]] = None                      # [min, max] inclusive
    has_coordinates: Optional[bool] = None
    max_coordinate_uncertainty_m: Optional[float] = None
    reproductive_condition: list[str] = Field(default_factory=list)  # flowering/fruiting/budding
    require_research_grade: bool = True                         # iNat: default to research-grade
    exclude_captive: bool = False                              # drop cultivated/captive records

    @field_validator("year_range")
    @classmethod
    def _year_range_len(cls, v):
        if v is not None and len(v) != 2:
            raise ValueError("year_range must be [min, max]")
        return v


class MediaSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_images: int = 500
    images_per_occurrence: Union[int, Literal["all"]] = "all"
    min_pixels: int = 0                  # minimum of (width, height); 0 = no minimum
    max_dimension: int = 1024            # cap longest edge (px), aspect kept; 0/neg = no cap
    prefer_size: str = "original"        # original | large (source-dependent)
    overharvest: float = 1.5             # harvest this multiple of max_images as candidates,
                                         # so duplicates and failed fetches can be topped up

    @property
    def buffer(self) -> int:
        """Candidate occurrences to harvest per class: a little over ``max_images`` so
        the downloader can top up past duplicates and failed fetches and still hit the
        target. Never less than ``max_images``."""
        return max(self.max_images, int(self.max_images * max(self.overharvest, 1.0)))


class SplitSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strategy: Literal["none", "random", "temporal", "geographic"] = "random"
    test_fraction: float = 0.2
    val_fraction: float = 0.0
    seed: int = 1312
    # random:
    stratify_by: Optional[str] = None    # any dataset column (e.g. "reproductive_condition")
    # temporal:
    test_after_year: Optional[int] = None  # records in/after this year -> test (overrides fraction)
    # geographic:
    cell_size_deg: float = 1.0           # spatial block size in degrees lat/long


class PublishSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repo_id: Optional[str] = None        # "owner/name"; owner falls back to config/token
    private: bool = True                 # PRIVATE BY DEFAULT
    push: bool = False                   # must explicitly opt in to hit the Hub


class DatasetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Optional[str] = None
    description: Optional[str] = None


class RequestSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    species: list[SpeciesSpec]           # one or many; many -> multi-class dataset
    sources: list[str] = Field(default_factory=lambda: ["inaturalist"])
    filters: FiltersSpec = Field(default_factory=FiltersSpec)
    media: MediaSpec = Field(default_factory=MediaSpec)
    output_dir: str = "out"
    split: SplitSpec = Field(default_factory=SplitSpec)
    publish: PublishSpec = Field(default_factory=PublishSpec)
    dataset: DatasetSpec = Field(default_factory=DatasetSpec)

    @field_validator("species", mode="before")
    @classmethod
    def _coerce_species(cls, v):
        # accept a single [species] table or a list of [[species]] tables
        if isinstance(v, dict):
            return [v]
        if not v:
            raise ValueError("at least one [species] (or [[species]]) is required")
        return v

    @staticmethod
    def label_of(sp: SpeciesSpec) -> str:
        return f"{sp.genus} {sp.species}"

    @property
    def is_multispecies(self) -> bool:
        return len({self.label_of(s) for s in self.species}) > 1

    @classmethod
    def from_toml(cls, path: str | Path) -> "RequestSpec":
        data = tomllib.loads(Path(path).read_text())
        # accept either [sources] enabled = [...] or top-level sources = [...]
        sources = data.pop("sources", None)
        if isinstance(sources, dict):
            sources = sources.get("enabled", ["inaturalist"])
        output = data.pop("output", {}) or {}
        return cls(
            species=data.get("species", {}),
            sources=sources or ["inaturalist"],
            filters=data.get("filters", {}),
            media=data.get("media", {}),
            output_dir=output.get("dir", data.get("output_dir", "out")),
            split=data.get("split", {}),
            publish=data.get("publish", {}),
            dataset=data.get("dataset", {}),
        )
