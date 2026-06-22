"""The Source interface — the modularity seam.

A Source turns a (species, filters) request into normalized ``OccurrenceRecord``s.
Adding a provider means implementing this one class and registering it; nothing
else in the pipeline changes. Source-native filtering (pushing filters into the
provider's query) is encouraged for efficiency, but the canonical
``filters.apply_filters`` always runs afterwards so behavior is uniform.
"""
from __future__ import annotations

import abc
from collections.abc import Iterator

from ..config import Config
from ..http import PoliteClient
from ..request import RequestSpec
from ..schema import OccurrenceRecord, TaxonRef


class Source(abc.ABC):
    name: str = "base"

    def __init__(self, config: Config, client: PoliteClient):
        self.config = config
        self.client = client

    @abc.abstractmethod
    def resolve_taxon(self, genus: str, species: str, *, taxon_key: str | None = None) -> TaxonRef:
        """Resolve genus/species to this source's taxonomy."""

    @abc.abstractmethod
    def harvest(self, spec: RequestSpec, taxon: TaxonRef) -> Iterator[OccurrenceRecord]:
        """Yield normalized occurrence records (metadata + media URLs, no bytes yet)."""

    def estimate_count(self, spec: RequestSpec, taxon: TaxonRef) -> int | None:
        """Optional: cheap upper bound on matching records (drives GBIF auto-select)."""
        return None
