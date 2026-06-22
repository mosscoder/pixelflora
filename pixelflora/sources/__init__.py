"""Source registry.

pixelflora currently targets a single provider — **iNaturalist** — but keeps the
``Source`` seam so another provider can be added later by implementing one class
and registering it here, with no changes elsewhere in the pipeline.
"""
from __future__ import annotations

from ..config import Config
from ..http import PoliteClient
from .base import Source
from .inaturalist import INaturalistSource

REGISTRY: dict[str, type[Source]] = {
    INaturalistSource.name: INaturalistSource,
}

DEFAULT_SOURCE = INaturalistSource.name


def get_source(name: str, config: Config, client: PoliteClient) -> Source:
    key = name.strip().lower()
    if key not in REGISTRY:
        raise KeyError(f"unknown source '{name}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[key](config, client)
