"""Assemble the image-level Hugging Face dataset and apply the split.

Each species becomes its own dataset *configuration* (a named subset of the
repository), and each configuration carries its own train and test split. This
lets a model load one species at a time and lets new species be added later as
new configurations without touching the ones already published.

A configuration is a ``DatasetDict`` stored as Parquet shards (via ``datasets``)
with an embedded ``Image`` feature, while the downloaded image files are also kept
on disk with a sidecar manifest, the "Parquet plus local files" layout chosen for
the project. We declare an explicit Features schema so the column types are stable
even when a whole column is empty for a given run.
"""
from __future__ import annotations

import re
from collections import OrderedDict

from datasets import Dataset, DatasetDict, Features, Image, Value

from .request import SplitSpec
from .schema import IMAGE_COLUMNS, OCCURRENCE_COLUMNS, OccurrenceRecord, explode_to_rows
from .splits import assign

_STRING = Value("string")
_INT = Value("int64")
_FLOAT = Value("float64")
_BOOL = Value("bool")

_TYPES = {
    "decimal_latitude": _FLOAT, "decimal_longitude": _FLOAT,
    "coordinate_uncertainty_m": _FLOAT,
    "coordinates_obscured": _BOOL,
    "year": _INT, "month": _INT, "day": _INT, "day_of_year": _INT,
    "num_identification_agreements": _INT, "num_identification_disagreements": _INT,
    "image_width": _INT, "image_height": _INT, "file_size": _INT,
}


def _features() -> Features:
    cols = {c: _TYPES.get(c, _STRING) for c in OCCURRENCE_COLUMNS}
    cols.update({c: _TYPES.get(c, _STRING) for c in IMAGE_COLUMNS})
    cols["image"] = Image()
    return Features(cols)


def build_rows(records: list[OccurrenceRecord]) -> list[dict]:
    rows: list[dict] = []
    for rec in records:
        for row in explode_to_rows(rec, only_downloaded=True):
            row["image"] = row["local_path"]  # datasets.Image loads from path
            rows.append(row)
    return rows


def config_name(label: str) -> str:
    """Turn a class label such as 'Lupinus sericeus' into a valid configuration
    name such as 'lupinus_sericeus'."""
    return re.sub(r"[^a-z0-9]+", "_", (label or "unspecified").lower()).strip("_")


def _to_dataset_dict(rows: list[dict], split: SplitSpec, features: Features) -> DatasetDict:
    labels = assign(rows, split)
    buckets: dict[str, list[dict]] = {}
    for i, row in enumerate(rows):
        buckets.setdefault(labels[i], []).append(row)
    dd = {}
    for name, items in buckets.items():
        columns = {k: [r.get(k) for r in items] for k in features}
        dd[name] = Dataset.from_dict(columns, features=features)
    return DatasetDict(dd)


def assemble_configs(records: list[OccurrenceRecord], split: SplitSpec) -> "OrderedDict[str, DatasetDict]":
    """Build one configuration per species, each with its own train and test split.

    Returns an ordered mapping of configuration name to ``DatasetDict``.
    """
    by_label: "OrderedDict[str, list[OccurrenceRecord]]" = OrderedDict()
    for rec in records:
        by_label.setdefault(rec.label or rec.scientific_name or "unspecified", []).append(rec)

    features = _features()
    configs: "OrderedDict[str, DatasetDict]" = OrderedDict()
    for label, recs in by_label.items():
        rows = build_rows(recs)
        if not rows:
            continue
        configs[config_name(label)] = _to_dataset_dict(rows, split, features)
    if not configs:
        raise RuntimeError("no successfully downloaded images to assemble")
    return configs
