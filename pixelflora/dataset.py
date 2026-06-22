"""Assemble the image-level Hugging Face dataset and apply the split.

The dataset is stored as Parquet shards (via ``datasets``) with an embedded
``Image`` feature, while the downloaded image files are also kept on disk with a
sidecar manifest — the "Parquet + local files" layout chosen for the project.
We declare an explicit Features schema so the column types are stable even when a
whole column is null for a given pull.
"""
from __future__ import annotations

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


def assemble(records: list[OccurrenceRecord], split: SplitSpec,
             group_by: str | None = None) -> DatasetDict:
    rows = build_rows(records)
    if not rows:
        raise RuntimeError("no successfully downloaded images to assemble")
    labels = assign(rows, split, group_by=group_by)
    features = _features()

    buckets: dict[str, list[dict]] = {}
    for i, row in enumerate(rows):
        buckets.setdefault(labels[i], []).append(row)

    dd = {}
    for name, items in buckets.items():
        columns = {k: [r.get(k) for r in items] for k in features}
        dd[name] = Dataset.from_dict(columns, features=features)
    return DatasetDict(dd)
