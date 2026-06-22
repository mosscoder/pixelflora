"""Train/test(/val) splitting strategies that operate on the flat image rows.

  * random      — seeded shuffle, optionally stratified to preserve the
                  proportions of a metadata column.
  * temporal    — hold out the most recent fraction by date (or everything in/
                  after ``test_after_year``); measures generalization over time.
  * geographic  — SPATIAL BLOCKING: bin rows into lat/long grid cells and assign
                  whole cells to a split, so photos from the same place never
                  straddle train/test. Avoids spatial-autocorrelation leakage.

Each returns ``{row_index: "train"|"validation"|"test"}``.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict

from .request import SplitSpec


def assign(rows: list[dict], spec: SplitSpec, group_by: str | None = None) -> dict[int, str]:
    """Assign each row to a split. When ``group_by`` is set (e.g. the class
    ``label`` for a multi-species dataset), the strategy is applied independently
    within each group, so every class is represented in every split regardless of
    strategy (no class ends up entirely in train or test)."""
    if not group_by:
        return _assign_flat(rows, spec)
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r.get(group_by)].append(i)
    out: dict[int, str] = {}
    for idxs in groups.values():
        local = _assign_flat([rows[i] for i in idxs], spec)
        for local_i, label in local.items():
            out[idxs[local_i]] = label
    return out


def _assign_flat(rows: list[dict], spec: SplitSpec) -> dict[int, str]:
    if spec.strategy == "none":
        return {i: "train" for i in range(len(rows))}
    if spec.strategy == "random":
        return _random(rows, spec)
    if spec.strategy == "temporal":
        return _temporal(rows, spec)
    if spec.strategy == "geographic":
        return _geographic(rows, spec)
    raise ValueError(f"unknown split strategy: {spec.strategy}")


def _three_way(n_test: float, n_val: float):
    """Return cumulative thresholds for test/val/train over a [0,1) position."""
    return n_test, n_test + n_val


def _label_at(pos: float, test_thr: float, val_thr: float) -> str:
    if pos < test_thr:
        return "test"
    if pos < val_thr:
        return "validation"
    return "train"


def _random(rows, spec) -> dict[int, str]:
    rng = random.Random(spec.seed)
    test_thr, val_thr = _three_way(spec.test_fraction, spec.val_fraction)
    groups: dict = defaultdict(list)
    for i, r in enumerate(rows):
        key = r.get(spec.stratify_by) if spec.stratify_by else "_all"
        groups[key].append(i)
    out: dict[int, str] = {}
    for idxs in groups.values():
        rng.shuffle(idxs)
        n = len(idxs)
        for rank, i in enumerate(idxs):
            out[i] = _label_at(rank / n, test_thr, val_thr)
    return out


def _temporal(rows, spec) -> dict[int, str]:
    if spec.test_after_year is not None:
        out = {}
        for i, r in enumerate(rows):
            y = r.get("year")
            out[i] = "test" if (y is not None and y >= spec.test_after_year) else "train"
        return out
    # fraction-based: most recent dates -> test, then validation, oldest -> train
    def sort_key(i):
        r = rows[i]
        return (r.get("event_date") or "", r.get("year") or 0)
    order = sorted(range(len(rows)), key=sort_key)  # oldest first
    n = len(order)
    n_test = round(spec.test_fraction * n)
    n_val = round(spec.val_fraction * n)
    out = {}
    for rank, i in enumerate(order):
        if rank >= n - n_test:
            out[i] = "test"
        elif rank >= n - n_test - n_val:
            out[i] = "validation"
        else:
            out[i] = "train"
    return out


def _geographic(rows, spec) -> dict[int, str]:
    cell = spec.cell_size_deg
    rng = random.Random(spec.seed)

    def cell_of(r):
        lat, lng = r.get("decimal_latitude"), r.get("decimal_longitude")
        if lat is None or lng is None:
            return None
        return (math.floor(lat / cell), math.floor(lng / cell))

    members: dict = defaultdict(list)
    no_geo: list[int] = []
    for i, r in enumerate(rows):
        c = cell_of(r)
        (no_geo if c is None else members[c]).append(i)

    cells = list(members.keys())
    rng.shuffle(cells)
    total = sum(len(members[c]) for c in cells)
    test_target = spec.test_fraction * total
    val_target = spec.val_fraction * total

    out: dict[int, str] = {}
    acc_test = acc_val = 0
    for c in cells:
        idxs = members[c]
        if acc_test < test_target:
            label, acc_test = "test", acc_test + len(idxs)
        elif acc_val < val_target:
            label, acc_val = "validation", acc_val + len(idxs)
        else:
            label = "train"
        for i in idxs:
            out[i] = label
    for i in no_geo:  # rows lacking coordinates default to train
        out[i] = "train"
    return out
