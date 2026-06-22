"""Offline unit checks for the source-agnostic core (no network).
Run: python -m tests.test_core   (or: pytest tests/)"""
from pixelflora import licenses as lic
from pixelflora.filters import apply_filters
from pixelflora.request import FiltersSpec, SplitSpec
from pixelflora.schema import ImageRef, OccurrenceRecord, explode_to_rows
from pixelflora.splits import assign


def _rec(oid, lat, lng, year, license_tag, status="ok"):
    img = ImageRef(image_id=f"{oid}-0", source_url="http://x/y.jpg",
                   license=license_tag, download_status=status)
    return OccurrenceRecord(
        source="t", occurrence_id=str(oid), decimal_latitude=lat,
        decimal_longitude=lng, year=year, event_date=f"{year}-06-15",
        basis_of_record="HUMAN_OBSERVATION", quality_grade="research",
        images=[img]).fill_derived()


def test_license_normalize():
    assert lic.normalize("cc-by-nc") == lic.CC_BY_NC
    assert lic.normalize("http://creativecommons.org/licenses/by-sa/4.0/") == lic.CC_BY_SA
    assert lic.normalize("CC0_1_0") == lic.CC0
    assert lic.normalize(None) == lic.ALL_RIGHTS_RESERVED
    assert lic.is_redistributable(lic.CC_BY) and not lic.is_redistributable(lic.CC_BY_NC)
    print("ok: license_normalize")


def test_derived_day_of_year():
    r = _rec(1, 46.0, -114.0, 2021, lic.CC0)
    assert r.day_of_year == 166 and r.month == 6
    print("ok: derived day_of_year")


def test_filters_license_optin():
    recs = [_rec(1, 46, -114, 2021, lic.CC0), _rec(2, 47, -113, 2019, lic.CC_BY_NC)]
    kept, _ = apply_filters(recs, FiltersSpec())              # no filter -> keep all
    assert len(kept) == 2
    kept, _ = apply_filters(recs, FiltersSpec(license=["CC0"]))  # opt-in keep only CC0
    assert len(kept) == 1 and kept[0].occurrence_id == "1"
    print("ok: filters_license_optin")


def test_geographic_no_leakage():
    # two tight clusters far apart -> whole cells must land in one split each
    recs = [_rec(i, 46.0 + i * 0.01, -114.0, 2020, lic.CC0) for i in range(10)]
    recs += [_rec(100 + i, 10.0 + i * 0.01, 20.0, 2020, lic.CC0) for i in range(10)]
    rows = [explode_to_rows(r)[0] for r in recs]
    labels = assign(rows, SplitSpec(strategy="geographic", test_fraction=0.5,
                                    cell_size_deg=1.0, seed=1))
    # each 1-degree cell is internally consistent
    by_cell = {}
    for i, row in enumerate(rows):
        cell = (int(row["decimal_latitude"]), int(row["decimal_longitude"]))
        by_cell.setdefault(cell, set()).add(labels[i])
    assert all(len(v) == 1 for v in by_cell.values()), by_cell
    print("ok: geographic_no_leakage")


def test_temporal_holds_out_recent():
    recs = [_rec(i, 46, -114, 2010 + i, lic.CC0) for i in range(10)]
    rows = [explode_to_rows(r)[0] for r in recs]
    labels = assign(rows, SplitSpec(strategy="temporal", test_fraction=0.3))
    test_years = sorted(rows[i]["year"] for i, l in labels.items() if l == "test")
    train_years = sorted(rows[i]["year"] for i, l in labels.items() if l == "train")
    assert min(test_years) > max(train_years)  # recent -> test
    print("ok: temporal_holds_out_recent")


def test_multispecies_parse():
    from pixelflora.request import RequestSpec
    multi = RequestSpec(species=[{"genus": "Lupinus", "species": "sericeus"},
                                 {"genus": "Lupinus", "species": "argenteus"}])
    assert multi.is_multispecies and len(multi.species) == 2
    single = RequestSpec(species={"genus": "Lupinus", "species": "sericeus"})  # dict coerced
    assert not single.is_multispecies and len(single.species) == 1
    print("ok: multispecies_parse")


def test_grouped_split_keeps_each_class_in_both():
    from collections import defaultdict
    rows = [{"label": lab, "decimal_latitude": 40 + i, "decimal_longitude": -110.0,
             "year": 2010 + i, "event_date": f"{2010 + i}-06-15"}
            for lab in ("A", "B") for i in range(8)]
    labels = assign(rows, SplitSpec(strategy="random", test_fraction=0.5, seed=1),
                    group_by="label")
    seen = defaultdict(set)
    for i, l in labels.items():
        seen[rows[i]["label"]].add(l)
    assert seen["A"] >= {"train", "test"} and seen["B"] >= {"train", "test"}, dict(seen)
    print("ok: grouped_split_keeps_each_class_in_both")


if __name__ == "__main__":
    for fn in [test_license_normalize, test_derived_day_of_year, test_filters_license_optin,
               test_geographic_no_leakage, test_temporal_holds_out_recent,
               test_multispecies_parse, test_grouped_split_keeps_each_class_in_both]:
        fn()
    print("ALL CORE TESTS PASSED")
