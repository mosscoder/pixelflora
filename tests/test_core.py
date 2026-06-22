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


def test_resumable_download_retries_only_failures():
    """The downloader reuses files already fetched ok, refetches only failures, and
    keeps dedup across runs (the resume / backfill feature)."""
    import io
    import tempfile

    from PIL import Image

    from pixelflora.download import download_images

    def _png(seed):
        im = Image.new("RGB", (300, 300), (seed % 256, (seed * 7) % 256, (seed * 13) % 256))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()

    class _FakeClient:
        def __init__(self, fail_urls):
            self.fail_urls = set(fail_urls)
            self.calls = []

        def get_bytes(self, url):
            self.calls.append(url)
            if url in self.fail_urls:
                raise RuntimeError("simulated CDN failure")
            return _png(int(url.split("/")[-1].split(".")[0]))

    def _mk(i):
        return OccurrenceRecord(
            source="t", occurrence_id=str(i), basis_of_record="HUMAN_OBSERVATION",
            label="Test species",
            images=[ImageRef(image_id=f"img{i}", source_url=f"http://h/{i}.png", license=lic.CC0)])

    with tempfile.TemporaryDirectory() as d:
        recs = [_mk(1), _mk(2), _mk(3)]
        c1 = _FakeClient(fail_urls={"http://h/2.png"})
        s1 = download_images(recs, d, c1, workers=2, min_pixels=0, max_dimension=0)
        assert s1["ok"] == 2 and s1["failed"] == 1, s1

        # persist prior per-image state as the manifest would, then re-harvest fresh objects
        prior = {img.image_id: {"download_status": img.download_status, "sha256": img.sha256,
                                "local_path": img.local_path, "width": img.width,
                                "height": img.height, "image_format": img.image_format,
                                "file_size": img.file_size}
                 for r in recs for img in r.images}
        recs2 = [_mk(1), _mk(2), _mk(3)]
        c2 = _FakeClient(fail_urls=set())                  # CDN healthy now
        s2 = download_images(recs2, d, c2, workers=2, min_pixels=0, max_dimension=0, prior=prior)
        assert s2["reused"] == 2, s2                       # img1 + img3 reused, not refetched
        assert s2["ok"] == 1 and s2["failed"] == 0, s2     # only img2 retried, now ok
        assert c2.calls == ["http://h/2.png"], c2.calls    # the sole network call was the failure
    print("ok: resumable_download_retries_only_failures")


def test_download_to_target_tops_up_past_dups_and_failures():
    """With a target, the downloader keeps pulling from the buffer until N unique
    images land, skipping failures and duplicates, and fetches no more than needed."""
    import io
    import tempfile

    from PIL import Image

    from pixelflora.download import download_images

    def _png(seed):
        im = Image.new("RGB", (300, 300), (seed % 256, (seed * 7) % 256, (seed * 13) % 256))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()

    class _FakeClient:
        def __init__(self, fail_urls=(), alias=None):
            self.fail_urls = set(fail_urls)
            self.alias = alias or {}
            self.calls = []

        def get_bytes(self, url):
            self.calls.append(url)
            if url in self.fail_urls:
                raise RuntimeError("simulated failure")
            seed_url = self.alias.get(url, url)
            return _png(int(seed_url.split("/")[-1].split(".")[0]))

    def _mk(i):
        return OccurrenceRecord(
            source="t", occurrence_id=str(i), basis_of_record="HUMAN_OBSERVATION",
            label="Test species",
            images=[ImageRef(image_id=f"img{i}", source_url=f"http://h/{i}.png", license=lic.CC0)])

    # 6 candidates in the buffer, target 3 unique. img2 fails; img4 is byte-identical to img1.
    recs = [_mk(i) for i in range(1, 7)]
    client = _FakeClient(fail_urls={"http://h/2.png"}, alias={"http://h/4.png": "http://h/1.png"})
    with tempfile.TemporaryDirectory() as d:
        stats = download_images(recs, d, client, workers=3, min_pixels=0,
                                max_dimension=0, target_per_class=3)
    assert stats["ok"] == 3, stats                          # img1, img3, img5
    assert stats["failed"] == 1, stats                      # img2
    assert stats["duplicate"] == 1, stats                   # img4 == img1 by bytes, not counted
    assert "http://h/6.png" not in client.calls, client.calls  # target met at img5; img6 untouched
    print("ok: download_to_target_tops_up_past_dups_and_failures")


if __name__ == "__main__":
    for fn in [test_license_normalize, test_derived_day_of_year, test_filters_license_optin,
               test_geographic_no_leakage, test_temporal_holds_out_recent,
               test_multispecies_parse, test_grouped_split_keeps_each_class_in_both,
               test_resumable_download_retries_only_failures,
               test_download_to_target_tops_up_past_dups_and_failures]:
        fn()
    print("ALL CORE TESTS PASSED")
