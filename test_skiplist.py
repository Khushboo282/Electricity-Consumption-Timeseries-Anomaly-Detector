"""
test_skiplist.py -- How we know the skip list works.

Run:   python -m pytest -v
Real-data tests need the CSV; point to it with the environment variable
STORE_DATA_CSV (default: store_data.csv in this folder, ./data, or cwd).
Without the file those tests are skipped and the rest still run.

STRATEGY (three layers)
  A. Skip list unit tests on small synthetic data (structure + edge cases).
  B. Randomised "property" test: skip list vs a trivially-correct reference.
  C. Store tests on a hand-made example and on the REAL dataset, comparing
     every answer with the brute-force linear scan, plus an independent
     check against pandas.
"""

import os
import random
from datetime import datetime

import pytest

from brute_force import brute_force_query_raw
from skiplist import SkipList
from timeseries_store import TimeSeriesStore, epoch_to_datetime, load_rows, to_epoch


# ====================================================================== #
# A. Skip list unit tests
# ====================================================================== #
def make_list(keys, seed=0):
    sl = SkipList(seed=seed)
    for k in keys:
        sl.insert(k, f"v{k}")
    return sl


def test_empty_list():
    sl = SkipList(seed=0)
    assert len(sl) == 0
    assert sl.search(5) == (False, None)
    assert sl.get(5, "x") == "x"
    assert sl.range_query(0, 100) == []
    assert list(sl) == []
    assert sl.check_invariants()


def test_invalid_parameters():
    with pytest.raises(ValueError):
        SkipList(max_level=0)
    with pytest.raises(ValueError):
        SkipList(p=0.0)
    with pytest.raises(ValueError):
        SkipList(p=1.0)


@pytest.mark.parametrize("order", ["ascending", "descending", "shuffled"])
def test_iteration_is_sorted_for_any_insertion_order(order):
    keys = list(range(500))
    if order == "descending":
        keys.reverse()
    elif order == "shuffled":
        random.Random(7).shuffle(keys)
    sl = make_list(keys)
    assert [k for k, _ in sl] == sorted(keys)
    assert len(sl) == 500
    assert sl.check_invariants()


def test_duplicate_key_updates_value_not_size():
    sl = SkipList(seed=0)
    assert sl.insert(10, "a") is True
    assert sl.insert(10, "b") is False        # existing key -> update
    assert len(sl) == 1
    assert sl.get(10) == "b"
    assert sl.check_invariants()


def test_search_hit_and_miss():
    sl = make_list([10, 20, 30])
    assert sl.search(20) == (True, "v20")
    assert 20 in sl
    for missing in (5, 15, 25, 35):           # below, between, between, above
        assert sl.search(missing) == (False, None)
        assert missing not in sl


def test_search_can_distinguish_stored_none():
    sl = SkipList(seed=0)
    sl.insert(1, None)
    assert sl.search(1) == (True, None)
    assert sl.search(2) == (False, None)


def test_range_query_bounds_are_inclusive():
    sl = make_list([10, 20, 30, 40, 50])
    assert sl.range_query(20, 40) == [(20, "v20"), (30, "v30"), (40, "v40")]


def test_range_query_edge_cases():
    sl = make_list([10, 20, 30, 40, 50])
    assert sl.range_query(30, 30) == [(30, "v30")]        # single key
    assert sl.range_query(31, 39) == []                    # between keys
    assert sl.range_query(40, 20) == []                    # lo > hi
    assert sl.range_query(-100, 5) == []                   # entirely below
    assert sl.range_query(60, 100) == []                   # entirely above
    assert len(sl.range_query(-100, 100)) == 5             # covers everything
    assert sl.range_query(5, 10) == [(10, "v10")]          # lo below min
    assert sl.range_query(50, 99) == [(50, "v50")]         # hi above max


def test_tuple_keys_group_by_first_element():
    sl = SkipList(seed=0)
    for b in (1, 2, 3):
        for t in (100, 200, 300):
            sl.insert((b, t), b * 1000 + t)
    # one "building" only; neighbours (1, *) and (3, *) must not leak in
    assert sl.range_query((2, 0), (2, 10**9)) == [
        ((2, 100), 2100), ((2, 200), 2200), ((2, 300), 2300)]


def test_randomised_against_reference_model():
    """Property test: many random inserts + queries vs a sorted dict."""
    rng = random.Random(123)
    sl = SkipList(seed=5)
    reference = {}
    for _ in range(3000):                       # small key domain -> duplicates
        k = rng.randint(0, 1500)
        v = rng.random()
        sl.insert(k, v)
        reference[k] = v
    assert len(sl) == len(reference)
    assert sl.check_invariants()
    ordered = sorted(reference.items())
    for _ in range(1000):
        lo = rng.randint(-50, 1550)
        hi = rng.randint(-50, 1550)
        expected = [(k, v) for k, v in ordered if lo <= k <= hi]
        assert sl.range_query(lo, hi) == expected


def test_structure_looks_logarithmic():
    """Half the nodes should reach level >= 2, and height ~ log2(n)."""
    n = 20000
    sl = SkipList(seed=11)
    for k in range(n):
        sl.insert(k, k)
    assert sl.check_invariants()
    assert 10 <= sl.height <= 20                # log2(20000) is about 14.3
    # count nodes taller than 1 by walking level 1 (index 1)
    node, tall = sl._head.forward[1], 0
    while node is not None:
        tall += 1
        node = node.forward[1]
    assert 0.45 * n < tall < 0.55 * n


# ====================================================================== #
# C1. Store tests on a small hand-made dataset (the example from our notes)
# ====================================================================== #
@pytest.fixture()
def toy_store():
    store = TimeSeriesStore(seed=0)
    temps = [22.1, 22.3, 22.2, 22.4, 31.8, 22.5, 22.6, 22.4, 13.2, 22.7]
    for minute, value in enumerate(temps):
        store.insert(1, f"2016-01-01 09:{minute:02d}", value)
    store.insert(2, "2016-01-01 09:03", 99.0)   # neighbouring building
    store.insert(0, "2016-01-01 09:03", -5.0)   # neighbouring building
    return store


def test_toy_range_query(toy_store):
    got = toy_store.query(1, "2016-01-01 09:02", "2016-01-01 09:06")
    assert got == [
        (datetime(2016, 1, 1, 9, 2), 22.2), (datetime(2016, 1, 1, 9, 3), 22.4),
        (datetime(2016, 1, 1, 9, 4), 31.8), (datetime(2016, 1, 1, 9, 5), 22.5),
        (datetime(2016, 1, 1, 9, 6), 22.6)]


def test_toy_other_buildings_never_leak(toy_store):
    everything = toy_store.query(1, "1900-01-01", "2100-01-01")
    assert len(everything) == 10
    assert all(value not in (99.0, -5.0) for _, value in everything)


def test_toy_edge_cases(toy_store):
    assert toy_store.query(1, "2016-01-01 10:00", "2016-01-01 11:00") == []  # gap
    assert toy_store.query(1, "2016-01-01 09:05", "2016-01-01 09:02") == []  # start > end
    assert toy_store.query(999, "2016-01-01", "2017-01-01") == []            # unknown building
    one = toy_store.query(1, "2016-01-01 09:08", "2016-01-01 09:08")         # single instant
    assert one == [(datetime(2016, 1, 1, 9, 8), 13.2)]
    assert toy_store.query(1, "2016-01-01 09:08:30", "2016-01-01 09:08:59") == []


def test_toy_accepts_several_time_formats(toy_store):
    as_str = toy_store.query(1, "2016-01-01 09:00", "2016-01-01 09:01")
    as_dt = toy_store.query(1, datetime(2016, 1, 1, 9, 0), datetime(2016, 1, 1, 9, 1))
    as_int = toy_store.query(1, to_epoch("2016-01-01 09:00"), to_epoch("2016-01-01 09:01"))
    assert as_str == as_dt == as_int and len(as_str) == 2


def test_toy_metadata(toy_store):
    assert toy_store.buildings() == [0, 1, 2]
    assert toy_store.span(1) == (datetime(2016, 1, 1, 9, 0), datetime(2016, 1, 1, 9, 9))
    assert toy_store.span(42) is None
    assert len(toy_store) == 12


def test_insert_same_reading_twice_overwrites(toy_store):
    toy_store.insert(1, "2016-01-01 09:04", 25.0)
    assert toy_store.query(1, "2016-01-01 09:04", "2016-01-01 09:04")[0][1] == 25.0
    assert len(toy_store) == 12


# ====================================================================== #
# C2. Real dataset: skip list vs brute force vs pandas
# ====================================================================== #
def _find_csv():
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [os.environ.get("STORE_DATA_CSV"),
                  os.path.join(here, "store_data.csv"),
                  os.path.join(here, "data", "store_data.csv"),
                  "store_data.csv"]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


@pytest.fixture(scope="module")
def real():
    path = _find_csv()
    if path is None:
        pytest.skip("store_data.csv not found (set STORE_DATA_CSV)")
    rows = load_rows(path)
    store = TimeSeriesStore.from_rows(rows, seed=42)
    return path, rows, store


def test_real_every_row_indexed_and_structure_valid(real):
    _, rows, store = real
    assert len(store) == len(rows) == 262719
    assert store.index.check_invariants()
    assert len(store.buildings()) == 192


def test_real_whole_history_per_building_matches_pandas(real):
    """Independent check (does not use our brute force): row counts per building."""
    import pandas as pd
    path, _, store = real
    counts = pd.read_csv(path).groupby("building_id").size().to_dict()
    for building_id, expected in counts.items():
        first, last = store.span(building_id)
        got = store.query(building_id, first, last)
        assert len(got) == expected, f"building {building_id}"
        times = [t for t, _ in got]
        assert times == sorted(times)           # time-ordered output


def test_real_random_queries_equal_brute_force(real):
    _, rows, store = real
    rng = random.Random(2024)
    widths_hours = [0, 1, 6, 24, 120, 24 * 40, 24 * 400]   # 0h = single instant
    checked = 0
    for _ in range(300):
        b, e, _v = rows[rng.randrange(len(rows))]
        w = rng.choice(widths_hours) * 3600
        lo = e - rng.choice([0, 3600, 7200, 86400])         # start before/at a reading
        hi = lo + w
        expected = brute_force_query_raw(rows, b, lo, hi)
        assert store.query_raw(b, lo, hi) == expected, (b, lo, hi)
        checked += 1
    assert checked == 300


def test_real_ranges_inside_and_across_gaps(real):
    """The data comes in 5-day blocks with ~25 day gaps; test around them."""
    _, rows, store = real
    b = 107
    # Jan 5 23:00 is the last hour of block 1; Feb 1 00:00 starts block 2.
    inside_gap = store.query(b, "2016-01-10", "2016-01-30")
    assert inside_gap == []
    across = store.query(b, "2016-01-05 20:00", "2016-02-01 03:00")
    expected = brute_force_query_raw(rows, b, to_epoch("2016-01-05 20:00"),
                                     to_epoch("2016-02-01 03:00"))
    assert len(across) == len(expected) > 0
    assert across == [(epoch_to_datetime(k[1]), v) for k, v in expected]


def test_real_unknown_building_and_out_of_range(real):
    _, _, store = real
    assert store.query(-1, "2016-01-01", "2016-12-31") == []
    assert store.query(107, "1999-01-01", "1999-12-31") == []
    assert store.query(107, "2030-01-01", "2030-12-31") == []
