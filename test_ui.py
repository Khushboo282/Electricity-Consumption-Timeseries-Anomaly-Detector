"""
test_ui.py -- Tests for the UI layer.

Run:  python -m pytest test_ui.py -v
Needs the data files for the integration tests (env STORE_DATA_CSV /
GROUND_TRUTH_CSV, or a ./data folder); those tests are skipped otherwise.

WHAT IS COVERED
  A. Helper functions: data blocks, chart frames with gaps, normal band,
     file lookup, timing helper.
  B. The normal band agrees exactly with the detector's decision.
  C. The TEAMMATE'S brute force and our skip list return identical rows
     for whole-hour windows (the UI's guarantee) and the known edge case
     where they can differ for non-whole-hour bounds is documented.
  D. A headless run of the whole Streamlit app (clicks through the controls).
"""

import os
import random
from datetime import datetime, timedelta

import pytest

from anomaly import AnomalyDetector, Baseline, compute_baseline, signed_z
from timeseries_store import TimeSeriesStore, to_epoch
from ui_helpers import (band_frame, brute_keys, brute_range, build_explore_chart,
                        data_blocks, find_data_file, load_brute_force, normal_range,
                        series_frame, time_best_us, to_brute_ts)

HOUR = 3600
DAY = 24 * HOUR
START = to_epoch("2016-01-01 00:00")


# ====================================================================== #
# A. Helpers
# ====================================================================== #
def test_data_blocks_split_on_large_gaps_only():
    block1 = [START + i * HOUR for i in range(120)]
    block2 = [START + 30 * DAY + i * HOUR for i in range(120)]
    assert data_blocks(block1 + block2) == [(block1[0], block1[-1]), (block2[0], block2[-1])]
    # a building missing one whole day INSIDE a block must not be split
    missing_day = [e for e in block1 if not (2 * DAY <= e - START < 3 * DAY)]
    assert data_blocks(missing_day) == [(missing_day[0], missing_day[-1])]
    assert data_blocks([]) == []
    assert data_blocks([START]) == [(START, START)]


def test_series_frame_breaks_line_at_gaps():
    t = datetime(2016, 1, 1)
    readings = [(t + timedelta(hours=h), 1.0 + h) for h in (0, 1, 2, 10, 11)]  # gap 2h -> 10h
    frame = series_frame(readings)
    assert len(frame) == 6                              # one empty row inserted
    assert frame["value"].isna().sum() == 1
    assert frame["value"].isna().iloc[3]                # right where the gap is
    assert len(series_frame(readings[:3])) == 3         # no gap -> unchanged
    assert series_frame([]).empty


def test_normal_range_uses_hourly_stats_with_fallback():
    baseline = Baseline(mean=100.0, std=10.0, n=500,
                        hourly={3: (50.0, 5.0, 60), 4: (70.0, 7.0, 3)})
    h3, h4 = START + 3 * HOUR, START + 4 * HOUR
    assert normal_range(baseline, "hourly", h3, 2.0) == (40.0, 60.0)       # own stats
    assert normal_range(baseline, "hourly", h4, 2.0) == (80.0, 120.0)      # too few -> building
    assert normal_range(baseline, "building", h3, 2.0) == (80.0, 120.0)
    assert normal_range(baseline, None, h3, 2.0) is None
    assert normal_range(Baseline(5.0, 0.0, 9, {}), "building", h3, 2.0) is None   # zero spread


def test_band_frame_has_one_row_per_reading_and_gap_breaks():
    t = datetime(2016, 1, 1)
    readings = [(t + timedelta(hours=h), 0.0) for h in (0, 1, 2, 12, 13)]
    values = [10.0 + (h % 24) for h in range(24 * 20)]
    baseline = compute_baseline([(START + i * HOUR, v) for i, v in enumerate(values)])
    frame = band_frame(readings, baseline, "hourly", 3.0)
    assert len(frame) == 6 and frame["lo"].isna().sum() == 1
    assert band_frame(readings, baseline, None, 3.0).empty


def test_find_data_file_prefers_env_then_data_folder(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "store_data.csv").write_text("x")
    monkeypatch.delenv("STORE_DATA_CSV", raising=False)
    assert find_data_file("store_data.csv", "STORE_DATA_CSV", extra_dirs=[str(data)]) == \
        str(data / "store_data.csv")
    other = tmp_path / "elsewhere.csv"
    other.write_text("y")
    monkeypatch.setenv("STORE_DATA_CSV", str(other))
    assert find_data_file("store_data.csv", "STORE_DATA_CSV") == str(other)
    monkeypatch.delenv("STORE_DATA_CSV")
    assert find_data_file("definitely_not_here_123.csv") is None


def test_time_best_us_is_positive_and_scales_with_work():
    fast = time_best_us(lambda: None, repeats=200)
    slow = time_best_us(lambda: sum(range(2000)), repeats=200)
    assert 0 <= fast < slow


# ====================================================================== #
# B. The band agrees with the detector (what you see = what is flagged)
# ====================================================================== #
def test_normal_band_matches_detector_decision():
    rng = random.Random(4)
    readings = []
    for i in range(24 * 30):
        daily = 100 + 40 * (1 if 8 <= i % 24 <= 18 else 0)
        readings.append((START + i * HOUR, daily + rng.gauss(0, 3)))
    readings[100] = (readings[100][0], readings[100][1] * 3)            # a spike
    baseline = compute_baseline(readings)
    for method in ("hourly", "building"):
        for k in (2.0, 4.0):
            for epoch, value in readings:
                lo, hi = normal_range(baseline, method, epoch, k)
                outside = not (lo <= value <= hi)
                assert outside == (abs(signed_z(baseline, method, epoch, value)) > k)


def test_build_explore_chart_produces_valid_spec():
    t = datetime(2016, 1, 1)
    readings = [(t + timedelta(hours=h), 10.0 + h) for h in range(10)]
    baseline = compute_baseline([(START + h * HOUR, 10.0 + h % 3) for h in range(240)])
    store = TimeSeriesStore.from_rows([(1, START + h * HOUR, 10.0 + h % 3) for h in range(240)], seed=0)
    anomalies = AnomalyDetector(store, method="hourly", k=1.0).detect(1, "2016-01-01", "2016-01-02")
    chart = build_explore_chart(readings, anomalies, baseline, "hourly", 3.0, [(t, 10.0)])
    spec = chart.to_dict()                              # must serialise without error
    assert "layer" in spec and len(spec["layer"]) >= 2


# ====================================================================== #
# C./D. Real data
# ====================================================================== #
def _find(env_name, filename):
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (os.environ.get(env_name), os.path.join(here, "data", filename),
                 os.path.join(here, filename), os.path.join("data", filename), filename):
        if path and os.path.exists(path):
            return path
    return None


@pytest.fixture(scope="module")
def both():
    path = _find("STORE_DATA_CSV", "store_data.csv")
    if path is None:
        pytest.skip("store_data.csv not found")
    return TimeSeriesStore.from_csv(path, seed=42), load_brute_force(path)


def test_brute_force_and_skip_list_agree_on_whole_hour_windows(both):
    store, brute = both
    rng = random.Random(77)
    buildings = store.buildings()
    checked = 0
    for _ in range(150):
        b = rng.choice(buildings)
        first, last = (to_epoch(x) for x in store.span(b))
        lo = first + HOUR * rng.randrange(0, (last - first) // HOUR)
        hi = lo + HOUR * rng.choice([0, 1, 6, 24, 72, 24 * 30, 24 * 400])
        results, elapsed_us, comparisons = brute_range(brute, b, lo, hi)
        assert brute_keys(results) == {k for k, _ in store.query_raw(b, lo, hi)}, (b, lo, hi)
        assert comparisons == len(store)           # the scan really inspects every row
        checked += 1
    assert checked == 150


def test_brute_force_values_match_skip_list_values(both):
    store, brute = both
    lo, hi = to_epoch("2016-01-01 00:00"), to_epoch("2016-01-03 00:00")
    skip = dict(store.query_raw(107, lo, hi))
    results, _, _ = brute_range(brute, 107, lo, hi)
    assert len(results) == len(skip) > 0
    for r in results:
        assert skip[(107, to_epoch(r.timestamp))] == r.value


def test_known_edge_case_non_whole_hour_bounds_can_differ(both):
    """Documents WHY the UI offers whole-hour windows only: the teammate's list
    filters on the coarse `ts` column (1,000 s steps), so a bound a few seconds
    before a reading can still include it."""
    store, brute = both
    b = 107
    span = store.span(b)
    epochs = [k[1] for k, _ in store.query_raw(b, to_epoch(span[0]), to_epoch(span[1]))]
    target = next(e for e in epochs if e % 1000 != 0)       # a reading not on a 1,000 s boundary
    lo, hi = target - 7200, target - 1                      # window ends 1 s BEFORE the reading
    exact = {k for k, _ in store.query_raw(b, lo, hi)}
    coarse = brute_keys(brute_range(brute, b, lo, hi)[0])
    assert (b, target) not in exact
    assert (b, target) in coarse
    assert to_brute_ts(target - 1) == to_brute_ts(target)


def test_app_runs_headlessly_and_responds_to_controls():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    store_csv = _find("STORE_DATA_CSV", "store_data.csv")
    if store_csv is None:
        pytest.skip("store_data.csv not found")
    os.environ["STORE_DATA_CSV"] = store_csv
    truth = _find("GROUND_TRUTH_CSV", "ground_truth.csv")
    if truth:
        os.environ["GROUND_TRUTH_CSV"] = truth
    app_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    at = AppTest.from_file(app_file, default_timeout=240).run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert int(metrics["Readings in window"].replace(",", "")) > 0
    assert [s.value for s in at.success] and "exactly the same" in at.success[0].value
    # change building, method and block; the app must keep working and stay consistent
    at.sidebar.selectbox[0].select(at.sidebar.selectbox[0].options[50]).run()
    at.sidebar.radio[0].set_value("Building baseline (all hours together)").run()
    at.sidebar.selectbox[1].select(0).run()
    assert not at.exception
    assert "exactly the same" in at.success[0].value
