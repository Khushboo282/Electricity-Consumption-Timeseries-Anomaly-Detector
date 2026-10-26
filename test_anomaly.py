"""
test_anomaly.py -- How we know the anomaly detector works.

Run:  python -m pytest test_anomaly.py -v
Real-data tests need the readings and labels (env vars STORE_DATA_CSV and
GROUND_TRUTH_CSV, or files next to this script); otherwise they are skipped.

STRATEGY
  A. Synthetic data with PLANTED spikes and drops whose positions we know:
     the detector must find them (and almost nothing else), for a tiny and a
     huge building alike.
  B. Behaviour and edge cases of the detector API.
  C. The evaluation maths on hand-computed examples.
  D. Real data: labels line up with readings, and simple logical guarantees.
"""

import math
import os
import random
from datetime import datetime, timedelta

import pytest

from anomaly import (AnomalyDetector, Baseline, compute_baseline, hour_of_day,
                     mean_and_std, signed_z)
from evaluation import (anomaly_runs, event_recall, load_ground_truth, restrict,
                        score, split_buildings)
from timeseries_store import TimeSeriesStore, to_epoch

HOUR = 3600
START = to_epoch("2016-01-01 00:00")


# ---------------------------------------------------------------------- #
# Synthetic data: daily cycle + small noise, with planted anomalies
# ---------------------------------------------------------------------- #
def make_series(scale, days=20, seed=0):
    """Return rows for one building and the planted spike/drop positions."""
    rng = random.Random(seed)
    values = []
    for i in range(days * 24):
        daily = 1.0 + 0.5 * math.sin(2 * math.pi * (i % 24) / 24)   # day/night cycle
        values.append(scale * daily * (1 + rng.gauss(0, 0.02)))
    spikes = [100, 250, 400]
    drops = [150, 333]
    for i in spikes:
        values[i] = values[i] * 4.0          # far above that hour's usual level
    for i in drops:
        values[i] = values[i] * 0.05         # collapses to ~5 % of normal
    return values, spikes, drops


@pytest.fixture()
def synthetic():
    rows, planted = [], {}
    for building, scale in ((1, 5.0), (2, 5000.0)):    # tiny and huge building
        values, spikes, drops = make_series(scale, seed=building)
        for i, v in enumerate(values):
            rows.append((building, START + i * HOUR, v))
        planted[building] = (spikes, drops)
    return TimeSeriesStore.from_rows(rows, seed=0), planted


# ====================================================================== #
# A. Planted anomalies are found
# ====================================================================== #
def _planted_times(planted_for_building):
    spikes, drops = planted_for_building
    to_time = lambda i: datetime(2016, 1, 1) + timedelta(hours=i)
    return [to_time(i) for i in spikes], [to_time(i) for i in drops]


@pytest.mark.parametrize("building", [1, 2])      # tiny and huge building
def test_planted_spikes_and_drops_are_flagged(synthetic, building):
    """Hourly method: every planted spike/drop is found, with the right kind."""
    store, planted = synthetic
    spikes, drops = _planted_times(planted[building])
    det = AnomalyDetector(store, method="hourly", k=4.0, sentinel_value=None)
    found = {a.time: a for a in det.detect_history(building)}
    for when in spikes:
        assert when in found and found[when].kind == "spike"
    for when in drops:
        assert when in found and found[when].kind == "drop"


def test_hourly_method_has_almost_no_false_alarms(synthetic):
    store, planted = synthetic
    det = AnomalyDetector(store, method="hourly", k=5.0, sentinel_value=None)
    spikes, drops = planted[1]
    assert len(det.detect_history(1)) <= len(spikes) + len(drops) + 1


def test_building_method_misses_what_the_daily_cycle_hides(synthetic):
    """Why the hourly baseline exists: a spike at a low-usage hour of the daily
    cycle (hour 16 here) stays inside the building-wide spread, so the
    building-level z-score misses it while the hourly z-score catches it."""
    store, planted = synthetic
    spikes, drops = _planted_times(planted[1])
    planted_times = set(spikes + drops)
    building_found = {a.time for a in AnomalyDetector(
        store, method="building", k=4.0, sentinel_value=None).detect_history(1)}
    hourly_found = {a.time for a in AnomalyDetector(
        store, method="hourly", k=4.0, sentinel_value=None).detect_history(1)}
    assert planted_times <= hourly_found
    assert len(planted_times & building_found) < len(planted_times & hourly_found)


# ====================================================================== #
# B. Detector behaviour and edge cases
# ====================================================================== #
def test_stuck_value_rule_takes_priority_and_can_be_disabled():
    rows = [(7, START + i * HOUR, 100.0 + (i % 5)) for i in range(48)]
    rows[10] = (7, rows[10][1], 1.0)
    store = TimeSeriesStore.from_rows(rows, seed=0)
    on = AnomalyDetector(store, method="building", k=50.0, sentinel_value=1.0)
    off = AnomalyDetector(store, method="building", k=50.0, sentinel_value=None)
    flagged = on.detect_history(7)
    assert [a.kind for a in flagged] == ["stuck-value"]
    assert "placeholder" in flagged[0].reason
    assert off.detect_history(7) == []


def test_detect_window_equals_filtered_history(synthetic):
    store, _ = synthetic
    det = AnomalyDetector(store, method="hourly", k=4.0, sentinel_value=None)
    start, end = "2016-01-05 00:00", "2016-01-12 23:00"
    window = det.detect(1, start, end)
    expected = [a for a in det.detect_history(1)
                if datetime(2016, 1, 5) <= a.time <= datetime(2016, 1, 12, 23)]
    assert window == expected and len(window) > 0


def test_detect_edge_cases(synthetic):
    store, _ = synthetic
    det = AnomalyDetector(store, method="hourly", k=4.0)
    assert det.detect(999, "2016-01-01", "2016-02-01") == []            # unknown building
    assert det.detect(1, "2016-01-10", "2016-01-05") == []              # start > end
    assert det.detect(1, "2030-01-01", "2030-02-01") == []              # no data in window
    assert det.baseline(999) is None


def test_flagged_keys_matches_detect_history(synthetic):
    store, _ = synthetic
    det = AnomalyDetector(store, method="hourly", k=4.0, sentinel_value=None)
    keys = det.flagged_keys([1, 2])
    expected = {(a.building_id, int((a.time - datetime(1970, 1, 1)).total_seconds()))
                for b in (1, 2) for a in det.detect_history(b)}
    assert keys == expected


def test_invalid_parameters(synthetic):
    store, _ = synthetic
    with pytest.raises(ValueError):
        AnomalyDetector(store, method="bogus")
    with pytest.raises(ValueError):
        AnomalyDetector(store, k=0)
    with pytest.raises(ValueError):
        signed_z(compute_baseline([(START, 1.0)]), "bogus", START, 1.0)


# ---------------------------------------------------------------------- #
# Baseline maths
# ---------------------------------------------------------------------- #
def test_mean_and_std_match_numpy():
    np = pytest.importorskip("numpy")
    rng = random.Random(1)
    values = [rng.gauss(50, 7) for _ in range(500)]
    assert len(set(values)) == 500          # genuinely varied data
    mean, std = mean_and_std(values)
    assert mean == pytest.approx(np.mean(values))
    assert std == pytest.approx(np.std(values, ddof=1))


def test_mean_and_std_degenerate_inputs():
    assert mean_and_std([]) == (0.0, 0.0)
    assert mean_and_std([5.0]) == (5.0, 0.0)
    assert mean_and_std([3.0, 3.0, 3.0]) == (3.0, 0.0)


def test_hourly_baseline_groups_by_hour_of_day():
    readings = [(START + i * HOUR, float(i % 24)) for i in range(24 * 3)]
    baseline = compute_baseline(readings)
    assert hour_of_day(START + 5 * HOUR) == 5
    assert baseline.hourly[5] == (5.0, 0.0, 3)         # three days, always 5.0
    assert baseline.n == 72


def test_signed_z_sign_and_zero_spread_guard():
    baseline = Baseline(mean=10.0, std=2.0, n=100, hourly={})
    assert signed_z(baseline, "building", START, 16.0) == pytest.approx(3.0)
    assert signed_z(baseline, "building", START, 4.0) == pytest.approx(-3.0)
    flat = Baseline(mean=10.0, std=0.0, n=100, hourly={})
    assert signed_z(flat, "building", START, 99.0) == 0.0     # cannot judge
    assert signed_z(baseline, None, START, 99.0) == 0.0


def test_hourly_falls_back_to_building_stats_for_thin_hours():
    baseline = Baseline(mean=10.0, std=2.0, n=100, hourly={0: (50.0, 1.0, 3)})
    # hour 0 has only 3 readings (< 10) -> building statistics are used
    assert signed_z(baseline, "hourly", START, 12.0, min_hour_count=10) == pytest.approx(1.0)
    assert signed_z(baseline, "hourly", START, 52.0, min_hour_count=3) == pytest.approx(2.0)


# ====================================================================== #
# C. Evaluation maths (hand-computed)
# ====================================================================== #
def test_score_hand_computed_example():
    truth = {(1, 10), (1, 20), (1, 30)}
    predicted = {(1, 10), (1, 20), (1, 99)}          # 2 right, 1 wrong, 1 missed
    s = score(predicted, truth)
    assert (s.tp, s.fp, s.fn) == (2, 1, 1)
    assert s.precision == pytest.approx(2 / 3)
    assert s.recall == pytest.approx(2 / 3)
    assert s.f1 == pytest.approx(2 / 3)
    assert s.flagged == 3


def test_score_degenerate_cases():
    assert score(set(), {(1, 1)}).f1 == 0.0
    assert score(set(), {(1, 1)}).precision == 0.0       # nothing flagged
    assert score({(1, 1)}, set()).recall == 0.0          # nothing to find
    perfect = score({(1, 1)}, {(1, 1)})
    assert (perfect.precision, perfect.recall, perfect.f1) == (1.0, 1.0, 1.0)


def test_anomaly_runs_and_event_recall():
    h = HOUR
    truth = {(1, 0), (1, h), (1, 2 * h),          # run of 3
             (1, 10 * h),                          # run of 1 (gap before it)
             (2, 0), (2, h)}                       # run of 2 in another building
    runs = anomaly_runs(truth)
    assert [len(r) for r in runs] == [3, 1, 2]
    assert event_recall({(1, h)}, runs) == pytest.approx(1 / 3)       # one run touched
    assert event_recall({(1, 0), (1, 10 * h), (2, h)}, runs) == 1.0
    assert event_recall(set(), runs) == 0.0
    assert event_recall({(1, 0)}, []) == 0.0


def test_split_buildings_is_disjoint_complete_and_repeatable():
    ids = list(range(100, 200))
    tune, test = split_buildings(ids, seed=1)
    assert set(tune).isdisjoint(test) and sorted(tune + test) == ids
    assert (tune, test) == split_buildings(ids, seed=1)
    assert (tune, test) != split_buildings(ids, seed=2)


def test_restrict_keeps_only_requested_buildings():
    assert restrict({(1, 5), (2, 5), (3, 5)}, [1, 3]) == {(1, 5), (3, 5)}


# ====================================================================== #
# D. Real data
# ====================================================================== #
def _find(env_name, filename):
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (os.environ.get(env_name), os.path.join(here, filename),
                 os.path.join(here, "data", filename), filename):
        if path and os.path.exists(path):
            return path
    return None


@pytest.fixture(scope="module")
def real():
    readings = _find("STORE_DATA_CSV", "store_data.csv")
    labels = _find("GROUND_TRUTH_CSV", "ground_truth.csv")
    if readings is None or labels is None:
        pytest.skip("store_data.csv / ground_truth.csv not found")
    store = TimeSeriesStore.from_csv(readings, seed=42)
    return store, load_ground_truth(labels)


def test_real_labels_line_up_with_readings(real):
    """Every labelled anomaly must be a reading that exists in the store."""
    store, truth = real
    assert len(truth) == 3144
    sample = random.Random(0).sample(sorted(truth), 300)
    for building, epoch in sample:
        assert store.query_raw(building, epoch, epoch), (building, epoch)
    assert {b for b, _ in truth} <= set(store.buildings())


def test_real_stuck_value_rule_is_precise(real):
    store, truth = real
    det = AnomalyDetector(store, method=None, sentinel_value=1.0)
    s = score(det.flagged_keys(), truth)
    assert s.precision > 0.95            # almost every reading == 1.0 is labelled
    assert s.flagged == 1195             # the number of readings equal to 1.0


def test_real_adding_zscores_never_lowers_recall(real):
    """Logical guarantee: the union of two rules finds at least what one finds."""
    store, truth = real
    ids = store.buildings()[:40]
    only_rule = AnomalyDetector(store, method=None, sentinel_value=1.0).flagged_keys(ids)
    combined = AnomalyDetector(store, method="hourly", k=5.0, sentinel_value=1.0).flagged_keys(ids)
    assert only_rule <= combined
    t = restrict(truth, ids)
    assert score(combined, t).recall >= score(only_rule, t).recall


def test_real_larger_k_flags_fewer_readings(real):
    store, _ = real
    ids = store.buildings()[:40]
    counts = [len(AnomalyDetector(store, method="hourly", k=k, sentinel_value=None)
                  .flagged_keys(ids)) for k in (3.0, 5.0, 8.0)]
    assert counts[0] >= counts[1] >= counts[2]
