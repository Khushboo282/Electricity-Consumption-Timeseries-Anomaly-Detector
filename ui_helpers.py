"""
ui_helpers.py -- Pure helper functions for the Streamlit app (app.py).

They live in their own module so they can be unit-tested without a browser.

CONTENTS
  find_data_file       locate store_data.csv / ground_truth.csv (the data folder)
  data_blocks          split a building's readings into its separate time blocks
  series_frame         readings -> DataFrame, with breaks where data is missing
  normal_range         the "normal band" (mean +/- k std) shown behind the chart
  time_best_us         robust timing of a function call
  load_brute_force     build the in-project BruteForceList from the CSV
  brute_range          run a range query on it with the same inputs as the skip list
  brute_keys           turn its results into (building, epoch) keys for comparison
"""

from __future__ import annotations

import csv
import os
import time
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import altair as alt
import pandas as pd

from anomaly import Baseline, hour_of_day
from timeseries_store import to_epoch

HERE = os.path.dirname(os.path.abspath(__file__))
SECONDS_PER_HOUR = 3600


# ---------------------------------------------------------------------- #
# Locating the data files
# ---------------------------------------------------------------------- #
def candidate_dirs(extra: Iterable[str] = ()) -> List[str]:
    """Folders searched for the data files, most specific first."""
    dirs = [*extra,
            os.path.join(HERE, "data"), HERE,
            os.path.join(HERE, "processed data"),
            os.path.join(os.getcwd(), "data"), os.getcwd(),
            os.path.join(os.path.dirname(HERE), "data")]
    seen, ordered = set(), []
    for d in dirs:
        d = os.path.abspath(d)
        if d not in seen:
            seen.add(d)
            ordered.append(d)
    return ordered


def find_data_file(filename: str, env_var: Optional[str] = None,
                   extra_dirs: Iterable[str] = ()) -> Optional[str]:
    """Return the path of ``filename`` or None.

    Order: the environment variable (if set), then the folders from
    :func:`candidate_dirs` (``data/`` next to the code first).
    """
    if env_var and os.environ.get(env_var) and os.path.isfile(os.environ[env_var]):
        return os.environ[env_var]
    for directory in candidate_dirs(extra_dirs):
        path = os.path.join(directory, filename)
        if os.path.isfile(path):
            return path
    return None


# ---------------------------------------------------------------------- #
# Data blocks and chart frames
# ---------------------------------------------------------------------- #
def data_blocks(epochs: Sequence[int], gap_seconds: int = 7 * 24 * SECONDS_PER_HOUR
                ) -> List[Tuple[int, int]]:
    """Group sorted epochs into blocks separated by gaps larger than ``gap_seconds``.

    The dataset has up to 12 blocks (one per month, about 5 days each) separated by
    gaps of roughly 25 days. The default threshold is 7 days, so a building that is
    merely missing a day or two INSIDE a block is not split into extra blocks.
    Returns ``[(first_epoch, last_epoch), ...]`` in time order.
    """
    blocks: List[Tuple[int, int]] = []
    for e in epochs:
        if blocks and e - blocks[-1][1] <= gap_seconds:
            blocks[-1] = (blocks[-1][0], e)
        else:
            blocks.append((e, e))
    return blocks


def series_frame(readings: Sequence[Tuple[object, float]],
                 gap_seconds: int = 2 * SECONDS_PER_HOUR) -> pd.DataFrame:
    """Readings ``[(datetime, value)]`` -> DataFrame(time, value).

    Where two consecutive readings are more than ``gap_seconds`` apart, an empty
    row is inserted so the chart line BREAKS instead of drawing a misleading
    straight line across missing data.
    """
    rows = []
    previous = None
    for when, value in readings:
        if previous is not None and (when - previous).total_seconds() > gap_seconds:
            rows.append({"time": previous + (when - previous) / 2, "value": None})
        rows.append({"time": when, "value": value})
        previous = when
    return pd.DataFrame(rows, columns=["time", "value"])


def normal_range(baseline: Baseline, method: Optional[str], epoch: int, k: float,
                 min_hour_count: int = 10) -> Optional[Tuple[float, float]]:
    """The interval ``(mean - k*std, mean + k*std)`` a reading must stay inside.

    Uses exactly the statistics the detector uses for that reading (hourly
    statistics when available, otherwise the building's). Returns None when no
    z-test applies (``method`` None or zero spread).
    """
    if method is None:
        return None
    mean, std = baseline.mean, baseline.std
    if method == "hourly":
        h_mean, h_std, h_n = baseline.hourly.get(hour_of_day(epoch), (0.0, 0.0, 0))
        if h_n >= min_hour_count and h_std > 0:
            mean, std = h_mean, h_std
    if std == 0:
        return None
    return mean - k * std, mean + k * std


def band_frame(readings: Sequence[Tuple[object, float]], baseline: Baseline,
               method: Optional[str], k: float,
               gap_seconds: int = 2 * SECONDS_PER_HOUR,
               min_hour_count: int = 10) -> pd.DataFrame:
    """DataFrame(time, lo, hi): the normal band for each reading's time.

    Breaks (empty rows) are inserted at data gaps, like :func:`series_frame`.
    Empty result if no z-test applies.
    """
    rows = []
    previous = None
    for when, _value in readings:
        rng = normal_range(baseline, method, to_epoch(when), k, min_hour_count)
        if rng is None:
            continue
        if previous is not None and (when - previous).total_seconds() > gap_seconds:
            rows.append({"time": previous + (when - previous) / 2, "lo": None, "hi": None})
        rows.append({"time": when, "lo": rng[0], "hi": rng[1]})
        previous = when
    return pd.DataFrame(rows, columns=["time", "lo", "hi"])


KIND_COLORS = {"spike": "#d62728", "drop": "#9467bd", "stuck-value": "#ff7f0e"}


def build_explore_chart(readings, anomalies, baseline, method, k,
                        truth_points=()) -> "alt.Chart":
    """The main chart: readings line, normal band, flagged points, ground truth.

    Args:
        readings: ``[(datetime, value)]`` from the skip-list range query.
        anomalies: list of :class:`anomaly.Anomaly` flagged in the window.
        baseline: the building's :class:`anomaly.Baseline` (for the band) or None.
        method, k: detector settings (decide the width of the normal band).
        truth_points: ``[(datetime, value)]`` labelled anomalous (drawn as diamonds).
    """
    layers = []
    if method is not None and baseline is not None:
        band = band_frame(readings, baseline, method, k)
        if not band.empty:
            layers.append(alt.Chart(band).mark_area(opacity=0.18, color="#4c78a8").encode(
                x="time:T", y="lo:Q", y2="hi:Q"))
    layers.append(alt.Chart(series_frame(readings)).mark_line(color="#4c78a8").encode(
        x=alt.X("time:T", title="Time"), y=alt.Y("value:Q", title="Meter reading")))
    if anomalies:
        flagged = pd.DataFrame([{"time": a.time, "value": a.value, "kind": a.kind,
                                 "z": round(a.score, 2), "why": a.reason}
                                for a in anomalies])
        layers.append(alt.Chart(flagged).mark_point(filled=True, size=90, opacity=1).encode(
            x="time:T", y="value:Q",
            color=alt.Color("kind:N", title="Flagged as",
                            scale=alt.Scale(domain=list(KIND_COLORS),
                                            range=list(KIND_COLORS.values()))),
            tooltip=["time:T", "value:Q", "kind:N", "z:Q", "why:N"]))
    if len(truth_points):
        gt = pd.DataFrame(list(truth_points), columns=["time", "value"])
        layers.append(alt.Chart(gt).mark_point(
            shape="diamond", filled=False, size=260, strokeWidth=2, color="black").encode(
            x="time:T", y="value:Q", tooltip=["time:T", "value:Q"]))
    return alt.layer(*layers)


# ---------------------------------------------------------------------- #
# Timing
# ---------------------------------------------------------------------- #
def time_best_us(func: Callable[[], object], repeats: int, rounds: int = 3) -> float:
    """Mean microseconds per call, best of ``rounds`` rounds of ``repeats`` calls."""
    best = float("inf")
    for _ in range(rounds):
        start = time.perf_counter()
        for _ in range(repeats):
            func()
        best = min(best, (time.perf_counter() - start) / repeats)
    return best * 1e6


# ---------------------------------------------------------------------- #
# Brute-force comparison list
# ---------------------------------------------------------------------- #
def load_brute_force(csv_path: str):
    """Build the in-project ``BruteForceList`` from the CSV."""
    from brute_force import BruteForceList

    brute = BruteForceList()
    with open(csv_path, mode="r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            brute.insert(ts=int(row["ts"]), timestamp=row["timestamp"],
                         value=float(row["meter_reading"]),
                         building_id=int(row["building_id"]))
    return brute


def to_brute_ts(epoch: int) -> int:
    """Convert exact epoch seconds to the CSV ``ts`` unit (epoch seconds // 1000).

    The ``ts`` column is a coarse version of the time (steps of 1,000 s). For
    query bounds on WHOLE HOURS and readings 3,600 s apart this gives exactly the
    same rows as exact times; for bounds in between it could differ at the edges.
    The UI therefore only offers whole-hour bounds.
    """
    return epoch // 1000


def brute_range(brute, building_id: int, start_epoch: int, end_epoch: int):
    """Teammate's range query; returns (results, elapsed_us, comparisons)."""
    return brute.range_query(to_brute_ts(start_epoch), to_brute_ts(end_epoch),
                             building_id=building_id)


def brute_keys(results) -> set:
    """Their DataPoint results -> set of ``(building_id, epoch_seconds)`` keys."""
    return {(r.building_id, to_epoch(r.timestamp)) for r in results}
