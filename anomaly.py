"""
anomaly.py -- Statistical anomaly detection on top of the skip-list store.

NO MACHINE LEARNING. Everything here is descriptive statistics plus a
hand-chosen threshold:

    1. Ask the skip list for a building's readings  (a RANGE QUERY).
    2. Describe what is "normal" for that building: mean and standard
       deviation (the standard deviation is the square root of the variance).
    3. Flag a reading when it is more than ``k`` standard deviations away from
       normal  (|z| > k, where z = (value - mean) / std).

Why per building?  Building scales differ enormously (typical readings range
from a few units to several thousand), so one global mean/std would flag the
big buildings constantly. Each building is compared only with itself.

Why raw min / max are not used on their own: the largest value of a dataset
can never exceed the largest value of that same dataset, so a max-based fence
computed from the data flags nothing. The variance/standard deviation is what
turns "typical spread" into a usable threshold.

TWO KINDS OF "NORMAL"
---------------------
  method="building" one mean/std per building (all hours together).
  method="hourly"   one mean/std per building AND hour of day. Electricity use
                    follows a daily cycle (night vs day), so comparing a 3 a.m.
                    reading with other 3 a.m. readings is fairer than comparing
                    it with the building's overall average.
  method=None       no z-score test at all (only the stuck-value rule below).

STUCK-VALUE RULE (dataset specific!)
------------------------------------
In this dataset a reading of exactly 1.0 behaves like a placeholder: 1,193 of
the 1,195 readings equal to 1.0 are labelled anomalies. ``sentinel_value=1.0``
flags them. This is NOT a statistical method; it is a data-specific rule that
we discovered by inspecting the data, and the report should say so. Set
``sentinel_value=None`` to switch it off.

THE SKIP LIST'S ROLE
--------------------
The detector never touches the raw rows. Every reading it looks at comes from
``store.query_raw(building, start, end)``:
  * the baseline = one range query over the building's whole history;
  * ``detect(building, start, end)`` = a range query over just that window,
    scored against the baseline (this is what the UI will call).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Set, Tuple

from timeseries_store import TimeSeriesStore, epoch_to_datetime, to_epoch

SECONDS_PER_HOUR = 3600

# (epoch_seconds, value) pairs of ONE building, in time order
Readings = List[Tuple[int, float]]


# ---------------------------------------------------------------------- #
# Baseline statistics
# ---------------------------------------------------------------------- #
def mean_and_std(values: List[float]) -> Tuple[float, float]:
    """Mean and SAMPLE standard deviation (divide by n - 1), two-pass.

    Returns std = 0.0 when fewer than 2 values are given.
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(variance)


def hour_of_day(epoch: int) -> int:
    """Hour of day 0..23 of an epoch-seconds value (timestamps are naive)."""
    return (epoch // SECONDS_PER_HOUR) % 24


@dataclass
class Baseline:
    """What 'normal' looks like for one building.

    Attributes:
        mean, std, n: over all readings of the building.
        hourly: ``{hour_of_day: (mean, std, n)}`` over the readings at that hour.
    """
    mean: float
    std: float
    n: int
    hourly: Dict[int, Tuple[float, float, int]]


def compute_baseline(readings: Readings) -> Baseline:
    """Build a :class:`Baseline` from ``[(epoch, value), ...]``."""
    values = [v for _, v in readings]
    mean, std = mean_and_std(values)
    by_hour: Dict[int, List[float]] = {}
    for epoch, value in readings:
        by_hour.setdefault(hour_of_day(epoch), []).append(value)
    hourly = {h: (*mean_and_std(vs), len(vs)) for h, vs in by_hour.items()}
    return Baseline(mean=mean, std=std, n=len(values), hourly=hourly)


def signed_z(baseline: Baseline, method: Optional[str], epoch: int,
             value: float, min_hour_count: int = 10) -> float:
    """Signed z-score of one reading (positive = above normal, negative = below).

    Returns 0.0 when no judgement is possible (``method`` is None, or the
    relevant standard deviation is zero, i.e. the building never varies).
    For ``method="hourly"``, an hour with fewer than ``min_hour_count``
    readings (or zero spread) falls back to the building-level statistics.
    """
    if method is None:
        return 0.0
    mean, std = baseline.mean, baseline.std
    if method == "hourly":
        h_mean, h_std, h_n = baseline.hourly.get(hour_of_day(epoch), (0.0, 0.0, 0))
        if h_n >= min_hour_count and h_std > 0:
            mean, std = h_mean, h_std
    elif method != "building":
        raise ValueError(f"unknown method {method!r}; use 'building', 'hourly' or None")
    if std == 0:
        return 0.0
    return (value - mean) / std


# ---------------------------------------------------------------------- #
# Result type
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class Anomaly:
    """One flagged reading.

    kind: "stuck-value" (hit the sentinel rule), "spike" (z > 0) or "drop" (z < 0).
    score: the signed z-score (0.0 for a stuck-value flag raised without a z-test).
    """
    building_id: int
    time: datetime
    value: float
    score: float
    kind: str

    @property
    def reason(self) -> str:
        if self.kind == "stuck-value":
            return f"reading equals the placeholder value {self.value:g}"
        return f"{self.kind}: {abs(self.score):.1f} standard deviations {'above' if self.score > 0 else 'below'} normal"


# ---------------------------------------------------------------------- #
# The detector
# ---------------------------------------------------------------------- #
class AnomalyDetector:
    """Flags unusual readings using range queries on a :class:`TimeSeriesStore`.

    Args:
        store: the skip-list backed store holding the readings.
        method: "hourly" (default), "building" or None  (see module docstring).
        k: threshold; a reading is flagged when ``|z| > k``.
        sentinel_value: also flag readings exactly equal to this value
            (default 1.0, dataset-specific); None disables the rule.
        min_hour_count: minimum readings an hour-of-day needs to have its own
            statistics (otherwise the building-level ones are used).

    Example:
        >>> det = AnomalyDetector(store, method="hourly", k=5.0)      # doctest: +SKIP
        >>> det.detect(107, "2016-01-01", "2016-01-05 23:00")         # doctest: +SKIP
    """

    def __init__(self, store: TimeSeriesStore, method: Optional[str] = "hourly",
                 k: float = 4.0, sentinel_value: Optional[float] = 1.0,
                 min_hour_count: int = 10) -> None:
        if method not in ("hourly", "building", None):
            raise ValueError("method must be 'hourly', 'building' or None")
        if k <= 0:
            raise ValueError("k must be positive")
        self.store = store
        self.method = method
        self.k = k
        self.sentinel_value = sentinel_value
        self.min_hour_count = min_hour_count
        self._baselines: Dict[int, Baseline] = {}

    # -- baseline ----------------------------------------------------------
    def history(self, building_id: int) -> Readings:
        """The building's whole history via ONE range query (time-ordered)."""
        span = self.store.span(building_id)
        if span is None:
            return []
        raw = self.store.query_raw(building_id, to_epoch(span[0]), to_epoch(span[1]))
        return [(key[1], value) for key, value in raw]

    def baseline(self, building_id: int) -> Optional[Baseline]:
        """Baseline of a building (computed once, then cached); None if unknown."""
        if building_id not in self._baselines:
            readings = self.history(building_id)
            if not readings:
                return None
            self._baselines[building_id] = compute_baseline(readings)
        return self._baselines[building_id]

    # -- scoring -----------------------------------------------------------
    def _judge(self, baseline: Baseline, epoch: int,
               value: float) -> Optional[Tuple[str, float]]:
        """Return (kind, z) if the reading is anomalous, else None."""
        z = signed_z(baseline, self.method, epoch, value, self.min_hour_count)
        if self.sentinel_value is not None and value == self.sentinel_value:
            return "stuck-value", z
        if abs(z) > self.k:
            return ("spike" if z > 0 else "drop"), z
        return None

    def detect(self, building_id: int, start, end) -> List[Anomaly]:
        """Anomalies of one building with ``start <= time <= end``.

        The window is fetched with a skip-list range query and scored against
        the building's whole-history baseline. Returns [] for an unknown
        building, an empty window or ``start > end``.
        """
        baseline = self.baseline(building_id)
        if baseline is None:
            return []
        raw = self.store.query_raw(building_id, to_epoch(start), to_epoch(end))
        found = []
        for (_, epoch), value in raw:
            verdict = self._judge(baseline, epoch, value)
            if verdict is not None:
                kind, z = verdict
                found.append(Anomaly(building_id, epoch_to_datetime(epoch),
                                     value, z, kind))
        return found

    def detect_history(self, building_id: int) -> List[Anomaly]:
        """Anomalies over the building's entire history."""
        span = self.store.span(building_id)
        if span is None:
            return []
        return self.detect(building_id, span[0], span[1])

    def flagged_keys(self, building_ids: Optional[Iterable[int]] = None
                     ) -> Set[Tuple[int, int]]:
        """Set of ``(building_id, epoch)`` flagged anomalous (for evaluation)."""
        if building_ids is None:
            building_ids = self.store.buildings()
        flagged: Set[Tuple[int, int]] = set()
        for b in building_ids:
            baseline = self.baseline(b)
            if baseline is None:
                continue
            for epoch, value in self.history(b):
                if self._judge(baseline, epoch, value) is not None:
                    flagged.add((b, epoch))
        return flagged
