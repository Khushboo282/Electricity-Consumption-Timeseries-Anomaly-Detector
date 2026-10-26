"""
timeseries_store.py -- Time-series store built on top of the skip list.

This module is the "storage + query layer" of the project. It turns the raw
CSV (building_id, timestamp, meter_reading) into skip-list entries and exposes
one friendly function:

    store.query(building_id, start, end)

DESIGN DECISIONS (and why)
--------------------------
1. Key = (building_id, epoch_seconds), value = meter_reading.
   The dataset contains 192 buildings that share the same timestamps, so a
   timestamp alone is not unique. With a tuple key, Python sorts first by
   building and then by time, so one building's readings form ONE contiguous
   run in the skip list and "building X between t1 and t2" is a single range
   query. A query for one building can never leak into the next building,
   because the range is bounded by (building, start) and (building, end).

2. The CSV column ``ts`` is ignored. It is a scaled epoch that can differ from
   ``timestamp`` by up to ~13 minutes, so it is not an exact time. We parse the
   ``timestamp`` column into integer seconds since 1970-01-01 instead (integers
   compare faster than strings or datetime objects).

3. Timestamps are treated as timezone-naive (the dataset has no time zone).

4. Raw values are stored untouched (no cleaning, no gap filling), see README.
"""

from __future__ import annotations

import numbers
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from skiplist import SkipList

_EPOCH = pd.Timestamp("1970-01-01")
_ONE_SECOND = pd.Timedelta(seconds=1)

# A raw row: (building_id, epoch_seconds, meter_reading)
Row = Tuple[int, int, float]
# A raw query result: ((building_id, epoch_seconds), meter_reading)
RawResult = Tuple[Tuple[int, int], float]


# ---------------------------------------------------------------------- #
# Time helpers
# ---------------------------------------------------------------------- #
def to_epoch(t: Any) -> int:
    """Convert a time to integer seconds since 1970-01-01 (naive UTC).

    Accepts an ``int`` (already epoch seconds), a string such as
    ``"2016-01-01 09:30"``, a ``datetime`` or a pandas ``Timestamp``.
    Timezone-aware values are converted to UTC and then made naive.
    """
    if isinstance(t, numbers.Integral) and not isinstance(t, bool):
        return int(t)
    stamp = pd.Timestamp(t)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return int((stamp - _EPOCH) // _ONE_SECOND)


def epoch_to_datetime(epoch: int) -> datetime:
    """Inverse of :func:`to_epoch` (returns a naive ``datetime``)."""
    return datetime(1970, 1, 1) + timedelta(seconds=int(epoch))


# ---------------------------------------------------------------------- #
# CSV loading
# ---------------------------------------------------------------------- #
def timestamps_to_epochs(column: "pd.Series") -> "pd.Series":
    """Vectorised :func:`to_epoch` for a whole pandas column of time strings.

    Used by every loader (readings AND ground truth) so that both files are
    converted in exactly the same way.
    """
    stamps = pd.to_datetime(column)
    return ((stamps - _EPOCH) // _ONE_SECOND).astype("int64")


def load_rows(csv_path: str) -> List[Row]:
    """Read the CSV into a list of ``(building_id, epoch_seconds, value)``.

    Rows are returned in FILE ORDER (the file is not sorted by time; the skip
    list does the sorting). Only ``building_id``, ``timestamp`` and
    ``meter_reading`` are used.

    Raises:
        ValueError: if a required column is missing.
    """
    needed = ["building_id", "timestamp", "meter_reading"]
    df = pd.read_csv(csv_path)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required column(s): {missing}")
    epochs = timestamps_to_epochs(df["timestamp"])
    return list(zip(df["building_id"].tolist(),
                    epochs.tolist(),
                    df["meter_reading"].tolist()))


# ---------------------------------------------------------------------- #
# The store
# ---------------------------------------------------------------------- #
class TimeSeriesStore:
    """In-memory time-series store indexed by a skip list.

    Args:
        max_level, p, seed: forwarded to :class:`skiplist.SkipList`.

    Typical use:
        >>> store = TimeSeriesStore.from_csv("store_data.csv", seed=42)   # doctest: +SKIP
        >>> store.query(107, "2016-01-01 09:00", "2016-01-01 12:00")      # doctest: +SKIP
        [(datetime.datetime(2016, 1, 1, 9, 0), 1.0), ...]
    """

    def __init__(self, max_level: int = 20, p: float = 0.5,
                 seed: Optional[int] = None) -> None:
        self._index = SkipList(max_level=max_level, p=p, seed=seed)
        # building_id -> [first_epoch, last_epoch]; handy for UI and for
        # "whole history of a building" queries in the anomaly detector.
        self._spans: Dict[int, List[int]] = {}

    # -- construction ---------------------------------------------------- #
    @classmethod
    def from_rows(cls, rows: Iterable[Row], max_level: int = 20, p: float = 0.5,
                  seed: Optional[int] = None) -> "TimeSeriesStore":
        """Build a store from ``(building_id, epoch_seconds, value)`` rows."""
        store = cls(max_level=max_level, p=p, seed=seed)
        for building_id, epoch, value in rows:
            store._insert_epoch(building_id, epoch, value)
        return store

    @classmethod
    def from_csv(cls, csv_path: str, max_level: int = 20, p: float = 0.5,
                 seed: Optional[int] = None) -> "TimeSeriesStore":
        """Load the CSV and index every row."""
        return cls.from_rows(load_rows(csv_path), max_level=max_level, p=p,
                             seed=seed)

    # -- writing --------------------------------------------------------- #
    def _insert_epoch(self, building_id: int, epoch: int, value: float) -> None:
        self._index.insert((building_id, epoch), value)
        span = self._spans.get(building_id)
        if span is None:
            self._spans[building_id] = [epoch, epoch]
        else:
            if epoch < span[0]:
                span[0] = epoch
            if epoch > span[1]:
                span[1] = epoch

    def insert(self, building_id: int, timestamp: Any, value: float) -> None:
        """Insert one reading. ``timestamp`` may be a str/datetime/epoch int.

        If a reading for the same building and time already exists, its value
        is replaced.
        """
        self._insert_epoch(int(building_id), to_epoch(timestamp), float(value))

    # -- reading --------------------------------------------------------- #
    def query_raw(self, building_id: int, start_epoch: int,
                  end_epoch: int) -> List[RawResult]:
        """Index-level query on integer epochs; returns raw skip-list pairs.

        Result shape: ``[((building_id, epoch), value), ...]`` sorted by time.
        This is what the benchmark times, so no conversion cost is included.
        """
        return self._index.range_query((building_id, start_epoch),
                                       (building_id, end_epoch))

    def query_raw_counted(self, building_id: int, start_epoch: int,
                          end_epoch: int) -> Tuple[List[RawResult], int]:
        """Return raw query results and the skip list's key-comparison count."""
        return self._index.range_query_counted(
            (building_id, start_epoch), (building_id, end_epoch))

    def query(self, building_id: int, start: Any,
              end: Any) -> List[Tuple[datetime, float]]:
        """All readings of one building with ``start <= time <= end``.

        Args:
            building_id: the building to look at.
            start, end: inclusive bounds; str / datetime / Timestamp / epoch int.

        Returns:
            ``[(datetime, meter_reading), ...]`` sorted by time. The list is
            empty if the building is unknown, if no reading falls inside the
            range (e.g. the range lies in one of the data gaps), or if
            ``start > end``.
        """
        raw = self.query_raw(int(building_id), to_epoch(start), to_epoch(end))
        return [(epoch_to_datetime(key[1]), value) for key, value in raw]

    # -- metadata helpers (UI / anomaly detection) ----------------------- #
    def buildings(self) -> List[int]:
        """Sorted list of building ids present in the store."""
        return sorted(self._spans)

    def span(self, building_id: int) -> Optional[Tuple[datetime, datetime]]:
        """(first, last) reading time of a building, or None if unknown."""
        span = self._spans.get(building_id)
        if span is None:
            return None
        return epoch_to_datetime(span[0]), epoch_to_datetime(span[1])

    def __len__(self) -> int:
        return len(self._index)

    @property
    def index(self) -> SkipList:
        """The underlying skip list (exposed for tests and reporting)."""
        return self._index
