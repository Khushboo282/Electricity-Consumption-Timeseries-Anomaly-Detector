"""
brute_force.py -- The linear-scan baseline (and our correctness oracle).

WHY THIS FILE EXISTS
--------------------
1. Correctness oracle: it is so simple that it is obviously right, so if the
   skip list returns exactly the same answers, the skip list is right.
2. Benchmark baseline: the problem statement asks us to compare the index with
   "a simple linear scan".

NOTE FOR THE TEAM: the brute-force part is owned by a teammate. This minimal
version only fixes the contract; a teammate's version can replace it as long as
it keeps the same function signature and result shape.

CONTRACT (shared with TimeSeriesStore.query_raw)
------------------------------------------------
    brute_force_query_raw(rows, building_id, start_epoch, end_epoch)
      -> [((building_id, epoch), value), ...]   sorted by time, inclusive ends,
         empty list if nothing matches or start > end.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Sequence, Tuple

Row = Tuple[int, int, float]
RawResult = Tuple[Tuple[int, int], float]


@dataclass
class BruteForcePoint:
    """One reading in the shape used by the Streamlit comparison tab."""

    ts: int
    timestamp: str
    value: float
    building_id: int


class BruteForceList:
    """Linear-scan list used by the UI to compare against the skip list."""

    def __init__(self) -> None:
        self._points: List[BruteForcePoint] = []

    def insert(self, ts: int, timestamp: str, value: float,
               building_id: int) -> None:
        self._points.append(BruteForcePoint(ts, timestamp, value, building_id))

    def range_query(self, start_ts: int, end_ts: int,
                    building_id: int) -> Tuple[List[BruteForcePoint], float, int]:
        """Return matching points, elapsed microseconds, and rows inspected."""
        started = time.perf_counter()
        matches = [point for point in self._points
                   if point.building_id == building_id
                   and start_ts <= point.ts <= end_ts]
        matches.sort(key=lambda point: point.timestamp)
        elapsed_us = (time.perf_counter() - started) * 1e6
        return matches, elapsed_us, len(self._points)


def brute_force_query_raw(rows: Sequence[Row], building_id: int,
                          start_epoch: int, end_epoch: int) -> List[RawResult]:
    """Scan EVERY row and keep those of ``building_id`` inside the time range.

    Cost is O(n) per query no matter how small the range is. The result is
    sorted by key so it can be compared directly with the skip list's output
    (the skip list returns keys in sorted order by construction).
    """
    matches = [((b, e), v) for b, e, v in rows
               if b == building_id and start_epoch <= e <= end_epoch]
    matches.sort(key=lambda item: item[0])
    return matches
