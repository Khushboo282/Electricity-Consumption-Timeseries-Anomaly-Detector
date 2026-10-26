"""
evaluation.py -- Measuring how accurate the anomaly detector is.

The ground truth file has one row per reading with ``anomaly`` = 0 or 1. Only
about 1.2 % of the readings are anomalies, so PLAIN ACCURACY IS MEANINGLESS
here: saying "nothing is ever anomalous" is already 98.8 % accurate. We report

    precision = of the readings we flagged, how many really were anomalies
    recall    = of the real anomalies, how many we flagged
    F1        = harmonic mean of the two (high only if BOTH are high)

and, because anomalies come in runs of consecutive hours (median length 3,
longest 36),

    event recall = of the real anomaly RUNS, the share where we flagged at
                   least one reading of the run.

Everything works on sets of ``(building_id, epoch_seconds)`` keys, the same key
the skip list uses, so predictions and labels line up exactly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import pandas as pd

from timeseries_store import timestamps_to_epochs

Key = Tuple[int, int]
SECONDS_PER_HOUR = 3600


def load_ground_truth(csv_path: str) -> Set[Key]:
    """Return the set of ``(building_id, epoch)`` labelled anomalous (== 1).

    Rows are matched to the readings by KEY, not by row position, so a
    different row order in the two files cannot silently misalign the labels.
    """
    df = pd.read_csv(csv_path)
    for column in ("building_id", "timestamp", "anomaly"):
        if column not in df.columns:
            raise ValueError(f"{csv_path} is missing required column {column!r}")
    df = df[df["anomaly"] == 1]
    epochs = timestamps_to_epochs(df["timestamp"])
    return set(zip(df["building_id"].tolist(), epochs.tolist()))


def restrict(keys: Iterable[Key], building_ids: Iterable[int]) -> Set[Key]:
    """Keep only the keys that belong to the given buildings."""
    wanted = set(building_ids)
    return {key for key in keys if key[0] in wanted}


@dataclass(frozen=True)
class Scores:
    """Confusion counts and the metrics derived from them."""
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float

    @property
    def flagged(self) -> int:
        return self.tp + self.fp


def score(predicted: Set[Key], truth: Set[Key]) -> Scores:
    """Precision / recall / F1 of a set of predicted anomalies.

    Convention: a metric whose denominator is 0 is reported as 0.0 (e.g.
    precision when nothing was flagged).
    """
    tp = len(predicted & truth)
    fp = len(predicted - truth)
    fn = len(truth - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return Scores(tp, fp, fn, precision, recall, f1)


def anomaly_runs(truth: Set[Key]) -> List[List[Key]]:
    """Group labelled anomalies into runs of consecutive hours (per building).

    Two anomalies belong to the same run when they are exactly one hour apart.
    A missing hour or a gap between the data blocks ends the run.
    """
    runs: List[List[Key]] = []
    current: List[Key] = []
    for key in sorted(truth):
        if current and key[0] == current[-1][0] and key[1] - current[-1][1] == SECONDS_PER_HOUR:
            current.append(key)
        else:
            if current:
                runs.append(current)
            current = [key]
    if current:
        runs.append(current)
    return runs


def event_recall(predicted: Set[Key], runs: Sequence[Sequence[Key]]) -> float:
    """Share of anomaly runs with at least one flagged reading (0.0 if no runs)."""
    if not runs:
        return 0.0
    caught = sum(1 for run in runs if any(key in predicted for key in run))
    return caught / len(runs)


def split_buildings(building_ids: Iterable[int], seed: int = 42
                    ) -> Tuple[List[int], List[int]]:
    """Split buildings into two disjoint halves: (tune, test).

    The threshold ``k`` is chosen on the TUNE half and then reported on the TEST
    half, which the choice never saw, so the reported accuracy is not inflated
    by tuning on the same data.
    """
    ids = sorted(set(building_ids))
    random.Random(seed).shuffle(ids)
    half = len(ids) // 2
    return sorted(ids[:half]), sorted(ids[half:])
