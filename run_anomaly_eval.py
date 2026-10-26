"""
run_anomaly_eval.py -- Measure anomaly-detection accuracy against ground truth.

Run:
    python run_anomaly_eval.py --store-csv store_data.csv --truth-csv ground_truth.csv

PROTOCOL (so the reported numbers can be trusted)
  1. Build the skip-list store from the readings; load the labels by KEY.
  2. Split the 192 buildings into two disjoint halves with a fixed seed:
     a TUNE half and a TEST half.
  3. For each detector configuration, choose the threshold k that gives the
     best F1 on the TUNE half only.
  4. Report precision / recall / F1 / event recall on the TEST half, which the
     choice of k never saw.
  The detectors are unsupervised: they never look at labels. Labels are used
  only to choose k and to score the result.

  One honest caveat: the stuck-value rule (reading == 1.0) was found by
  inspecting the data together with the labels, so it is not a blind result.

OUTPUT (in --out-dir)
  anomaly_results.csv       final table (test half)
  anomaly_tuning_sweep.csv  precision/recall/F1 for every k on the tune half
"""

from __future__ import annotations

import argparse
import csv
import os
import time

from anomaly import AnomalyDetector, signed_z
from evaluation import (anomaly_runs, event_recall, load_ground_truth, restrict,
                        score, split_buildings)
from timeseries_store import TimeSeriesStore

K_GRID = [2, 2.5, 3, 3.5, 4, 5, 6, 8, 10]
SENTINEL = 1.0

# (label, z-score method or None, use the stuck-value rule?)
CONFIGS = [
    ("building z-score", "building", False),
    ("hourly z-score", "hourly", False),
    ("stuck-value rule only (==1.0)", None, True),
    ("building z-score + stuck-value", "building", True),
    ("hourly z-score + stuck-value", "hourly", True),
]


def precompute(store, truth, building_ids):
    """One pass: for every reading keep (key, |z_building|, |z_hourly|, is_sentinel, label).

    Readings come from skip-list range queries (one whole-history query per
    building, via the detector), exactly as the real detector would see them.
    """
    helper = AnomalyDetector(store)          # used only for history()/baseline()
    table = []
    for b in building_ids:
        baseline = helper.baseline(b)
        for epoch, value in helper.history(b):
            key = (b, epoch)
            table.append((key,
                          abs(signed_z(baseline, "building", epoch, value)),
                          abs(signed_z(baseline, "hourly", epoch, value)),
                          value == SENTINEL,
                          key in truth))
    return table


def predicted_from(table, method, use_sentinel, k):
    """Set of flagged keys for one configuration and threshold."""
    column = {"building": 1, "hourly": 2}.get(method)
    flagged = set()
    for row in table:
        hit = (column is not None and row[column] > k) or (use_sentinel and row[3])
        if hit:
            flagged.add(row[0])
    return flagged


def main(args):
    t0 = time.perf_counter()
    store = TimeSeriesStore.from_csv(args.store_csv, seed=args.seed)
    truth = load_ground_truth(args.truth_csv)
    print(f"Loaded {len(store):,} readings, {len(truth):,} labelled anomalies "
          f"({len(truth) / len(store):.2%}) in {time.perf_counter() - t0:.1f}s")

    tune_ids, test_ids = split_buildings(store.buildings(), seed=args.seed)
    print(f"Buildings: {len(tune_ids)} for tuning k, {len(test_ids)} held out for testing")

    tune_table = precompute(store, truth, tune_ids)
    test_table = precompute(store, truth, test_ids)
    tune_truth = restrict(truth, tune_ids)
    test_truth = restrict(truth, test_ids)
    test_runs = anomaly_runs(test_truth)
    print(f"Test half: {len(test_truth):,} anomalous readings in {len(test_runs)} runs\n")

    sweep_rows, result_rows = [], []
    for label, method, use_sentinel in CONFIGS:
        grid = K_GRID if method is not None else [None]
        best_k, best_f1 = None, -1.0
        for k in grid:
            s = score(predicted_from(tune_table, method, use_sentinel,
                                     k if k is not None else float("inf")), tune_truth)
            sweep_rows.append({"config": label, "k": k, "tune_precision": round(s.precision, 4),
                               "tune_recall": round(s.recall, 4), "tune_f1": round(s.f1, 4)})
            if s.f1 > best_f1:
                best_k, best_f1 = k, s.f1

        predicted = predicted_from(test_table, method, use_sentinel,
                                   best_k if best_k is not None else float("inf"))
        # Consistency gate: the real AnomalyDetector must flag exactly the same
        # readings as this fast evaluation path.
        detector = AnomalyDetector(store, method=method, k=best_k or 1.0,
                                   sentinel_value=SENTINEL if use_sentinel else None)
        if detector.flagged_keys(test_ids) != predicted:
            raise AssertionError(f"detector and evaluation disagree for {label!r}")

        s = score(predicted, test_truth)
        row = {"config": label, "k": best_k, "tune_f1": round(best_f1, 4),
               "test_precision": round(s.precision, 4), "test_recall": round(s.recall, 4),
               "test_f1": round(s.f1, 4),
               "test_event_recall": round(event_recall(predicted, test_runs), 4),
               "flagged": s.flagged, "tp": s.tp, "fp": s.fp, "fn": s.fn}
        result_rows.append(row)

    header = (f"{'configuration':<34}{'k':>5}{'precision':>11}{'recall':>9}"
              f"{'F1':>8}{'event rec.':>12}{'flagged':>9}")
    print(header + "\n" + "-" * len(header))
    for r in result_rows:
        k = "-" if r["k"] is None else f"{r['k']:g}"
        print(f"{r['config']:<34}{k:>5}{r['test_precision']:>11.3f}{r['test_recall']:>9.3f}"
              f"{r['test_f1']:>8.3f}{r['test_event_recall']:>12.3f}{r['flagged']:>9}")

    os.makedirs(args.out_dir, exist_ok=True)
    for name, rows in (("anomaly_results.csv", result_rows),
                       ("anomaly_tuning_sweep.csv", sweep_rows)):
        with open(os.path.join(args.out_dir, name), "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nSaved anomaly_results.csv and anomaly_tuning_sweep.csv in {args.out_dir}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    data_dir = os.path.join(here, "data")
    ap.add_argument("--store-csv", default=os.path.join(data_dir, "store_data.csv"))
    ap.add_argument("--truth-csv", default=os.path.join(data_dir, "ground_truth.csv"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=here)
    main(ap.parse_args())
