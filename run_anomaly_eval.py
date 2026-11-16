"""
run_anomaly_eval.py -- Measure anomaly-detection accuracy against ground truth.

Run:
    python run_anomaly_eval.py --store-csv store_data.csv --truth-csv ground_truth.csv

PROTOCOL (so the reported numbers can be trusted)
  1. Build the skip-list store from the readings; load the labels by KEY.
  2. Split the 192 buildings into two disjoint halves with a fixed seed:
     a TUNE half and a TEST half.
  3. For each detector configuration, choose the threshold k (and the gap
     limit of the gap-edge rule) that gives the best F1 on the TUNE half only.
  4. Report precision / recall / F1 / event recall on the TEST half, which the
     choice of k never saw.
  5. Robustness: repeat steps 2-4 for ``--splits`` different random splits and
     report the mean / min / max of the test scores, so the result does not
     hinge on one lucky (or unlucky) choice of test buildings.
  The detectors are unsupervised: they never look at labels. Labels are used
  only to choose k and to score the result.

  One honest caveat: the stuck-value rule (reading == 1.0) and the gap-edge
  rule (reading next to a short hole in the data) were found by inspecting
  the data together with the labels, so they are not blind results.

  The HIGH-RECALL configuration is not chosen by F1: it deliberately trades
  precision for recall by also flagging the readings within one hour of a
  flagged reading (neighbour_hours=1).

OUTPUT (in --out-dir)
  anomaly_results.csv            final table (test half of the main split)
  anomaly_tuning_sweep.csv       precision/recall/F1 for every k on the tune half
  anomaly_split_robustness.csv   mean/min/max test scores over --splits splits
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import time

from anomaly import AnomalyDetector, signed_z
from evaluation import (anomaly_runs, event_recall, load_ground_truth, restrict,
                        score, split_buildings)
from timeseries_store import TimeSeriesStore

K_GRID = [2, 2.5, 3, 3.5, 4, 5, 6, 8, 10]
GAP_GRID = [6, 12, 24, 48]      # candidate max_gap_hours for the gap-edge rule
SENTINEL = 1.0
INF = float("inf")
HOUR = 3600

# (label, z-score method or None, stuck-value rule?, gap-edge rule?, neighbour_hours)
CONFIGS = [
    ("building z-score", "building", False, False, 0),
    ("hourly z-score", "hourly", False, False, 0),
    ("stuck-value rule only (==1.0)", None, True, False, 0),
    ("building z-score + stuck-value", "building", True, False, 0),
    ("hourly z-score + stuck-value", "hourly", True, False, 0),
    ("stuck-value + gap-edge", None, True, True, 0),
    ("hourly z-score + stuck-value + gap-edge", "hourly", True, True, 0),
    ("HIGH-RECALL: stuck + gap-edge + 1h neighbours", None, True, True, 1),
]


def precompute(store, truth, building_ids):
    """One pass: for every reading keep
    (key, |z_building|, |z_hourly|, is_sentinel, label, gap_hours).

    gap_hours = the smallest hole (in hours, >= 2) between this reading and its
    previous / next reading, or inf if both neighbours are 1 hour away. The
    gap-edge rule with max_gap_hours=G flags the reading iff gap_hours <= G.

    Readings come from skip-list range queries (one whole-history query per
    building, via the detector), exactly as the real detector would see them.
    Returns ``{building_id: [row, ...]}``.
    """
    helper = AnomalyDetector(store)          # used only for history()/baseline()
    table = {}
    for b in building_ids:
        baseline = helper.baseline(b)
        history = helper.history(b)
        epochs = [epoch for epoch, _ in history]
        rows = []
        for i, (epoch, value) in enumerate(history):
            key = (b, epoch)
            holes = [g for g in ((epoch - epochs[i - 1]) if i > 0 else 0,
                                 (epochs[i + 1] - epoch) if i + 1 < len(epochs) else 0)
                     if g >= 2 * HOUR]
            rows.append((key,
                         abs(signed_z(baseline, "building", epoch, value)),
                         abs(signed_z(baseline, "hourly", epoch, value)),
                         value == SENTINEL,
                         key in truth,
                         min(holes) / HOUR if holes else INF))
        table[b] = rows
    return table


def predicted_from(table, building_ids, method, use_sentinel, k,
                   max_gap=None, neighbour_hours=0):
    """Set of flagged keys for one configuration on the given buildings."""
    column = {"building": 1, "hourly": 2}.get(method)
    flagged = set()
    for b in building_ids:
        rows = table[b]
        primary = [row[0][1] for row in rows
                   if (column is not None and row[column] > k) or (use_sentinel and row[3])
                   or (max_gap is not None and row[5] <= max_gap)]
        flagged.update((b, e) for e in primary)
        if neighbour_hours and primary:
            # two-pointer sweep: readings within neighbour_hours of a primary flag
            reach, j = neighbour_hours * HOUR, 0
            for (_, epoch), *_ in rows:
                while j < len(primary) and primary[j] < epoch - reach:
                    j += 1
                if j < len(primary) and primary[j] <= epoch + reach:
                    flagged.add((b, epoch))
    return flagged


def tune_and_test(table, truth, tune_ids, test_ids, config, sweep_rows=None):
    """Pick k / gap on the tune buildings, score on the test buildings."""
    label, method, use_sentinel, use_gap, neighbours = config
    tune_truth, test_truth = restrict(truth, tune_ids), restrict(truth, test_ids)
    k_grid = K_GRID if method is not None else [None]
    gap_grid = GAP_GRID if use_gap else [None]
    best_k, best_gap, best_f1 = None, None, -1.0
    for k in k_grid:
        for gap in gap_grid:
            s = score(predicted_from(table, tune_ids, method, use_sentinel,
                                     k if k is not None else INF, gap, neighbours), tune_truth)
            if sweep_rows is not None:
                sweep_rows.append({"config": label, "k": k, "max_gap_hours": gap,
                                   "tune_precision": round(s.precision, 4),
                                   "tune_recall": round(s.recall, 4), "tune_f1": round(s.f1, 4)})
            if s.f1 > best_f1:
                best_k, best_gap, best_f1 = k, gap, s.f1
    predicted = predicted_from(table, test_ids, method, use_sentinel,
                               best_k if best_k is not None else INF, best_gap, neighbours)
    return best_k, best_gap, best_f1, predicted, score(predicted, test_truth)


def main(args):
    t0 = time.perf_counter()
    store = TimeSeriesStore.from_csv(args.store_csv, seed=args.seed)
    truth = load_ground_truth(args.truth_csv)
    print(f"Loaded {len(store):,} readings, {len(truth):,} labelled anomalies "
          f"({len(truth) / len(store):.2%}) in {time.perf_counter() - t0:.1f}s")
    table = precompute(store, truth, store.buildings())

    # ---- main split ---------------------------------------------------------
    tune_ids, test_ids = split_buildings(store.buildings(), seed=args.seed)
    print(f"Buildings: {len(tune_ids)} for tuning k, {len(test_ids)} held out for testing")
    test_truth = restrict(truth, test_ids)
    test_runs = anomaly_runs(test_truth)
    print(f"Test half: {len(test_truth):,} anomalous readings in {len(test_runs)} runs\n")

    sweep_rows, result_rows = [], []
    for config in CONFIGS:
        label, method, use_sentinel, _, neighbours = config
        best_k, best_gap, best_f1, predicted, s = tune_and_test(
            table, truth, tune_ids, test_ids, config, sweep_rows)
        # Consistency gate: the real AnomalyDetector must flag exactly the same
        # readings as this fast evaluation path.
        detector = AnomalyDetector(store, method=method, k=best_k or 1.0,
                                   sentinel_value=SENTINEL if use_sentinel else None,
                                   max_gap_hours=best_gap, neighbour_hours=neighbours)
        if detector.flagged_keys(test_ids) != predicted:
            raise AssertionError(f"detector and evaluation disagree for {label!r}")
        result_rows.append({
            "config": label, "k": best_k, "max_gap_hours": best_gap,
            "neighbour_hours": neighbours, "tune_f1": round(best_f1, 4),
            "test_precision": round(s.precision, 4), "test_recall": round(s.recall, 4),
            "test_f1": round(s.f1, 4),
            "test_event_recall": round(event_recall(predicted, test_runs), 4),
            "flagged": s.flagged, "tp": s.tp, "fp": s.fp, "fn": s.fn})

    header = (f"{'configuration':<47}{'k':>5}{'gap':>5}{'precision':>11}{'recall':>9}"
              f"{'F1':>8}{'event rec.':>12}{'flagged':>9}")
    print(header + "\n" + "-" * len(header))
    for r in result_rows:
        k = "-" if r["k"] is None else f"{r['k']:g}"
        gap = "-" if r["max_gap_hours"] is None else f"{r['max_gap_hours']}"
        print(f"{r['config']:<47}{k:>5}{gap:>5}{r['test_precision']:>11.3f}{r['test_recall']:>9.3f}"
              f"{r['test_f1']:>8.3f}{r['test_event_recall']:>12.3f}{r['flagged']:>9}")

    # ---- robustness over several random splits ------------------------------
    robust_rows = []
    if args.splits > 0:
        seeds = list(range(args.splits))
        per_config = {c[0]: [] for c in CONFIGS}
        for seed in seeds:
            tune_s, test_s = split_buildings(store.buildings(), seed=seed)
            runs_s = anomaly_runs(restrict(truth, test_s))
            for config in CONFIGS:
                *_, predicted, s = tune_and_test(table, truth, tune_s, test_s, config)
                per_config[config[0]].append(
                    (s.precision, s.recall, s.f1, event_recall(predicted, runs_s)))

        print(f"\nRobustness: test scores over {len(seeds)} random tune/test splits "
              f"(seeds {seeds[0]}..{seeds[-1]}), mean [min - max]")
        header = f"{'configuration':<47}{'precision':>20}{'recall':>20}{'F1':>20}{'event rec.':>20}"
        print(header + "\n" + "-" * len(header))
        for label, scores in per_config.items():
            row = {"config": label, "splits": len(seeds)}
            cells = []
            for i, name in enumerate(("precision", "recall", "f1", "event_recall")):
                vals = [sc[i] for sc in scores]
                row.update({f"{name}_mean": round(statistics.mean(vals), 4),
                            f"{name}_std": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
                            f"{name}_min": round(min(vals), 4),
                            f"{name}_max": round(max(vals), 4)})
                cells.append(f"{statistics.mean(vals):.3f} [{min(vals):.2f}-{max(vals):.2f}]")
            robust_rows.append(row)
            print(f"{label:<47}" + "".join(f"{c:>20}" for c in cells))

    os.makedirs(args.out_dir, exist_ok=True)
    outputs = [("anomaly_results.csv", result_rows), ("anomaly_tuning_sweep.csv", sweep_rows)]
    if robust_rows:
        outputs.append(("anomaly_split_robustness.csv", robust_rows))
    for name, rows in outputs:
        with open(os.path.join(args.out_dir, name), "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nSaved {', '.join(n for n, _ in outputs)} in {args.out_dir}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    data_dir = os.path.join(here, "data")
    ap.add_argument("--store-csv", default=os.path.join(data_dir, "store_data.csv"))
    ap.add_argument("--truth-csv", default=os.path.join(data_dir, "ground_truth.csv"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits", type=int, default=10,
                    help="number of extra random splits for the robustness check (0 = skip)")
    ap.add_argument("--out-dir", default=here)
    main(ap.parse_args())
