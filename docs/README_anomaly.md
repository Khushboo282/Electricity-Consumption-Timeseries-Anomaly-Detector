# Anomaly Detection (Part 2)

Flags unusual readings **without machine learning**, using only descriptive
statistics (mean, variance/standard deviation) and a threshold, on top of the
skip-list store from Part 1. Every reading the detector looks at comes from a
skip-list range query (`store.query_raw`), as the problem statement requires.

## Files

| File | Purpose |
|---|---|
| `05_anomaly.py` | `AnomalyDetector`, baseline statistics, z-score maths, the `Anomaly` result type. |
| `06_evaluation.py` | Loads ground truth by key; precision / recall / F1; event recall; tune/test building split. |
| `07_run_anomaly_eval.py` | Runs the whole evaluation and writes `anomaly_results.csv` and `anomaly_tuning_sweep.csv`, and prints the 10-split robustness table. |
| `06_test_anomaly.py` | 25 tests: planted anomalies, edge cases, the gap-edge and high-recall rules, metric maths, real-data checks. |

## How it works

1. **Baseline.** One range query fetches a building's whole history; we compute
   its mean and standard deviation (sample std, `n - 1`). With `method="hourly"`
   we also compute them per hour of day, because electricity use follows a daily
   cycle (a 3 a.m. reading should be compared with other 3 a.m. readings).
2. **Score.** `z = (value - mean) / std`. Positive = above normal, negative = below.
3. **Flag.** `|z| > k` -> **spike** (z > 0) or **drop** (z < 0).
4. **Stuck-value rule (dataset-specific).** A reading of exactly `1.0` is
   flagged as `stuck-value`. In this data 1,193 of the 1,195 readings equal to
   1.0 are labelled anomalies, so it behaves like a placeholder. This is **not
   statistics**; it is a rule we found by inspecting the data and labels. Report
   it separately and say so. Switch off with `sentinel_value=None`.
5. **Gap-edge rule (dataset-specific).** The data is hourly. When a building has
   a *short* hole in its data (2 to `max_gap_hours` = 6 missing hours), the
   readings on **both sides** of the hole are flagged as `gap-edge`: readings
   around a meter outage are usually unreliable. Long holes (weeks) are just
   the breaks between the data blocks and are ignored. The previous / next
   reading is found by **binary search** in the building's sorted timestamps
   (cached from the baseline range query), so it works even when the neighbour
   lies outside the requested window. Like the stuck-value rule, it was found
   by inspecting the data with its labels. Switch off with `max_gap_hours=None`.
6. **High-recall mode (off in the class, on by default in the app).** With `neighbour_hours=1`,
   every reading within 1 hour of a reading flagged by rules 3-5 is also
   flagged, as `neighbour`. Anomalies come in runs of consecutive hours and the
   rules often catch only part of a run, so this finds more real anomalies, at
   the cost of more false alarms.

Priority when a reading matches several rules: stuck-value, then spike/drop,
then gap-edge, then neighbour.

Why **per building**: building scales differ by orders of magnitude, so one
global mean/std would flag the large buildings constantly. Why **not raw
min/max**: the maximum of a dataset can never be exceeded by that same dataset,
so a fence built from it flags nothing; the standard deviation supplies the
usable threshold.

Edge behaviour: a building (or hour) with zero spread cannot be judged and gets
`z = 0`; an hour with fewer than 10 readings falls back to the building-level
statistics; unknown building, empty window, or `start > end` return `[]`.

```python
from importlib import import_module   # file names start with a digit
_timeseries_store = import_module("02_timeseries_store")
TimeSeriesStore = _timeseries_store.TimeSeriesStore
_anomaly = import_module("05_anomaly")
AnomalyDetector = _anomaly.AnomalyDetector

store = TimeSeriesStore.from_csv("store_data.csv", seed=42)
det = AnomalyDetector(store, method="hourly", k=5.0)      # sentinel_value=1.0, max_gap_hours=6 by default
for a in det.detect(107, "2016-01-01", "2016-01-05 23:00"):   # range query for the window
    print(a.time, a.value, a.kind, a.reason)

# best F1 setting, and the high-recall variant
best = AnomalyDetector(store, method=None)                      # stuck-value + gap-edge
high_recall = AnomalyDetector(store, method=None, neighbour_hours=1)
```

## Evaluation protocol

* Plain accuracy is useless (only 1.2 % of readings are anomalies; "never flag
  anything" scores 98.8 %), so we report **precision, recall, F1** and
  **event recall** (share of anomaly *runs* where at least one reading was flagged).
* Labels are matched to readings **by key** `(building, time)`, never by row position.
* The detectors never see labels. Labels are used only to choose `k` and to score.
* The 192 buildings are split into two disjoint halves (fixed seed). `k` (grid
  2 ... 10) and the gap limit (grid 6, 12, 24, 48 hours) are chosen by best F1 on
  the **tune** half; the reported numbers are on the **test** half.
* The high-recall mode is not chosen by F1 (F1 would always switch it off); it
  is reported as a separate operating point with `neighbour_hours=1`.
* The sweep is saved in `anomaly_tuning_sweep.csv`.
* **Robustness:** the whole tune-then-test procedure is repeated over 10 random
  splits (seeds 0-9, option `--splits`), printed at the end of the run.
* A consistency gate checks that the real `AnomalyDetector` flags exactly the same
  readings as the fast evaluation path.

## Results

Default split (seed 42): 96 test buildings, 1,540 anomalous readings in 385 runs.

| Configuration | k | Gap (h) | Precision | Recall | F1 | Event recall |
|---|---|---|---|---|---|---|
| building z-score | 2.5 | - | 0.270 | 0.210 | 0.236 | 0.242 |
| hourly z-score | 3 | - | 0.349 | 0.226 | 0.274 | 0.239 |
| stuck-value rule only (== 1.0) | - | - | 1.000 | 0.299 | 0.461 | 0.096 |
| building z-score + stuck-value | 5 | - | 0.922 | 0.309 | 0.463 | 0.127 |
| hourly z-score + stuck-value | 5 | - | 0.927 | 0.314 | 0.469 | 0.135 |
| **stuck-value + gap-edge** | - | 6 | **0.896** | **0.437** | **0.588** | **0.616** |
| hourly z-score + stuck-value + gap-edge | 5 | 6 | 0.855 | 0.445 | 0.585 | 0.629 |
| high-recall mode (+ 1 h neighbours) | - | 6 | 0.702 | 0.498 | 0.583 | 0.616 |

**These numbers depend on which buildings land in the test half.** Over 10
random splits (seeds 0-9), test scores, mean [min - max]:

| Configuration | Precision | Recall | F1 | Event recall |
|---|---|---|---|---|
| building z-score | 0.25 [0.16-0.31] | 0.25 [0.18-0.33] | 0.25 [0.17-0.32] | 0.25 [0.23-0.28] |
| hourly z-score | 0.32 [0.21-0.48] | 0.30 [0.22-0.37] | 0.30 [0.24-0.36] | 0.28 [0.22-0.33] |
| stuck-value rule only | 1.00 [1.00-1.00] | 0.36 [0.28-0.42] | 0.53 [0.43-0.59] | 0.12 [0.09-0.15] |
| building z-score + stuck-value | 0.95 [0.85-0.99] | 0.38 [0.29-0.44] | 0.54 [0.45-0.60] | 0.16 [0.12-0.20] |
| hourly z-score + stuck-value | 0.90 [0.80-0.95] | 0.39 [0.31-0.46] | 0.54 [0.46-0.60] | 0.18 [0.15-0.24] |
| **stuck-value + gap-edge** | **0.81 [0.68-0.91]** | **0.52 [0.41-0.58]** | **0.63 [0.56-0.71]** | **0.66 [0.61-0.72]** |
| hourly z-score + stuck-value + gap-edge | 0.79 [0.66-0.89] | 0.52 [0.41-0.60] | 0.63 [0.56-0.71] | 0.67 [0.61-0.74] |
| high-recall mode | 0.62 [0.48-0.72] | 0.58 [0.47-0.65] | 0.59 [0.54-0.68] | 0.66 [0.61-0.72] |

The default split (seed 42) is one of the less favourable ones for the
gap-edge rules. The gap-edge rules beat the previous best (hourly z-score +
stuck-value) on every one of the 10 splits. Report the mean and range, e.g.
"F1 0.63 (0.56 to 0.71 over 10 splits)", rather than a single number.

## What the results say (for the report)

* **Z-scores alone are weak here.** Precision is only 0.2-0.5: most flags are
  false alarms: plenty of readings the labels call normal are still far from a
  simple mean/std. Raising `k` improves precision but recall collapses (see the
  sweep).
* **The stuck-value rule is very precise (about 1.0) but limited:** it finds
  30-42 % of anomalous readings, and only 10-15 % of anomaly *runs*, because it
  catches the long flat outages but not the short ones.
* **The gap-edge rule is the biggest single improvement:** recall rises from
  0.39 to 0.52 and event recall from 0.18 to 0.66 (10-split means), because
  it catches short outages that the stuck-value rule misses. Precision drops
  from 0.90 to 0.81.
* **Once both rules are on, the z-score adds almost nothing** (F1 0.63 with or
  without it).
* **High-recall mode** trades precision (0.62) for recall (0.58). Choose it
  when a missed anomaly costs more than checking a false alarm. For
  electricity readings that is usually the case (a missed meter fault means
  wrong bills, wasted energy or failing equipment; a false alarm costs a few
  minutes of checking), so **the app uses it by default**. Pushing further
  (2 h neighbours: recall 0.60, precision 0.49; z > 2.5 + 1 h neighbours:
  recall 0.70, precision 0.25) adds more false alarms than real finds.
  Report the F1-best setting as the main accuracy result and high-recall
  mode as the operating choice.
* **Hourly vs building baseline:** the hourly baseline is a little better
  (F1 0.274 vs 0.236 for z-scores alone) but the difference is small and
  within the split-to-split variation; do not over-claim it. Its clear benefit
  appears on data with a strong daily cycle (see the planted-anomaly test).
* **Ceiling.** About 62 % of the labelled anomalies are NOT at the 1.0 value.
  Of the ones still missed after both rules, about 94 % are within 3 standard
  deviations of that building's usual reading for that hour, and some labels
  look arbitrary. Value-only statistics cannot separate those from normal
  readings, which caps recall. State this as a limitation, not a failure.
* **What else we tried and rejected** (all scored lower F1 on the test half):
  robust median/MAD z-scores, rolling 7-day baselines, hour-to-hour change
  scores, flat-line runs of repeated values, sudden 2x / 0.5x jumps, and a
  cross-building rule (many buildings unusual at the same hour). Each added
  more false alarms than real anomalies.

## How we know it works

* **Planted anomalies:** synthetic buildings (one tiny, one huge) with a daily
  cycle and known spikes/drops; the hourly detector finds every one with the
  right `kind`, and raises almost no false alarms. A second test shows the
  building-level z-score misses a spike hidden at a low-usage hour that the
  hourly one catches.
* **Maths:** mean/std checked against NumPy; z-score sign and zero-spread guard;
  precision/recall/F1 and run/event recall checked against hand-computed examples.
* **Real data:** every labelled anomaly exists in the store (labels line up);
  the stuck-value rule flags exactly the 1,195 readings equal to 1.0 with
  precision above 0.95; adding z-scores to the rule can never lower recall;
  a larger `k` never flags more readings.
* **New rules:** the gap-edge rule flags both sides of a 6-hour hole but not a
  500-hour block break, respects `max_gap_hours`, and still works when the
  neighbour is outside the window; high-recall mode flags exactly the readings
  within N hours of a rule flag, including across the window edge.
* **Consistency gate:** for every configuration, the real `AnomalyDetector`
  flags exactly the same readings as the fast evaluation path.
* **Mutation checks:** we deliberately broke the code four ways (spike/drop
  swapped, population instead of sample std, wrong hour of day, inverted
  threshold). The suite failed each time. (One test was initially too weak and
  let the std mutation through; it was fixed.)

```bash
STORE_DATA_CSV=store_data.csv GROUND_TRUTH_CSV=ground_truth.csv python -m pytest -v
python 07_run_anomaly_eval.py --store-csv store_data.csv --truth-csv ground_truth.csv
```

## Limitations

* The baseline uses the building's whole history, including the readings being
  judged (standard for unsupervised offline detection, but not a live/streaming
  setup). It uses no labels.
* The stuck-value and gap-edge rules are dataset-specific and were found by
  inspecting the data with its labels, so they are not blind results.
* Time is naive hourly data in 5-day blocks; a reading cannot be judged against
  its immediate neighbours across a data gap, so we deliberately use a
  whole-history baseline instead of a rolling window (a 24-hour rolling z-score
  scored F1 about 0.04 in our early experiments because of the daily cycle).
