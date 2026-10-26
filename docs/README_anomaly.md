# Anomaly Detection (Part 2)

Flags unusual readings **without machine learning**, using only descriptive
statistics (mean, variance/standard deviation) and a threshold, on top of the
skip-list store from Part 1. Every reading the detector looks at comes from a
skip-list range query (`store.query_raw`), as the problem statement requires.

## Files

| File | Purpose |
|---|---|
| `anomaly.py` | `AnomalyDetector`, baseline statistics, z-score maths, the `Anomaly` result type. |
| `evaluation.py` | Loads ground truth by key; precision / recall / F1; event recall; tune/test building split. |
| `run_anomaly_eval.py` | Runs the whole evaluation and writes `anomaly_results.csv` and `anomaly_tuning_sweep.csv`. |
| `test_anomaly.py` | 23 tests: planted anomalies, edge cases, metric maths, real-data checks. |

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

Why **per building**: building scales differ by orders of magnitude, so one
global mean/std would flag the large buildings constantly. Why **not raw
min/max**: the maximum of a dataset can never be exceeded by that same dataset,
so a fence built from it flags nothing; the standard deviation supplies the
usable threshold.

Edge behaviour: a building (or hour) with zero spread cannot be judged and gets
`z = 0`; an hour with fewer than 10 readings falls back to the building-level
statistics; unknown building, empty window, or `start > end` return `[]`.

```python
from timeseries_store import TimeSeriesStore
from anomaly import AnomalyDetector

store = TimeSeriesStore.from_csv("store_data.csv", seed=42)
det = AnomalyDetector(store, method="hourly", k=5.0)      # sentinel_value=1.0 by default
for a in det.detect(107, "2016-01-01", "2016-01-05 23:00"):   # range query for the window
    print(a.time, a.value, a.kind, a.reason)
```

## Evaluation protocol

* Plain accuracy is useless (only 1.2 % of readings are anomalies; "never flag
  anything" scores 98.8 %), so we report **precision, recall, F1** and
  **event recall** (share of anomaly *runs* where at least one reading was flagged).
* Labels are matched to readings **by key** `(building, time)`, never by row position.
* The detectors never see labels. Labels are used only to choose `k` and to score.
* The 192 buildings are split into two disjoint halves (fixed seed). `k` is chosen
  by best F1 on the **tune** half; the reported numbers are on the **test** half.
* The sweep over `k` is saved in `anomaly_tuning_sweep.csv`.
* A consistency gate checks that the real `AnomalyDetector` flags exactly the same
  readings as the fast evaluation path.

## Results

Default split (seed 42): 96 test buildings, 1,540 anomalous readings in 385 runs.

| Configuration | k | Precision | Recall | F1 | Event recall |
|---|---|---|---|---|---|
| building z-score | 2.5 | 0.270 | 0.210 | 0.236 | 0.242 |
| hourly z-score | 3 | 0.349 | 0.226 | 0.274 | 0.239 |
| stuck-value rule only (== 1.0) | - | 1.000 | 0.299 | 0.461 | 0.096 |
| building z-score + stuck-value | 5 | 0.922 | 0.309 | 0.463 | 0.127 |
| hourly z-score + stuck-value | 5 | 0.927 | 0.314 | 0.469 | 0.135 |

**These numbers depend on which buildings land in the test half.** Re-running
with six different split seeds (1-5 and 42) gave:

| Configuration | F1 range over six splits |
|---|---|
| hourly z-score | 0.24 - 0.35 |
| stuck-value rule only | 0.46 - 0.59 |
| hourly z-score + stuck-value | 0.47 - 0.60 |

The default split (seed 42) is the least favourable of the six. Report a range,
or all splits, rather than a single number.

## What the results say (for the report)

* **Z-scores alone are weak here.** Precision is only 0.2-0.5: most flags are
  false alarms: plenty of readings the labels call normal are still far from a
  simple mean/std. Raising `k` improves precision but recall collapses (see the
  sweep).
* **The stuck-value rule is very precise (about 1.0) but limited:** it finds
  30-42 % of anomalous readings, and only 10-15 % of anomaly *runs*, because it
  catches the long flat outages but not the short ones.
* **Hourly vs building baseline:** the hourly baseline is a little better
  (F1 0.274 vs 0.236 for z-scores alone) but the difference is small and
  within the split-to-split variation; do not over-claim it. Its clear benefit
  appears on data with a strong daily cycle (see the planted-anomaly test).
* **Ceiling.** About 62 % of the labelled anomalies are NOT at the 1.0 value,
  and in our early check their typical value sat close to that building's usual
  reading for that hour. Value-only statistics cannot separate those from normal
  readings, which caps recall. State this as a limitation, not a failure.

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
* **Mutation checks:** we deliberately broke the code four ways (spike/drop
  swapped, population instead of sample std, wrong hour of day, inverted
  threshold). The suite failed each time. (One test was initially too weak and
  let the std mutation through; it was fixed.)

```bash
STORE_DATA_CSV=store_data.csv GROUND_TRUTH_CSV=ground_truth.csv python -m pytest -v
python run_anomaly_eval.py --store-csv store_data.csv --truth-csv ground_truth.csv
```

## Limitations

* The baseline uses the building's whole history, including the readings being
  judged (standard for unsupervised offline detection, but not a live/streaming
  setup). It uses no labels.
* The stuck-value rule is dataset-specific and was found by inspecting the data
  with its labels, so it is not a blind result.
* Time is naive hourly data in 5-day blocks; a reading cannot be judged against
  its immediate neighbours across a data gap, so we deliberately use a
  whole-history baseline instead of a rolling window (a 24-hour rolling z-score
  scored F1 about 0.04 in our early experiments because of the daily cycle).
