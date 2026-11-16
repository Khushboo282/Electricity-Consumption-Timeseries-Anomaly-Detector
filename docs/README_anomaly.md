# Anomaly Detection

Flags unusual electricity readings **without machine learning**, using only averages, standard
deviations and a few simple rules. Every reading the detector looks at comes from a skip list
range query.

## Files

| File | What it does |
|---|---|
| `05_anomaly.py` | The detector (`AnomalyDetector`) and the statistics behind it. |
| `06_evaluation.py` | Loads the labels; computes precision, recall, F1; splits buildings into tune/test halves. |
| `07_run_anomaly_eval.py` | Runs the full evaluation; writes `anomaly_results.csv` and `anomaly_tuning_sweep.csv`. |
| `06_test_anomaly.py` | 25 tests. |

## How it works

**Step 1: learn what "normal" is.** One range query fetches a building's whole history. From it
we compute:

- the **mean** (average reading), and
- the **standard deviation**, or **std** (how much readings usually vary around the mean).

With `method="hourly"` we compute these **separately for each hour of the day**, because
electricity use follows a daily cycle: a 3 a.m. reading should be compared with other 3 a.m.
readings, not with the daily average.

**Step 2: score each reading.** The **z-score** says how many standard deviations a reading is
from normal:

```
z = (reading - mean) / std
```

Example: building 159 at 14:00 normally averages 13.07 with std 6.26. A reading of 48.73 gives
`z = (48.73 - 13.07) / 6.26 = 5.70`.

**Step 3: flag it.** A reading is flagged if any of these rules applies:

| Rule | Flags a reading when | Shown as |
|---|---|---|
| z-score | `z > k` or `z < -k` | **spike** (above) / **drop** (below) |
| Stuck value | it is exactly `1.0` | **stuck-value** |
| Gap edge | it sits right before or after a hole of 2 to 6 missing hours | **gap-edge** |
| High-recall mode (optional) | it is within 1 hour of a reading flagged above | **neighbour** |

With `k = 5` the example above (z = 5.70) is flagged as a spike.

### About the two dataset rules

- **Stuck value.** 1,195 readings are exactly `1.0`, and 1,193 of them are labelled anomalies.
  So `1.0` works like a "meter broken" placeholder. Turn it off with `sentinel_value=None`.
- **Gap edge.** The data is hourly. When a building is missing a few hours, the readings on
  both sides of that hole are usually bad too (for example, a meter outage). Long gaps of weeks
  are just the breaks between data blocks and are ignored. To find a reading's previous and next
  neighbour quickly, we use binary search on the building's sorted times. Turn it off with
  `max_gap_hours=None`.

These two rules are **not statistics**. We found them by looking at the data together with its
labels, so always present them as dataset-specific rules.

### Why these choices

- **Per building:** buildings differ hugely in size (building 107 averages about 395, building
  159 about 10). One shared average would flag the big buildings all the time.
- **Why not just min / max:** a dataset's own maximum can never be exceeded by that same data,
  so a fence built from it flags nothing. The standard deviation gives a usable threshold.
- **Edge cases:** if a building (or hour) never varies (std = 0), we can't judge it, so z = 0.
  An hour with fewer than 10 readings uses the building's overall statistics instead.

## Using the detector

```python
from importlib import import_module   # file names start with a digit
TimeSeriesStore = import_module("02_timeseries_store").TimeSeriesStore
AnomalyDetector = import_module("05_anomaly").AnomalyDetector

store = TimeSeriesStore.from_csv("data/store_data.csv", seed=42)

det = AnomalyDetector(store, method="hourly", k=5.0)   # stuck-value and gap-edge rules are on by default
for a in det.detect(107, "2016-01-01", "2016-01-05 23:00"):
    print(a.time, a.value, a.kind, a.reason)

best = AnomalyDetector(store, method=None)                       # best F1: the two rules only
high_recall = AnomalyDetector(store, method=None, neighbour_hours=1)
```

| Method | Returns |
|---|---|
| `detect(building, start, end)` | the anomalies in a time window |
| `detect_history(building)` | the anomalies over the building's whole history |
| `flagged_keys(building_ids)` | the set of flagged `(building, time)` keys, used for evaluation |

## How accuracy is measured

Only 1.2 % of readings are anomalies, so "never flag anything" would already be 98.8 %
accurate. Plain accuracy is useless here, so we use:

- **Precision**: of the readings we flagged, the share that were real anomalies.
- **Recall**: of the real anomalies, the share we found.
- **F1**: combines both into one number (it is high only when both are high).
- **Event recall**: anomalies come in runs of consecutive hours. This is the share of runs
  where we flagged at least one reading.

Small example: we flag 10 readings, 7 are real, and there are 20 real anomalies in total.
Precision = 7/10 = 0.70, recall = 7/20 = 0.35, F1 = 2 × 0.70 × 0.35 / (0.70 + 0.35) = 0.47.

**To keep the numbers honest:**

1. The detector never sees the labels. Labels are only used to pick `k` and to score.
2. The 192 buildings are split into two halves. `k` (tried: 2 to 10) and the gap limit (tried:
   6, 12, 24, 48 hours) are chosen on the **tune** half. Scores are reported on the **test**
   half, which the choice never saw.
3. All of this is repeated for 10 different random splits, so one lucky split can't fool us.
4. A final check confirms that the real `AnomalyDetector` flags exactly the same readings as
   the faster code used during the evaluation.

```bash
python 07_run_anomaly_eval.py      # about 1 minute
```

## Results

### Main result: averaged over 10 random splits

Test scores, average [lowest - highest]:

| Method | Precision | Recall | F1 | Event recall |
|---|---|---|---|---|
| building z-score | 0.25 [0.16-0.31] | 0.25 [0.18-0.33] | 0.25 [0.17-0.32] | 0.25 [0.23-0.28] |
| hourly z-score | 0.32 [0.21-0.48] | 0.30 [0.22-0.37] | 0.30 [0.24-0.36] | 0.28 [0.22-0.33] |
| stuck-value rule only | 1.00 [1.00-1.00] | 0.36 [0.28-0.42] | 0.53 [0.43-0.59] | 0.12 [0.09-0.15] |
| building z-score + stuck-value | 0.95 [0.85-0.99] | 0.38 [0.29-0.44] | 0.54 [0.45-0.60] | 0.16 [0.12-0.20] |
| hourly z-score + stuck-value | 0.90 [0.80-0.95] | 0.39 [0.31-0.46] | 0.54 [0.46-0.60] | 0.18 [0.15-0.24] |
| **stuck-value + gap-edge** | **0.81 [0.68-0.91]** | **0.52 [0.41-0.58]** | **0.63 [0.56-0.71]** | **0.66 [0.61-0.72]** |
| hourly z-score + stuck-value + gap-edge | 0.79 [0.66-0.89] | 0.52 [0.41-0.60] | 0.63 [0.56-0.71] | 0.67 [0.61-0.74] |
| high-recall mode | 0.62 [0.48-0.72] | 0.58 [0.47-0.65] | 0.59 [0.54-0.68] | 0.66 [0.61-0.72] |

In the report, quote the average and range, for example "F1 0.63 (0.56 to 0.71 over 10 splits)".

### One split in detail (seed 42, saved in `anomaly_results.csv`)

96 test buildings, 1,540 anomalous readings in 385 runs. This split is one of the less
favourable ones for the gap-edge rule.

| Method | k | Gap (h) | Precision | Recall | F1 | Event recall |
|---|---|---|---|---|---|---|
| building z-score | 2.5 | - | 0.270 | 0.210 | 0.236 | 0.242 |
| hourly z-score | 3 | - | 0.349 | 0.226 | 0.274 | 0.239 |
| stuck-value rule only | - | - | 1.000 | 0.299 | 0.461 | 0.096 |
| building z-score + stuck-value | 5 | - | 0.922 | 0.309 | 0.463 | 0.127 |
| hourly z-score + stuck-value | 5 | - | 0.927 | 0.314 | 0.469 | 0.135 |
| **stuck-value + gap-edge** | - | 6 | **0.896** | **0.437** | **0.588** | **0.616** |
| hourly z-score + stuck-value + gap-edge | 5 | 6 | 0.855 | 0.445 | 0.585 | 0.629 |
| high-recall mode | - | 6 | 0.702 | 0.498 | 0.583 | 0.616 |

### What the results mean

- **The z-score alone is weak.** Precision is only about 0.25 to 0.32: many perfectly normal
  readings are still far from a simple average.
- **The stuck-value rule is almost always right, but finds only about a third** of anomalous
  readings. It catches long flat outages, not short events.
- **The gap-edge rule is the biggest improvement.** Recall goes from 0.39 to 0.52, and runs
  caught from 0.18 to 0.66, while precision drops only slightly (0.90 to 0.81). It beat the
  previous best method on every one of the 10 splits.
- **Once both rules are on, the z-score adds almost nothing** (F1 is 0.63 either way).
- **Hourly vs building baseline:** hourly is slightly better, but the difference is within the
  normal split-to-split variation, so don't over-claim it.
- **There is a ceiling.** Of the anomalies still missed, about 94 % look normal by value (within
  3 standard deviations), and some labels look arbitrary. No value-based method can find those.
  This is a limit of the data, not a bug.
- **Things we tried that did worse:** median-based z-scores, rolling 7-day averages,
  hour-to-hour change, repeated values, sudden 2x jumps, and "many buildings unusual at the same
  hour". Each added more false alarms than real finds.

### Why the app uses high-recall mode

For electricity meters, missing a real problem usually costs more than a false alarm:

- **Missed anomaly:** a broken meter gives wrong bills, and faulty equipment keeps wasting
  energy, until someone notices.
- **False alarm:** someone looks at a chart for a minute and decides it's fine.

High-recall mode finds more real anomalies (recall 0.58 vs 0.52) at the cost of more false
alarms (precision 0.62 vs 0.81). Going further (2-hour neighbours, or lower `k`) adds more false
alarms than real finds. In the report, give the F1-best method as the main result and present
high-recall mode as the practical choice.

## How we know it works

- **Planted anomalies:** fake buildings with spikes and drops at known places. The detector
  finds all of them, labels each correctly, and raises almost no false alarms.
- **Why hourly exists:** a test shows a spike at a quiet hour that the building-level method
  misses but the hourly method catches.
- **Maths:** mean and std checked against NumPy; precision, recall and F1 checked against
  hand-worked examples.
- **Real data:** labels line up with readings; the stuck-value rule flags exactly the 1,195
  readings equal to 1.0; a larger `k` never flags more readings.
- **Gap-edge and high-recall rules:** tested on small hand-made examples, including neighbours
  just outside the requested window.
- **Deliberate bugs:** we broke the code on purpose four ways. The tests failed every time.

```bash
python -m pytest 06_test_anomaly.py -v
```

## Limitations

- "Normal" is computed from the building's whole history, including the readings being judged.
  That's fine for analysing past data, but not for live monitoring.
- The stuck-value and gap-edge rules only fit this dataset, and we found them by looking at the
  labels.
- We use the whole history rather than a rolling window. Because the data comes in 5-day blocks
  with a strong daily cycle, a 24-hour rolling z-score scored F1 of only about 0.04 in early tests.
