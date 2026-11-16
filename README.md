# Electricity Consumption Time-Series Anomaly Detector

A small time-series database written from scratch in Python. It stores hourly electricity
meter readings, answers "all readings of building X between time A and time B" with a
**skip list index** instead of scanning every row, and flags unusual spikes and drops with
**plain statistics (no machine learning)**. An interactive **Streamlit** app ties it together.

Course project for DS603, Data Structure Algorithms in Python.

## Highlights

- **Skip list built from scratch** (no `bisect`, `sortedcontainers` or tree libraries): insert, search and range query in expected O(log n + k).
- **Roughly 800 to 2,000 times faster than a linear scan** on 262,719 readings: about 6 to 17 microseconds per query versus 10 to 14 milliseconds (see the benchmark below).
- **Statistical anomaly detection** per building (mean and standard deviation, optionally per hour of day), built on top of the fast range queries.
- **Honest evaluation** against labelled ground truth: precision, recall, F1, with the threshold tuned on one half of the buildings and scored on the other half.
- **Interactive app**: query any building and time window, see flagged anomalies on a chart, and compare skip list and brute force live.
- **59 automated tests**, including a comparison of every skip list answer with a brute-force scan.

## Dataset

| | |
|---|---|
| Source | [Kaggle: Energy Anomaly Detection](https://www.kaggle.com/competitions/energy-anomaly-detection/data) |
| Readings | 262,719 hourly electricity meter readings (`data/store_data.csv`) |
| Buildings | 192 |
| Period | 12 blocks of about 5 days each (one per month of 2016) |
| Labels | `data/ground_truth.csv`: 3,144 readings (1.2 %) labelled anomalous |

## How it works

```mermaid
flowchart LR
    CSV["store_data.csv"] --> DP["Data processing<br/>keys = (building, time)"]
    DP --> SL["Skip list index"]
    DP --> BF["Brute force<br/>linear scan"]
    SL --> AD["Anomaly detector"]
    GT["ground_truth.csv"] --> EV["Evaluation<br/>precision / recall / F1"]
    AD --> EV
    SL --> UI["Streamlit app"]
    BF --> UI
    AD --> UI
```

**1. Index.** Every reading is stored in a skip list under the key `(building_id, time)`, so one
building's readings sit together in time order. A skip list is a sorted linked list with random
"express lanes": a search moves right along the top lane while the next key is still too small,
then drops down a level, and repeats.

```
Level 3:  HEAD ---------------------------> 40 ------------------> None
Level 2:  HEAD ---------> 10 -------------> 40 ------> 70 -------> None
Level 1:  HEAD --> 5 ---> 10 --> 25 ------> 40 --> 55 --> 70 --> 90 -> None
```

**2. Range query.** "Building B between t1 and t2" finds the first key at or after `(B, t1)` in
about log n steps, then walks forward until it passes `(B, t2)`. Example: building 107 over 24 hours
returns 24 readings after 47 key comparisons, while the brute force reads all 262,719 rows.

**3. Anomaly detection.** Fetch a building's history with one range query, compute its mean and
standard deviation, and flag any reading whose z-score `(value - mean) / std` exceeds a threshold `k`
(spike if above, drop if below). The baseline can be per building or per hour of day, because
electricity use follows a daily cycle. A separate, dataset-specific rule also flags readings of exactly
`1.0`, which behave like a placeholder value in this data, and a second one flags the readings on
either side of a short hole (2 to 6 missing hours) in a building's data, found by binary search in
the building's sorted timestamps.

## Results

### Query speed

Skip list versus linear scan, same queries, same rows, pure Python, best of several runs:

![Benchmark: skip list vs linear scan](benchmark_plot.png)

The linear scan grows with the number of stored readings; the skip list stays almost flat.
Exact numbers are in `benchmark_results.csv`. Every skip list answer is checked against the brute
force before it is timed.

### Anomaly detection accuracy

Threshold `k` and gap limit chosen on 96 buildings, scored on the 96 held-out buildings:

| Method | k | Gap (h) | Precision | Recall | F1 | Event recall |
|---|---|---|---|---|---|---|
| building z-score | 2.5 | - | 0.270 | 0.210 | 0.236 | 0.242 |
| hourly z-score | 3 | - | 0.349 | 0.226 | 0.274 | 0.239 |
| stuck-value rule only (reading = 1.0) | - | - | 1.000 | 0.299 | 0.461 | 0.096 |
| building z-score + stuck-value rule | 5 | - | 0.922 | 0.309 | 0.463 | 0.127 |
| hourly z-score + stuck-value rule | 5 | - | 0.927 | 0.314 | 0.469 | 0.135 |
| **stuck-value + gap-edge rules** | - | 6 | **0.896** | **0.437** | **0.588** | **0.616** |
| hourly z-score + stuck-value + gap-edge | 5 | 6 | 0.855 | 0.445 | 0.585 | 0.629 |
| high-recall mode (rules + 1 h neighbours) | - | 6 | 0.702 | 0.498 | 0.583 | 0.616 |

The gap-edge rule raised recall from 0.31 to 0.44 and event recall (anomaly runs with at least one
flag) from 0.14 to 0.62, at a small cost in precision. Adding the z-score on top of the two rules
barely changes F1. High-recall mode (`neighbour_hours=1`) also flags the hour before and after
every flagged reading: it finds more real anomalies but raises more false alarms. **The app uses it
by default**, because for electricity readings a missed fault (wrong bills, wasted energy, failing
equipment) usually costs more than a few minutes spent checking a false alarm.

**Robustness.** One split can be lucky or unlucky, so the whole protocol (tune on one half, test on
the other) is repeated over 10 random splits of the buildings. Test scores, mean [min - max]:

| Method | Precision | Recall | F1 | Event recall |
|---|---|---|---|---|
| hourly z-score | 0.32 [0.21-0.48] | 0.30 [0.22-0.37] | 0.30 [0.24-0.36] | 0.28 [0.22-0.33] |
| stuck-value rule only | 1.00 [1.00-1.00] | 0.36 [0.28-0.42] | 0.53 [0.43-0.59] | 0.12 [0.09-0.15] |
| hourly z-score + stuck-value | 0.90 [0.80-0.95] | 0.39 [0.31-0.46] | 0.54 [0.46-0.60] | 0.18 [0.15-0.24] |
| **stuck-value + gap-edge** | **0.81 [0.68-0.91]** | **0.52 [0.41-0.58]** | **0.63 [0.56-0.71]** | **0.66 [0.61-0.72]** |
| high-recall mode | 0.62 [0.48-0.72] | 0.58 [0.47-0.65] | 0.59 [0.54-0.68] | 0.66 [0.61-0.72] |

`python 07_run_anomaly_eval.py` prints this table for every configuration. The split used for the first table
(seed 42) is one of the less favourable ones for the gap-edge rules. Plain accuracy is not reported
because only 1.2 % of readings are anomalies, so a detector that flags nothing would already score
98.8 %.

## Getting started

Requires Python 3.9 or newer (tested on 3.12 and 3.13).

```bash
git clone https://github.com/<your-username>/Electricity-Consumption-Timeseries-Anomaly-Detector.git
cd Electricity-Consumption-Timeseries-Anomaly-Detector
pip install -r requirements.txt
```

Make sure the two data files are in the `data/` folder:

```
data/store_data.csv
data/ground_truth.csv
```

Then run the parts in this order (the file names are numbered the same way):

```bash
python -m pytest 03_test_skiplist.py   # 3. test the skip list, store and brute force
python 04_benchmark.py                 # 4. skip list vs brute force timing
python -m pytest 06_test_anomaly.py    # 6. test the anomaly detector and metrics
python 07_run_anomaly_eval.py          # 7. anomaly detection accuracy
streamlit run 09_app.py                # 9. the interactive app
python -m pytest 09_test_ui.py         # 9. test the app and its helpers
```

`python -m pytest` runs all three test files at once. Files 01-03, 05-06 and 08 are
modules used by the others; they are not run on their own. Because the file names
start with a digit, Python code loads them with
`importlib.import_module("02_timeseries_store")` instead of a plain `import`.

### Using the store from Python

```python
from importlib import import_module   # file names start with a digit
_timeseries_store = import_module("02_timeseries_store")
TimeSeriesStore = _timeseries_store.TimeSeriesStore
_anomaly = import_module("05_anomaly")
AnomalyDetector = _anomaly.AnomalyDetector

store = TimeSeriesStore.from_csv("data/store_data.csv", seed=42)

# all readings of building 107 between two times
store.query(107, "2016-01-01 09:00", "2016-01-02 08:00")

# anomalies in a window (hourly baseline, threshold k = 3)
detector = AnomalyDetector(store, method="hourly", k=3.0)
for a in detector.detect(159, "2016-08-01", "2016-08-05 23:00"):
    print(a.time, a.value, a.kind, round(a.score, 1))
```

## The app

`streamlit run 09_app.py` opens four tabs:

1. **Explore & detect**: readings, the normal band, and flagged spikes, drops and stuck values, with the ground-truth anomalies available as an overlay.
2. **Skip list vs brute force**: the same query on both methods, with time, work done and a check that the results are identical.
3. **Accuracy**: correct flags, false alarms and misses for the current window, plus precision, recall and F1 across all buildings.
4. **About**: a short explanation of the method.

Tip: with the default threshold the detector flags few spikes and drops. Set `k = 3` and try
buildings 159 (August block), 183 or 890 to see them.

<!-- Add screenshots here once you have them, for example:
![App screenshot](docs/images/app.png) -->

## Project structure

```
.
├── README.md
├── requirements.txt
├── data/
│   ├── store_data.csv
│   └── ground_truth.csv
├── docs/
│   ├── README_team.md       detailed guide to every part
│   ├── README_data_structure.md  the skip list
│   ├── README_anomaly.md    anomaly detection and its evaluation
│   └── README_ui.md         the Streamlit app
├── pytest.ini               lets pytest find the numbered test files
├── 01_skiplist.py           the skip list index
├── 02_timeseries_store.py   data loading, keys and the query layer
├── 03_brute_force.py        linear-scan baseline and correctness oracle
├── 03_test_skiplist.py      TEST: skip list, store and brute force
├── 04_benchmark.py          RUN: skip list vs brute force timing
├── 05_anomaly.py            statistical anomaly detector
├── 06_evaluation.py         ground-truth loading and accuracy metrics
├── 06_test_anomaly.py       TEST: anomaly detector and metrics
├── 07_run_anomaly_eval.py   RUN: full accuracy evaluation
├── 08_ui_helpers.py         helper functions for the app
├── 09_app.py                RUN: Streamlit app
├── 09_test_ui.py            TEST: app and helpers
├── benchmark_results.csv    generated by 04_benchmark.py
├── benchmark_plot.png       generated by 04_benchmark.py
├── anomaly_results.csv      generated by 07_run_anomaly_eval.py
└── anomaly_tuning_sweep.csv generated by 07_run_anomaly_eval.py
```

## Testing

```bash
python -m pytest
```

The 61 tests cover the skip list (unit, edge cases, randomised comparison against a reference),
the store and brute force (every answer compared on real data, including ranges inside the data gaps),
the anomaly detector (planted spikes and drops with known positions, plus hand-computed metrics),
and the app (a headless run that clicks through the controls). Tests that need the CSV files are
skipped if `data/` is missing.

## Limitations

- Everything is in memory and in pure Python; there is no persistence to disk and no `delete` operation.
- A NumPy-based scan would be faster than our pure-Python brute force, so the speed-up figures compare against a plain Python scan.
- About 62 % of the labelled anomalies are not at the `1.0` value, and many of them look normal by value (about 94 % of the still-missed ones are within 3 standard deviations of normal). Statistics on values alone cannot separate them from normal readings, which caps recall at about 0.45.
- The `1.0` rule and the gap-edge rule are specific to this dataset and were found by inspecting the data with its labels, so they are not blind results.
- The baseline uses a building's whole history, which suits offline analysis but not live streaming.

## Documentation

See [`docs/README_team.md`](docs/README_team.md) for a walkthrough of each part (data processing,
brute force, skip list, anomaly detection, UI) with worked examples.

## Team

- `<Name>`: `<part>`
- `<Name>`: `<part>`
- `<Name>`: `<part>`

## Reference

W. Pugh, "Skip Lists: A Probabilistic Alternative to Balanced Trees", *Communications of the ACM* 33(6), 1990.
