# Time-Series Store with Anomaly Detection

A small time-series database for hourly electricity-meter readings, built for a
Data Structures & Algorithms course project.

* **Part 1: fast range queries.** A **skip list written from scratch** indexes
  every reading, so "all readings of building *B* between two times" does not
  scan the whole dataset. It is compared against a linear scan.
* **Part 2: anomaly detection.** A statistical detector (no machine learning)
  flags unusual spikes and drops, fetching its data through the same range
  queries, and is scored against labelled ground truth.
* **Part 3: user interface.** *(planned: Streamlit app in `app/`)*

## Highlights

* 262,719 readings, 192 buildings, indexed in about 2 seconds.
* A range query takes roughly **2 to 13 microseconds** on the skip list. A
  linear scan takes about **0.3 ms at 10,000 readings and 11 ms at all 262,719**,
  so the skip list is roughly **800 to 2,000 times faster at full size**
  (pure Python, same queries and rows for both; your machine will differ, the
  shape should not).
* Anomaly detection reaches **F1 of about 0.47 to 0.60** (varies with which
  buildings are held out) when the z-score test is combined with a
  dataset-specific "stuck at 1.0" rule; z-scores alone reach **0.24 to 0.35**.
  See [docs/anomaly.md](docs/anomaly.md) for the honest limits of these numbers.
* 47 automated tests, including comparison with a brute-force oracle and
  deliberate-bug ("mutation") checks.

## Setup

```bash
git clone <your-repo-url>
cd timeseries-anomaly-store
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Put `store_data.csv` and `ground_truth.csv` in `data/` (see [data/README.md](data/README.md)).

## Usage

```python
from tsstore import TimeSeriesStore, AnomalyDetector

store = TimeSeriesStore.from_csv("data/store_data.csv", seed=42)

# Part 1: range query (both ends inclusive, results in time order)
store.query(107, "2016-01-01 09:00", "2016-01-01 12:00")

# Part 2: anomalies in a window
detector = AnomalyDetector(store, method="hourly", k=5.0)
for a in detector.detect(107, "2016-01-01", "2016-01-05 23:00"):
    print(a.time, a.value, a.kind, a.reason)
```

```bash
python -m pytest                     # run all tests
python scripts/benchmark.py          # skip list vs linear scan -> results/
python scripts/run_anomaly_eval.py   # accuracy vs ground truth  -> results/
```

## Repository structure

```
timeseries-anomaly-store/
├── README.md                  this file
├── pyproject.toml             package + pytest configuration
├── requirements.txt           dependencies
├── .gitignore
├── src/tsstore/               the library
│   ├── __init__.py
│   ├── skiplist.py            skip list (insert, search, range_query), from scratch
│   ├── timeseries_store.py    CSV loading, keys, store.query(building, start, end)
│   ├── brute_force.py         linear-scan baseline and correctness oracle
│   ├── anomaly.py             z-score anomaly detector using range queries
│   └── evaluation.py          precision / recall / F1 / event recall, building split
├── scripts/
│   ├── benchmark.py           skip list vs linear scan benchmark
│   └── run_anomaly_eval.py    tune on half the buildings, test on the other half
├── tests/
│   ├── test_skiplist.py       skip list + store tests (24)
│   └── test_anomaly.py        detector + evaluation tests (23)
├── data/                      CSV inputs (git-ignored) + README
├── results/                   benchmark and accuracy outputs (CSV, PNG)
├── docs/
│   ├── skiplist.md            Part 1 design, verification, benchmark
│   └── anomaly.md             Part 2 method, protocol, results, limitations
└── app/                       (planned) Streamlit UI
```

## Design in one paragraph

Readings are keyed by `(building_id, epoch_seconds)`, so each building's data is
one contiguous, time-ordered run in the skip list and a query for one building
is a single range query that cannot leak into another building. The skip list
gives O(log n + k) range queries (k = number of results) against O(n) for a
scan. The detector computes each building's mean and standard deviation from
one range query, then flags readings more than *k* standard deviations away,
optionally per hour of day, plus a separate rule for readings stuck at `1.0`.

## Known limitations

* Everything is in memory; nothing is persisted.
* Pure-Python timings: a NumPy-vectorised scan would be much faster than our
  scan, so say which baseline you compare against.
* About 62 % of the labelled anomalies are not at the `1.0` value, and value
  statistics alone cannot separate those from normal readings, so recall is limited.
* The stuck-at-`1.0` rule is dataset-specific and was found by inspecting the data.

## References

* W. Pugh, *Skip Lists: A Probabilistic Alternative to Balanced Trees*,
  Communications of the ACM 33(6), 1990.

## Team

<!-- add names and who built which part -->
