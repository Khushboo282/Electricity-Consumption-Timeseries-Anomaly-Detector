# Skip-List Index for the Time-Series Store

Part 1 of the project (querying): a **skip list written from scratch** that
answers "all readings of building *B* between time *t1* and *t2*" without
scanning the whole dataset.

## Files

| File | Purpose |
|---|---|
| `skiplist.py` | The skip list itself (`insert`, `search`, `get`, `range_query`, `check_invariants`). Generic: works with any comparable keys. |
| `timeseries_store.py` | Storage/query layer: loads the CSV, builds the keys, exposes `query(building_id, start, end)`. |
| `brute_force.py` | Linear-scan baseline. Also the "correctness oracle" for the tests. (Teammate's version can replace it, same signature.) |
| `test_skiplist.py` | 24 tests: unit, edge cases, randomized, real-data comparison. |
| `benchmark.py` | Skip list vs linear scan; writes `benchmark_results.csv` and `benchmark_plot.png`. |

Requirements: Python 3.9+ (tested on 3.12), `pandas` (CSV parsing), `pytest` (tests), `matplotlib` (plot, optional).

## Quick start

```python
from timeseries_store import TimeSeriesStore

store = TimeSeriesStore.from_csv("store_data.csv", seed=42)   # ~2 s for 262,719 rows
store.query(107, "2016-01-01 09:00", "2016-01-01 12:00")
# -> [(datetime(2016, 1, 1, 9, 0), 1.0), (datetime(2016, 1, 1, 10, 0), 1.0),
#     (datetime(2016, 1, 1, 11, 0), 1.0), (datetime(2016, 1, 1, 12, 0), 1.0)]

store.buildings()      # [107, 108, 111, 112, 117, ...]  192 ids (for a UI dropdown)
store.span(107)        # (datetime(2016, 1, 1, 0, 0), datetime(2016, 12, 5, 14, 0))
```

`query` returns `[(datetime, meter_reading), ...]` sorted by time, both ends
inclusive. It returns an empty list for an unknown building, a range with no
data (e.g. inside one of the data gaps), or `start > end`.

## Design

**Key = `(building_id, epoch_seconds)`, value = `meter_reading`.**
The file has 192 buildings sharing the same timestamps, so time alone is not
unique. Python compares tuples element by element, so each building's readings
form one contiguous, time-ordered run inside the skip list. A query for one
building is then a single range query bounded by `(B, t1)` and `(B, t2)`; it
cannot leak into a neighbouring building.

Other choices:

* The CSV column `ts` is **ignored**: it can differ from `timestamp` by up to
  ~13 minutes. We parse `timestamp` instead.
* Times are timezone-naive; stored as integer seconds (fast to compare).
* Raw values are stored untouched: no cleaning, no gap filling, no
  normalisation. The `1.0` readings are kept because they are labelled
  anomalies in `ground_truth.csv`.
* The file is not sorted by time; that is fine, the skip list sorts as it inserts.

### How the skip list works

A skip list is a sorted linked list with extra "express lanes". Each node gets
a random tower height (each extra level with probability `p = 0.5`). To find a
key we start at the top lane, move right while the next key is smaller than the
target, then drop a level, and repeat. On the bottom level we are standing just
before the first key that is `>=` the target.

`range_query(lo, hi)`: (1) descend to the first key `>= lo`; (2) walk the bottom
level collecting keys until one exceeds `hi`.

| Operation | Expected cost |
|---|---|
| `insert`, `search`, `get` | O(log n) |
| `range_query` | O(log n + k), *k* = results returned |
| memory | O(n) |
| brute-force scan (for comparison) | O(n) for **every** query |

Reference: W. Pugh, *Skip Lists: A Probabilistic Alternative to Balanced Trees*,
CACM 33(6), 1990.

Parameters: `max_level=20` (about log2 of the largest n we expect; 2^18 is
262,144), `p=0.5`, `seed` for reproducibility. Not implemented on purpose:
`delete` (the project never removes readings).

## How we know it works

1. **Correctness against an oracle.** On real data, 300 random queries (widths
   from a single instant to over a year, start times before/at readings) must
   return exactly the same list as the brute-force scan. The benchmark repeats
   this check on every query it times and aborts on any mismatch.
2. **Edge cases:** empty list, duplicate keys, key absent/present, range below
   all data / above all data / between two keys / single instant / `lo > hi` /
   covering everything, unknown building, ranges inside and across the 25-day
   gaps, neighbouring buildings never leaking into a result.
3. **Structure checks:** `check_invariants()` verifies the bottom level is
   strictly sorted with exactly `len` keys and that each higher level is a
   sorted sub-list of the one below. A randomized test compares thousands of
   inserts and queries with a plain dictionary.
4. **Independent check:** for all 192 buildings, a whole-history query returns
   exactly the row count that `pandas.groupby` reports.
5. **Do the tests actually catch bugs?** We deliberately broke the code three
   ways (upper bound exclusive, search overshooting equal keys, duplicates not
   detected); the suite failed each time.

Run everything:

```bash
STORE_DATA_CSV=path/to/store_data.csv python -m pytest -v
python -m doctest skiplist.py
python benchmark.py --csv path/to/store_data.csv
```

## Benchmark results

Pure-Python, same queries and rows for both methods, best of 3 repeats, 50
queries per cell (run in the development sandbox; **your numbers will differ**
with your machine, but the shape should not):

| n readings | window | results/query | skip list | linear scan | speed-up |
|---|---|---|---|---|---|
| 10,000 | 24 h | 1.7 | 2.5 µs | 286 µs | ~110x |
| 50,000 | 24 h | 4.9 | 3.3 µs | 1.6 ms | ~490x |
| 100,000 | 24 h | 8.9 | 8.6 µs | 5.4 ms | ~630x |
| 262,719 | 24 h | 20.9 | 6.3 µs | 10.6 ms | ~1,700x |

The scan time grows with n; the skip list time barely moves (its height grows
only from 14 to 17 levels). Full table: `benchmark_results.csv`; chart:
`benchmark_plot.png` (log scale on the y-axis).

## Limitations and notes for the report

* Everything is in memory; nothing is persisted to disk.
* Python skip lists have large constant factors, and a NumPy-vectorised scan
  would be much faster than our pure-Python scan. Say which brute force you
  compare against.
* Timings are noisy (other programs, CPU caches). The benchmark copies each
  dataset into contiguous memory and takes the best of several repeats, but
  small differences (e.g. 6 vs 9 µs) are not meaningful.
* The data is hourly, in 5-day blocks with ~25-day gaps, so a 30-minute window
  returns at most one reading per building.
* Hooks for the anomaly part: `store.span(b)` gives a building's full time
  range (one range query returns its whole history), and any time-based window
  is just `store.query(b, t - window, t)`.
