# The Skip List Index

This part answers "all readings of building B between time t1 and t2" **without looking at
every row**. It uses a skip list that we wrote from scratch (no `bisect`, `sortedcontainers` or
tree libraries).

## Files

| File | What it does |
|---|---|
| `02_skiplist.py` | The skip list: `insert`, `get`, `search`, `range_query`, `check_invariants`. Works with any sortable keys. |
| `03_timeseries_store.py` | Loads the CSV, turns each row into a key, and offers `query(building_id, start, end)`. |
| `04_brute_force.py` | The linear scan: checks every row. Used to measure speed and to check the skip list's answers. |
| `04_test_skiplist.py` | 24 tests. |
| `05_benchmark.py` | Times skip list vs linear scan; writes `benchmark_results.csv` and `benchmark_plot.png`. |

## Quick start

```python
from importlib import import_module   # file names start with a digit
TimeSeriesStore = import_module("03_timeseries_store").TimeSeriesStore

store = TimeSeriesStore.from_csv("data/store_data.csv", seed=42)   # about 2 s
store.query(107, "2016-01-01 09:00", "2016-01-01 12:00")
# -> [(datetime(2016, 1, 1, 9, 0), 1.0), (datetime(2016, 1, 1, 10, 0), 1.0),
#     (datetime(2016, 1, 1, 11, 0), 1.0), (datetime(2016, 1, 1, 12, 0), 1.0)]

store.buildings()      # [107, 108, 111, ...]  all 192 building ids
store.span(107)        # first and last reading time of building 107
```

`query` returns `[(time, reading), ...]` sorted by time. Both ends are included. You get an empty
list for an unknown building, a time range with no data, or `start > end`.

## The key: `(building_id, time)`

All 192 buildings share the same timestamps, so time alone is not unique. We use the pair
`(building_id, time)` as the key. Python compares pairs item by item (building first, then
time), so in the skip list:

```
(107, 09:00) (107, 10:00) (107, 11:00) ... (108, 00:00) (108, 01:00) ...
```

One building's readings sit together, in time order. "Building 107 from t1 to t2" is then
**one** range query from `(107, t1)` to `(107, t2)`, and it can never pick up another building.

Other choices:

- **Time is stored as whole seconds since 1970** (for example `1451606400`). Whole numbers
  compare quickly, and "one hour later" is just `+ 3600`.
- **The CSV's `ts` column is not used.** It's a rounded version of the time that can be off by
  up to about 13 minutes. We read the exact `timestamp` column instead.
- **Readings are stored exactly as they are**: no cleaning, no filling of missing hours. The
  `1.0` readings are kept on purpose, because the labels mark them as anomalies.
- **The file doesn't need to be sorted.** The skip list puts every key in order as it's
  inserted.

## How a skip list works

A skip list is a **sorted linked list with express lanes**:

```
Level 3:  HEAD ---------------------------> 40 ------------------> None
Level 2:  HEAD ---------> 10 -------------> 40 ------> 70 -------> None
Level 1:  HEAD --> 5 ---> 10 --> 25 ------> 40 --> 55 --> 70 --> 90 -> None
```

Level 1 holds every key. Each higher level holds roughly half the keys of the level below.

**Searching.** Start at the top left. Move right while the next key is still smaller than the
one you want; when you can't, drop down one level. On the bottom level, the next node is the
first key that is greater than or equal to your target.

**Range query for 20 to 60** on the picture:

1. Level 3: the next key is 40, which isn't smaller than 20, so drop down.
2. Level 2: move to 10 (smaller than 20). The next key is 40, which isn't, so drop down.
3. Level 1: the next key is 25. This is the first key >= 20.
4. Walk right collecting 25, 40, 55. Stop at 70 (bigger than 60).

Most of the list is never touched.

**How tall is each node?** When a key is inserted we flip a coin: heads, the node also goes one
level higher; flip again, and so on. About half the nodes reach level 2, a quarter reach level 3,
and so on. This randomness keeps the list balanced on average without any rebalancing code,
which makes it much simpler to write than a balanced tree.

| Operation | Expected cost |
|---|---|
| `insert`, `search`, `get` | O(log n) |
| `range_query` | O(log n + k), where k = number of results |
| linear scan, for comparison | O(n) for **every** query |

Settings: `max_level=20` (enough for about a million keys), `p=0.5` (the coin flip), and `seed`
so results are reproducible. We left out `delete` on purpose, because the project never removes
readings.

## How we know it is correct

1. **Same answers as the linear scan.** On the real data, 300 random queries must return exactly
   the same list as the brute force. The benchmark also checks every query it times.
2. **Edge cases**: empty list, duplicate keys, ranges below, above or between all data, `lo > hi`,
   unknown building, ranges inside the gaps between data blocks, and no "leaking" into the next
   building.
3. **Structure checks**: `check_invariants()` checks that every level is sorted and that each
   level only contains keys from the level below.
4. **Random testing**: thousands of random inserts and queries compared with a plain dictionary.
5. **Do the tests catch bugs?** We broke the code on purpose three ways (excluded the upper bound,
   made search overshoot, stopped detecting duplicates). The tests failed each time.

```bash
python -m pytest 04_test_skiplist.py -v
python 05_benchmark.py
```

## Benchmark results

Time per query for a 24-hour window, pure Python, best of 3 runs. Your numbers will be different
on your machine, but the pattern should be the same:

| Readings stored | Results per query | Skip list | Linear scan | Speed-up |
|---|---|---|---|---|
| 10,000 | 1.7 | 4.3 µs | 0.46 ms | ~110x |
| 50,000 | 4.9 | 6.3 µs | 2.7 ms | ~420x |
| 100,000 | 8.9 | 5.5 µs | 4.7 ms | ~870x |
| 262,719 | 20.9 | 17.4 µs | 10.7 ms | ~620x |

The linear scan's time grows with the number of readings. The skip list's time barely changes.
All window sizes are in `benchmark_results.csv` and `benchmark_plot.png`.

**How the comparison is kept fair:** both methods get the same rows and the same queries, in the
same language, and return the same result format. Each time is the best of several runs.
Smaller datasets are random samples of the full file.

## Limitations

- Everything is in memory; nothing is saved to disk.
- A NumPy-based scan would be much faster than our pure-Python one, so say which scan we compare
  against.
- Timings vary from run to run, so small differences (for example 6 µs vs 9 µs) don't mean
  anything.
