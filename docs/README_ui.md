# The App (Streamlit)

An interactive page that shows the whole project at work: pick a building and a time window, see
its readings and flagged anomalies on a chart, compare the skip list with the brute force, and
check accuracy against the labels.

## Run it

```bash
pip install -r requirements.txt
streamlit run 10_app.py
```

It opens in your browser. The first start takes a few seconds while the skip list is built.

The app looks for the CSV files in the `data/` folder. If yours are somewhere else, set the
environment variables `STORE_DATA_CSV` / `GROUND_TRUTH_CSV`, or type the paths in the sidebar
under **Data files**. Without `ground_truth.csv` the app still works, but the accuracy views are
turned off.

## Sidebar controls

| Control | What it does |
|---|---|
| Building | pick one of the 192 buildings |
| Data block | the data comes in blocks of about 5 days (one per month); pick one, or "All data" |
| Time window | slider (whole hours) inside the chosen block |
| Method | compare a reading with that hour's normal, with the building's overall normal, or skip the z-score |
| Threshold k | how many standard deviations away counts as unusual (larger k means fewer flags) |
| Also flag readings equal to 1.0 | the stuck-value rule |
| Also flag readings next to a short data gap | the gap-edge rule |
| High-recall mode | also flags the hour before and after each flagged reading (on by default) |
| Show ground-truth anomalies | draws the labelled anomalies as black diamonds |

## Tabs

1. **Explore & detect**: the chart, plus a table of flagged readings.
2. **Skip list vs brute force**: runs the same query both ways and shows the time each took,
   how much work each did, and a check that both returned the same rows.
3. **Accuracy**: correct flags, false alarms and misses for the current window. You can also
   score all 192 buildings, and see the held-out results from `08_run_anomaly_eval.py`.
4. **About**: a short explanation.

## Reading the chart

| Mark | Meaning |
|---|---|
| Blue line | the readings (it breaks where data is missing) |
| Shaded band | the "normal" range: mean ± k standard deviations |
| Red dot | spike (too high) |
| Purple dot | drop (too low) |
| Orange dot | stuck value (exactly 1.0) |
| Green dot | gap edge (next to a short hole in the data) |
| Grey dot | high-recall neighbour (within 1 hour of another flag) |
| Black diamond | a real anomaly according to the labels |

A diamond **without** a dot is an anomaly we missed. A dot **without** a diamond is a false
alarm.

Tip: with the default `k = 5` you'll mostly see orange and green dots. To see red and purple
ones, set `k = 3` and pick building 159 (August block), 183 or 890.

## Good to know

- The brute force in the app (the teammate's `BruteForceList`) compares times using the CSV's
  rounded `ts` column. With whole-hour windows, which is all the slider allows, it returns exactly
  the same rows as the skip list. `10_test_ui.py` checks this on 150 random windows.
- "Score every building" includes the buildings the threshold was tuned on, so it is a bit
  optimistic. The held-out table is the one to report.
- Timings depend on your machine and on what else is running.

## How it is built

- Streamlit **re-runs `10_app.py` from top to bottom on every click**. The slow steps (building
  the skip list, reading the labels) are cached with `@st.cache_resource`, so they only happen
  once.
- `09_ui_helpers.py` holds the logic that can be tested without a browser: finding the data files,
  splitting the data into blocks, preparing the chart data, and drawing the chart.

## Tests

```bash
python -m pytest 10_test_ui.py     # 12 tests, including a headless run of the whole app
```
