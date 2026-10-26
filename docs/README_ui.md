# UI (Streamlit app)

An interactive front end for the whole project: query a building's readings
through the skip list, see anomalies on a chart, compare against the brute-force
scan, and check accuracy against the ground truth.

## Run it

```
project folder/
  app.py  ui_helpers.py  skiplist.py  timeseries_store.py  anomaly.py  evaluation.py
  brute_force_demo.py        <- the teammate's brute force (used unmodified)
  data/
    store_data.csv
    ground_truth.csv
```

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app looks for the CSVs in `data/` next to `app.py` first (then the project
folder, `processed data/`, and the current folder). To use other locations set the
environment variables `STORE_DATA_CSV` / `GROUND_TRUTH_CSV`, or type the paths in
the sidebar under *Data files*. Without `ground_truth.csv` the app still works; only
the accuracy views are disabled. First start takes a few seconds to build the index.

## Sidebar controls

* **Building** (192) and **Data block**: the data comes in up to 12 blocks of about 5
  days (one per month); pick one or "All data".
* **Time window**: a slider in whole hours inside the chosen block.
* **Method**: hourly baseline, building baseline, or the stuck-value rule only.
* **Threshold k**: how many standard deviations count as unusual.
* **Also flag readings equal to 1.0**: the dataset-specific placeholder rule.
* **Show ground-truth anomalies**: black diamonds on the chart.

## Tabs

1. **Explore & detect**: readings line, shaded normal band (mean ± k standard
   deviations, the same one the detector uses), coloured dots for flagged spikes /
   drops / stuck values, and a table of flagged readings.
2. **Skip list vs brute force**: the same query run on both, with time per query,
   key comparisons vs rows inspected, speed-up, and a check that both return exactly
   the same rows. Below it, the saved benchmark across dataset sizes.
3. **Accuracy**: TP / FP / FN for the current window; optionally precision / recall / F1
   over all 192 buildings with the current settings; and the held-out table from
   `run_anomaly_eval.py`.
4. **About**: short explanation and caveats.

## Notes

* The brute force (`brute_force_demo.py`) filters on the CSV's coarse `ts` column
  (1,000 s steps); the skip list uses the exact `timestamp`. For whole-hour windows
  (all the slider allows) both return identical rows; `test_ui.py` checks 150 random
  windows and also documents the non-whole-hour case where they could differ.
* "Score every building" uses all buildings, including the ones the threshold was
  tuned on, so it is optimistic. Report the held-out table.
* Timings are best-of-several in pure Python and vary by machine and by what else
  is running.
* `ui_helpers.py` holds the logic that can be tested without a browser;
  `test_ui.py` also drives the whole app headlessly (clicks through the controls).
  The app was tested that way and its chart was rendered to an image and inspected,
  but it was not opened in a real browser during development.

## Tests

```bash
python -m pytest            # all three parts: 60 tests
python -m pytest test_ui.py
```
