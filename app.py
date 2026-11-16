"""
app.py -- Streamlit UI for the Time-Series Store with Anomaly Detection.

Run:   streamlit run app.py
(The data files are looked up in ./data next to this file; see the sidebar.)

WHAT THE APP SHOWS
  Tab 1  Explore & detect : pick a building and a time window; the skip list
                            answers the range query, the detector flags
                            spikes / drops / stuck values on the chart.
  Tab 2  Skip list vs brute force : the SAME query run on our skip list and on
                            the teammate's linear-scan list, timed side by side,
                            with a check that both return identical rows.
  Tab 3  Accuracy : precision / recall / F1 against ground_truth.csv.
  Tab 4  About : how it works and the caveats.
"""

from __future__ import annotations

import datetime as dt
import inspect
import os
import time

import pandas as pd
import streamlit as st

from anomaly import AnomalyDetector
from evaluation import anomaly_runs, event_recall, load_ground_truth, score
from timeseries_store import TimeSeriesStore, to_epoch
from ui_helpers import (HERE, brute_keys, brute_range, build_explore_chart,
                        data_blocks, find_data_file, load_brute_force, time_best_us)

st.set_page_config(page_title="Time-Series Store + Anomaly Detection", layout="wide")

# Streamlit renamed use_container_width -> width="stretch" in recent versions.
_STRETCH_CHART = ({"width": "stretch"}
                  if "width" in inspect.signature(st.altair_chart).parameters
                  else {"use_container_width": True})
_STRETCH_TABLE = ({"width": "stretch"}
                  if "width" in inspect.signature(st.dataframe).parameters
                  else {"use_container_width": True})

METHODS = {
    "Hourly baseline (per hour of day)": "hourly",
    "Building baseline (all hours together)": "building",
    "Stuck-value rule only (no z-score)": None,
}


# ---------------------------------------------------------------------- #
# Cached loaders (run once per file, not on every click)
# ---------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Building the skip-list index (about 2 s)...")
def load_store(csv_path: str) -> TimeSeriesStore:
    return TimeSeriesStore.from_csv(csv_path, seed=42)


@st.cache_resource(show_spinner="Loading the brute-force list...")
def load_brute(csv_path: str):
    return load_brute_force(csv_path)


@st.cache_resource(show_spinner="Reading ground truth...")
def load_truth(csv_path: str):
    return load_ground_truth(csv_path)


@st.cache_data(show_spinner="Scoring all buildings (a few seconds)...")
def evaluate_all(store_csv, truth_csv, method, k, sentinel):
    store, truth = load_store(store_csv), load_truth(truth_csv)
    predicted = AnomalyDetector(store, method=method, k=k,
                                sentinel_value=sentinel).flagged_keys()
    s = score(predicted, truth)
    return {"tp": s.tp, "fp": s.fp, "fn": s.fn, "precision": s.precision,
            "recall": s.recall, "f1": s.f1, "flagged": s.flagged,
            "event_recall": event_recall(predicted, anomaly_runs(truth))}


# ---------------------------------------------------------------------- #
# Sidebar: data files
# ---------------------------------------------------------------------- #
st.title("Time-Series Store with Anomaly Detection")
st.caption("Skip-list index for fast range queries, with statistical anomaly "
           "detection on top. Hourly electricity meter readings, 192 buildings.")

with st.sidebar:
    st.header("Data")
    with st.expander("Data files", expanded=False):
        store_default = find_data_file("store_data.csv", "STORE_DATA_CSV") or ""
        truth_default = find_data_file("ground_truth.csv", "GROUND_TRUTH_CSV") or ""
        store_csv = st.text_input("store_data.csv", store_default)
        truth_csv = st.text_input("ground_truth.csv (optional)", truth_default)
        st.caption(f"Looked in `data/` next to the code first ({os.path.join(HERE, 'data')}).")

if not store_csv or not os.path.isfile(store_csv):
    st.error("Could not find **store_data.csv**. Put it in a `data` folder next to "
             "`app.py`, set the STORE_DATA_CSV environment variable, or type its "
             "path in the sidebar (Data files).")
    st.stop()

store = load_store(store_csv)
brute = load_brute(store_csv)
has_truth = bool(truth_csv) and os.path.isfile(truth_csv)
truth = load_truth(truth_csv) if has_truth else set()

# ---------------------------------------------------------------------- #
# Sidebar: query + detector settings
# ---------------------------------------------------------------------- #
with st.sidebar:
    st.header("Query")
    building = st.selectbox("Building", store.buildings(), index=0)

    probe = AnomalyDetector(store)                 # only used to fetch the history
    history = probe.history(building)
    blocks = data_blocks([e for e, _ in history])
    labels = ["All data (every block)"] + [
        f"{to_dt:%b %d} to {end_dt:%b %d, %Y}"
        for to_dt, end_dt in ((dt.datetime(1970, 1, 1) + dt.timedelta(seconds=a),
                               dt.datetime(1970, 1, 1) + dt.timedelta(seconds=b))
                              for a, b in blocks)]
    block_choice = st.selectbox("Data block", range(len(labels)),
                                format_func=lambda i: labels[i], index=1 if blocks else 0)
    first_epoch = history[0][0] if block_choice == 0 else blocks[block_choice - 1][0]
    last_epoch = history[-1][0] if block_choice == 0 else blocks[block_choice - 1][1]
    t_min = dt.datetime(1970, 1, 1) + dt.timedelta(seconds=first_epoch)
    t_max = dt.datetime(1970, 1, 1) + dt.timedelta(seconds=last_epoch)
    if t_min < t_max:
        window = st.slider("Time window (whole hours)", min_value=t_min, max_value=t_max,
                           value=(t_min, t_max), step=dt.timedelta(hours=1),
                           format="MMM DD, HH:mm", key=f"win_{building}_{block_choice}")
    else:
        window = (t_min, t_max)
        st.info("This block has a single reading.")

    st.header("Anomaly detection")
    method_label = st.radio("Method", list(METHODS), index=0)
    method = METHODS[method_label]
    k = st.slider("Threshold k (standard deviations)", 1.5, 10.0, 5.0, 0.5,
                  disabled=method is None,
                  help="A reading is flagged when it is more than k standard "
                       "deviations from normal. Larger k = fewer, surer flags.")
    use_sentinel = st.checkbox("Also flag readings equal to 1.0", value=True,
                               help="Dataset-specific rule: 1.0 behaves like a "
                                    "placeholder value in this data (not statistics).")
    show_truth = st.checkbox("Show ground-truth anomalies (diamonds)", value=has_truth,
                             disabled=not has_truth)

sentinel = 1.0 if use_sentinel else None
detector = AnomalyDetector(store, method=method, k=k, sentinel_value=sentinel)
start_epoch, end_epoch = to_epoch(window[0]), to_epoch(window[1])

# The core operation: ONE skip-list range query for the window
readings = store.query(building, window[0], window[1])
anomalies = detector.detect(building, window[0], window[1])

tab_explore, tab_timing, tab_accuracy, tab_about = st.tabs(
    ["Explore & detect", "Skip list vs brute force", "Accuracy", "About"])

# ====================================================================== #
# Tab 1: explore & detect
# ====================================================================== #
with tab_explore:
    window_truth = [(when, value) for when, value in readings
                    if (building, to_epoch(when)) in truth] if has_truth else []
    c1, c2, c3 = st.columns(3)
    c1.metric("Readings in window", f"{len(readings):,}")
    c2.metric("Anomalies flagged", f"{len(anomalies):,}")
    c3.metric("Labelled anomalies in window", f"{len(window_truth):,}" if has_truth else "n/a")

    if not readings:
        st.warning("No readings in this window.")
    else:
        chart = build_explore_chart(readings, anomalies, detector.baseline(building), method, k,
                                    window_truth if show_truth else ())
        st.altair_chart(chart.interactive(), **_STRETCH_CHART)
        caption = ("Blue line: readings. Shaded band: the normal range the detector uses "
                   "(mean ± k standard deviations). Coloured dots: flagged by the detector.")
        if show_truth:
            caption += " Black diamonds: readings labelled anomalous in the ground truth."
        st.caption(caption)

        if anomalies:
            st.subheader("Flagged readings")
            table = pd.DataFrame([{"Time": a.time, "Reading": a.value, "Type": a.kind,
                                   "z-score": round(a.score, 2), "Why": a.reason}
                                  for a in anomalies])
            st.dataframe(table, hide_index=True, **_STRETCH_TABLE)
        else:
            st.info("Nothing flagged in this window with the current settings.")

# ====================================================================== #
# Tab 2: skip list vs brute force
# ====================================================================== #
with tab_timing:
    st.subheader("Same query, two methods")
    st.write(f"Query: building **{building}**, from **{window[0]:%Y-%m-%d %H:%M}** to "
             f"**{window[1]:%Y-%m-%d %H:%M}** (the window selected in the sidebar).")

    skip_results, skip_comparisons = store.query_raw_counted(building, start_epoch, end_epoch)
    skip_us = time_best_us(lambda: store.query_raw(building, start_epoch, end_epoch), repeats=300)
    brute_results, _, brute_comparisons = brute_range(brute, building, start_epoch, end_epoch)
    brute_us = min(brute_range(brute, building, start_epoch, end_epoch)[1] for _ in range(5))

    identical = {key for key, _ in skip_results} == brute_keys(brute_results)
    a, b, c = st.columns(3)
    with a:
        st.markdown("**Skip list (ours)**")
        st.metric("Time per query", f"{skip_us:,.1f} µs")
        st.metric("Key comparisons", f"{skip_comparisons:,}")
    with b:
        st.markdown("**Brute force (linear scan)**")
        st.metric("Time per query", f"{brute_us:,.1f} µs")
        st.metric("Rows inspected", f"{brute_comparisons:,}")
    with c:
        st.markdown("**Result**")
        st.metric("Speed-up", f"{brute_us / skip_us:,.0f}x" if skip_us > 0 else "n/a")
        st.metric("Rows returned", f"{len(skip_results):,}")

    if identical:
        st.success(f"Both methods returned exactly the same {len(skip_results):,} readings.")
    else:
        st.error(f"Results differ: skip list {len(skip_results)} rows, "
                 f"brute force {len(brute_results)} rows.")
    st.caption("The skip list descends express lanes straight to the start of the "
               "range and then walks forward, so its work depends on how many readings "
               "are returned, not on the 262,719 stored. The linear scan checks every "
               "row for every query. Times are best-of-several runs in pure Python and "
               "will vary by machine.")

    bench_csv = os.path.join(HERE, "benchmark_results.csv")
    bench_png = os.path.join(HERE, "benchmark_plot.png")
    if os.path.isfile(bench_csv):
        st.subheader("Benchmark across dataset sizes")
        if os.path.isfile(bench_png):
            st.image(bench_png)
        st.dataframe(pd.read_csv(bench_csv), hide_index=True, **_STRETCH_TABLE)

# ====================================================================== #
# Tab 3: accuracy
# ====================================================================== #
with tab_accuracy:
    if not has_truth:
        st.info("Add ground_truth.csv (see Data files in the sidebar) to measure accuracy.")
    else:
        st.subheader("Accuracy against ground truth")
        st.write("Plain accuracy is meaningless here (only 1.2% of readings are anomalies), "
                 "so we report **precision** (flags that were right), **recall** (real "
                 "anomalies found) and **F1** (both together).")
        window_truth_keys = {(building, to_epoch(w)) for w, _ in window_truth}
        window_pred = {(a.building_id, to_epoch(a.time)) for a in anomalies}
        ws = score(window_pred, window_truth_keys)
        st.markdown("**This building and window**")
        w1, w2, w3, w4 = st.columns(4)
        w1.metric("Correct flags (TP)", ws.tp)
        w2.metric("False alarms (FP)", ws.fp)
        w3.metric("Missed (FN)", ws.fn)
        w4.metric("F1", f"{ws.f1:.2f}")

        st.markdown("**All 192 buildings, current settings**")
        if st.checkbox("Score every building with these settings"):
            r = evaluate_all(store_csv, truth_csv, method, k, sentinel)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Precision", f"{r['precision']:.3f}")
            m2.metric("Recall", f"{r['recall']:.3f}")
            m3.metric("F1", f"{r['f1']:.3f}")
            m4.metric("Anomaly runs caught", f"{r['event_recall']:.1%}")
            st.caption(f"{r['tp']:,} correct flags, {r['fp']:,} false alarms, {r['fn']:,} missed "
                       f"({r['flagged']:,} readings flagged). These numbers include the "
                       "buildings the threshold was tuned on, so they are optimistic; the "
                       "held-out results below are the ones to report.")

        results_csv = os.path.join(HERE, "anomaly_results.csv")
        if os.path.isfile(results_csv):
            st.markdown("**Held-out result (threshold tuned on half the buildings, "
                        "scored on the other half)**")
            held = pd.read_csv(results_csv)[["config", "k", "test_precision", "test_recall",
                                             "test_f1", "test_event_recall"]]
            held.columns = ["Configuration", "k", "Precision", "Recall", "F1", "Runs caught"]
            st.dataframe(held, hide_index=True, **_STRETCH_TABLE)
            st.caption("These F1 values come from one building split and move between "
                       "about 0.47 and 0.60 for the best configuration across splits; "
                       "see README_anomaly.md.")

# ====================================================================== #
# Tab 4: about
# ====================================================================== #
with tab_about:
    st.markdown("""
**How it works**

1. **Index.** Every reading is stored in a skip list written from scratch, keyed by
   `(building, time)`, so one building's readings sit together in time order.
2. **Query.** "Building B between t1 and t2" is one range query: descend the
   skip list's express lanes to `(B, t1)`, then walk forward until past `(B, t2)`.
3. **Detection (no machine learning).** The building's whole history is fetched with a
   range query and summarised by its mean and standard deviation (optionally per hour
   of day). A reading more than **k** standard deviations away is flagged as a spike or
   a drop.
4. **Stuck-value rule.** A reading of exactly `1.0` is flagged too. This is specific to
   this dataset, where `1.0` behaves like a placeholder; it is not a statistical method.

**Things to know**

- The data is hourly and comes in 5-day blocks with long gaps in between; the window
  slider therefore works inside one block (or across all of them).
- Many labelled anomalies look ordinary by value, so value-based statistics cannot
  find them all. Expect modest recall.
- The brute-force list filters on the CSV's coarse `ts` column, while the skip list uses
  the exact `timestamp`. With whole-hour windows (what the slider allows) both return
  the same rows.
""")
