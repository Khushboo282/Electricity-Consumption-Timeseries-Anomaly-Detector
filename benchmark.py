"""
benchmark.py -- Skip list vs linear scan.

Run:  python benchmark.py --csv store_data.csv
      (optional: --sizes 10000 50000 100000 262719 --queries 50 --repeats 3)

WHAT IT MEASURES
  * build time: how long it takes to insert n rows into the skip list;
  * query time: average time of ONE range query "building b, from t to t+w",
    for three window widths w (6 hours, 1 day, 5 days), on the skip list and on
    the brute-force scan, at several dataset sizes n.

HOW IT KEEPS THE COMPARISON FAIR
  * Both methods answer the SAME queries on the SAME rows, in the same
    language (pure Python) and with the same result format.
  * Every answer from the skip list is compared with the brute-force answer;
    the benchmark aborts if they ever differ (a fast wrong answer is useless).
  * Queries start at real reading times of random buildings so they hit data.
  * Each timing is the best (minimum) of several repeats, which reduces noise
    from other programs running on the machine.
  * Smaller datasets are random samples of the full file (so every building is
    still represented), drawn with a fixed seed for reproducibility, kept in
    file order and copied into contiguous memory (see ``compact``) so the linear
    scan is not penalised by an artificial memory layout.

OUTPUT
  benchmark_results.csv and benchmark_plot.png in --out-dir (default: here).
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import time

from brute_force import brute_force_query_raw
from timeseries_store import TimeSeriesStore, load_rows

WINDOWS_HOURS = [6, 24, 120]


def compact(rows):
    """Copy rows into brand-new tuples/ints/floats allocated back to back.

    A sample drawn from the big list still points at objects scattered all over
    memory, so scanning it touches as much memory as scanning the whole file.
    That would make the linear scan look slower than it really is for smaller
    n. Rebuilding the objects gives every dataset size the same, contiguous
    memory layout, so scan time grows honestly with n.
    """
    return [(b + 0, e + 0, v + 0.0) for b, e, v in rows]


def make_queries(rows, window_hours, count, rng):
    """Random (building, start, end) queries that start at a real reading."""
    queries = []
    for _ in range(count):
        b, e, _v = rows[rng.randrange(len(rows))]
        queries.append((b, e, e + window_hours * 3600))
    return queries


def best_mean_seconds(func, queries, repeats):
    """Mean time per query, taking the best of ``repeats`` passes."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        for q in queries:
            func(*q)
        best = min(best, (time.perf_counter() - start) / len(queries))
    return best


def run(csv_path, sizes, n_queries, repeats, seed):
    all_rows = load_rows(csv_path)
    sizes = [min(s, len(all_rows)) for s in sizes]
    results = []
    for n in sizes:
        rng = random.Random(seed)
        if n == len(all_rows):
            chosen = all_rows
        else:
            # Sample row POSITIONS and keep them in file order.
            positions = sorted(rng.sample(range(len(all_rows)), n))
            chosen = [all_rows[i] for i in positions]
        rows = compact(chosen)

        t0 = time.perf_counter()
        store = TimeSeriesStore.from_rows(rows, seed=seed)
        build_s = time.perf_counter() - t0
        print(f"\nn = {n:>7,}  build = {build_s:.2f}s  "
              f"skip-list height = {store.index.height}")

        for w in WINDOWS_HOURS:
            queries = make_queries(rows, w, n_queries, random.Random(seed + w))

            # Correctness gate: identical answers on every benchmark query.
            sizes_returned = []
            for q in queries:
                expected = brute_force_query_raw(rows, *q)
                got = store.query_raw(*q)
                if got != expected:
                    raise AssertionError(f"MISMATCH for query {q}")
                sizes_returned.append(len(got))

            skip_s = best_mean_seconds(store.query_raw, queries, repeats)
            brute_s = best_mean_seconds(
                lambda b, lo, hi: brute_force_query_raw(rows, b, lo, hi),
                queries, repeats)
            results.append({
                "n": n, "window_hours": w,
                "avg_results": round(sum(sizes_returned) / len(sizes_returned), 1),
                "build_seconds": round(build_s, 3),
                "skiplist_us": round(skip_s * 1e6, 2),
                "brute_force_us": round(brute_s * 1e6, 2),
                "speedup_x": round(brute_s / skip_s, 1),
                "answers_match": True,
            })
            r = results[-1]
            print(f"  window {w:>3}h | results/query {r['avg_results']:>6} | "
                  f"skip list {r['skiplist_us']:>9.1f} us | "
                  f"brute force {r['brute_force_us']:>10.1f} us | "
                  f"speed-up {r['speedup_x']:>7.1f}x")
    return results


def save(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    csv_file = os.path.join(out_dir, "benchmark_results.csv")
    with open(csv_file, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved {csv_file}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping the plot.")
        return
    fig, axes = plt.subplots(1, len(WINDOWS_HOURS), figsize=(13, 4), sharey=True)
    for ax, w in zip(axes, WINDOWS_HOURS):
        rows = [r for r in results if r["window_hours"] == w]
        ns = [r["n"] for r in rows]
        ax.plot(ns, [r["brute_force_us"] for r in rows], "o-", label="Brute force (linear scan)")
        ax.plot(ns, [r["skiplist_us"] for r in rows], "s-", label="Skip list")
        ax.set_yscale("log")
        ax.set_title(f"Query window = {w} hours")
        ax.set_xlabel("Number of stored readings (n)")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("Time per query (microseconds, log scale)")
    axes[0].legend()
    fig.suptitle("Range query time: skip list vs linear scan")
    fig.tight_layout()
    png = os.path.join(out_dir, "benchmark_plot.png")
    fig.savefig(png, dpi=140)
    print(f"Saved {png}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=os.path.join(here, "store_data.csv"))
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[10_000, 50_000, 100_000, 262_719])
    ap.add_argument("--queries", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=here)
    args = ap.parse_args()
    save(run(args.csv, args.sizes, args.queries, args.repeats, args.seed),
         args.out_dir)
