#!/usr/bin/env python3
"""
benchmark.py

Measures random-access retrieval speed for LMDB vs HDF5 event stores.

Run from the repo root or as a module after building the database:
  python -m event_database.build_db
  python -m event_database.benchmark
"""

import argparse
import time

import numpy as np

from event_database.access import HDF5EventDB, LMDBEventDB

N_EVENTS = 1000


def bench(db, idx_lists: list) -> np.ndarray:
    times = []
    for idxs in idx_lists:
        t0 = time.perf_counter()
        db.get_batch(idxs)
        times.append(time.perf_counter() - t0)
    return np.array(times)


def fmt(times: np.ndarray) -> str:
    m, s = times.mean(), times.std()
    if m < 1.0:
        return f"{m * 1000:6.1f} ±{s * 1000:4.1f} ms"
    return f"{m:6.3f} ±{s:.3f} s "


def main(n_trials: int = 20) -> None:
    rng = np.random.default_rng(0)
    batch_sizes = [1, 8, 64]

    # Pre-draw all trial index sets
    trial_idx = {
        bs: [rng.choice(N_EVENTS, bs, replace=False) for _ in range(n_trials)]
        for bs in batch_sizes
    }

    lmdb_db = LMDBEventDB()
    hdf5_db = HDF5EventDB()

    # Warm up page cache with one pass each
    for bs in batch_sizes:
        lmdb_db.get_batch(trial_idx[bs][0])
        hdf5_db.get_batch(trial_idx[bs][0])

    print(f"{'=' * 52}")
    print(f"  Benchmark — {n_trials} trials, n_events={N_EVENTS}")
    print(f"{'=' * 52}")
    print(f"  {'batch':>5}  {'LMDB (per-event)':>18}  {'HDF5 (1000-event file)':>22}")
    print(f"  {'-' * 5}  {'-' * 18}  {'-' * 22}")

    for bs in batch_sizes:
        lmdb_t = bench(lmdb_db, trial_idx[bs])
        hdf5_t = bench(hdf5_db, trial_idx[bs])
        speedup = hdf5_t.mean() / lmdb_t.mean()
        print(f"  {bs:>5}  {fmt(lmdb_t)}  {fmt(hdf5_t)}   ({speedup:.1f}× faster)")

    lmdb_db.close()
    hdf5_db.close()

    print()
    print("  Notes")
    print("  - Warm page-cache: repeated reads of the same pages are free.")
    print("  - LMDB advantage grows with batch size: HDF5 pays per-call Python")
    print("    overhead into the C library for every non-contiguous chunk access.")
    print("  - On NERSC Lustre both are single files — copy to $TMPDIR at job")
    print("    start for best performance regardless of backend.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Benchmark LMDB vs HDF5 event-store reads.")
    ap.add_argument("--trials", type=int, default=20)
    args = ap.parse_args()
    main(args.trials)
