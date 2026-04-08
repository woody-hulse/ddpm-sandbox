#!/usr/bin/env python3
"""
event_db_benchmark.py

Builds two event database layouts from data/tritium_ss.h5 (default 1000 events):

  event_db/single/
    events.h5        — all waveforms in one HDF5 file, chunk=(1,253,1000)
    waveforms.bin    — flat binary memmap, one contiguous file
    metadata.npz     — scalar fields alongside the memmap

  event_db/multi/
    event_000000.npy — one .npy waveform file per event
    event_000001.npy
    ...
    metadata.npz     — shared scalar fields

Then benchmarks random-access retrieval across layouts and batch sizes.

Run from the project root:
    python event_db_benchmark.py
    python event_db_benchmark.py --rebuild --n 1000 --trials 30
"""

import os
import time
import shutil
import argparse

import numpy as np
import h5py


SOURCE_H5   = "data/tritium_ss.h5"
DB_ROOT     = "event_db"
N_EVENTS    = 1000
WF_SHAPE    = (253, 1000)          # (channels, time_bins)
META_KEYS   = ["deltamu_us_squared", "dt", "s2w1090_ns", "xc", "yc", "orig_idx"]

SINGLE_DIR  = f"{DB_ROOT}/single"
MULTI_DIR   = f"{DB_ROOT}/multi"
SINGLE_H5   = f"{SINGLE_DIR}/events.h5"
SINGLE_BIN  = f"{SINGLE_DIR}/waveforms.bin"
SINGLE_META = f"{SINGLE_DIR}/metadata.npz"
MULTI_META  = f"{MULTI_DIR}/metadata.npz"


# ── Build ─────────────────────────────────────────────────────────────────────

def build(n: int = N_EVENTS, seed: int = 42, force: bool = False) -> int:
    if force and os.path.exists(DB_ROOT):
        shutil.rmtree(DB_ROOT)
    os.makedirs(SINGLE_DIR, exist_ok=True)
    os.makedirs(MULTI_DIR, exist_ok=True)

    print(f"Extracting {n} events from {SOURCE_H5} ...")
    with h5py.File(SOURCE_H5, "r") as src:
        total = src["waveforms"].shape[0]
        rng   = np.random.default_rng(seed)
        idx   = np.sort(rng.choice(total, n, replace=False))
        wf    = src["waveforms"][idx]           # (N, 253, 1000) float32
        meta  = {k: src[k][idx] for k in META_KEYS}

    # ── single HDF5, chunk=one event ─────────────────────────────────────────
    if not os.path.exists(SINGLE_H5):
        print("  building single-file HDF5 (chunk per event, no compression) ...")
        with h5py.File(SINGLE_H5, "w") as f:
            f.create_dataset("waveforms", data=wf,
                             chunks=(1, *WF_SHAPE), compression=None)
            for k, v in meta.items():
                f.create_dataset(k, data=v)
        mb = os.path.getsize(SINGLE_H5) / 1e6
        print(f"    {SINGLE_H5}  ({mb:.0f} MB)")

    # ── single flat memmap ────────────────────────────────────────────────────
    if not os.path.exists(SINGLE_BIN):
        print("  building single-file memmap ...")
        mm = np.memmap(SINGLE_BIN, dtype="float32", mode="w+", shape=(n, *WF_SHAPE))
        mm[:] = wf
        del mm
        np.savez(SINGLE_META, **meta)
        mb = os.path.getsize(SINGLE_BIN) / 1e6
        print(f"    {SINGLE_BIN}  ({mb:.0f} MB)")

    # ── per-event .npy files ──────────────────────────────────────────────────
    n_existing = sum(1 for fn in os.listdir(MULTI_DIR) if fn.startswith("event_"))
    if n_existing < n:
        print(f"  building {n} per-event .npy files ...")
        for i in range(n):
            path = f"{MULTI_DIR}/event_{i:06d}.npy"
            if not os.path.exists(path):
                np.save(path, wf[i])
        np.savez(MULTI_META, **meta)
        each_kb = os.path.getsize(f"{MULTI_DIR}/event_000000.npy") / 1e3
        print(f"    {n} files × {each_kb:.0f} KB  ({n * each_kb / 1e3:.0f} MB total)")

    print("Databases ready.\n")
    return n


# ── Benchmark ─────────────────────────────────────────────────────────────────

def _fmt(mean_s: float, std_s: float) -> str:
    if mean_s < 1.0:
        return f"{mean_s * 1000:6.1f} ±{std_s * 1000:4.1f} ms"
    return f"{mean_s:6.3f} ±{std_s:.3f} s "


def benchmark(n_events: int, batch_sizes=(1, 8, 64), n_trials: int = 20) -> None:
    rng = np.random.default_rng(99)

    # Pre-draw all trial indices; sort each so h5py and memmap get monotone access
    trial_idx = {
        bs: [np.sort(rng.choice(n_events, bs, replace=False)) for _ in range(n_trials)]
        for bs in batch_sizes
    }

    # Open memmap once; keep open for all trials
    mm = np.memmap(SINGLE_BIN, dtype="float32", mode="r", shape=(n_events, *WF_SHAPE))

    print(f"{'=' * 66}")
    print(f"  Benchmark — {n_trials} trials per cell, n_events={n_events}")
    print(f"{'=' * 66}")
    print(f"  {'batch':>5}  {'HDF5 single':>17}  {'memmap single':>17}  {'multi .npy':>17}")
    print(f"  {'-' * 5}  {'-' * 17}  {'-' * 17}  {'-' * 17}")

    for bs in batch_sizes:
        idx_list = trial_idx[bs]

        # HDF5 single file
        h5_times = []
        with h5py.File(SINGLE_H5, "r") as f:
            ds = f["waveforms"]
            for idx in idx_list:
                t0 = time.perf_counter()
                _ = ds[idx]
                h5_times.append(time.perf_counter() - t0)

        # memmap single file
        mm_times = []
        for idx in idx_list:
            t0 = time.perf_counter()
            _ = mm[idx].copy()
            mm_times.append(time.perf_counter() - t0)

        # multi-file directory
        mf_times = []
        for idx in idx_list:
            t0 = time.perf_counter()
            _ = np.stack([np.load(f"{MULTI_DIR}/event_{i:06d}.npy") for i in idx])
            mf_times.append(time.perf_counter() - t0)

        h5_m, h5_s = np.mean(h5_times), np.std(h5_times)
        mm_m, mm_s = np.mean(mm_times), np.std(mm_times)
        mf_m, mf_s = np.mean(mf_times), np.std(mf_times)

        print(f"  {bs:>5}  {_fmt(h5_m, h5_s)}  {_fmt(mm_m, mm_s)}  {_fmt(mf_m, mf_s)}")

    del mm

    print()
    print("  Layouts")
    print(f"    single HDF5 : {SINGLE_H5}")
    print(f"    single mmap : {SINGLE_BIN}")
    print(f"    multi .npy  : {MULTI_DIR}/event_NNNNNN.npy")
    print()
    print("  Notes")
    print("    - Results reflect warm OS page-cache; cold-cache reads are slower.")
    print("    - On NERSC Lustre, multi-file overhead is dominated by metadata ops")
    print("      (stat/open/close per file) rather than data transfer — expect")
    print("      10-100x more latency per event than seen here on a local SSD.")
    print("    - For Lustre: copy the single-file layouts to $TMPDIR at job start.")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build and benchmark event DB layouts.")
    ap.add_argument("--rebuild", action="store_true",
                    help="Delete and rebuild event_db/ from scratch.")
    ap.add_argument("--n",      type=int, default=N_EVENTS,
                    help=f"Number of events to sample (default {N_EVENTS}).")
    ap.add_argument("--trials", type=int, default=20,
                    help="Benchmark trials per cell (default 20).")
    args = ap.parse_args()

    n = build(n=args.n, force=args.rebuild)
    benchmark(n, n_trials=args.trials)
