#!/usr/bin/env python3
"""
build_db.py

One-time setup: extracts N events from a tritium_ss HDF5 source and populates
two databases under db/:

  db/lmdb/     — LMDB key-value store, one record per event
  db/events.h5 — HDF5 single file, all events, chunked 1-per-event (baseline)

Each LMDB record:
  key   : 4-byte big-endian event index
  value : waveform bytes (253 × 1000 × float32) + metadata (6 × float32)
          metadata order: deltamu_us_squared, dt, s2w1090_ns, xc, yc, orig_idx

Run from the repo root or as a module:
  python -m event_database.build_db
  python -m event_database.build_db --source /path/to/tritium_ss.h5 --n 1000 --rebuild
"""

import argparse
import os
import shutil
import struct
from pathlib import Path

import h5py
import lmdb
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = PACKAGE_ROOT.parent / "data" / "tritium_ss.h5"
DEFAULT_DB_ROOT = PACKAGE_ROOT / "db"
N_DEFAULT = 1000
WF_SHAPE = (253, 1000)
WF_BYTES = 253 * 1000 * 4
META_KEYS = ["deltamu_us_squared", "dt", "s2w1090_ns", "xc", "yc", "orig_idx"]


def build(source: str | Path, n: int, seed: int = 42, db_root: str | Path = DEFAULT_DB_ROOT) -> None:
    source = Path(source)
    db_root = Path(db_root)
    db_root.mkdir(parents=True, exist_ok=True)

    print(f"Reading {n} events from {source} ...")
    with h5py.File(source, "r") as f:
        total = f["waveforms"].shape[0]
        rng   = np.random.default_rng(seed)
        idx   = np.sort(rng.choice(total, n, replace=False))
        wf    = f["waveforms"][idx]                                   # (N, 253, 1000) float32
        meta  = {k: f[k][idx].astype(np.float32) for k in META_KEYS}

    # meta_arr[i] = [deltamu, dt, s2w1090, xc, yc, orig_idx] for event i
    meta_arr = np.stack([meta[k] for k in META_KEYS], axis=1)        # (N, 6) float32

    # ── LMDB: one record per event ────────────────────────────────────────────
    lmdb_path = db_root / "lmdb"
    if not lmdb_path.exists():
        print("Building LMDB (one record per event) ...")
        bytes_per_record = WF_BYTES + len(META_KEYS) * 4
        map_size = bytes_per_record * n * 3
        env = lmdb.open(str(lmdb_path), map_size=map_size, subdir=True)
        with env.begin(write=True) as txn:
            for i in range(n):
                key = struct.pack(">I", i)
                val = wf[i].astype(np.float32).tobytes() + meta_arr[i].tobytes()
                txn.put(key, val)
        env.sync()
        env.close()
        size = sum(
            os.path.getsize(lmdb_path / fn)
            for fn in os.listdir(lmdb_path)
        )
        print(f"  {lmdb_path}  ({size / 1e6:.0f} MB, {n} records)")

    # ── HDF5: all events in one file, chunk = one event (baseline) ────────────
    h5_path = db_root / "events.h5"
    if not h5_path.exists():
        print("Building HDF5 baseline (all events, chunk=1) ...")
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("waveforms", data=wf,
                             chunks=(1, *WF_SHAPE), compression=None)
            for k, v in meta.items():
                f.create_dataset(k, data=v)
        print(f"  {h5_path}  ({os.path.getsize(h5_path) / 1e6:.0f} MB)")

    print("Done.\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build LMDB and HDF5 event-store baselines.")
    ap.add_argument("--source",  default=str(DEFAULT_SOURCE),
                    help="Path to source tritium_ss.h5")
    ap.add_argument("--n",       type=int, default=N_DEFAULT,
                    help="Number of events to extract")
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--db-root", default=str(DEFAULT_DB_ROOT),
                    help="Directory where the benchmark databases will be written")
    ap.add_argument("--rebuild", action="store_true",
                    help="Delete db/ and rebuild from scratch")
    args = ap.parse_args()

    if args.rebuild and os.path.exists(args.db_root):
        shutil.rmtree(args.db_root)

    build(args.source, args.n, args.seed, args.db_root)
