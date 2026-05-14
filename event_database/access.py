"""
access.py

Unified random-access interface for the two event database backends.

Usage:
    from event_database import LMDBEventDB, HDF5EventDB

    db = LMDBEventDB()          # or HDF5EventDB()
    event  = db.get(42)         # single event  → dict
    batch  = db.get_batch([0, 7, 99])   # multiple events → dict of arrays
    db.close()

Returned dict keys:
    waveform          np.ndarray  (253, 1000) or (B, 253, 1000) for batches
    deltamu_us_squared float / np.ndarray
    dt                float / np.ndarray
    s2w1090_ns        float / np.ndarray
    xc                float / np.ndarray
    yc                float / np.ndarray
    orig_idx          float / np.ndarray
"""

import struct
from pathlib import Path
from typing import Sequence

import h5py
import lmdb
import numpy as np

DB_ROOT = Path(__file__).resolve().parent / "db"
WF_SHAPE = (253, 1000)
WF_BYTES = 253 * 1000 * 4
META_KEYS = ["deltamu_us_squared", "dt", "s2w1090_ns", "xc", "yc", "orig_idx"]


def _unpack(val: bytes) -> dict:
    """Deserialize one LMDB value into a waveform array + metadata dict."""
    wf       = np.frombuffer(val[:WF_BYTES], dtype=np.float32).reshape(WF_SHAPE).copy()
    meta_arr = np.frombuffer(val[WF_BYTES:], dtype=np.float32)
    return {"waveform": wf, **dict(zip(META_KEYS, meta_arr.tolist()))}


class LMDBEventDB:
    """Random-access event store backed by LMDB."""

    def __init__(self, path: str | Path = DB_ROOT / "lmdb"):
        self._env = lmdb.open(str(path), readonly=True, lock=False, subdir=True)

    def get(self, idx: int) -> dict:
        with self._env.begin() as txn:
            val = txn.get(struct.pack(">I", idx))
        if val is None:
            raise KeyError(idx)
        return _unpack(val)

    def get_batch(self, idxs: Sequence[int]) -> dict:
        waveforms, metas = [], {k: [] for k in META_KEYS}
        with self._env.begin() as txn:
            for idx in idxs:
                val = txn.get(struct.pack(">I", int(idx)))
                event = _unpack(val)
                waveforms.append(event["waveform"])
                for k in META_KEYS:
                    metas[k].append(event[k])
        return {
            "waveform": np.stack(waveforms),
            **{k: np.array(v, dtype=np.float32) for k, v in metas.items()},
        }

    def __len__(self) -> int:
        with self._env.begin() as txn:
            return txn.stat()["entries"]

    def close(self) -> None:
        self._env.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class HDF5EventDB:
    """Random-access event store backed by a single chunked HDF5 file."""

    def __init__(self, path: str | Path = DB_ROOT / "events.h5"):
        self._path = str(path)

    def get(self, idx: int) -> dict:
        with h5py.File(self._path, "r") as f:
            wf   = np.array(f["waveforms"][idx], dtype=np.float32)
            meta = {k: float(f[k][idx]) for k in META_KEYS}
        return {"waveform": wf, **meta}

    def get_batch(self, idxs: Sequence[int]) -> dict:
        idxs = np.sort(idxs)          # h5py requires monotone fancy-index
        with h5py.File(self._path, "r") as f:
            wf   = np.array(f["waveforms"][idxs], dtype=np.float32)
            meta = {k: np.array(f[k][idxs], dtype=np.float32) for k in META_KEYS}
        return {"waveform": wf, **meta}

    def __len__(self) -> int:
        with h5py.File(self._path, "r") as f:
            return f["waveforms"].shape[0]

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
