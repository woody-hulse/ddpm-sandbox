"""
Compress and transform LZ/Tritium H5 data.

Combines functionality for:
1. Channel reduction (--channels N): Select N nearest PMTs, or sum all if N=1
2. Time window cropping (--time-window, --time-start)
3. Activity filtering (--activity-threshold)
4. Multi-scatter event generation (--ms): Combine pairs of events with time shifts

Usage examples:
    # Select 42 PMTs, crop time, filter by activity
    python compress.py --channels 42 --time-window 500

    # Single-node mode (sum all channels)
    python compress.py --channels 1

    # Generate MS events from existing SS data
    python compress.py --ms --n-ms-events 10000 --delta-min -30 --delta-max 30

    # Full pipeline: compress then generate MS
    python compress.py --channels 42 --time-window 500 --ms --n-ms-events 10000
"""
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


NS_PER_BIN = 10.0  # Each z (time) bin represents 10ns


@dataclass(frozen=True)
class ChannelPositions:
    positions_cm: np.ndarray  # (C,2) float32
    raw_units_per_cm: float   # used only for writing TA_PMTs_xy = positions_cm * raw_units_per_cm


def _load_channel_positions_like_lz(path: Path) -> ChannelPositions:
    with h5py.File(path, "r") as f:
        if "TA_PMTs_xy" in f:
            raw = np.asarray(f["TA_PMTs_xy"][:], dtype=np.float32)
            return ChannelPositions(positions_cm=raw / 10.0, raw_units_per_cm=10.0)
        if "positions" in f:
            raw = np.asarray(f["positions"][:], dtype=np.float32)
            return ChannelPositions(positions_cm=raw, raw_units_per_cm=10.0)
        if "xy" in f:
            raw = np.asarray(f["xy"][:], dtype=np.float32)
            return ChannelPositions(positions_cm=raw, raw_units_per_cm=10.0)
        raise ValueError(f"Could not find channel positions in {path}; expected one of TA_PMTs_xy/positions/xy.")


def _select_pmts_near_point(positions_cm: np.ndarray, point_xy_cm: Tuple[float, float], n: int) -> np.ndarray:
    pos = np.asarray(positions_cm, dtype=np.float32)
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError(f"positions must be (C,2), got {pos.shape}")
    n = int(n)
    if n < 1:
        raise ValueError("N must be >= 1")
    if n > pos.shape[0]:
        raise ValueError(f"N={n} exceeds available PMTs={pos.shape[0]}")
    p = np.asarray(point_xy_cm, dtype=np.float32).reshape(1, 2)
    d2 = np.sum((pos - p) ** 2, axis=1)
    idx = np.argsort(d2)[:n]
    return idx.astype(np.int64, copy=False)


def _copy_attrs(src, dst) -> None:
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def _create_like(src: h5py.Dataset, out_file: h5py.File, name: str, shape) -> h5py.Dataset:
    chunks = src.chunks
    if chunks is not None:
        shape_t = tuple(int(d) for d in shape)
        chunks = tuple(min(int(c), int(s)) for c, s in zip(chunks, shape_t))
    return out_file.create_dataset(
        name,
        shape=shape,
        dtype=src.dtype,
        chunks=chunks,
        compression=src.compression,
        compression_opts=src.compression_opts,
        shuffle=src.shuffle,
        fletcher32=src.fletcher32,
    )


def compress_to_single_node(
    *,
    tritium_in: Path,
    out_dir: Optional[Path] = None,
    time_window: Optional[int] = None,
    time_start: Optional[int] = None,
    batch_rows: int = 512,
) -> Tuple[Path, Path]:
    """Compress waveforms by summing over all channels (single-node mode).

    Input shape: (S, C, T) where C = channels (spatial), T = time.
    Output shape: (S, 1, T') where T' is optionally cropped.

    Returns (tritium_out, pmt_xy_out) paths.
    """
    tritium_in = Path(tritium_in)
    out_dir = Path(out_dir) if out_dir is not None else tritium_in.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(tritium_in, "r") as fin:
        if "waveforms" not in fin:
            raise ValueError(f"{tritium_in} must contain a waveforms dataset.")
        wf_in = fin["waveforms"]
        if wf_in.ndim != 3:
            raise ValueError(f"Expected waveforms to be (S,C,T); got shape={wf_in.shape}")
        s, c, t = map(int, wf_in.shape)

        tw = int(time_window) if time_window is not None else t
        if tw < 1 or tw > t:
            raise ValueError(f"time_window must be in [1, {t}], got {tw}")
        ts = int(time_start) if time_start is not None else (t - tw) // 2
        ts = max(0, min(ts, t - tw))
        te = ts + tw

        suffix = f"_c1_t{tw}" if tw != t else "_c1"
        tritium_out = out_dir / f"{tritium_in.stem}{suffix}.h5"
        pmt_xy_out = out_dir / f"pmt_xy_c1.h5"

        with h5py.File(tritium_out, "w") as fout:
            _copy_attrs(fin, fout)
            fout.attrs["shrink_mode"] = "single_node"
            fout.attrs["original_channels"] = int(c)
            fout.attrs["original_time"] = int(t)
            fout.attrs["time_start"] = int(ts)
            fout.attrs["time_end"] = int(te)

            wf_out = fout.create_dataset(
                "waveforms",
                shape=(s, 1, tw),
                dtype=np.float32,
                chunks=(min(batch_rows, s), 1, tw),
                compression=wf_in.compression,
                compression_opts=wf_in.compression_opts,
            )
            _copy_attrs(wf_in, wf_out)

            for start in range(0, s, int(batch_rows)):
                end = min(s, start + int(batch_rows))
                block = np.asarray(wf_in[start:end, :, ts:te], dtype=np.float64)
                sums = block.sum(axis=1, keepdims=True).astype(np.float32)
                wf_out[start:end, :, :] = sums

            for name, obj in fin.items():
                if name == "waveforms":
                    continue
                if not isinstance(obj, h5py.Dataset):
                    continue
                ds_in = obj
                ds_out = _create_like(ds_in, fout, name, shape=ds_in.shape)
                _copy_attrs(ds_in, ds_out)
                for start in range(0, ds_in.shape[0], int(batch_rows)):
                    end = min(ds_in.shape[0], start + int(batch_rows))
                    ds_out[start:end, ...] = ds_in[start:end, ...]

            if ("dt" in fout) and (fout["dt"].ndim == 1) and (fout["dt"].shape[0] == s):
                fout["dt"][:] = np.asarray(fout["dt"][:], dtype=np.float32) - float(ts)

    with h5py.File(pmt_xy_out, "w") as fpos:
        ds = fpos.create_dataset("TA_PMTs_xy", data=np.array([[0.0, 0.0]], dtype=np.float32))
        ds.attrs["note"] = "Single-node graph: one PMT at origin (0, 0)."

    print(f"Compressed {s} events from ({c},{t}) to single-node ({1},{tw}).")
    print(f"Wrote {tritium_out}")
    print(f"Wrote {pmt_xy_out}")
    return tritium_out, pmt_xy_out


def compress_select_channels(
    *,
    tritium_in: Path,
    pmt_xy_in: Path,
    n_channels: int = 42,
    time_window: int = 500,
    time_start: Optional[int] = None,
    activity_threshold: float = 0.0,
    activity_hist_bins: int = 60,
    center_xy_cm: Optional[Tuple[float, float]] = None,
    sample_events: int = 3,
    out_dir: Optional[Path] = None,
    batch_rows: int = 512,
) -> Tuple[Path, Path, Path]:
    """Compress by selecting N nearest PMTs and cropping time window."""
    tritium_in = Path(tritium_in)
    pmt_xy_in = Path(pmt_xy_in)
    out_dir = Path(out_dir) if out_dir is not None else tritium_in.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    n = int(n_channels)
    tritium_out = out_dir / f"{tritium_in.stem}_c{n}_t{int(time_window)}.h5"
    pmt_xy_out = out_dir / f"pmt_xy_c{n}.h5"
    plot_out = out_dir / f"selected_pmts_c{n}_t{int(time_window)}.png"

    chpos = _load_channel_positions_like_lz(pmt_xy_in)

    with h5py.File(tritium_in, "r") as fin:
        if "waveforms" not in fin:
            raise ValueError(f"{tritium_in} must contain a waveforms dataset.")
        wf_in = fin["waveforms"]
        if center_xy_cm is None:
            center = np.mean(chpos.positions_cm, axis=0)
            target_xy = (float(center[0]), float(center[1]))
        else:
            target_xy = (float(center_xy_cm[0]), float(center_xy_cm[1]))
        keep_idx = _select_pmts_near_point(chpos.positions_cm, target_xy, n)
        keep_idx_sorted = np.sort(keep_idx)
        keep_perm = np.searchsorted(keep_idx_sorted, keep_idx)

        if wf_in.ndim != 3:
            raise ValueError(f"Expected waveforms to be (S,C,T); got shape={wf_in.shape}")
        s, c, t = map(int, wf_in.shape)
        if c != int(chpos.positions_cm.shape[0]):
            raise ValueError(f"Waveforms channels={c} does not match PMT positions count={chpos.positions_cm.shape[0]}")

        tw = int(time_window)
        if tw < 1 or tw > t:
            raise ValueError(f"time_window must be in [1, {t}], got {tw}")
        ts = int(time_start) if time_start is not None else (t - tw) // 2
        ts = max(0, min(ts, t - tw))
        te = ts + tw

        thr = float(activity_threshold)
        activity = np.empty((s,), dtype=np.float32)
        keep_mask = np.zeros((s,), dtype=bool)
        for start in range(0, s, int(batch_rows)):
            end = min(s, start + int(batch_rows))
            block = wf_in[start:end, :, ts:te][:, keep_idx_sorted, :]
            block = block[:, keep_perm, :]
            a = np.sum(block, axis=(1, 2), dtype=np.float64)
            activity[start:end] = a.astype(np.float32, copy=False)
            keep_mask[start:end] = (a >= thr)

        keep_rows = np.nonzero(keep_mask)[0].astype(np.int64, copy=False)
        s_out = int(keep_rows.shape[0])

        print(f"Events kept: {s_out} / {s} (removed: {s - s_out}) with activity_threshold={thr}")

        with h5py.File(tritium_out, "w") as fout:
            _copy_attrs(fin, fout)
            fout.attrs["pmt_subset_n"] = n
            fout.attrs["time_start"] = int(ts)
            fout.attrs["time_end"] = int(te)
            fout.attrs["activity_threshold"] = float(thr)

            wf_out = _create_like(wf_in, fout, "waveforms", shape=(s_out, n, tw))
            _copy_attrs(wf_in, wf_out)

            for out_start in range(0, s_out, int(batch_rows)):
                out_end = min(s_out, out_start + int(batch_rows))
                rows = keep_rows[out_start:out_end]
                block = wf_in[rows, :, ts:te][:, keep_idx_sorted, :]
                wf_out[out_start:out_end, :, :] = block[:, keep_perm, :]

            for name, obj in fin.items():
                if name == "waveforms":
                    continue
                if not isinstance(obj, h5py.Dataset):
                    continue
                ds_in = obj
                if ds_in.ndim >= 1 and ds_in.shape[0] == s:
                    out_shape = (s_out,) + tuple(int(d) for d in ds_in.shape[1:])
                else:
                    out_shape = ds_in.shape
                ds_out = _create_like(ds_in, fout, name, shape=out_shape)
                _copy_attrs(ds_in, ds_out)
                if ds_in.ndim >= 1 and ds_in.shape[0] == s:
                    for out_start in range(0, s_out, int(batch_rows)):
                        out_end = min(s_out, out_start + int(batch_rows))
                        rows = keep_rows[out_start:out_end]
                        ds_out[out_start:out_end, ...] = ds_in[rows, ...]
                else:
                    ds_out[...] = ds_in[...]

            if ("dt" in fout) and (fout["dt"].ndim == 1) and (fout["dt"].shape[0] == s_out):
                fout["dt"][:] = np.asarray(fout["dt"][:], dtype=np.float32) - float(ts)

        k = int(sample_events)
        if k > 0 and s_out > 0:
            sample_dir = out_dir / f"sample_events_c{n}_t{tw}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            kept_activity = activity[keep_rows]
            order = np.argsort(kept_activity)
            if k == 1:
                sel = [int(order[len(order) // 2])]
            else:
                anchors = np.linspace(0, len(order) - 1, num=min(k, len(order)), dtype=int)
                sel = [int(order[a]) for a in anchors]

            dt_in = fin["dt"][:] if "dt" in fin else None
            xc_in = fin["xc"][:] if "xc" in fin else None
            yc_in = fin["yc"][:] if "yc" in fin else None

            pos_sel = chpos.positions_cm[keep_idx]
            for j, kept_i in enumerate(sel):
                src_row = int(keep_rows[kept_i])
                wf = wf_in[src_row:src_row + 1, :, ts:te][:, keep_idx_sorted, :]
                wf = wf[:, keep_perm, :].squeeze(0).astype(np.float32, copy=False)
                a = float(kept_activity[kept_i])
                dt_val = float(dt_in[src_row]) - float(ts) if (dt_in is not None and np.ndim(dt_in) == 1) else None
                xc_val = float(xc_in[src_row]) if (xc_in is not None and np.ndim(xc_in) == 1) else None
                yc_val = float(yc_in[src_row]) if (yc_in is not None and np.ndim(yc_in) == 1) else None

                fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

                im = axes[0].imshow(wf, aspect="auto", origin="lower", cmap="viridis")
                if dt_val is not None and np.isfinite(dt_val):
                    axes[0].axvline(dt_val, color="red", linestyle="--", linewidth=2, label="dt")
                axes[0].set_title(f"Waveform (C={n}, T={tw})\nidx={src_row}, activity={a:.2f}")
                axes[0].set_xlabel("time (cropped)")
                axes[0].set_ylabel("PMT (selected)")
                fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

                pmt_int = wf.sum(axis=1)
                sc = axes[1].scatter(pos_sel[:, 0], pos_sel[:, 1], c=pmt_int, s=50, cmap="magma")
                axes[1].scatter([target_xy[0]], [target_xy[1]], s=80, c="cyan", marker="x", linewidths=2, label="center")
                axes[1].set_aspect("equal", adjustable="box")
                axes[1].set_xlabel("x (cm)")
                axes[1].set_ylabel("y (cm)")
                axes[1].set_title("XY projection (sum over time)")
                fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.04)

                z = wf.sum(axis=0)
                axes[2].plot(np.arange(tw), z, color="black", linewidth=1.5)
                if dt_val is not None and np.isfinite(dt_val):
                    axes[2].axvline(dt_val, color="red", linestyle="--", linewidth=2)
                axes[2].set_xlabel("time (cropped)")
                axes[2].set_ylabel("sum over PMTs")
                title = "Z/time projection"
                if xc_val is not None and yc_val is not None:
                    title += f"\n(xc,yc)=({xc_val:.2f},{yc_val:.2f})"
                axes[2].set_title(title)

                fig.tight_layout()
                fig.savefig(sample_dir / f"event_{j}_src{src_row}.png", dpi=180)
                plt.close(fig)

    pos_sel_cm = chpos.positions_cm[keep_idx]
    with h5py.File(pmt_xy_out, "w") as fpos:
        ds = fpos.create_dataset("TA_PMTs_xy", data=(pos_sel_cm * float(chpos.raw_units_per_cm)).astype(np.float32))
        ds.attrs["note"] = "Subset of PMTs chosen by distance to detector-center (mean xy)."

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5.5))

    ax0.hist(activity, bins=int(activity_hist_bins), color="0.25", alpha=0.85)
    ax0.axvline(float(activity_threshold), color="crimson", linestyle="--", linewidth=2, label="threshold")
    ax0.set_xlabel("sum(waveforms) over selected PMTs + time window")
    ax0.set_ylabel("events")
    ax0.set_title(f"Activity histogram\nkept={s_out}/{s} (removed={s - s_out})")
    ax0.legend(frameon=False)

    ax1.scatter(chpos.positions_cm[:, 0], chpos.positions_cm[:, 1], s=12, c="lightgray", label="All PMTs")
    ax1.scatter(pos_sel_cm[:, 0], pos_sel_cm[:, 1], s=22, c="crimson", label=f"Selected C={n}")
    ax1.scatter([target_xy[0]], [target_xy[1]], s=90, c="dodgerblue", marker="x", linewidths=2, label="Center")
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel("x (cm)")
    ax1.set_ylabel("y (cm)")
    ax1.set_title(f"Selected PMTs + time window [{ts},{te})")
    ax1.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(plot_out, dpi=180)
    plt.close(fig)

    print(f"Wrote {tritium_out}")
    print(f"Wrote {pmt_xy_out}")
    print(f"Wrote {plot_out}")
    return tritium_out, pmt_xy_out, plot_out


def shift_waveform(waveform: np.ndarray, shift_bins: int) -> np.ndarray:
    """Shift waveform along time axis with zero-padding."""
    if waveform.ndim == 1:
        T = waveform.shape[0]
        shifted = np.zeros_like(waveform)
        if shift_bins == 0:
            return waveform.copy()
        elif shift_bins > 0:
            if shift_bins < T:
                shifted[shift_bins:] = waveform[:T - shift_bins]
        else:
            shift_bins = -shift_bins
            if shift_bins < T:
                shifted[:T - shift_bins] = waveform[shift_bins:]
        return shifted
    
    C, T = waveform.shape
    shifted = np.zeros_like(waveform)
    if shift_bins == 0:
        return waveform.copy()
    elif shift_bins > 0:
        if shift_bins < T:
            shifted[:, shift_bins:] = waveform[:, :T - shift_bins]
    else:
        shift_bins = -shift_bins
        if shift_bins < T:
            shifted[:, :T - shift_bins] = waveform[:, shift_bins:]
    return shifted


def generate_ms_events(
    ss_path: Path,
    out_path: Path,
    n_ms_events: int = 10000,
    delta_min: int = -30,
    delta_max: int = 30,
    seed: int = 42,
) -> Path:
    """Generate multi-scatter events from single-scatter data.
    
    Args:
        ss_path: Path to single-scatter H5 file
        out_path: Path for output MS H5 file
        n_ms_events: Number of MS events to generate
        delta_min, delta_max: Time shift range in bins
        seed: Random seed
    
    Returns:
        Path to output file
    """
    np.random.seed(seed)
    
    with h5py.File(ss_path, 'r') as f:
        ss_waveforms = f['waveforms'][:]
        ss_xc = f['xc'][:].astype(np.float32)
        ss_yc = f['yc'][:].astype(np.float32)
        ss_dt = f['dt'][:].astype(np.float32)
    
    n_ss = len(ss_waveforms)
    n_channels, n_time = ss_waveforms.shape[1], ss_waveforms.shape[2]
    
    print(f"Generating {n_ms_events} MS events from {n_ss} SS events...")
    print(f"  Input shape: ({n_ss}, {n_channels}, {n_time})")
    print(f"  Delta range: [{delta_min}, {delta_max}] bins = [{delta_min * NS_PER_BIN:.0f}, {delta_max * NS_PER_BIN:.0f}] ns")
    
    ms_waveforms = np.zeros((n_ms_events, n_channels, n_time), dtype=ss_waveforms.dtype)
    ms_delta_mu = np.zeros(n_ms_events, dtype=np.float32)
    ms_delta_bins = np.zeros(n_ms_events, dtype=np.int32)
    ms_idx1 = np.zeros(n_ms_events, dtype=np.int64)
    ms_idx2 = np.zeros(n_ms_events, dtype=np.int64)
    ms_xc1 = np.zeros(n_ms_events, dtype=np.float32)
    ms_yc1 = np.zeros(n_ms_events, dtype=np.float32)
    ms_dt1 = np.zeros(n_ms_events, dtype=np.float32)
    ms_xc2 = np.zeros(n_ms_events, dtype=np.float32)
    ms_yc2 = np.zeros(n_ms_events, dtype=np.float32)
    ms_dt2 = np.zeros(n_ms_events, dtype=np.float32)
    
    for i in tqdm(range(n_ms_events), desc="Generating MS events"):
        idx1 = np.random.randint(0, n_ss)
        idx2 = np.random.randint(0, n_ss)
        while idx2 == idx1:
            idx2 = np.random.randint(0, n_ss)
        
        delta_bins = np.random.randint(delta_min, delta_max + 1)
        
        wf1 = ss_waveforms[idx1]
        wf2 = ss_waveforms[idx2]
        wf2_shifted = shift_waveform(wf2, delta_bins)
        ms_wf = wf1 + wf2_shifted
        
        ms_waveforms[i] = ms_wf
        ms_delta_mu[i] = delta_bins * NS_PER_BIN
        ms_delta_bins[i] = delta_bins
        ms_idx1[i] = idx1
        ms_idx2[i] = idx2
        ms_xc1[i] = ss_xc[idx1]
        ms_yc1[i] = ss_yc[idx1]
        ms_dt1[i] = ss_dt[idx1]
        ms_xc2[i] = ss_xc[idx2]
        ms_yc2[i] = ss_yc[idx2]
        ms_dt2[i] = ss_dt[idx2]
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('waveforms', data=ms_waveforms, compression='gzip')
        f.create_dataset('delta_mu', data=ms_delta_mu, compression='gzip')
        f.create_dataset('delta_bins', data=ms_delta_bins, compression='gzip')
        f.create_dataset('idx1', data=ms_idx1, compression='gzip')
        f.create_dataset('idx2', data=ms_idx2, compression='gzip')
        f.create_dataset('xc1', data=ms_xc1, compression='gzip')
        f.create_dataset('yc1', data=ms_yc1, compression='gzip')
        f.create_dataset('dt1', data=ms_dt1, compression='gzip')
        f.create_dataset('xc2', data=ms_xc2, compression='gzip')
        f.create_dataset('yc2', data=ms_yc2, compression='gzip')
        f.create_dataset('dt2', data=ms_dt2, compression='gzip')
        f.attrs['ns_per_bin'] = NS_PER_BIN
        f.attrs['description'] = 'Multi-scatter events generated from single-scatter data'
        f.attrs['source_file'] = str(ss_path)
    
    print(f"Saved MS dataset to {out_path}")
    print(f"  Events: {n_ms_events}")
    print(f"  Output shape: {ms_waveforms.shape}")
    print(f"  Delta mu range: [{ms_delta_mu.min():.1f}, {ms_delta_mu.max():.1f}] ns")
    
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description='Compress and transform LZ/Tritium H5 data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Select 42 channels, crop time window
  python compress.py --channels 42 --time-window 500

  # Single-node mode (sum all channels into 1)
  python compress.py --channels 1

  # Generate MS events only (from existing compressed data)
  python compress.py --ms -i data/tritium_ss_c42_t500.h5 -o data/tritium_ms_c42.h5

  # Full pipeline: compress then generate MS
  python compress.py --channels 42 --time-window 500 --ms --n-ms-events 10000
        """
    )
    
    parser.add_argument('-i', '--input', type=str, default='data/tritium_ss.h5',
                        help='Input H5 file path')
    parser.add_argument('--pmt-xy', type=str, default='data/pmt_xy.h5',
                        help='Input H5 with PMT xy positions')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output file path (auto-generated if not specified)')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Output directory (defaults to input file directory)')
    
    parser.add_argument('--channels', type=int, default=None,
                        help='Number of channels to keep (1 = sum all channels; >1 = select N nearest PMTs)')
    parser.add_argument('--time-window', type=int, default=500,
                        help='Number of time points to keep')
    parser.add_argument('--time-start', type=int, default=None,
                        help='Start index for time crop (default: centered)')
    parser.add_argument('--activity-threshold', type=float, default=500.0,
                        help='Drop events with activity below this threshold')
    parser.add_argument('--activity-hist-bins', type=int, default=60,
                        help='Bins for activity histogram')
    parser.add_argument('--sample-events', type=int, default=3,
                        help='Number of sample events to plot (0 to disable)')
    parser.add_argument('--center-x', type=float, default=None,
                        help='Override center x (cm) for PMT selection')
    parser.add_argument('--center-y', type=float, default=None,
                        help='Override center y (cm) for PMT selection')
    parser.add_argument('--batch-rows', type=int, default=512,
                        help='Rows per chunk when copying waveforms')
    
    parser.add_argument('--ms', action='store_true',
                        help='Generate multi-scatter events after compression')
    parser.add_argument('--n-ms-events', type=int, default=10000,
                        help='Number of MS events to generate')
    parser.add_argument('--delta-min', type=int, default=-30,
                        help='Minimum time shift in bins for MS events')
    parser.add_argument('--delta-max', type=int, default=30,
                        help='Maximum time shift in bins for MS events')
    parser.add_argument('--ms-seed', type=int, default=42,
                        help='Random seed for MS event generation')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    out_dir = Path(args.out_dir) if args.out_dir else input_path.parent
    
    ss_output_path = None
    
    if args.channels is not None:
        if args.channels == 1:
            print("="*60)
            print("Compressing to single-node (sum all channels)")
            print("="*60)
            ss_output_path, _ = compress_to_single_node(
                tritium_in=input_path,
                out_dir=out_dir,
                time_window=args.time_window,
                time_start=args.time_start,
                batch_rows=args.batch_rows,
            )
        else:
            print("="*60)
            print(f"Selecting {args.channels} channels")
            print("="*60)
            if (args.center_x is None) ^ (args.center_y is None):
                raise ValueError("Provide both --center-x and --center-y, or neither.")
            center_xy = (float(args.center_x), float(args.center_y)) if args.center_x is not None else None
            
            ss_output_path, _, _ = compress_select_channels(
                tritium_in=input_path,
                pmt_xy_in=Path(args.pmt_xy),
                n_channels=args.channels,
                time_window=args.time_window,
                time_start=args.time_start,
                activity_threshold=args.activity_threshold,
                activity_hist_bins=args.activity_hist_bins,
                center_xy_cm=center_xy,
                sample_events=args.sample_events,
                out_dir=out_dir,
                batch_rows=args.batch_rows,
            )
    
    if args.ms:
        print("\n" + "="*60)
        print("Generating multi-scatter events")
        print("="*60)
        
        ms_input = ss_output_path if ss_output_path else input_path
        
        if args.output:
            ms_output = Path(args.output)
        else:
            stem = ms_input.stem.replace('_ss', '_ms').replace('tritium_', 'tritium_ms_')
            if '_ms' not in stem:
                stem = stem + '_ms'
            ms_output = out_dir / f"{stem}.h5"
        
        generate_ms_events(
            ss_path=ms_input,
            out_path=ms_output,
            n_ms_events=args.n_ms_events,
            delta_min=args.delta_min,
            delta_max=args.delta_max,
            seed=args.ms_seed,
        )
    
    if args.channels is None and not args.ms:
        print("No operation specified. Use --channels N to compress or --ms to generate MS events.")
        print("Run with --help for more information.")


if __name__ == '__main__':
    main()
