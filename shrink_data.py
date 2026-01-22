import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import matplotlib.pyplot as plt


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


def shrink_lz_data(
    *,
    tritium_in: Path,
    pmt_xy_in: Path,
    n: int = 42,
    time_window: int = 500,
    time_start: Optional[int] = None,
    activity_threshold: float = 0.0,
    activity_hist_bins: int = 60,
    center_xy_cm: Optional[Tuple[float, float]] = None,
    sample_events: int = 3,
    out_dir: Optional[Path] = None,
    batch_rows: int = 512,
) -> Tuple[Path, Path, Path]:
    tritium_in = Path(tritium_in)
    pmt_xy_in = Path(pmt_xy_in)
    out_dir = Path(out_dir) if out_dir is not None else tritium_in.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    tritium_out = out_dir / f"tritium_ss_{int(n)}.h5"
    pmt_xy_out = out_dir / f"pmt_xy_{int(n)}.h5"
    plot_out = out_dir / f"selected_pmts_{int(n)}_t{int(time_window)}.png"

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
        keep_idx = _select_pmts_near_point(chpos.positions_cm, target_xy, int(n))
        # h5py requires fancy indices to be increasing; keep output order as "closest first"
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
            fout.attrs["pmt_subset_n"] = int(n)
            fout.attrs["time_start"] = int(ts)
            fout.attrs["time_end"] = int(te)
            fout.attrs["activity_threshold"] = float(thr)

            wf_out = _create_like(wf_in, fout, "waveforms", shape=(s_out, int(n), tw))
            _copy_attrs(wf_in, wf_out)

            for out_start in range(0, s_out, int(batch_rows)):
                out_end = min(s_out, out_start + int(batch_rows))
                rows = keep_rows[out_start:out_end]  # increasing
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

            # If dt exists and is per-sample, shift into the new window coordinates
            if ("dt" in fout) and (fout["dt"].ndim == 1) and (fout["dt"].shape[0] == s_out):
                fout["dt"][:] = np.asarray(fout["dt"][:], dtype=np.float32) - float(ts)

        # Sample event plots (from kept events) in a style similar to ddpm_sparse.py
        k = int(sample_events)
        if k > 0 and s_out > 0:
            sample_dir = out_dir / f"sample_events_{int(n)}_t{int(tw)}"
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
                wf = wf[:, keep_perm, :].squeeze(0).astype(np.float32, copy=False)  # (N,tw)
                a = float(kept_activity[kept_i])
                dt_val = float(dt_in[src_row]) - float(ts) if (dt_in is not None and np.ndim(dt_in) == 1) else None
                xc_val = float(xc_in[src_row]) if (xc_in is not None and np.ndim(xc_in) == 1) else None
                yc_val = float(yc_in[src_row]) if (yc_in is not None and np.ndim(yc_in) == 1) else None

                fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

                im = axes[0].imshow(wf, aspect="auto", origin="lower", cmap="viridis")
                if dt_val is not None and np.isfinite(dt_val):
                    axes[0].axvline(dt_val, color="red", linestyle="--", linewidth=2, label="dt")
                axes[0].set_title(f"Waveform (N={int(n)}, T={int(tw)})\nidx={src_row}, activity={a:.2f}")
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
                axes[2].plot(np.arange(int(tw)), z, color="black", linewidth=1.5)
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
        ds.attrs["note"] = "Subset of PMTs chosen by distance to detector-center (mean xy); stored in TA_PMTs_xy units."

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5.5))

    # Left: activity histogram in the cropped region, with threshold
    ax0.hist(activity, bins=int(activity_hist_bins), color="0.25", alpha=0.85)
    ax0.axvline(float(activity_threshold), color="crimson", linestyle="--", linewidth=2, label="threshold")
    ax0.set_xlabel("sum(waveforms) over selected PMTs + time window")
    ax0.set_ylabel("events")
    ax0.set_title(f"Activity histogram\nkept={int(s_out)}/{int(s)} (removed={int(s - s_out)})")
    ax0.legend(frameon=False)

    # Right: PMT positions with selected PMTs and the chosen center point overlaid
    ax1.scatter(chpos.positions_cm[:, 0], chpos.positions_cm[:, 1], s=12, c="lightgray", label="All PMTs")
    ax1.scatter(pos_sel_cm[:, 0], pos_sel_cm[:, 1], s=22, c="crimson", label=f"Selected N={int(n)}")
    ax1.scatter([target_xy[0]], [target_xy[1]], s=90, c="dodgerblue", marker="x", linewidths=2, label="Center")
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel("x (cm)")
    ax1.set_ylabel("y (cm)")
    ax1.set_title(f"Selected PMTs + time window [{ts},{te})")
    ax1.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(plot_out, dpi=180)
    plt.close(fig)

    return tritium_out, pmt_xy_out, plot_out


def shrink_to_single_node(
    *,
    tritium_in: Path,
    out_dir: Optional[Path] = None,
    batch_rows: int = 512,
) -> Path:
    """Shrink waveforms by collapsing the spatial (channel) dimension.

    Input shape: (S, C, T) where C = channels (spatial), T = time.
    Output shape: (S, 1, T) where each time bin is the sum over all channels.

    This produces a single-node graph per layer (time step) representing the
    total XY cross-section at that layer. No event pruning is applied.

    Returns the path to the output H5 file.
    """
    tritium_in = Path(tritium_in)
    out_dir = Path(out_dir) if out_dir is not None else tritium_in.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    tritium_out = out_dir / f"{tritium_in.stem}_single_node.h5"

    with h5py.File(tritium_in, "r") as fin:
        if "waveforms" not in fin:
            raise ValueError(f"{tritium_in} must contain a waveforms dataset.")
        wf_in = fin["waveforms"]
        if wf_in.ndim != 3:
            raise ValueError(f"Expected waveforms to be (S,C,T); got shape={wf_in.shape}")
        s, c, t = map(int, wf_in.shape)

        with h5py.File(tritium_out, "w") as fout:
            _copy_attrs(fin, fout)
            fout.attrs["shrink_mode"] = "single_node"
            fout.attrs["original_channels"] = int(c)
            fout.attrs["original_time"] = int(t)

            wf_out = fout.create_dataset(
                "waveforms",
                shape=(s, 1, t),
                dtype=np.float32,
                chunks=(min(batch_rows, s), 1, t),
                compression=wf_in.compression,
                compression_opts=wf_in.compression_opts,
            )
            _copy_attrs(wf_in, wf_out)

            for start in range(0, s, int(batch_rows)):
                end = min(s, start + int(batch_rows))
                block = np.asarray(wf_in[start:end], dtype=np.float64)
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

    print(f"Shrunk {s} events from ({c},{t}) to single-node ({1},{t}). Wrote {tritium_out}")
    return tritium_out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reduce LZ/Tritium SS H5 data to N center PMTs and a middle time window, with activity filtering.")
    p.add_argument("--tritium-in", type=str, default="data/tritium_ss.h5", help="Input H5 with waveforms/xc/yc/dt.")
    p.add_argument("--pmt-xy-in", type=str, default="data/pmt_xy.h5", help="Input H5 with PMT xy positions.")
    p.add_argument("-N", "--n", type=int, default=42, help="Number of PMTs to keep.")
    p.add_argument("--time-window", type=int, default=500, help="Number of time points to keep (middle window by default).")
    p.add_argument("--time-start", type=int, default=None, help="Start index for time crop; defaults to centered window.")
    p.add_argument("--activity-threshold", type=float, default=500.0, help="Drop events with summed activity below this threshold (in cropped region).")
    p.add_argument("--activity-hist-bins", type=int, default=60, help="Bins for the saved activity histogram.")
    p.add_argument("--sample-events", type=int, default=3, help="Number of kept events to plot (0 disables).")
    p.add_argument("--center-x", type=float, default=None, help="Override center x (cm) for PMT selection. Default: mean PMT x.")
    p.add_argument("--center-y", type=float, default=None, help="Override center y (cm) for PMT selection. Default: mean PMT y.")
    p.add_argument("--out-dir", type=str, default=None, help="Output directory (defaults to input tritium dir).")
    p.add_argument("--batch-rows", type=int, default=512, help="Rows per chunk when copying waveforms.")
    p.add_argument("--single-node", action="store_true", help="Shrink to single-node graph (sum all channels and time).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.single_node:
        tritium_out = shrink_to_single_node(
            tritium_in=Path(args.tritium_in),
            out_dir=Path(args.out_dir) if args.out_dir else None,
            batch_rows=int(args.batch_rows),
        )
        print(f"Wrote {tritium_out}")
        return

    if (args.center_x is None) ^ (args.center_y is None):
        raise ValueError("Provide both --center-x and --center-y, or neither.")
    center_xy = (float(args.center_x), float(args.center_y)) if args.center_x is not None else None
    tritium_out, pmt_xy_out, plot_out = shrink_lz_data(
        tritium_in=Path(args.tritium_in),
        pmt_xy_in=Path(args.pmt_xy_in),
        n=int(args.n),
        time_window=int(args.time_window),
        time_start=int(args.time_start) if args.time_start is not None else None,
        activity_threshold=float(args.activity_threshold),
        activity_hist_bins=int(args.activity_hist_bins),
        center_xy_cm=center_xy,
        sample_events=int(args.sample_events),
        out_dir=Path(args.out_dir) if args.out_dir else None,
        batch_rows=int(args.batch_rows),
    )
    print(f"Wrote {tritium_out}")
    print(f"Wrote {pmt_xy_out}")
    print(f"Wrote {plot_out}")


if __name__ == "__main__":
    main()

