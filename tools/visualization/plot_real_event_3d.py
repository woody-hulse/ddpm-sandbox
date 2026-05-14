#!/usr/bin/env python3
"""
Plot a real event as active nodes only:

- left: 3D scatter over (x, y, time) using a viridis amplitude scale
- right: 2D overhead view of active PMTs using the same color normalization

This always uses the full 253-channel SS dataset and PMT geometry:

- `data/tritium_ss.h5`
- `data/pmt_xy.h5`
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle

from config import default_config
from plot_style import apply_style, compact_layout

FULL_TRITIUM_PATH = "data/tritium_ss.h5"
FULL_PMT_PATH = "data/pmt_xy.h5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-idx", type=int, default=25,
                        help="Event index to plot.")
    parser.add_argument("--threshold-quantile", type=float, default=0.95,
                        help="Keep nodes with amplitude >= this quantile of positive amplitudes.")
    parser.add_argument("--min-amplitude", type=float, default=None,
                        help="Optional absolute lower bound; overrides the quantile if larger.")
    parser.add_argument("--ns-per-bin", type=float, default=default_config.ms_data.ns_per_bin,
                        help="Time conversion from bins to ns.")
    parser.add_argument("--gray-floor", type=float, default=1e-12,
                        help="Values below this level are rendered gray in the overhead panel.")
    parser.add_argument("--output-dir", type=str, default="figures/real_event_3d",
                        help="Directory for output figures.")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Raster dpi for saved png.")
    return parser.parse_args()


def load_pmt_positions(path: str) -> np.ndarray:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Channel positions file not found: {path}")

    if file_path.suffix in {".h5", ".hdf5"}:
        with h5py.File(file_path, "r") as f:
            if "TA_PMTs_xy" in f:
                return (f["TA_PMTs_xy"][:] / 10.0).astype(np.float32)
            if "positions" in f:
                return f["positions"][:].astype(np.float32)
            if "xy" in f:
                return f["xy"][:].astype(np.float32)
            raise ValueError(f"No recognized xy dataset in {path}; available keys: {list(f.keys())}")

    if file_path.suffix == ".npy":
        return np.load(file_path).astype(np.float32)
    if file_path.suffix == ".npz":
        data = np.load(file_path)
        key = "positions" if "positions" in data else list(data.keys())[0]
        return data[key].astype(np.float32)
    if file_path.suffix in {".txt", ".csv"}:
        return np.loadtxt(file_path, delimiter=",").astype(np.float32)

    raise ValueError(f"Unsupported channel positions format: {path}")


def load_event_waveform(path: str, event_idx: int) -> np.ndarray:
    with h5py.File(path, "r") as f:
        total = int(f["waveforms"].shape[0])
        if event_idx < 0 or event_idx >= total:
            raise IndexError(f"event_idx={event_idx} out of range for dataset with {total} events")
        return np.asarray(f["waveforms"][event_idx], dtype=np.float32)


def active_mask_from_waveform(
    waveform: np.ndarray,
    threshold_quantile: float,
    min_amplitude: float | None,
) -> tuple[np.ndarray, float]:
    if not (0.0 < threshold_quantile < 1.0):
        raise ValueError("--threshold-quantile must lie in (0, 1)")

    positive = waveform[waveform > 0]
    if positive.size == 0:
        raise ValueError("Selected event has no positive waveform entries.")

    threshold = float(np.quantile(positive, threshold_quantile))
    if min_amplitude is not None:
        threshold = max(threshold, float(min_amplitude))

    mask = waveform >= threshold
    if not np.any(mask):
        flat_idx = np.argmax(waveform)
        mask.flat[int(flat_idx)] = True
        threshold = float(waveform.flat[int(flat_idx)])
    return mask, threshold


def detector_plot_radii(pmt_xy: np.ndarray) -> tuple[float, float, float]:
    actual_radius = float(np.max(np.linalg.norm(pmt_xy, axis=1)))
    boundary_radius = actual_radius + 7.0
    plot_radius = boundary_radius + 2.0
    return actual_radius, boundary_radius, plot_radius


def plot_waveform_3d_scatter(
    ax3d: plt.Axes,
    waveform: np.ndarray,
    pmt_xy: np.ndarray,
    *,
    ns_per_bin: float,
    threshold_quantile: float = 0.95,
    min_amplitude: float | None = None,
    norm: Normalize | None = None,
    cmap=None,
    title: str | None = None,
):
    if waveform.shape[0] != pmt_xy.shape[0]:
        raise ValueError(
            f"Waveform channel count ({waveform.shape[0]}) does not match PMT positions ({pmt_xy.shape[0]})."
        )

    active_mask, threshold = active_mask_from_waveform(
        waveform,
        threshold_quantile=threshold_quantile,
        min_amplitude=min_amplitude,
    )

    ch_idx, time_idx = np.nonzero(active_mask)
    amplitudes = waveform[ch_idx, time_idx]
    times_ns = time_idx.astype(np.float32) * float(ns_per_bin)
    x = pmt_xy[ch_idx, 0]
    y = pmt_xy[ch_idx, 1]

    if cmap is None:
        cmap = plt.get_cmap("viridis")
    if norm is None:
        vmax = float(max(amplitudes.max(), waveform.max(), 1e-8))
        norm = Normalize(vmin=0.0, vmax=vmax)

    scatter = ax3d.scatter(
        x, y, times_ns,
        c=amplitudes,
        cmap=cmap,
        norm=norm,
        s=10,
        alpha=0.72,
        linewidths=0.0,
        depthshade=False,
        rasterized=True,
    )

    _, _, plot_radius = detector_plot_radii(pmt_xy)
    ax3d.set_xlim(-plot_radius, plot_radius)
    ax3d.set_ylim(-plot_radius, plot_radius)
    ax3d.set_zlim(0.0, float(waveform.shape[1] - 1) * float(ns_per_bin))
    ax3d.set_xlabel("x (cm)", labelpad=-1)
    ax3d.set_ylabel("y (cm)", labelpad=2)
    ax3d.set_zlabel("Time (ns)", labelpad=6, fontsize=10)
    ax3d.view_init(elev=22, azim=-58)
    ax3d.set_box_aspect((1.0, 1.0, 1.8))
    ax3d.grid(False)
    ax3d.tick_params(axis="x", pad=1)
    ax3d.tick_params(axis="y", pad=1)
    ax3d.tick_params(axis="z", pad=2)
    ax3d.set_zticks(np.arange(0, waveform.shape[1] * float(ns_per_bin), 2000.0))
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False
    ax3d.xaxis.pane.set_edgecolor((1, 1, 1, 0))
    ax3d.yaxis.pane.set_edgecolor((1, 1, 1, 0))
    ax3d.zaxis.pane.set_edgecolor((1, 1, 1, 0))
    if title:
        ax3d.set_title(title, pad=10, fontweight="bold")

    return {
        "scatter": scatter,
        "threshold": float(threshold),
        "active_nodes": int(active_mask.sum()),
        "active_pmts": int(np.count_nonzero(active_mask.any(axis=1))),
        "amplitude_min": float(amplitudes.min()),
        "amplitude_max": float(amplitudes.max()),
    }


def main() -> None:
    args = parse_args()

    apply_style()
    waveform = load_event_waveform(FULL_TRITIUM_PATH, args.event_idx)
    pmt_xy = load_pmt_positions(FULL_PMT_PATH)
    if waveform.shape[0] != pmt_xy.shape[0]:
        raise ValueError(
            f"Waveform channel count ({waveform.shape[0]}) does not match PMT positions ({pmt_xy.shape[0]})."
        )

    active_mask, threshold = active_mask_from_waveform(
        waveform,
        threshold_quantile=args.threshold_quantile,
        min_amplitude=args.min_amplitude,
    )

    ch_idx, time_idx = np.nonzero(active_mask)
    amplitudes = waveform[ch_idx, time_idx]
    active_channels = np.flatnonzero(active_mask.any(axis=1))
    channel_peak = waveform.max(axis=1)
    vmax = float(max(amplitudes.max(), channel_peak.max()))
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.get_cmap("viridis")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"real_event_3d_event{args.event_idx:05d}"
    out_png = output_dir / f"{stem}.png"
    out_pdf = output_dir / f"{stem}.pdf"

    fig = plt.figure(figsize=(7.4, 6.2))
    gs = GridSpec(
        1, 2, figure=fig,
        width_ratios=[0.78, 1.0],
        left=0.06, right=0.90, bottom=0.08, top=0.89, wspace=0.18,
    )
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_xy = fig.add_subplot(gs[0, 1])

    panel_info = plot_waveform_3d_scatter(
        ax3d,
        waveform,
        pmt_xy,
        ns_per_bin=float(args.ns_per_bin),
        threshold_quantile=args.threshold_quantile,
        min_amplitude=args.min_amplitude,
        norm=norm,
        cmap=cmap,
    )
    sc3d = panel_info["scatter"]
    _, boundary_radius, plot_radius = detector_plot_radii(pmt_xy)

    ax_xy.scatter(
        pmt_xy[:, 0],
        pmt_xy[:, 1],
        s=78,
        c="#D3DAE0",
        alpha=1.0,
        linewidths=0.0,
        zorder=2,
    )
    colored_channels = np.flatnonzero(channel_peak >= float(args.gray_floor))
    sc2d = ax_xy.scatter(
        pmt_xy[colored_channels, 0],
        pmt_xy[colored_channels, 1],
        c=channel_peak[colored_channels],
        cmap=cmap,
        norm=norm,
        s=78,
        alpha=1.0,
        linewidths=0.0,
        zorder=3,
    )
    boundary = Circle((0.0, 0.0), boundary_radius, fill=False,
                      edgecolor="#D3DAE0", linewidth=1.0, alpha=0.9, zorder=1)
    ax_xy.add_patch(boundary)
    ax_xy.set_aspect("equal")
    ax_xy.set_xlim(-plot_radius, plot_radius)
    ax_xy.set_ylim(-plot_radius, plot_radius)
    ax_xy.set_xlabel("x (cm)")
    ax_xy.set_ylabel("y (cm)", labelpad=-6)
    ax_xy.grid(False)
    ax_xy.tick_params(axis="both", pad=2)

    bbox3d = ax3d.get_position()
    bbox_xy = ax_xy.get_position()
    title_y = max(bbox3d.y1, bbox_xy.y1) + 0.003
    fig.text(
        0.5 * (bbox3d.x0 + bbox3d.x1),
        title_y,
        "Sample event",
        ha="center",
        va="bottom",
        fontsize=11,
    )
    fig.text(
        0.5 * (bbox_xy.x0 + bbox_xy.x1),
        title_y,
        "Overhead view",
        ha="center",
        va="bottom",
        fontsize=11,
    )

    cbar = fig.colorbar(sc3d, ax=[ax3d, ax_xy], fraction=0.022, pad=0.012, shrink=0.88)
    cbar.set_label("Amplitude (AU)")

    compact_layout(fig, rect=(0.0, 0.0, 1.0, 0.985), pad=0.25, h_pad=0.25, w_pad=0.25)

    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")
    print(f"Event index: {args.event_idx}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Active 3D nodes: {panel_info['active_nodes']}")
    print(f"Active PMTs: {panel_info['active_pmts']}")
    print(f"Amplitude range: [{panel_info['amplitude_min']:.4f}, {panel_info['amplitude_max']:.4f}]")


if __name__ == "__main__":
    main()
