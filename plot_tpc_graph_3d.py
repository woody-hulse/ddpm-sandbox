#!/usr/bin/env python3
"""
Create a paper-ready visualization of the 3D TPC graph for the full 253-PMT array.

The figure is designed for readability rather than brute-force rendering of every
edge in the full 253 x 1000 graph. It shows:

- a clean 3D view of the graph over the full z extent,
- a matching overhead XY view of a single layer,
- a restrained highlight of one representative local neighborhood so the
  graph structure stays visible.

Usage:
    python plot_tpc_graph_3d.py
    python plot_tpc_graph_3d.py --num-z 1000 --radius 15 --z-hops 5
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import h5py
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from config import default_config
from plot_style import COLORS, apply_style


FULL_GEOMETRY_PATH = "data/pmt_xy.h5"
CONTEXT_NODE = "#B7C0C8"
CONTEXT_EDGE = "#D3DAE0"
LAYER_NODE = "#7E8C98"
LAYER_EDGE = "#A8B4BF"
HOP_EDGE = COLORS["diffae"]
SPATIAL_HIGHLIGHT = COLORS["ae"]
FOCUS_NODE = COLORS["truth"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel-positions", type=str, default=FULL_GEOMETRY_PATH,
                        help="Path to full PMT xy coordinates (defaults to the 253-PMT array).")
    parser.add_argument("--radius", type=float, default=default_config.graph.radius,
                        help="Within-layer radius connectivity in cm.")
    parser.add_argument("--z-hops", type=int, default=default_config.graph.z_hops,
                        help="Number of connected z hops above/below each node.")
    parser.add_argument("--z-sep", type=float, default=default_config.graph.z_sep,
                        help="Configured graph z spacing (reported in annotations).")
    parser.add_argument("--num-z", type=int, default=1000,
                        help="Number of z layers / time bins in the plotted graph.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Seed for the highlighted random node.")
    parser.add_argument("--focus-stride", type=int, default=1,
                        help="Subsample highlighted focus-neighborhood edges by hop layer to reduce clutter.")
    parser.add_argument("--time-min-ns", type=float, default=4750.0,
                        help="Lower bound of the displayed time window in ns.")
    parser.add_argument("--time-max-ns", type=float, default=5250.0,
                        help="Upper bound of the displayed time window in ns.")
    parser.add_argument("--output-dir", type=str, default="plots",
                        help="Directory for output files.")
    parser.add_argument("--dpi", type=int, default=300, help="Raster dpi for saved outputs.")
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


def build_radius_graph(xy: np.ndarray, radius: float) -> np.ndarray:
    diff = xy[:, None, :] - xy[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    adjacency = (dist2 <= radius * radius).astype(np.uint8)
    np.fill_diagonal(adjacency, 0)
    adjacency = np.maximum(adjacency, adjacency.T)
    return adjacency


def undirected_edges(adjacency: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.where(np.triu(adjacency, k=1) > 0)
    return rows.astype(np.int32), cols.astype(np.int32)


def nearest_neighbor_stats(xy: np.ndarray) -> tuple[float, float, float]:
    diff = xy[:, None, :] - xy[None, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(distances, np.inf)
    nn = distances.min(axis=1)
    return float(nn.min()), float(np.median(nn)), float(nn.max())


def add_segments(ax, segments: np.ndarray, *, color: str, alpha: float, linewidth: float) -> None:
    if segments.size == 0:
        return
    collection = Line3DCollection(segments, colors=color, linewidths=linewidth, alpha=alpha)
    ax.add_collection3d(collection)


def make_edge_segments(
    xy: np.ndarray,
    z_value: float,
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    start = np.column_stack([xy[rows], np.full(rows.shape[0], z_value, dtype=np.float32)])
    end = np.column_stack([xy[cols], np.full(cols.shape[0], z_value, dtype=np.float32)])
    return np.stack([start, end], axis=1)


def iter_cross_segments(
    xy: np.ndarray,
    focus_idx: int,
    neighbor_idx: np.ndarray,
    center_layer: int,
    z_hops: int,
    stride: int,
) -> Iterable[tuple[np.ndarray, float]]:
    base = xy[focus_idx]
    cross_targets = np.concatenate([[focus_idx], neighbor_idx])
    cross_targets = cross_targets[::max(1, stride)]

    for hop in range(1, z_hops + 1):
        alpha = 0.34 - 0.04 * (hop - 1)
        for sign in (-1, 1):
            target_layer = center_layer + sign * hop
            target_xyz = np.column_stack([
                xy[cross_targets, 0],
                xy[cross_targets, 1],
                np.full(cross_targets.shape[0], target_layer, dtype=np.float32),
            ])
            start_xyz = np.column_stack([
                np.full(cross_targets.shape[0], base[0], dtype=np.float32),
                np.full(cross_targets.shape[0], base[1], dtype=np.float32),
                np.full(cross_targets.shape[0], center_layer, dtype=np.float32),
            ])
            yield np.stack([start_xyz, target_xyz], axis=1), alpha


def configure_3d_axis(
    ax,
    xy: np.ndarray,
    time_min_ns: float,
    time_max_ns: float,
) -> None:
    radius = 1.08 * float(np.max(np.linalg.norm(xy, axis=1)))
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_zlim(time_min_ns, time_max_ns)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_zlabel("Time (ns)")
    ax.view_init(elev=22, azim=-58)
    ax.set_box_aspect((1.0, 1.0, 0.7))
    ax.grid(False)
    z_ticks = np.linspace(time_min_ns, time_max_ns, num=5)
    ax.set_zticks(z_ticks)
    ax.set_zticklabels([f"{int(t):d}" for t in z_ticks])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor((1, 1, 1, 0))
    ax.yaxis.pane.set_edgecolor((1, 1, 1, 0))
    ax.zaxis.pane.set_edgecolor((1, 1, 1, 0))


def main() -> None:
    args = parse_args()
    if args.num_z < 2:
        raise ValueError("--num-z must be at least 2")
    if args.z_hops < 1:
        raise ValueError("--z-hops must be at least 1")
    if args.time_max_ns <= args.time_min_ns:
        raise ValueError("--time-max-ns must be greater than --time-min-ns")

    apply_style()
    xy = load_pmt_positions(args.channel_positions)
    adjacency = build_radius_graph(xy, args.radius)
    edge_r, edge_c = undirected_edges(adjacency)

    n_pmts = int(xy.shape[0])
    n_nodes = n_pmts * args.num_z
    degrees = adjacency.sum(axis=1).astype(int)
    rng = np.random.default_rng(args.seed)
    focus_idx = int(rng.integers(0, n_pmts))
    focus_layer = int(rng.integers(0, args.num_z))
    focus_neighbors = np.flatnonzero(adjacency[focus_idx] > 0).astype(np.int32)
    layer_edges_directed = int(adjacency.sum())
    cross_edges_directed = layer_edges_directed + n_pmts
    total_nnz = (
        args.num_z * layer_edges_directed
        + 2 * sum((args.num_z - hop) * cross_edges_directed for hop in range(1, args.z_hops + 1))
        + args.num_z * n_pmts
    )
    nn_min, nn_med, nn_max = nearest_neighbor_stats(xy)
    detector_radius = float(np.max(np.linalg.norm(xy, axis=1)))

    ns_per_bin = float(default_config.ms_data.ns_per_bin)
    time_values = np.arange(args.num_z, dtype=np.float32) * ns_per_bin
    layer_mask = (time_values >= args.time_min_ns) & (time_values <= args.time_max_ns)
    visible_layers = np.flatnonzero(layer_mask).astype(np.int32)
    if visible_layers.size == 0:
        raise ValueError("Requested time window does not overlap the available graph.")
    interior_pmts = np.flatnonzero(np.linalg.norm(xy, axis=1) <= 0.72 * detector_radius)
    if interior_pmts.size > 0:
        focus_idx = int(interior_pmts[rng.integers(0, interior_pmts.size)])
        focus_neighbors = np.flatnonzero(adjacency[focus_idx] > 0).astype(np.int32)
    if focus_layer < visible_layers[0] or focus_layer > visible_layers[-1]:
        focus_layer = int(visible_layers[len(visible_layers) // 2])
    focus_time = float(time_values[focus_layer])
    focus_layers = np.arange(focus_layer - args.z_hops, focus_layer + args.z_hops + 1, dtype=np.int32)
    focus_layers = focus_layers[(focus_layers >= 0) & (focus_layers < args.num_z)]
    focus_times = time_values[focus_layers]
    focus_cross_targets = np.concatenate([[focus_idx], focus_neighbors])[::max(1, args.focus_stride)]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"tpc_graph_3d_253pmt_z{args.num_z}"
    out_png = output_dir / f"{stem}.png"
    out_pdf = output_dir / f"{stem}.pdf"

    fig = plt.figure(figsize=(8.8, 4.5))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.65, 1.0], left=0.055, right=0.985, bottom=0.16, top=0.88, wspace=0.08)
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_xy = fig.add_subplot(gs[0, 1])

    configure_3d_axis(ax3d, xy, args.time_min_ns, args.time_max_ns)

    focus_xy = xy[focus_idx]
    visible_times = time_values[visible_layers]
    ax3d.scatter(
        np.tile(xy[:, 0], visible_layers.size),
        np.tile(xy[:, 1], visible_layers.size),
        np.repeat(visible_times, n_pmts),
        s=0.14,
        c=CONTEXT_NODE,
        alpha=0.075,
        depthshade=False,
        rasterized=True,
        zorder=1,
    )

    visible_edge_segments = np.concatenate(
        [make_edge_segments(xy, float(time_values[layer]), edge_r, edge_c) for layer in visible_layers],
        axis=0,
    )
    add_segments(ax3d, visible_edge_segments, color=LAYER_EDGE, alpha=0.12, linewidth=0.45)

    # Local spatial stencil at the highlighted node.
    same_layer_segments = np.stack([
        np.column_stack([
            np.full(focus_neighbors.shape[0], focus_xy[0], dtype=np.float32),
            np.full(focus_neighbors.shape[0], focus_xy[1], dtype=np.float32),
            np.full(focus_neighbors.shape[0], focus_time, dtype=np.float32),
        ]),
        np.column_stack([
            xy[focus_neighbors, 0],
            xy[focus_neighbors, 1],
            np.full(focus_neighbors.shape[0], focus_time, dtype=np.float32),
        ]),
    ], axis=1)
    add_segments(ax3d, same_layer_segments, color=SPATIAL_HIGHLIGHT, alpha=0.85, linewidth=1.55)

    for segments, alpha in iter_cross_segments(
        xy, focus_idx, focus_neighbors, focus_layer, args.z_hops, args.focus_stride
    ):
        segments[:, :, 2] *= ns_per_bin
        add_segments(ax3d, segments, color=HOP_EDGE, alpha=min(0.92, alpha + 0.18), linewidth=1.45)

    for layer_time in focus_times:
        ax3d.scatter(
            focus_xy[0],
            focus_xy[1],
            float(layer_time),
            s=16 if layer_time == focus_time else 10,
            c=FOCUS_NODE if layer_time == focus_time else HOP_EDGE,
            alpha=0.98 if layer_time == focus_time else 0.80,
            depthshade=False,
            zorder=6,
        )

    non_focus_times = focus_times[np.abs(focus_times - focus_time) > 1e-6]
    if non_focus_times.size > 0:
        cross_x = np.tile(xy[focus_cross_targets, 0], non_focus_times.size)
        cross_y = np.tile(xy[focus_cross_targets, 1], non_focus_times.size)
        cross_z = np.repeat(non_focus_times, focus_cross_targets.size)
        ax3d.scatter(
            cross_x,
            cross_y,
            cross_z,
            s=9,
            c=HOP_EDGE,
            alpha=0.72,
            depthshade=False,
            zorder=5,
        )

    ax3d.scatter(
        xy[focus_neighbors, 0],
        xy[focus_neighbors, 1],
        np.full(focus_neighbors.shape[0], focus_time, dtype=np.float32),
        s=14,
        c=SPATIAL_HIGHLIGHT,
        alpha=0.95,
        depthshade=False,
        zorder=5,
    )

    handles = [
        Line2D([0], [0], color=SPATIAL_HIGHLIGHT, lw=1.6, label="same-layer neighbors"),
        Line2D([0], [0], color=HOP_EDGE, lw=1.2, label="cross-layer edges"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=FOCUS_NODE, markersize=6, label="highlighted node"),
    ]
    ax3d.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.00, -0.14))

    # Overhead view.
    for r, c in zip(edge_r, edge_c):
        ax_xy.plot([xy[r, 0], xy[c, 0]], [xy[r, 1], xy[c, 1]],
                   color=LAYER_EDGE, alpha=0.30, linewidth=0.60, zorder=1)
    for neigh in focus_neighbors:
        ax_xy.plot([focus_xy[0], xy[neigh, 0]], [focus_xy[1], xy[neigh, 1]],
                   color=SPATIAL_HIGHLIGHT, alpha=0.90, linewidth=1.25, zorder=3)
    ax_xy.scatter(xy[:, 0], xy[:, 1], s=16, c=CONTEXT_NODE, alpha=0.78, linewidths=0.0, zorder=2)
    ax_xy.scatter(xy[focus_neighbors, 0], xy[focus_neighbors, 1], s=28,
                  c=SPATIAL_HIGHLIGHT, alpha=0.98, linewidths=0.0, zorder=4)
    ax_xy.scatter([focus_xy[0]], [focus_xy[1]], s=34,
                  c=FOCUS_NODE, alpha=0.98, linewidths=0.0, zorder=5)
    overhead_radius = 1.05 * detector_radius
    ax_xy.set_xlim(-overhead_radius, overhead_radius)
    ax_xy.set_ylim(-overhead_radius, overhead_radius)
    ax_xy.set_aspect("equal")
    ax_xy.set_xlabel("x (cm)")
    ax_xy.set_ylabel("y (cm)")
    ax_xy.grid(False)

    bbox3d = ax3d.get_position()
    bbox_xy = ax_xy.get_position()
    title_y = max(bbox3d.y1, bbox_xy.y1) + 0.008
    fig.text(
        0.5 * (bbox3d.x0 + bbox3d.x1),
        title_y,
        f"TPC Subgraph: {int(args.time_min_ns):,}ns - {int(args.time_max_ns):,}ns",
        ha="center",
        va="bottom",
        fontsize=11,
    )
    fig.text(
        0.5 * (bbox_xy.x0 + bbox_xy.x1),
        title_y,
        f"Overhead View at {int(round(focus_time)):,}ns",
        ha="center",
        va="bottom",
        fontsize=11,
    )

    fig.savefig(out_png, dpi=args.dpi)
    fig.savefig(out_pdf)
    plt.close(fig)

    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")
    print(f"PMTs: {n_pmts}, z bins: {args.num_z}, nodes: {n_nodes}")
    print(f"Within-layer radius degree min/median/max: {degrees.min()}/{int(np.median(degrees))}/{degrees.max()}")
    print(f"Nearest-neighbour spacing min/median/max: {nn_min:.2f}/{nn_med:.2f}/{nn_max:.2f} cm")
    print(f"Approximate sparse adjacency nnz (including self-loops): {total_nnz}")


if __name__ == "__main__":
    main()
