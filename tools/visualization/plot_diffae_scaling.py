#!/usr/bin/env python3
"""
Plot current-config model memory scaling at requested node counts.

Workflow:
- Measure feasible anchor points on the current layered graph geometry
  (fixed 42 channels, varying time bins T so N = 42 * T).
- Fit memory-vs-node-count on linear scale for DiffAE and DiffAE light.
- Fit AE memory against its exact parameter count, since under the current
  configuration AE memory is dominated by the MLP decoder weights.
- Add a hypothetical dense-adjacency DiffAE line by augmenting the sparse
  DiffAE curve with dense adjacency storage at every graph scale.

This keeps the current model configuration fixed, including latent size, while
making it possible to show scaling out to node counts that are too large to
measure directly in this environment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig"))

REPO_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_ANCHOR_TIME_POINTS = [16, 32, 64, 128, 256, 500]
DEFAULT_TARGET_NODES = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
LZ_EVENT_SIZE = 253_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--measure", action="store_true", help="Internal: run one isolated measurement.")
    p.add_argument("--model", choices=["diffae", "light", "ae"], help="Internal: model variant to measure.")
    p.add_argument("--time-point", type=int, help="Internal: number of time bins to use.")
    p.add_argument(
        "--anchor-time-points",
        type=int,
        nargs="+",
        default=DEFAULT_ANCHOR_TIME_POINTS,
        help="Empirical anchor time bins; node count is 42 x T under the current geometry.",
    )
    p.add_argument(
        "--target-nodes",
        type=int,
        nargs="+",
        default=DEFAULT_TARGET_NODES,
        help="Requested input/output node counts to show in the final plot.",
    )
    p.add_argument(
        "--output",
        default="figures/scaling/diffae_node_scaling_loglog.png",
        help="Output PNG path for the requested-node memory figure.",
    )
    p.add_argument(
        "--json-output",
        default="figures/scaling/diffae_node_scaling_results.json",
        help="Output JSON path for anchor measurements, fits, and requested-node results.",
    )
    return p.parse_args()


def _load_channel_positions(channel_positions_path: str):
    import h5py
    import numpy as np

    path = Path(channel_positions_path)
    if path.suffix not in {".h5", ".hdf5"}:
        raise ValueError(f"Unsupported channel position file: {path}")
    with h5py.File(path, "r") as f:
        if "TA_PMTs_xy" in f:
            return (f["TA_PMTs_xy"][:] / 10.0).astype(np.float32)
        if "positions" in f:
            return f["positions"][:].astype(np.float32)
        if "xy" in f:
            return f["xy"][:].astype(np.float32)
        raise ValueError(f"Could not find channel positions in {path}. Keys: {list(f.keys())}")


def _build_graph(cfg: Any, time_points: int):
    import numpy as np
    import torch
    from data_loader import create_3d_adjacency_matrix_sparse_, compute_spatial_laplacian_pe

    xy = _load_channel_positions(cfg.paths.channel_positions)
    graph = create_3d_adjacency_matrix_sparse_(
        xy,
        num_layers=time_points,
        r_within=cfg.graph.radius,
        positions_xy_profile=xy,
        z_hops=cfg.graph.z_hops,
        self_loops=True,
        z_spacing=cfg.graph.z_sep,
        weighted=cfg.graph.weighted_edges,
    )
    if cfg.graph.lpe_dim > 0:
        lpe_spatial = compute_spatial_laplacian_pe(xy, cfg.graph.radius, k=cfg.graph.lpe_dim)
        if lpe_spatial.shape[1] > 0:
            graph.lpe = torch.from_numpy(np.tile(lpe_spatial, (time_points, 1)))
    return xy, graph


def _param_count(*modules: Any) -> int:
    total = 0
    for module in modules:
        if module is None:
            continue
        total += sum(p.numel() for p in module.parameters() if p.requires_grad)
    return int(total)


def run_measurement(model: str, time_point: int) -> dict[str, Any]:
    import gc
    import threading
    import time
    import warnings

    warnings.filterwarnings("ignore")

    import psutil
    import torch

    from config import default_config as cfg
    from diffae_light import (
        LightGraphDiffusionDecoder,
        LightGraphEncoder,
        LightLatentDecoder,
        build_graph_pyramid,
    )
    from graphae import GraphAutoencoder, GraphEncoder, SimpleGraphDecoder
    from models.graph_unet import GraphDDPMUNet

    gc.collect()
    proc = psutil.Process(os.getpid())
    baseline_rss = proc.memory_info().rss
    peak_rss = {"value": baseline_rss}
    keep_sampling = {"value": True}

    def sample_peak() -> None:
        while keep_sampling["value"]:
            try:
                rss = proc.memory_info().rss
                if rss > peak_rss["value"]:
                    peak_rss["value"] = rss
            except Exception:
                pass
            time.sleep(0.01)

    sampler = threading.Thread(target=sample_peak, daemon=True)
    sampler.start()
    t0 = time.perf_counter()

    try:
        torch.manual_seed(0)
        xy, graph = _build_graph(cfg, time_point)
        n_channels = int(xy.shape[0])
        n_nodes = int(n_channels * time_point)
        n_edges = int(graph.adjacency._nnz())
        batch_size = int(cfg.training.batch_size)
        lpe_dim = 0 if graph.lpe is None else int(graph.lpe.size(1))

        x = torch.randn(batch_size * n_nodes, cfg.model.in_dim)
        z = torch.randn(batch_size, cfg.encoder.latent_dim)
        cond_full = torch.randn(batch_size, cfg.encoder.latent_dim + cfg.conditioning.time_dim)

        if model == "diffae":
            encoder = GraphEncoder(
                in_dim=cfg.model.in_dim,
                hidden_dim=cfg.encoder.hidden_dim,
                latent_dim=cfg.encoder.latent_dim,
                depth=cfg.encoder.depth,
                blocks_per_stage=cfg.encoder.blocks_per_stage,
                pool_ratio=cfg.encoder.pool_ratio,
                dropout=cfg.encoder.dropout,
                pos_dim=cfg.model.pos_dim,
                use_stochastic=cfg.encoder.use_stochastic,
                lpe_dim=lpe_dim,
            ).eval()
            decoder = GraphDDPMUNet(
                in_dim=cfg.model.in_dim,
                cond_dim=cfg.encoder.latent_dim + cfg.conditioning.time_dim,
                hidden_dim=cfg.model.hidden_dim,
                depth=cfg.model.depth,
                blocks_per_stage=cfg.model.blocks_per_stage,
                pool_ratio=cfg.model.pool_ratio,
                out_dim=cfg.model.out_dim,
                dropout=cfg.model.dropout,
                pos_dim=cfg.model.pos_dim,
                pos_dropout=cfg.model.pos_dropout,
                skip_scale=cfg.model.skip_scale,
            ).eval()
            regressive = None
            if cfg.encoder.use_regressive_head:
                regressive = SimpleGraphDecoder(
                    latent_dim=cfg.encoder.latent_dim,
                    hidden_dim=cfg.encoder.regressive_hidden_dim,
                    out_dim=cfg.model.out_dim,
                    n_nodes=n_nodes,
                    depth=cfg.encoder.depth,
                    blocks_per_stage=cfg.encoder.blocks_per_stage,
                    dropout=cfg.encoder.dropout,
                    pos_dim=cfg.model.pos_dim,
                ).eval()

            param_count = _param_count(encoder, decoder, regressive)
            with torch.inference_mode():
                encoder(x, graph.adjacency, graph.positions_xyz, batch_size=batch_size, lpe=graph.lpe)
                decoder(x, graph.adjacency, cond_full, graph.positions_xyz, batch_size=batch_size)
                if regressive is not None:
                    regressive(z, graph.adjacency, graph.positions_xyz, batch_size=batch_size)

        elif model == "light":
            decoder_pyramid = build_graph_pyramid(
                graph=graph,
                n_channels=n_channels,
                n_time_points=time_point,
                depth=cfg.model.depth,
                pool_ratio=cfg.model.pool_ratio,
                pos_dim=cfg.model.pos_dim,
                weighted_edges=cfg.graph.weighted_edges,
                device=torch.device("cpu"),
            )
            encoder_pyramid = build_graph_pyramid(
                graph=graph,
                n_channels=n_channels,
                n_time_points=time_point,
                depth=cfg.encoder.depth,
                pool_ratio=cfg.encoder.pool_ratio,
                pos_dim=cfg.model.pos_dim,
                weighted_edges=cfg.graph.weighted_edges,
                device=torch.device("cpu"),
            )

            encoder = LightGraphEncoder(
                in_dim=cfg.model.in_dim,
                hidden_dim=cfg.encoder.hidden_dim,
                latent_dim=cfg.encoder.latent_dim,
                latent_head_dim=cfg.encoder.latent_head_dim,
                anchor_dim=cfg.encoder.latent_anchor_dim,
                anchor_count=cfg.encoder.latent_anchor_count,
                anchor_value_dim=cfg.encoder.latent_anchor_value_dim,
                num_scales=len(encoder_pyramid.levels),
                blocks_per_stage=cfg.encoder.blocks_per_stage,
                dropout=cfg.encoder.dropout,
                pos_dim=cfg.model.pos_dim,
                lpe_dim=lpe_dim,
                use_stochastic=cfg.encoder.use_stochastic,
            ).eval()
            decoder = LightGraphDiffusionDecoder(
                in_dim=cfg.model.in_dim,
                out_dim=cfg.model.out_dim,
                hidden_dim=cfg.model.hidden_dim,
                cond_dim=cfg.encoder.latent_dim + cfg.conditioning.time_dim,
                num_scales=len(decoder_pyramid.levels),
                blocks_per_stage=cfg.model.blocks_per_stage,
                dropout=cfg.model.dropout,
                pos_dim=cfg.model.pos_dim,
                pos_dropout=cfg.model.pos_dropout,
                skip_scale=cfg.model.skip_scale,
                anchor_count=cfg.encoder.latent_anchor_count,
                anchor_value_dim=max(cfg.encoder.latent_anchor_value_dim, cfg.model.hidden_dim // 4),
            ).eval()
            regressive = None
            if cfg.encoder.use_regressive_head:
                regressive = LightLatentDecoder(
                    out_dim=cfg.model.out_dim,
                    hidden_dim=cfg.encoder.regressive_hidden_dim,
                    cond_dim=cfg.encoder.latent_dim,
                    num_scales=len(decoder_pyramid.levels),
                    blocks_per_stage=cfg.encoder.blocks_per_stage,
                    dropout=cfg.encoder.dropout,
                    pos_dim=cfg.model.pos_dim,
                    anchor_count=cfg.encoder.latent_anchor_count,
                    anchor_value_dim=max(
                        cfg.encoder.latent_anchor_value_dim,
                        cfg.encoder.regressive_hidden_dim // 4,
                    ),
                ).eval()

            param_count = _param_count(encoder, decoder, regressive)
            with torch.inference_mode():
                encoder(x, encoder_pyramid, batch_size=batch_size)
                decoder(x, decoder_pyramid, cond_full, batch_size=batch_size)
                if regressive is not None:
                    regressive(z, decoder_pyramid, batch_size=batch_size)

        else:
            graph_ae = GraphAutoencoder(
                in_dim=cfg.model.in_dim,
                hidden_dim=cfg.encoder.hidden_dim,
                latent_dim=cfg.encoder.latent_dim,
                n_nodes=n_nodes,
                depth=cfg.encoder.depth,
                blocks_per_stage=cfg.encoder.blocks_per_stage,
                pool_ratio=cfg.encoder.pool_ratio,
                dropout=cfg.encoder.dropout,
                pos_dim=cfg.model.pos_dim,
                out_dim=cfg.model.out_dim,
            ).eval()

            param_count = _param_count(graph_ae)
            with torch.inference_mode():
                graph_ae(x, graph.adjacency, graph.positions_xyz, batch_size=batch_size)

        runtime_s = time.perf_counter() - t0
    finally:
        keep_sampling["value"] = False
        sampler.join()

    peak_bytes = int(peak_rss["value"])
    return {
        "model": {"diffae": "DiffAE", "light": "DiffAE (light)", "ae": "AE"}[model],
        "time_points": int(time_point),
        "n_channels": int(n_channels),
        "n_nodes": int(n_nodes),
        "n_edges": int(n_edges),
        "latent_dim": int(cfg.encoder.latent_dim),
        "batch_size": int(batch_size),
        "param_count": int(param_count),
        "peak_rss_bytes": peak_bytes,
        "memory_cost_bytes": int(max(0, peak_bytes - baseline_rss)),
        "runtime_s": float(runtime_s),
    }


def _run_subprocess_measurement(model: str, time_point: int) -> dict[str, Any]:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig"))
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--measure",
        "--model",
        model,
        "--time-point",
        str(time_point),
    ]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"No measurement output for {model} @ T={time_point}")
    return json.loads(lines[-1])


def collect_results(time_points: list[int]) -> dict[str, list[dict[str, Any]]]:
    results = {"DiffAE": [], "DiffAE (light)": [], "AE": []}
    for model_key, label in [("diffae", "DiffAE"), ("light", "DiffAE (light)"), ("ae", "AE")]:
        for time_point in time_points:
            measurement = _run_subprocess_measurement(model_key, time_point)
            results[label].append(measurement)
    return results


def fit_log_slope(xs: list[float], ys: list[float]) -> float:
    import numpy as np

    x = np.log10(np.asarray(xs, dtype=float))
    y = np.log10(np.asarray(ys, dtype=float))
    return float(np.polyfit(x, y, deg=1)[0])


def format_node_tick(n: int) -> str:
    if n < 1_000:
        return str(n)
    exp = len(str(n)) - 1
    if n == 10 ** exp:
        return f"1e{exp}"
    return f"{n:,}"


def fit_linear_memory_model(anchor_results: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, float]]:
    import numpy as np

    fits: dict[str, dict[str, float]] = {}
    for label in ["DiffAE", "DiffAE (light)", "AE"]:
        series = anchor_results[label]
        xs = np.asarray([item["n_nodes"] for item in series], dtype=float)
        ys = np.asarray([item["memory_cost_bytes"] for item in series], dtype=float)
        slope, intercept = np.polyfit(xs, ys, deg=1)
        fits[label] = {
            "slope_bytes_per_node": float(slope),
            "intercept_bytes": float(intercept),
            "anchor_loglog_slope": fit_log_slope(xs.tolist(), ys.tolist()),
        }
    return fits


def evaluate_requested_nodes_sparse(
    anchor_results: dict[str, list[dict[str, Any]]],
    memory_fits: dict[str, dict[str, float]],
    target_nodes: list[int],
) -> dict[str, list[dict[str, Any]]]:
    requested: dict[str, list[dict[str, Any]]] = {}
    for label in ["DiffAE", "DiffAE (light)", "AE"]:
        series = anchor_results[label]
        first = series[0]
        slope = memory_fits[label]["slope_bytes_per_node"]
        intercept = memory_fits[label]["intercept_bytes"]
        requested[label] = []
        for n_nodes in target_nodes:
            memory_cost = max(0.0, intercept + slope * float(n_nodes))
            requested[label].append(
                {
                    "model": label,
                    "n_nodes": int(n_nodes),
                    "latent_dim": int(first["latent_dim"]),
                    "batch_size": int(first["batch_size"]),
                    "param_count": int(first["param_count"]),
                    "memory_cost_bytes": int(round(memory_cost)),
                }
            )
    return requested


def pooled_node_sizes(n_nodes: int, depth: int, pool_ratio: float) -> list[int]:
    sizes = [int(n_nodes)]
    cur = int(n_nodes)
    for _ in range(depth):
        cur = max(1, int(math.ceil(cur * pool_ratio)))
        sizes.append(cur)
    return sizes


def diffae_dense_adjacency_bytes(n_nodes: int, *, batch_size: int) -> int:
    from config import default_config as cfg

    enc_sizes = pooled_node_sizes(n_nodes, int(cfg.encoder.depth), float(cfg.encoder.pool_ratio))
    dec_sizes = pooled_node_sizes(n_nodes, int(cfg.model.depth), float(cfg.model.pool_ratio))

    total_entries = 0
    total_entries += sum((batch_size * n) ** 2 for n in enc_sizes)
    total_entries += sum((batch_size * n) ** 2 for n in dec_sizes)
    if cfg.encoder.use_regressive_head:
        total_entries += (batch_size * n_nodes) ** 2
    return int(4 * total_entries)


def evaluate_dense_diffae_requested(
    requested_results: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    dense_rows: list[dict[str, Any]] = []
    for row in requested_results["DiffAE"]:
        dense_rows.append(
            {
                **row,
                "model": "DiffAE (dense)",
                "memory_cost_bytes": int(
                    row["memory_cost_bytes"]
                    + diffae_dense_adjacency_bytes(
                        int(row["n_nodes"]),
                        batch_size=int(row["batch_size"]),
                    )
                ),
            }
        )
    return dense_rows


def plot_results(
    requested_results: dict[str, list[dict[str, Any]]],
    output_path: Path,
    *,
    current_n: int,
) -> dict[str, dict[str, float]]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from plot_style import COLORS, apply_style, compact_layout

    apply_style()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    colors = {
        "DiffAE": COLORS["diffae"],
        "DiffAE (light)": "#E8A1AA",
        "AE": COLORS["ae"],
        "DiffAE (dense)": COLORS["diffae"],
    }

    fig, ax = plt.subplots(1, 1, figsize=(7.6, 4.6))
    loglog_slopes: dict[str, dict[str, float]] = {"memory_cost_bytes": {}}

    tick_nodes = None
    for label in ["DiffAE", "DiffAE (light)", "AE", "DiffAE (dense)"]:
        series = requested_results[label]
        xs = [item["n_nodes"] for item in series]
        ys = [item["memory_cost_bytes"] for item in series]
        mem_gib = [y / (1024 ** 3) for y in ys]

        if tick_nodes is None:
            tick_nodes = xs

        ax.plot(
            xs,
            mem_gib,
            marker="o",
            linestyle="--" if label == "DiffAE (dense)" else "-",
            color=colors[label],
        )
        loglog_slopes["memory_cost_bytes"][label] = fit_log_slope(xs, ys)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Input / output size (nodes)")
    ax.set_ylabel("Memory requirement (GiB)")
    ax.set_title("Input Size vs. Memory Requirement for Model Training.")
    ax.axvline(current_n, color=COLORS["baseline"], linestyle=":", linewidth=1.2, alpha=0.9)
    if tick_nodes is not None:
        ax.set_xticks(tick_nodes)
        ax.set_xticklabels([format_node_tick(n) for n in tick_nodes])
    ax.set_xlim(left=70)

    legend_handles = [
        Line2D([0], [0], color=colors["DiffAE"], linewidth=1.5, label="DiffAE"),
        Line2D([0], [0], color=colors["DiffAE (light)"], linewidth=1.5, label="DiffAE (light)"),
        Line2D([0], [0], color=colors["DiffAE (dense)"], linewidth=1.5, linestyle="--", label="DiffAE (dense)"),
        Line2D([0], [0], color=colors["AE"], linewidth=1.5, label="AE"),
        Line2D([0], [0], color=COLORS["baseline"], linewidth=1.2, linestyle=":", label="LZ event size"),
    ]
    ax.legend(handles=legend_handles, loc="upper left")

    compact_layout(fig)
    fig.savefig(output_path)
    plt.close(fig)
    return loglog_slopes


def main() -> None:
    args = parse_args()
    if args.measure:
        if args.model is None or args.time_point is None:
            raise SystemExit("--measure requires --model and --time-point")
        print(json.dumps(run_measurement(args.model, args.time_point)))
        return

    anchor_time_points = sorted({int(tp) for tp in args.anchor_time_points if int(tp) >= 1})
    target_nodes = sorted({int(n) for n in args.target_nodes if int(n) >= 1})

    anchor_results = collect_results(anchor_time_points)
    sparse_memory_fits = fit_linear_memory_model(anchor_results)
    requested_results = evaluate_requested_nodes_sparse(anchor_results, sparse_memory_fits, target_nodes)
    requested_results["DiffAE (dense)"] = evaluate_dense_diffae_requested(requested_results)

    current_n = LZ_EVENT_SIZE
    loglog_slopes = plot_results(
        requested_results,
        Path(args.output),
        current_n=current_n,
    )

    payload = {
        "meta": {
            "anchor_time_points": anchor_time_points,
            "target_nodes": target_nodes,
            "anchor_node_formula": "N = 42 x T",
            "current_time_points": 500,
            "current_n_nodes": current_n,
        },
        "sparse_memory_fits": sparse_memory_fits,
        "requested_loglog_slopes": loglog_slopes,
        "anchor_results": anchor_results,
        "requested_results": requested_results,
    }
    json_path = Path(args.json_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2))

    print(f"Saved plot: {Path(args.output).resolve()}")
    print(f"Saved data: {json_path.resolve()}")
    for model, fit in sparse_memory_fits.items():
        print(
            f"sparse memory fit [{model}]: "
            f"bytes ~= {fit['intercept_bytes']:.3e} + {fit['slope_bytes_per_node']:.3e} * N"
        )


if __name__ == "__main__":
    main()
