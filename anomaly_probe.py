#!/usr/bin/env python3
"""
anomaly_probe.py — Three scatter plots comparing anomaly separability.

For each of three spaces (RQ metrics, AE latent, DiffAE latent) the figure
shows:
  • N real SS events  — small grey dots (background distribution)
  • 1 prototype per anomaly type — large coloured markers, labelled

The anomaly set is intentionally subtle. Instead of using grossly altered
events that every representation can reject, these prototypes preserve most
low-order marginals while violating joint spatial-temporal coherence patterns
that a generative latent space should model more explicitly.

Two anomaly families:

  SPATIAL  — preserve the summed z-profile but perturb the charge pattern in a
             locally inconsistent way.
    spatial_residual_boost, azimuthal_charge_roll

  TEMPORAL — preserve total charge and much of the global envelope, but break
             channel-to-channel timing consistency.
    radial_time_shear, sector_time_split, subset_late_charge
"""

import argparse
import copy
import glob
import os

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from config import default_config
from diffae import DiffAEContext
from graphae import GraphAEContext
from compare_rqs import collect_rqs, compute_rqs, wf_to_z_profile
from plot_style import apply_style, COLORS, compact_layout

apply_style()

# ---------------------------------------------------------------------------
# Anomaly registry
# ---------------------------------------------------------------------------

ANOMALY_TYPES = [
    # (key,                       display label,            colour,    family)
    ("spatial_residual_boost",    "Spatial residual boost", "#2166AC", "spatial"),
    ("azimuthal_charge_roll",     "Azimuthal charge roll",  "#D6604D", "spatial"),
    ("radial_time_shear",         "Radial time shear",      "#1A9850", "temporal"),
    ("sector_time_split",         "Sector time split",      "#762A83", "temporal"),
    ("subset_late_charge",        "Subset late charge",     "#B35806", "temporal"),
]

ANOMALY_LABELS = {t[0]: t[1] for t in ANOMALY_TYPES}
ANOMALY_COLORS = {t[0]: t[2] for t in ANOMALY_TYPES}
ANOMALY_FAMILY = {t[0]: t[3] for t in ANOMALY_TYPES}
SPATIAL_TYPES  = [t[0] for t in ANOMALY_TYPES if t[3] == "spatial"]
TEMPORAL_TYPES = [t[0] for t in ANOMALY_TYPES if t[3] == "temporal"]

PLOT_DPI = 300


# ---------------------------------------------------------------------------
# Subtle anomaly constructors
# ---------------------------------------------------------------------------

def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapz(y, x))


def _channel_weights(wf_ct: np.ndarray) -> np.ndarray:
    ch = wf_ct.sum(axis=1).astype(np.float64)
    total = float(ch.sum())
    if total <= 1e-8:
        return np.full(wf_ct.shape[0], 1.0 / max(wf_ct.shape[0], 1), dtype=np.float32)
    return (ch / total).astype(np.float32)


def _renormalize_positive(values: np.ndarray, target_sum: float) -> np.ndarray:
    out = np.clip(np.asarray(values, dtype=np.float64), 0.0, None)
    cur = float(out.sum())
    if cur > 1e-8 and target_sum > 0.0:
        out *= target_sum / cur
    return out.astype(np.float32)


def _neighbor_average(values: np.ndarray, channel_positions: np.ndarray, k: int = 4) -> np.ndarray:
    if len(values) <= 1:
        return values.astype(np.float32)
    diff = channel_positions[:, None, :] - channel_positions[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    order = np.argsort(dist, axis=1)
    k_eff = min(max(1, k), len(values) - 1)
    nbr_idx = order[:, 1 : 1 + k_eff]
    return values[nbr_idx].mean(axis=1).astype(np.float32)


def _shift_trace(trace: np.ndarray, shift_bins: float) -> np.ndarray:
    t = np.arange(trace.shape[0], dtype=np.float64)
    shifted = np.interp(t - float(shift_bins), t, trace.astype(np.float64), left=0.0, right=0.0)
    target = float(np.sum(trace))
    cur = float(np.sum(shifted))
    if cur > 1e-8 and target > 0.0:
        shifted *= target / cur
    return shifted.astype(np.float32)


def _event_centered_radii(channel_positions: np.ndarray) -> np.ndarray:
    center = channel_positions.mean(axis=0)
    return np.linalg.norm(channel_positions - center[None, :], axis=1).astype(np.float32)


def _event_angles(channel_positions: np.ndarray) -> np.ndarray:
    center = channel_positions.mean(axis=0)
    return np.arctan2(channel_positions[:, 1] - center[1], channel_positions[:, 0] - center[0]).astype(np.float32)


def compute_spatial_weights(
    channel_positions: np.ndarray,
    mode: str,
    xc: float = 0.0,
    yc: float = 0.0,
) -> np.ndarray:
    del xc, yc
    C = len(channel_positions)
    dummy = np.ones(C, dtype=np.float32) / max(C, 1)

    if mode == "spatial_residual_boost":
        local = _neighbor_average(dummy, channel_positions, k=min(4, C - 1))
        w = dummy + 0.35 * (dummy - local)

    elif mode == "azimuthal_charge_roll":
        angles = _event_angles(channel_positions)
        idx = np.argsort(angles)
        rolled = np.roll(dummy[idx], max(2, C // 10))
        w = dummy.copy()
        w[idx] = 0.7 * dummy[idx] + 0.3 * rolled

    else:
        raise ValueError(f"Unknown spatial mode: {mode!r}")

    s = float(w.sum())
    return (w / s).astype(np.float32) if s > 1e-8 else dummy


def make_spatial_anomaly(
    wf_ct: np.ndarray,
    channel_positions: np.ndarray,
    mode: str,
    xc: float = 0.0,
    yc: float = 0.0,
) -> np.ndarray:
    del xc, yc
    z = wf_ct.sum(axis=0).astype(np.float32)
    ch_real = _channel_weights(wf_ct)

    if mode == "spatial_residual_boost":
        local = _neighbor_average(ch_real, channel_positions, k=min(4, len(ch_real) - 1))
        residual = ch_real - local
        w = _renormalize_positive(ch_real + 0.55 * residual, target_sum=1.0)

    elif mode == "azimuthal_charge_roll":
        angles = _event_angles(channel_positions)
        idx = np.argsort(angles)
        rolled = np.roll(ch_real[idx], max(2, len(ch_real) // 10))
        w = ch_real.copy()
        w[idx] = 0.68 * ch_real[idx] + 0.32 * rolled
        w = _renormalize_positive(w, target_sum=1.0)

    else:
        w_target = compute_spatial_weights(channel_positions, mode)
        w = _renormalize_positive(0.75 * ch_real + 0.25 * w_target, target_sum=1.0)

    return (w[:, None] * z[None, :]).astype(np.float32)


def make_temporal_anomaly(
    wf_ct: np.ndarray,
    mode: str,
    rqs: dict,
    channel_positions: np.ndarray,
) -> np.ndarray:
    del rqs
    C, T = wf_ct.shape
    ch_charge = wf_ct.sum(axis=1).astype(np.float32)
    total = float(np.sum(ch_charge))
    if np.any(ch_charge > 0):
        thresh = max(float(np.percentile(ch_charge[ch_charge > 0], 35.0)), 1e-6)
        active = np.flatnonzero(ch_charge > thresh)
    else:
        active = np.arange(C)

    if mode == "radial_time_shear":
        radii = _event_centered_radii(channel_positions)
        r_norm = (radii - float(radii.mean())) / max(float(radii.std()), 1e-6)
        shifts = np.clip(np.rint(2.8 * r_norm), -4, 4)
        if np.any(ch_charge > 0):
            shifts = shifts - np.rint(np.average(shifts, weights=np.clip(ch_charge, 1e-6, None)))
        shifted = np.stack([_shift_trace(wf_ct[ch], shifts[ch]) for ch in range(C)], axis=0)
        out = 0.58 * wf_ct + 0.42 * shifted

    elif mode == "sector_time_split":
        angles = _event_angles(channel_positions)
        sector_sign = np.sign(np.sin(2.0 * angles))
        sector_sign[sector_sign == 0.0] = 1.0
        shifts = 3.0 * sector_sign
        if np.any(ch_charge > 0):
            shifts = shifts - np.average(shifts, weights=np.clip(ch_charge, 1e-6, None))
        shifted = np.stack([_shift_trace(wf_ct[ch], shifts[ch]) for ch in range(C)], axis=0)
        out = 0.60 * wf_ct + 0.40 * shifted

    elif mode == "subset_late_charge":
        out = wf_ct.astype(np.float32).copy()
        active_sorted = active[np.argsort(ch_charge[active])[::-1]] if active.size > 0 else active
        subset = active_sorted[::3] if active_sorted.size > 0 else active_sorted
        delay = min(12, max(6, T // 18))
        frac = 0.18
        for ch in subset:
            late = _shift_trace(wf_ct[ch], delay)
            out[ch] = (1.0 - frac) * wf_ct[ch] + frac * late

    else:
        raise ValueError(f"Unknown temporal mode: {mode!r}")

    out = np.clip(out, 0.0, None).astype(np.float64)
    cur_total = float(np.sum(out))
    if cur_total > 1e-8 and total > 0.0:
        out *= total / cur_total
    return out.astype(np.float32)


def make_anomaly(
    wf_ct: np.ndarray,
    channel_positions: np.ndarray,
    mode: str,
    xc: float = 0.0,
    yc: float = 0.0,
    rqs: dict | None = None,
) -> np.ndarray:
    if mode in SPATIAL_TYPES:
        return make_spatial_anomaly(wf_ct, channel_positions, mode, xc, yc)
    assert rqs is not None
    return make_temporal_anomaly(wf_ct, mode, rqs, channel_positions)


def plot_anomaly_examples(
    wf_base: np.ndarray,
    proto_wfs: dict[str, np.ndarray],
    channel_positions: np.ndarray,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    def _plot_family(modes: list[str], title: str, filename: str) -> None:
        entries = [("base", "Base event", wf_base)] + [(mode, ANOMALY_LABELS[mode], proto_wfs[mode]) for mode in modes]
        fig, axes = plt.subplots(len(entries), 2, figsize=(10.1, 2.45 * len(entries)), squeeze=False)
        vmax = max(float(wf_ct.sum(axis=1).max()) for _, _, wf_ct in entries)
        vmax = max(vmax, 1e-8)
        scatter = None
        for row, (_, label, wf_ct) in enumerate(entries):
            charge = wf_ct.sum(axis=1)
            trace = wf_ct.sum(axis=0)

            ax_xy = axes[row, 0]
            scatter = ax_xy.scatter(
                channel_positions[:, 0],
                channel_positions[:, 1],
                c=charge,
                cmap="viridis",
                vmin=0.0,
                vmax=vmax,
                s=64,
                edgecolors="k",
                linewidths=0.2,
            )
            ax_xy.set_aspect("equal")
            ax_xy.set_title(f"{label} charge map", fontweight="bold")
            ax_xy.set_xlabel("x (cm)")
            ax_xy.set_ylabel("y (cm)")
            ax_xy.grid(False)

            ax_t = axes[row, 1]
            ax_t.plot(np.arange(trace.shape[0]), trace, color=COLORS["truth"], linewidth=1.4)
            ax_t.fill_between(np.arange(trace.shape[0]), trace, color=COLORS["truth"], alpha=0.10)
            ax_t.set_title(f"{label} summed waveform", fontweight="bold")
            ax_t.set_xlabel("Time bin")
            ax_t.set_ylabel("Amplitude")

        fig.subplots_adjust(left=0.07, right=0.91, bottom=0.06, top=0.91, wspace=0.26, hspace=0.34)
        cax = fig.add_axes([0.925, 0.18, 0.014, 0.66])
        cbar = fig.colorbar(scatter, cax=cax)
        cbar.ax.set_ylabel("Integrated charge")
        fig.suptitle(title, fontweight="bold", y=0.965)
        fig.savefig(os.path.join(output_dir, filename), dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)

    _plot_family(SPATIAL_TYPES, "Charge-pattern anomaly prototypes", "anomaly_examples_spatial.png")
    _plot_family(TEMPORAL_TYPES, "Timing-coherence anomaly prototypes", "anomaly_examples_temporal.png")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def to_flat(wf_batch: np.ndarray) -> np.ndarray:
    """(B, C, T) → (B, T*C) layer-major."""
    B = wf_batch.shape[0]
    return np.transpose(wf_batch, (0, 2, 1)).reshape(B, -1)


@torch.no_grad()
def encode_batch(wf_batch: np.ndarray, ctx, batch_size: int = 8) -> np.ndarray:
    """
    (B, C, T) → (B, latent_dim).

    Accepts either DiffAEContext (has ctx.encoder / ctx.ema_encoder, returns
    z as first element of a 3-tuple) or GraphAEContext (has ctx.model /
    ctx.ema_model; uses model.encode which returns (z, pool_indices)).
    """
    B      = wf_batch.shape[0]
    wf_col = np.transpose(wf_batch, (0, 2, 1)).reshape(B, -1, 1).astype(np.float32)
    wf_col = ctx.data_stats.normalize(wf_col)

    # Resolve the callable and call convention for each context type.
    is_graphae = isinstance(ctx, GraphAEContext)
    if is_graphae:
        net = ctx.ema_model if ctx.ema_model is not None else ctx.model
    else:
        net = ctx.ema_encoder if ctx.ema_encoder is not None else ctx.encoder
    net.eval()

    all_z = []
    for start in range(0, B, batch_size):
        end = min(start + batch_size, B)
        bs  = end - start
        x   = torch.from_numpy(wf_col[start:end]).to(ctx.device)
        xf  = x.view(bs * ctx.n_nodes, 1)
        if is_graphae:
            z, _ = net.encode(xf, ctx.A_sparse, ctx.pos, batch_size=bs)
        else:
            z, _, _ = net(xf, ctx.A_sparse, ctx.pos, batch_size=bs)
        all_z.append(z.cpu().numpy())
    return np.concatenate(all_z, axis=0)


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

def fit_reduce(
    Z_real: np.ndarray,
    Z_protos: np.ndarray,
    perplexity: int = 30,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Embed Z_real with t-SNE, then position Z_protos by fitting a 1-NN
    look-up into the existing embedding (nearest neighbour in the original
    space, inherit its 2-D coordinate).  This avoids re-running t-SNE for
    the prototype points while still placing them correctly relative to the
    real-event cloud.
    Returns (Z_real_2d, Z_protos_2d).
    """
    tsne = TSNE(n_components=2, perplexity=perplexity,
                random_state=seed, init="pca", learning_rate="auto")
    Z_real_2d = tsne.fit_transform(Z_real)

    # Place each prototype at the 2-D position of its nearest real neighbour.
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1).fit(Z_real)
    _, idx = nn.kneighbors(Z_protos)
    Z_protos_2d = Z_real_2d[idx[:, 0]]
    return Z_real_2d, Z_protos_2d


# ---------------------------------------------------------------------------
# Anomaly scoring — Mahalanobis distance + percentile rank
# ---------------------------------------------------------------------------

def _mahal_distances(
    Z_real: np.ndarray,
    Z_queries: np.ndarray,
    reg: float = 1e-4,
) -> np.ndarray:
    """
    Mahalanobis distance of each row in Z_queries from the real distribution.

    Uses the empirical covariance of Z_real plus a small ridge to ensure
    invertibility even when n_samples < n_dims (shouldn't be an issue here
    but makes the call robust).
    """
    mu  = Z_real.mean(axis=0)
    cov = np.cov(Z_real.T) + reg * np.eye(Z_real.shape[1])
    cov_inv = np.linalg.inv(cov)
    delta = Z_queries - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", delta, cov_inv, delta))


def anomaly_scores(
    Z_real: np.ndarray,
    Z_protos: np.ndarray,
    reg: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    d_protos : (n_protos,)  Mahalanobis distance of each prototype
    pct_rank : (n_protos,)  % of real events with *smaller* distance
                             (0 = inside the core, 100 = most extreme outlier)
    """
    d_real   = _mahal_distances(Z_real, Z_real,    reg)
    d_protos = _mahal_distances(Z_real, Z_protos,  reg)
    pct_rank = np.array([(dp > d_real).mean() * 100.0 for dp in d_protos])
    return d_protos, pct_rank


# ---------------------------------------------------------------------------
# Anomaly heatmap
# ---------------------------------------------------------------------------

def anomaly_heatmap(
    scores: dict[str, np.ndarray | None],   # space_label → (n_types,) pct_rank
    anomaly_labels: list[str],              # display names, len = n_types
    path: str,
) -> None:
    """
    Heatmap: rows = anomaly types, columns = representation space.
    Cell colour = percentile rank of prototype Mahalanobis distance
    (0 = indistinguishable from core, 100 = maximally anomalous).
    Annotated with the numeric value.
    """
    space_labels  = [k for k, v in scores.items() if v is not None]
    valid_arrays  = [scores[k] for k in space_labels]           # all non-None
    data = np.column_stack(valid_arrays)                        # (n_types, n_spaces)

    n_types, n_spaces = data.shape
    fig, ax = plt.subplots(figsize=(2.15 + 1.55 * n_spaces, 0.42 * n_types + 1.0))

    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)

    # Annotate each cell
    for r in range(n_types):
        for c in range(n_spaces):
            v = data[r, c]
            txt_col = "white" if v > 70 else "black"
            ax.text(c, r, f"{v:.0f}%", ha="center", va="center",
                    fontsize=9, color=txt_col)

    ax.set_xticks(range(n_spaces))
    ax.set_xticklabels(space_labels, fontsize=10)
    ax.set_yticks(range(n_types))
    ax.set_yticklabels(anomaly_labels, fontsize=9)
    ax.set_title("Anomaly percentile rank by representation space", fontsize=11, fontweight="bold")
    ax.set_xlabel("Representation space", fontsize=10)

    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.03)
    cb.set_label("Percentile rank of Mahalanobis distance", fontsize=8)
    cb.set_ticks([0, 25, 50, 75, 100])

    # Thin separator lines between family groups (spatial/temporal)
    n_spatial = sum(1 for t in ANOMALY_TYPES if t[3] == "spatial")
    ax.axhline(n_spatial - 0.5, color="white", linewidth=2)

    compact_layout(fig)
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  {os.path.basename(path)}")


def print_score_table(
    scores: dict[str, np.ndarray | None],
    d_scores: dict[str, np.ndarray | None],
) -> None:
    """Print a formatted table of Mahalanobis distances and percentile ranks."""
    space_labels = [k for k, v in scores.items() if v is not None]
    col_w = max(len(k) for k in space_labels) + 4

    header = f"{'Anomaly type':<26}" + "".join(f"{k:>{col_w}}" for k in space_labels)
    print("\n" + "=" * len(header))
    print("Mahalanobis distance  (percentile rank)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    prev_family = None
    for i, (atype, label, _, family) in enumerate(ANOMALY_TYPES):
        if prev_family is not None and family != prev_family:
            print()
        prev_family = family
        row = f"{label:<26}"
        for k in space_labels:
            s_arr = scores[k]
            d_arr = d_scores[k]
            if s_arr is not None and d_arr is not None:
                d   = float(d_arr[i])
                pct = float(s_arr[i])
                row += f"{d:>{col_w - 8}.2f} ({pct:4.0f}%)"
            else:
                row += f"{'—':>{col_w}}"
        print(row)
    print("=" * len(header))


# ---------------------------------------------------------------------------
# Scatter plot
# ---------------------------------------------------------------------------

def scatter_plot(
    Z_real_2d: np.ndarray,
    Z_protos_2d: np.ndarray,
    title: str,
    subtitle: str,
    path: str,
) -> None:
    """
    Grey in-distribution cloud + red anomaly dots (single "Anomaly" legend entry).
    """
    fig, ax = plt.subplots(figsize=(5.45, 4.45))

    ax.scatter(
        Z_real_2d[:, 0], Z_real_2d[:, 1],
        c=COLORS["baseline"], s=6, alpha=0.40, linewidths=0,
        zorder=1, label="In distribution",
        rasterized=True,
    )
    ax.scatter(
        Z_protos_2d[:, 0], Z_protos_2d[:, 1],
        c=COLORS["diffae"], s=60, alpha=1.0, linewidths=0,
        zorder=3, label="Anomaly",
    )

    ax.set_xlabel("t-SNE 1", fontsize=10)
    ax.set_ylabel("t-SNE 2", fontsize=10)
    ax.set_title(f"{title}\n{subtitle}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, markerscale=1.4, loc="best",
              handlelength=0.8, borderpad=0.5)

    compact_layout(fig)
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  {os.path.basename(path)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Anomaly scatter plots: RQ space + AE latent + DiffAE latent"
    )
    parser.add_argument("--n-events",   type=int, default=500,
                        help="Number of real SS events for background cloud")
    parser.add_argument("--batch-size",      type=int, default=8)
    parser.add_argument("--ae-latent-dim",   type=int, default=64,
                        help="Latent dimension for the AE model (default 64)")
    parser.add_argument("--diffae-latent-dim", type=int, default=64,
                        help="Latent dimension for the DiffAE model (default 64)")
    parser.add_argument("--output-dir", type=str, default="anomaly_results")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--perplexity",  type=int, default=30,
                        help="t-SNE perplexity (default 30)")
    parser.add_argument("--use-cached-latents", action="store_true",
                        help="Load Z_real from pre-encoded H5 files instead of encoding on-the-fly")
    parser.add_argument("--ae-latents",     type=str, default=None,
                        help="Path to GraphAE encoded latents H5 (auto-detected if omitted)")
    parser.add_argument("--diffae-latents", type=str, default=None,
                        help="Path to DiffAE encoded latents H5 (auto-detected if omitted)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)

    cfg = copy.deepcopy(default_config)

    # -----------------------------------------------------------------------
    # 1. Load real SS events
    # -----------------------------------------------------------------------
    print(f"Loading {args.n_events} real SS events from {cfg.paths.tritium_h5} …")
    with h5py.File(cfg.paths.tritium_h5, "r") as f:
        n_total = int(f["waveforms"].shape[0])
        rng     = np.random.default_rng(args.seed)
        idx     = np.sort(rng.choice(n_total, size=args.n_events, replace=False))
        wf_real = np.array(f["waveforms"][idx], dtype=np.float32)   # (N, C, T)
        xc_real = np.array(f["xc"][idx], dtype=np.float32)
        yc_real = np.array(f["yc"][idx], dtype=np.float32)

    N, C, T = wf_real.shape
    print(f"  shape {wf_real.shape}")

    # Load channel positions from DiffAEContext (needed before building models)
    print("Building contexts to get graph / channel positions …")
    cfg_diffae = copy.deepcopy(default_config)
    cfg_diffae.encoder.latent_dim = args.diffae_latent_dim
    ctx_diffae = DiffAEContext.build(cfg_diffae, for_training=True,
                                     verbose=False, use_ms_data=False)
    channel_pos = ctx_diffae.loader.channel_positions   # (C, 2)

    # -----------------------------------------------------------------------
    # 2. Pick one prototype base event (closest to median integral)
    # -----------------------------------------------------------------------
    print("Selecting prototype base event …")
    wf_flat_all = to_flat(wf_real)
    integrals   = wf_flat_all.sum(axis=1)
    med_int     = float(np.median(integrals))
    base_idx    = int(np.argmin(np.abs(integrals - med_int)))
    wf_base     = wf_real[base_idx]         # (C, T)
    xc_base     = float(xc_real[base_idx])
    yc_base     = float(yc_real[base_idx])

    z_base      = wf_to_z_profile(wf_flat_all[base_idx], C, T)
    rqs_base    = compute_rqs(z_base)
    if rqs_base is None:
        rqs_base = {k: 0.0 for k in
                    ["peak_amplitude", "peak_time", "total_integral",
                     "rise_time", "fall_time", "fwhm", "width_10_90", "std_dev"]}
    print(f"  Base event #{base_idx}  integral={integrals[base_idx]:.1f}  "
          f"(dataset median={med_int:.1f})")

    # -----------------------------------------------------------------------
    # 3. Generate anomaly prototypes  (one per type, all from base event)
    # -----------------------------------------------------------------------
    print("Generating anomaly prototypes …")
    proto_wfs  = {}  # atype → (C, T)
    for atype, _, _, _ in ANOMALY_TYPES:
        proto_wfs[atype] = make_anomaly(
            wf_base, channel_pos, atype,
            xc=xc_base, yc=yc_base, rqs=rqs_base,
        )

    # Stack into (n_types, C, T) for batch encoding
    proto_types_list = [t[0] for t in ANOMALY_TYPES]
    wf_protos = np.stack([proto_wfs[k] for k in proto_types_list], axis=0)  # (10, C, T)

    # -----------------------------------------------------------------------
    # 4. Compute RQs for real events + prototypes
    # -----------------------------------------------------------------------
    print("Computing RQs …")
    rq_real_dict   = collect_rqs(wf_flat_all, C, T)
    rq_proto_dict  = collect_rqs(to_flat(wf_protos), C, T)

    rq_names = list(rq_real_dict.keys())

    def to_mat(d: dict) -> np.ndarray:
        return np.column_stack([d[k] for k in rq_names]).astype(np.float32)

    RQ_real   = to_mat(rq_real_dict)    # (N, 8)
    RQ_protos = to_mat(rq_proto_dict)   # (n_types, 8)

    # Impute rare NaNs with column median
    for col in range(RQ_real.shape[1]):
        bad = ~np.isfinite(RQ_real[:, col])
        if bad.any():
            RQ_real[bad, col] = float(np.nanmedian(RQ_real[:, col]))
    for col in range(RQ_protos.shape[1]):
        bad = ~np.isfinite(RQ_protos[:, col])
        if bad.any():
            RQ_protos[bad, col] = float(np.nanmedian(RQ_real[:, col]))

    # -----------------------------------------------------------------------
    # 5. Load DiffAE encoder and encode
    # -----------------------------------------------------------------------
    print(f"\nLoading DiffAE encoder (latent_dim={args.diffae_latent_dim}) …")
    ckpt_diffae = ctx_diffae.latest_checkpoint()
    if ckpt_diffae is None:
        print(f"  WARNING: no DiffAE checkpoint found in {ctx_diffae.checkpoint_dir}")
        Z_diffae_real   = None
        Z_diffae_protos = None
    else:
        ep = ctx_diffae.load_checkpoint(ckpt_diffae, load_optim=False)
        print(f"  Loaded {os.path.basename(ckpt_diffae)} (epoch {ep})")

        # Background cloud: from cached latents or live encoding
        if args.use_cached_latents:
            diffae_h5 = args.diffae_latents or os.path.join(
                ctx_diffae.checkpoint_dir, cfg.paths.diffae_latents_file
            )
            if os.path.exists(diffae_h5):
                with h5py.File(diffae_h5, "r") as f:
                    all_z = np.array(f["latents"], dtype=np.float32)
                rng = np.random.default_rng(args.seed)
                idx = rng.choice(len(all_z), size=min(args.n_events, len(all_z)), replace=False)
                Z_diffae_real = all_z[idx]
                print(f"  Loaded Z_real from cache ({diffae_h5}): {Z_diffae_real.shape}")
            else:
                print(f"  WARNING: cached latents not found at {diffae_h5}, encoding on-the-fly")
                Z_diffae_real = encode_batch(wf_real, ctx_diffae, args.batch_size)
        else:
            print("  Encoding real events …")
            Z_diffae_real = encode_batch(wf_real, ctx_diffae, args.batch_size)

        print("  Encoding prototypes …")
        Z_diffae_protos = encode_batch(wf_protos, ctx_diffae, args.batch_size)
        print(f"  z shape: {Z_diffae_real.shape}")

    # -----------------------------------------------------------------------
    # 6. Load GraphAE encoder and encode
    # -----------------------------------------------------------------------
    print(f"\nLoading GraphAE encoder (latent_dim={args.ae_latent_dim}) …")
    cfg_ae = copy.deepcopy(default_config)
    cfg_ae.encoder.latent_dim = args.ae_latent_dim

    # Find the checkpoint before building so we can infer hidden_dim from the
    # saved weights, which may differ from the current config default.
    _graphae_subdir = cfg_ae.paths.graph_ae_subdir.format(latent_dim=args.ae_latent_dim)
    _graphae_ckpt_dir = os.path.join(cfg_ae.paths.checkpoint_dir, _graphae_subdir)
    _graphae_files = glob.glob(os.path.join(_graphae_ckpt_dir, "graphae_epoch_*.pt"))

    if _graphae_files:
        def _epoch_num(p):
            try:
                return int(os.path.splitext(os.path.basename(p))[0].split("_")[-1])
            except (ValueError, IndexError):
                return -1
        _ckpt_peek = max(_graphae_files, key=_epoch_num)
        _state = torch.load(_ckpt_peek, map_location="cpu", weights_only=False)
        # encoder.in_proj.weight has shape [hidden_dim, in_dim=1]
        _hidden_dim = int(_state["model"]["encoder.in_proj.weight"].shape[0])
        cfg_ae.encoder.hidden_dim = _hidden_dim
        print(f"  Inferred hidden_dim={_hidden_dim} from checkpoint weights")

    ctx_ae = GraphAEContext.build(cfg_ae, for_training=True, verbose=False,
                                  use_ms_data=False)
    ckpt_ae = ctx_ae.latest_checkpoint()
    if ckpt_ae is None:
        print(f"  WARNING: no GraphAE checkpoint found in {ctx_ae.checkpoint_dir}")
        Z_ae_real   = None
        Z_ae_protos = None
    else:
        ep = ctx_ae.load_checkpoint(ckpt_ae, load_optim=False)
        print(f"  Loaded {os.path.basename(ckpt_ae)} (epoch {ep})")

        # Background cloud: from cached latents or live encoding
        if args.use_cached_latents:
            ae_h5 = args.ae_latents or os.path.join(
                ctx_ae.checkpoint_dir, cfg.paths.graphae_latents_file
            )
            if os.path.exists(ae_h5):
                with h5py.File(ae_h5, "r") as f:
                    all_z = np.array(f["latents"], dtype=np.float32)
                rng = np.random.default_rng(args.seed + 1)
                idx = rng.choice(len(all_z), size=min(args.n_events, len(all_z)), replace=False)
                Z_ae_real = all_z[idx]
                print(f"  Loaded Z_real from cache ({ae_h5}): {Z_ae_real.shape}")
            else:
                print(f"  WARNING: cached latents not found at {ae_h5}, encoding on-the-fly")
                Z_ae_real = encode_batch(wf_real, ctx_ae, args.batch_size)
        else:
            print("  Encoding real events …")
            Z_ae_real = encode_batch(wf_real, ctx_ae, args.batch_size)

        print("  Encoding prototypes …")
        Z_ae_protos = encode_batch(wf_protos, ctx_ae, args.batch_size)
        print(f"  z shape: {Z_ae_real.shape}")

    # -----------------------------------------------------------------------
    # 7. Dimensionally reduce + plot
    # -----------------------------------------------------------------------
    print("\nGenerating plots (t-SNE) …")

    # --- Plot 1: RQ space ---
    RQ_real_2d, RQ_protos_2d = fit_reduce(
        RQ_real, RQ_protos, perplexity=args.perplexity, seed=args.seed
    )
    scatter_plot(
        Z_real_2d   = RQ_real_2d,
        Z_protos_2d = RQ_protos_2d,
        title       = "RQ space",
        subtitle    = f"t-SNE of 8 pulse-shape metrics  ({N} real events)",
        path        = os.path.join(args.output_dir, "scatter_rq.png"),
    )

    # --- Plot 2: GraphAE latent ---
    if Z_ae_real is not None:
        ae_real_2d, ae_protos_2d = fit_reduce(
            Z_ae_real, Z_ae_protos, perplexity=args.perplexity, seed=args.seed
        )
        scatter_plot(
            Z_real_2d   = ae_real_2d,
            Z_protos_2d = ae_protos_2d,
            title       = f"GraphAE latent space  (z={args.ae_latent_dim})",
            subtitle    = (f"t-SNE of {Z_ae_real.shape[1]}-dim encoder output  "
                           f"({N} real events)"),
            path        = os.path.join(args.output_dir, "scatter_ae.png"),
        )

    # --- Plot 3: DiffAE latent ---
    if Z_diffae_real is not None:
        diffae_real_2d, diffae_protos_2d = fit_reduce(
            Z_diffae_real, Z_diffae_protos, perplexity=args.perplexity, seed=args.seed
        )
        scatter_plot(
            Z_real_2d   = diffae_real_2d,
            Z_protos_2d = diffae_protos_2d,
            title       = f"DiffAE latent space  (z={args.diffae_latent_dim})",
            subtitle    = (f"t-SNE of {Z_diffae_real.shape[1]}-dim encoder output  "
                           f"({N} real events)"),
            path        = os.path.join(args.output_dir, "scatter_diffae.png"),
        )

    # -----------------------------------------------------------------------
    # 8. Anomaly scoring: Mahalanobis distance + percentile ranks
    # -----------------------------------------------------------------------
    print("\nComputing anomaly scores …")

    # Standardise RQ matrix before Mahalanobis (columns have very different scales)
    rq_std = RQ_real.std(axis=0, keepdims=True) + 1e-8
    RQ_real_n   = (RQ_real   - RQ_real.mean(axis=0)) / rq_std
    RQ_protos_n = (RQ_protos - RQ_real.mean(axis=0)) / rq_std

    d_rq,  pct_rq  = anomaly_scores(RQ_real_n, RQ_protos_n)

    d_ae,     pct_ae     = (anomaly_scores(Z_ae_real,     Z_ae_protos)
                             if Z_ae_real is not None else (None, None))
    d_diffae, pct_diffae = (anomaly_scores(Z_diffae_real, Z_diffae_protos)
                             if Z_diffae_real is not None else (None, None))

    scores_map  = {"RQ metrics": pct_rq,  "GraphAE": pct_ae,  "DiffAE": pct_diffae}
    d_score_map = {"RQ metrics": d_rq,    "GraphAE": d_ae,    "DiffAE": d_diffae}

    proto_labels = [t[1] for t in ANOMALY_TYPES]
    print_score_table(scores_map, d_score_map)

    anomaly_heatmap(
        scores        = scores_map,
        anomaly_labels = proto_labels,
        path          = os.path.join(args.output_dir, "anomaly_scores.png"),
    )

    print(f"\nDone → {args.output_dir}/")
    print("  scatter_rq.png     — RQ space (poor separability expected)")
    print("  scatter_ae.png     — GraphAE latent space")
    print("  scatter_diffae.png — DiffAE latent space")
    print("  anomaly_scores.png — Mahalanobis percentile-rank heatmap")


if __name__ == "__main__":
    main()
