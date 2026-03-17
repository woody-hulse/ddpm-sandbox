"""
eval_recon.py — Reconstruction quality evaluation for DiffAE vs GraphAE.

Loads N events, runs both models, and computes a comprehensive suite of metrics
organised into four categories:

  Physics-motivated
    mse, mae, poisson_deviance, total_charge_rel_error,
    temporal_centroid_error, peak_position_error_ns, channel_hit_jaccard

  Structural
    ssim, pearson_r

  Distribution-level (over the full evaluation set)
    wasserstein1 on total_charge / peak_time / channel_centroid_{x,y}
    ks_pvalue on the same marginals

  DiffAE stochasticity (requires --n-samples > 1)
    multi_sample_std  — pixel-wise std across K independent samples
    rank_histogram    — where does the true event rank among K DiffAE samples?
    energy_dispersion — std(total_charge of K samples) vs sqrt(true_charge)

Usage:
    python eval_recon.py                         # auto-detect checkpoints
    python eval_recon.py --n-events 4096 --n-samples 8 --output-dir eval_out
    python eval_recon.py --skip-graphae          # DiffAE only
    python eval_recon.py --skip-diffae           # GraphAE only
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from scipy.stats import wasserstein_distance
from tqdm import tqdm

from config import Config, default_config, get_config
from diffae import DiffAEContext, sample_diffae
from diffusion.schedule import build_cosine_schedule
from graphae import GraphAEContext, reconstruct_graphae  # type: ignore[import]
from lz_data_loader import TritiumSSDataLoader
from plot_style import apply_style, COLORS, MODEL_COLORS  # noqa: F401

try:
    from skimage.metrics import structural_similarity as _ssim_fn
    _SKIMAGE = True
except ImportError:
    _ssim_fn = None  # type: ignore[assignment]
    _SKIMAGE = False


# ---------------------------------------------------------------------------
# Data layout helpers
# ---------------------------------------------------------------------------

def to_2d(x_flat: np.ndarray, n_channels: int, n_time_points: int) -> np.ndarray:
    """(B, N) → (B, C, T) via per-sample Fortran-order reshape."""
    B = x_flat.shape[0]
    return np.stack(
        [x_flat[b].reshape(n_channels, n_time_points, order="F") for b in range(B)]
    )


# ---------------------------------------------------------------------------
# Per-sample metrics — all take (B, C, T) numpy arrays, return (B,)
# ---------------------------------------------------------------------------

def metric_mse(rec: np.ndarray, true: np.ndarray) -> np.ndarray:
    return np.mean((rec - true) ** 2, axis=(1, 2))


def metric_mae(rec: np.ndarray, true: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(rec - true), axis=(1, 2))


def metric_poisson_deviance(
    rec: np.ndarray, true: np.ndarray, eps: float = 1e-3
) -> np.ndarray:
    """
    Poisson deviance: D = 2 * Σ [ y*log(y/ŷ) - (y - ŷ) ]
    Appropriate goodness-of-fit for count data (photon hits).
    rec is clipped to eps to avoid log(0); true zeros contribute 0.
    """
    rec_clip = np.clip(rec, eps, None)
    term = np.where(true > 0, true * np.log(true / rec_clip) - (true - rec_clip), -(true - rec_clip))
    return 2.0 * term.sum(axis=(1, 2))


def metric_total_charge_rel_error(
    rec: np.ndarray, true: np.ndarray, eps: float = 1.0
) -> np.ndarray:
    """Relative error in total integrated charge (S2 area proxy)."""
    q_true = true.sum(axis=(1, 2))
    q_rec = rec.sum(axis=(1, 2))
    return np.abs(q_rec - q_true) / (q_true + eps)


def metric_temporal_centroid_error(
    rec: np.ndarray, true: np.ndarray, ns_per_bin: float = 10.0
) -> np.ndarray:
    """
    Error in charge-weighted temporal centroid, summed over channels, in ns.
    centroid = Σ_t [ t * Σ_c x[c,t] ] / Σ_t Σ_c x[c,t]
    """
    B, C, T = rec.shape
    t_axis = np.arange(T, dtype=np.float64)
    tot_true = true.sum(axis=1)        # (B, T)
    tot_rec  = rec.sum(axis=1)         # (B, T)
    denom_true = tot_true.sum(axis=1) + 1e-8   # (B,)
    denom_rec  = tot_rec.sum(axis=1)  + 1e-8
    cent_true = (tot_true * t_axis).sum(axis=1) / denom_true
    cent_rec  = (tot_rec  * t_axis).sum(axis=1) / denom_rec
    return np.abs(cent_rec - cent_true) * ns_per_bin


def metric_peak_position_error(
    rec: np.ndarray, true: np.ndarray,
    ns_per_bin: float = 10.0, top_k: int = 4,
) -> np.ndarray:
    """
    Mean argmax-in-time mismatch (in ns) for the top-k brightest channels.
    """
    B, C, T = rec.shape
    ch_energy = true.sum(axis=2)            # (B, C)
    errors = np.zeros(B, dtype=np.float64)
    for b in range(B):
        k = min(top_k, C)
        top_chs = np.argsort(ch_energy[b])[-k:]
        peak_true = np.argmax(true[b][top_chs], axis=1)   # (k,)
        peak_rec  = np.argmax(rec[b][top_chs],  axis=1)   # (k,)
        errors[b] = np.mean(np.abs(peak_rec - peak_true)) * ns_per_bin
    return errors


def metric_channel_hit_jaccard(
    rec: np.ndarray, true: np.ndarray, threshold_quantile: float = 0.8
) -> np.ndarray:
    """
    Jaccard similarity of the spatial hit pattern.
    A channel is considered "hit" if its integrated charge exceeds a threshold
    derived from the true event's charge distribution.
    """
    ch_true = true.sum(axis=2)   # (B, C)
    ch_rec  = rec.sum(axis=2)    # (B, C)
    scores = np.zeros(len(ch_true), dtype=np.float64)
    for b in range(len(ch_true)):
        thresh = np.quantile(ch_true[b], threshold_quantile)
        hit_t = ch_true[b] >= thresh
        hit_r = ch_rec[b]  >= thresh
        union = (hit_t | hit_r).sum()
        inter = (hit_t & hit_r).sum()
        scores[b] = inter / max(union, 1)
    return scores


def metric_ssim(rec: np.ndarray, true: np.ndarray) -> np.ndarray:
    """SSIM on the (C, T) 2D map per event."""
    B = rec.shape[0]
    scores = np.zeros(B, dtype=np.float64)
    for b in range(B):
        r = rec[b].astype(np.float64)
        t = true[b].astype(np.float64)
        data_range = max(t.max() - t.min(), r.max() - r.min(), 1e-8)
        if _SKIMAGE and _ssim_fn is not None:
            scores[b] = _ssim_fn(t, r, data_range=data_range)  # type: ignore[operator]
        else:
            # Simple SSIM approximation: luminance * contrast * structure
            mu_t, mu_r = t.mean(), r.mean()
            sig_t = t.std() + 1e-8
            sig_r = r.std() + 1e-8
            sig_tr = np.mean((t - mu_t) * (r - mu_r))
            C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
            lum = (2 * mu_t * mu_r + C1) / (mu_t ** 2 + mu_r ** 2 + C1)
            cs  = (2 * sig_t * sig_r + C2) / (sig_t ** 2 + sig_r ** 2 + C2)
            struct = (sig_tr + C2 / 2) / (sig_t * sig_r + C2 / 2)
            scores[b] = lum * cs * struct
    return scores


def metric_pearson_r(rec: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-sample Pearson correlation coefficient on the flattened (C*T) vector."""
    B = rec.shape[0]
    r_flat = rec.reshape(B, -1).astype(np.float64)
    t_flat = true.reshape(B, -1).astype(np.float64)
    r_mu = r_flat.mean(axis=1, keepdims=True)
    t_mu = t_flat.mean(axis=1, keepdims=True)
    num = ((r_flat - r_mu) * (t_flat - t_mu)).sum(axis=1)
    den = np.sqrt(((r_flat - r_mu) ** 2).sum(axis=1) * ((t_flat - t_mu) ** 2).sum(axis=1)) + 1e-12
    return num / den


# ---------------------------------------------------------------------------
# Distribution-level metrics — operate on ensemble arrays
# ---------------------------------------------------------------------------

def physics_marginals(
    x_2d: np.ndarray,
    channel_positions: np.ndarray,
    ns_per_bin: float = 10.0,
) -> Dict[str, np.ndarray]:
    """Extract per-event physics scalars from a (B, C, T) array."""
    B, C, T = x_2d.shape
    t_axis = np.arange(T, dtype=np.float64)
    total_charge = x_2d.sum(axis=(1, 2))                        # (B,)
    ch_sum = x_2d.sum(axis=2)                                   # (B, C) — per-channel energy
    tot_t  = x_2d.sum(axis=1)                                   # (B, T) — time profile
    denom  = tot_t.sum(axis=1) + 1e-8
    peak_time = (tot_t * t_axis).sum(axis=1) / denom * ns_per_bin   # (B,) centroid in ns
    # charge-weighted spatial centroid
    ch_norm = ch_sum / (ch_sum.sum(axis=1, keepdims=True) + 1e-8)  # (B, C)
    cx = (ch_norm * channel_positions[:C, 0]).sum(axis=1)
    cy = (ch_norm * channel_positions[:C, 1]).sum(axis=1)
    return {
        "total_charge": total_charge,
        "peak_time_ns": peak_time,
        "centroid_x":   cx,
        "centroid_y":   cy,
    }


def distribution_metrics(
    marg_rec: Dict[str, np.ndarray],
    marg_true: Dict[str, np.ndarray],
) -> Dict[str, float]:
    out = {}
    for key in marg_true:
        r, t = marg_rec[key], marg_true[key]
        out[f"W1_{key}"]    = float(wasserstein_distance(r, t))
        ks = stats.ks_2samp(r, t)
        out[f"KS_pval_{key}"] = float(ks.pvalue)  # type: ignore[attr-defined]
    return out


# ---------------------------------------------------------------------------
# DiffAE stochasticity metrics
# ---------------------------------------------------------------------------

def multi_sample_metrics(
    samples: np.ndarray,   # (K, B, C, T)  — K independent DiffAE samples
    true: np.ndarray,      # (B, C, T)
    ns_per_bin: float = 10.0,
) -> Dict[str, np.ndarray]:
    """
    Metrics that require multiple independent samples of the same event.

    rank_histogram_counts: verification rank histogram (Talagrand diagram).
      For each event, rank the true observation within K samples.
      Uniform = calibrated; peaked = under-dispersed; U-shaped = over-dispersed.

    multi_sample_std: mean pixel-wise std across K samples.

    energy_dispersion_ratio: std(total_charge of K samples) / sqrt(true_charge).
      For Poisson-calibrated reconstruction, ratio ≈ 1.
    """
    K, B, C, T = samples.shape

    # Pixel-wise std across K samples, averaged over pixels
    pix_std = samples.std(axis=0).mean(axis=(1, 2))   # (B,)

    # Energy dispersion
    k_charges = samples.sum(axis=(2, 3))               # (K, B)
    true_charge = true.sum(axis=(1, 2))                # (B,)
    energy_dispersion_ratio = k_charges.std(axis=0) / (np.sqrt(true_charge) + 1e-8)  # (B,)

    # Rank histogram: for each pixel, where does true fall among K samples?
    # Aggregate over pixels to get a single rank per event.
    # rank ∈ {0, ..., K}: number of samples that are less than truth.
    flat_samples = samples.reshape(K, B, -1)      # (K, B, C*T)
    flat_true    = true.reshape(B, 1, -1)         # (B, 1, C*T)
    ranks = (flat_samples < flat_true.transpose(1, 0, 2)).sum(axis=0).mean(axis=1)  # (B,)
    # Bin into K+1 bins
    rank_counts = np.histogram(ranks, bins=np.linspace(0, K, K + 2))[0]

    return {
        "multi_sample_std":         pix_std,
        "energy_dispersion_ratio":  energy_dispersion_ratio,
        "rank_histogram_counts":    rank_counts,
    }


# ---------------------------------------------------------------------------
# Reconstruction runners
# ---------------------------------------------------------------------------

def reconstruct_diffae_batch(
    ctx: DiffAEContext,
    x_norm: torch.Tensor,   # (B, N, 1)
    schedule: dict,
    cfg: Config,
) -> np.ndarray:
    """Single stochastic reconstruction. Returns (B, C, T) denormalised."""
    assert ctx.ema_encoder is not None and ctx.ema_decoder is not None and ctx.ema_latent_proj is not None
    rec = sample_diffae(
        encoder=ctx.ema_encoder,
        decoder=ctx.ema_decoder,
        latent_proj=ctx.ema_latent_proj,
        schedule=schedule,
        A_sparse=ctx.A_sparse,
        pos=ctx.pos,
        time_dim=cfg.conditioning.time_dim,
        x_ref=x_norm,
        parametrization=cfg.diffusion.parametrization,
    )                                                   # (B, 1, N)
    rec_np = ctx.data_stats.denormalize(rec.cpu().numpy()[:, 0, :])  # (B, N)
    return np.clip(rec_np, 0, None)


def reconstruct_graphae_batch(
    ctx: GraphAEContext,
    x_norm: torch.Tensor,   # (B, N, 1)
) -> np.ndarray:
    """Returns (B, N) denormalised."""
    rec = reconstruct_graphae(ctx.ema_model, ctx.A_sparse, ctx.pos, x_norm)  # (B, 1, N)
    rec_np = ctx.data_stats.denormalize(rec.cpu().numpy()[:, 0, :])           # (B, N)
    return np.clip(rec_np, 0, None)


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

def evaluate_all(
    rec_2d: np.ndarray,     # (B, C, T)
    true_2d: np.ndarray,    # (B, C, T)
    channel_positions: np.ndarray,
    ns_per_bin: float = 10.0,
) -> Dict[str, np.ndarray]:
    """Run all per-sample metrics and return dict of (B,) arrays."""
    return {
        "mse":                   metric_mse(rec_2d, true_2d),
        "mae":                   metric_mae(rec_2d, true_2d),
        "poisson_deviance":      metric_poisson_deviance(rec_2d, true_2d),
        "total_charge_rel_err":  metric_total_charge_rel_error(rec_2d, true_2d),
        "temporal_centroid_err": metric_temporal_centroid_error(rec_2d, true_2d, ns_per_bin),
        "peak_pos_err_ns":       metric_peak_position_error(rec_2d, true_2d, ns_per_bin),
        "channel_hit_jaccard":   metric_channel_hit_jaccard(rec_2d, true_2d),
        "ssim":                  metric_ssim(rec_2d, true_2d),
        "pearson_r":             metric_pearson_r(rec_2d, true_2d),
    }


# ---------------------------------------------------------------------------
# Printing & plotting
# ---------------------------------------------------------------------------

def print_table(
    per_sample: Dict[str, Dict[str, np.ndarray]],
    dist: Dict[str, Dict[str, float]],
) -> None:
    models = list(per_sample.keys())

    # Per-sample metrics
    metric_keys = list(next(iter(per_sample.values())).keys())
    w = max(len(m) for m in models) + 2
    mw = 28

    print("\n" + "=" * (w + len(models) * 22))
    print("Per-sample metrics (mean ± std)")
    print("=" * (w + len(models) * 22))
    header = f"{'Metric':<{mw}}" + "".join(f"  {m:>20}" for m in models)
    print(header)
    print("-" * len(header))

    higher_better = {"channel_hit_jaccard", "ssim", "pearson_r"}
    for k in metric_keys:
        row = f"{k:<{mw}}"
        for m in models:
            arr = per_sample[m][k]
            row += f"  {arr.mean():>9.4f} ±{arr.std():>8.4f}"
        print(row)

    # Distribution-level
    print("\n" + "=" * (w + len(models) * 22))
    print("Distribution-level metrics")
    print("=" * (w + len(models) * 22))
    dist_keys = list(next(iter(dist.values())).keys())
    header2 = f"{'Metric':<{mw}}" + "".join(f"  {m:>20}" for m in models)
    print(header2)
    print("-" * len(header2))
    for k in dist_keys:
        row = f"{k:<{mw}}"
        for m in models:
            row += f"  {dist[m][k]:>20.4f}"
        print(row)
    print("=" * (w + len(models) * 22))


def plot_results(
    per_sample: Dict[str, Dict[str, np.ndarray]],
    dist: Dict[str, Dict[str, float]],
    true_2d: np.ndarray,
    rec_2d_dict: Dict[str, np.ndarray],
    stoch_metrics: Optional[Dict[str, np.ndarray]],
    channel_positions: np.ndarray,
    ns_per_bin: float,
    output_dir: str,
) -> None:
    apply_style()
    os.makedirs(output_dir, exist_ok=True)
    models = list(per_sample.keys())
    col = {m: c for m, c in zip(models, MODEL_COLORS)}

    # 1. Per-sample metric histograms (one panel per metric)
    metric_keys = list(next(iter(per_sample.values())).keys())
    n_metrics = len(metric_keys)
    n_cols_h = min(5, n_metrics)
    n_rows_h = (n_metrics + n_cols_h - 1) // n_cols_h
    fig, axes = plt.subplots(n_rows_h, n_cols_h,
                             figsize=(4.5 * n_cols_h, 3.2 * n_rows_h))
    axes_flat = np.array(axes).flatten()
    metric_labels = {
        "mse": "MSE", "mae": "MAE",
        "poisson_deviance": "Poisson deviance",
        "total_charge_rel_err": "Charge rel. error",
        "temporal_centroid_err": "Centroid error (ns)",
        "peak_pos_err_ns": "Peak pos. error (ns)",
        "channel_hit_jaccard": "Channel Jaccard",
        "ssim": "SSIM", "pearson_r": "Pearson r",
    }
    for ax, k in zip(axes_flat, metric_keys):
        for m in models:
            arr = per_sample[m][k]
            ax.hist(arr, bins=50, alpha=0.60, color=col[m], density=True,
                    label=m, edgecolor='none')
        ax.set_title(metric_labels.get(k, k))
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend()
    for ax in axes_flat[n_metrics:]:
        ax.set_visible(False)
    fig.suptitle("Per-sample reconstruction metric distributions")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "metric_histograms.png"))
    plt.close(fig)

    # 2. Physics marginal scatter: true vs reconstructed (total_charge, peak_time)
    true_marg = physics_marginals(true_2d, channel_positions, ns_per_bin)
    scatter_keys = ["total_charge", "peak_time_ns"]
    scatter_labels = {"total_charge": "Total charge (a.u.)", "peak_time_ns": "Peak time (ns)"}
    fig, axes = plt.subplots(len(models), len(scatter_keys),
                             figsize=(4.8 * len(scatter_keys), 4.0 * len(models)),
                             squeeze=False)
    for row, m in enumerate(models):
        rec_marg = physics_marginals(rec_2d_dict[m], channel_positions, ns_per_bin)
        for col_idx, key in enumerate(scatter_keys):
            ax = axes[row, col_idx]
            t, r = true_marg[key], rec_marg[key]
            lim = max(np.abs(t).max(), np.abs(r).max()) * 1.05
            ax.scatter(t, r, s=2, alpha=0.20, color=col[m], rasterized=True, edgecolors='none')
            ax.plot([-lim, lim], [-lim, lim], color=COLORS["truth"],
                    linestyle='--', linewidth=0.9, alpha=0.7)
            ax.set_xlabel(f"True {scatter_labels[key]}")
            ax.set_ylabel(f"Reconstructed {scatter_labels[key]}")
            ax.set_title(m)
            ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "physics_marginals.png"))
    plt.close(fig)

    # 3. Distribution-level: overlaid marginal histograms
    marginal_keys = list(true_marg.keys())
    marg_labels = {
        "total_charge": "Total charge (a.u.)",
        "peak_time_ns": "Peak time (ns)",
        "centroid_x": "Centroid x (mm)",
        "centroid_y": "Centroid y (mm)",
    }
    fig, axes = plt.subplots(1, len(marginal_keys),
                             figsize=(4.2 * len(marginal_keys), 3.8))
    if len(marginal_keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, marginal_keys):
        ax.hist(true_marg[key], bins=50, alpha=0.50, color=COLORS["truth"],
                density=True, label="Truth", edgecolor='none')
        for m in models:
            rec_marg = physics_marginals(rec_2d_dict[m], channel_positions, ns_per_bin)
            w1 = dist[m].get(f"W1_{key}", float("nan"))
            ax.hist(rec_marg[key], bins=50, alpha=0.45, color=col[m], density=True,
                    label=f"{m}  (W₁={w1:.2f})", edgecolor='none')
        ax.set_xlabel(marg_labels.get(key, key))
        ax.set_ylabel("Density")
        ax.legend(handlelength=1.2)
    fig.suptitle("Marginal distributions: truth vs. reconstructions")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "marginal_distributions.png"))
    plt.close(fig)

    # 4. DiffAE stochasticity: rank histogram + energy dispersion
    if stoch_metrics is not None:
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))

        rhist: np.ndarray = stoch_metrics["rank_histogram_counts"]  # type: ignore[assignment,index]
        K_bins = len(rhist)
        axes[0].bar(range(K_bins), rhist, width=0.85,
                    color=COLORS["diffae"], alpha=0.75, edgecolor='none')
        axes[0].axhline(len(true_2d) / K_bins, color=COLORS["truth"],
                        linestyle="--", linewidth=1.0, label="Uniform (ideal)")
        axes[0].set_xlabel("Rank bin")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Rank histogram (Talagrand diagram)")
        axes[0].legend()

        edisp: np.ndarray = stoch_metrics["energy_dispersion_ratio"]  # type: ignore[assignment,index]
        axes[1].hist(edisp, bins=50, color=COLORS["diffae"], alpha=0.75,
                     density=True, edgecolor='none')
        axes[1].axvline(1.0, color=COLORS["truth"], linestyle="--",
                        linewidth=1.0, label="Poisson ideal (1.0)")
        axes[1].set_xlabel(r"$\sigma_K(Q) \,/\, \sqrt{\bar{Q}}$")
        axes[1].set_ylabel("Density")
        axes[1].set_title("Energy dispersion ratio")
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "diffae_stochasticity.png"))
        plt.close(fig)


# ---------------------------------------------------------------------------
# Context loading helpers
# ---------------------------------------------------------------------------

def load_diffae(cfg: Config, device: torch.device) -> Optional[DiffAEContext]:
    ctx = DiffAEContext.build(cfg, for_training=False, verbose=True)
    ckpt = ctx.latest_checkpoint()
    if ckpt is None:
        print("WARNING: no DiffAE checkpoint found — skipping DiffAE.")
        return None
    epoch = ctx.load_checkpoint(ckpt, load_optim=False)
    print(f"  DiffAE: loaded epoch {epoch} from {os.path.basename(ckpt)}")
    assert ctx.ema_encoder is not None and ctx.ema_decoder is not None and ctx.ema_latent_proj is not None
    ctx.ema_encoder.eval()
    ctx.ema_decoder.eval()
    ctx.ema_latent_proj.eval()
    return ctx


def load_graphae(cfg: Config, device: torch.device) -> Optional[GraphAEContext]:
    configured_dim = cfg.encoder.latent_dim
    candidates = [configured_dim, configured_dim // 2, configured_dim * 2]

    for ldim in candidates:
        if ldim < 1:
            continue
        probe_cfg = get_config(latent_dim=ldim) if ldim != configured_dim else cfg
        ctx = GraphAEContext.build(probe_cfg, for_training=False, verbose=False)
        ckpt = ctx.latest_checkpoint()
        if ckpt is not None:
            if ldim != configured_dim:
                print(
                    f"WARNING: no GraphAE checkpoint for latent_dim={configured_dim}. "
                    f"Falling back to latent_dim={ldim} "
                    f"(found {os.path.basename(ckpt)})."
                )
            epoch = ctx.load_checkpoint(ckpt, load_optim=False)
            print(f"  GraphAE: loaded epoch {epoch} from {os.path.basename(ckpt)}"
                  f"  (latent_dim={ldim})")
            ctx.ema_model.eval()
            return ctx

    print(
        f"WARNING: no GraphAE checkpoint found for latent_dim in "
        f"{candidates} — skipping GraphAE."
    )
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruction quality evaluation")
    parser.add_argument("--n-events", type=int, default=4096,
                        help="Number of events to evaluate (default 4096)")
    parser.add_argument("--n-samples", type=int, default=1,
                        help="Independent DiffAE samples per event for stochasticity metrics (default 1)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default="eval_recon_out")
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--skip-diffae",  action="store_true")
    parser.add_argument("--skip-graphae", action="store_true")
    parser.add_argument("--use-ms-data",  action="store_true",
                        help="Use MS (multi-scatter) events instead of SS for evaluation")
    args = parser.parse_args()

    cfg = default_config if args.latent_dim is None else get_config(latent_dim=args.latent_dim)
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}  |  latent_dim: {cfg.encoder.latent_dim}")

    # -------------------------------------------------------------------
    # Load data (use SS by default for clean physics ground truth)
    # -------------------------------------------------------------------
    if args.use_ms_data:
        from lz_data_loader import OnlineMSBatcher
        loader = OnlineMSBatcher(
            cfg.paths.tritium_h5,
            cfg.paths.channel_positions,
            delta_min=cfg.ms_data.delta_min,
            delta_max=cfg.ms_data.delta_max,
            ns_per_bin=cfg.ms_data.ns_per_bin,
        )
    else:
        loader = TritiumSSDataLoader(cfg.paths.tritium_h5, cfg.paths.channel_positions)

    n_channels    = loader.n_channels
    n_time_points = loader.n_time_points
    channel_positions = loader.channel_positions  # (C, 2)
    ns_per_bin = cfg.ms_data.ns_per_bin

    print(f"Graph: {n_channels} channels × {n_time_points} time bins = {n_channels*n_time_points} nodes")

    # -------------------------------------------------------------------
    # Collect events
    # -------------------------------------------------------------------
    print(f"\nCollecting {args.n_events} events...")
    all_wf: List[np.ndarray] = []
    collected = 0
    while collected < args.n_events:
        bsz = min(args.batch_size, args.n_events - collected)
        wf, *_ = loader.get_batch(bsz)
        all_wf.append(wf)
        collected += bsz

    wf_all = np.concatenate(all_wf, axis=0)[:args.n_events]   # (N_ev, N_nodes, 1)
    true_flat = wf_all[:, :, 0]                                 # (N_ev, N_nodes)
    true_2d   = to_2d(true_flat, n_channels, n_time_points)    # (N_ev, C, T)
    print(f"  true_2d shape: {true_2d.shape}")

    # -------------------------------------------------------------------
    # Load models
    # -------------------------------------------------------------------
    diffae_ctx  = None if args.skip_diffae  else load_diffae(cfg, device)
    graphae_ctx = None if args.skip_graphae else load_graphae(cfg, device)

    schedule = build_cosine_schedule(cfg.diffusion.timesteps, device)

    # -------------------------------------------------------------------
    # Run reconstructions
    # -------------------------------------------------------------------
    per_sample:  Dict[str, Dict[str, np.ndarray]] = {}
    dist_metrics: Dict[str, Dict[str, float]]     = {}
    rec_2d_dict:  Dict[str, np.ndarray]            = {}

    def _run_recon(name: str, rec_flat: np.ndarray) -> None:
        r2d = to_2d(rec_flat, n_channels, n_time_points)
        rec_2d_dict[name] = r2d
        per_sample[name]  = evaluate_all(r2d, true_2d, channel_positions, ns_per_bin)
        marg_rec  = physics_marginals(r2d, channel_positions, ns_per_bin)
        marg_true = physics_marginals(true_2d, channel_positions, ns_per_bin)
        dist_metrics[name] = distribution_metrics(marg_rec, marg_true)

    N_ev = len(true_flat)

    if diffae_ctx is not None:
        print("\nRunning DiffAE reconstructions...")
        rec_batches: List[np.ndarray] = []
        for start in tqdm(range(0, N_ev, args.batch_size), desc="DiffAE", ncols=90):
            end = min(start + args.batch_size, N_ev)
            wf_b = wf_all[start:end]
            wf_norm = diffae_ctx.data_stats.normalize(wf_b).astype(np.float32)
            x_t = torch.from_numpy(wf_norm).to(device)   # (B, N, 1)
            rec = reconstruct_diffae_batch(diffae_ctx, x_t, schedule, cfg)  # (B, N)
            rec_batches.append(rec)
        diffae_rec_flat = np.concatenate(rec_batches, axis=0)
        _run_recon("DiffAE", diffae_rec_flat)

    if graphae_ctx is not None:
        print("\nRunning GraphAE reconstructions...")
        rec_batches = []
        for start in tqdm(range(0, N_ev, args.batch_size), desc="GraphAE", ncols=90):
            end = min(start + args.batch_size, N_ev)
            wf_b = wf_all[start:end]
            wf_norm = graphae_ctx.data_stats.normalize(wf_b).astype(np.float32)
            x_t = torch.from_numpy(wf_norm).to(device)   # (B, N, 1)
            rec = reconstruct_graphae_batch(graphae_ctx, x_t)   # (B, N)
            rec_batches.append(rec)
        graphae_rec_flat = np.concatenate(rec_batches, axis=0)
        _run_recon("GraphAE", graphae_rec_flat)

    # -------------------------------------------------------------------
    # DiffAE stochasticity (K > 1 samples)
    # -------------------------------------------------------------------
    stoch_metrics: Optional[Dict[str, np.ndarray]] = None
    if diffae_ctx is not None and args.n_samples > 1:
        K = args.n_samples
        print(f"\nRunning {K} independent DiffAE samples for stochasticity metrics...")
        k_samples: List[np.ndarray] = []
        for k in range(K):
            k_batches: List[np.ndarray] = []
            for start in tqdm(range(0, N_ev, args.batch_size), desc=f"Sample {k+1}/{K}", ncols=90):
                end = min(start + args.batch_size, N_ev)
                wf_b = wf_all[start:end]
                wf_norm = diffae_ctx.data_stats.normalize(wf_b).astype(np.float32)
                x_t = torch.from_numpy(wf_norm).to(device)
                rec = reconstruct_diffae_batch(diffae_ctx, x_t, schedule, cfg)
                k_batches.append(rec)
            k_samples.append(to_2d(np.concatenate(k_batches, axis=0), n_channels, n_time_points))

        samples_arr = np.stack(k_samples, axis=0)   # (K, N_ev, C, T)
        stoch_metrics = multi_sample_metrics(samples_arr, true_2d, ns_per_bin)

        # Print stochasticity summary
        pix_std = stoch_metrics["multi_sample_std"]
        edisp   = stoch_metrics["energy_dispersion_ratio"]
        print(f"\n  DiffAE stochasticity (K={K}):")
        print(f"    Mean pixel-wise std across samples:   {pix_std.mean():.4f} ± {pix_std.std():.4f}")
        print(f"    Energy dispersion ratio (ideal=1.0):  {edisp.mean():.4f} ± {edisp.std():.4f}")

    # -------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------
    if not per_sample:
        print("No models ran — nothing to report.")
        return

    print_table(per_sample, dist_metrics)
    plot_results(
        per_sample, dist_metrics, true_2d, rec_2d_dict,
        stoch_metrics, channel_positions, ns_per_bin, args.output_dir,
    )
    print(f"\nPlots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
