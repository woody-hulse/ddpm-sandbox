"""
Plot RQ distributional overlays comparing original events to DiffAE and AE reconstructions.

Generates two figures:
  - diffae_rq_distributions.png : original vs DiffAE per-RQ histogram overlays
  - ae_rq_distributions.png     : original vs AE per-RQ histogram overlays

Usage:
    python plot_rq_distributions.py
    python plot_rq_distributions.py --n-samples 1000 --batch-size 16
    python plot_rq_distributions.py --output-dir results/rq_dist --n-bins 60
"""

import os
import argparse
from typing import Dict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from plot_style import apply_style, COLORS
from config import default_config
from ae import AEContext, reconstruct_ae
from diffae import DiffAEContext, sample_diffae
from compare_rqs import collect_rqs, RQ_UNITS

apply_style()

PLOT_DPI = 300

RQ_DISPLAY_NAMES = {
    "peak_amplitude": "Peak Amplitude",
    "peak_time":      "Peak Time",
    "total_integral": "Total Integral",
    "rise_time":      "Rise Time",
    "fall_time":      "Fall Time",
    "fwhm":           "FWHM",
    "width_10_90":    "Width 10-90",
    "std_dev":        "Std Dev",
}

N_COLS = 4


def _hist_range(true_vals: np.ndarray, gen_vals: np.ndarray, clip_pct: float = 99.5):
    """Compute a shared histogram range clipping outliers."""
    combined = np.concatenate([true_vals, gen_vals])
    finite = combined[np.isfinite(combined)]
    if len(finite) < 2:
        return None
    lo = np.percentile(finite, 100 - clip_pct)
    hi = np.percentile(finite, clip_pct)
    margin = max((hi - lo) * 0.05, 1e-8)
    return (lo - margin, hi + margin)


def plot_distributions(
    rq_true: Dict[str, np.ndarray],
    rq_gen: Dict[str, np.ndarray],
    label_true: str,
    label_gen: str,
    color_true: str,
    color_gen: str,
    output_path: str,
    n_bins: int = 50,
    title: str = "",
):
    """
    Plot a grid of histogram overlays (one subplot per RQ).

    Each subplot shows the original and generated distributions as
    semi-transparent filled histograms, mirroring the style of the
    reference figure (probability density, stacked legends).
    """
    rq_names = list(rq_true.keys())
    n_rqs = len(rq_names)
    n_rows = (n_rqs + N_COLS - 1) // N_COLS

    fig, axes = plt.subplots(
        n_rows, N_COLS,
        figsize=(4.5 * N_COLS, 3.5 * n_rows),
        squeeze=False,
    )

    for idx, rq_name in enumerate(rq_names):
        row, col = divmod(idx, N_COLS)
        ax = axes[row][col]

        true_vals = rq_true[rq_name]
        gen_vals  = rq_gen[rq_name]

        t_finite = true_vals[np.isfinite(true_vals)]
        g_finite = gen_vals[np.isfinite(gen_vals)]

        if len(t_finite) < 2 or len(g_finite) < 2:
            ax.set_visible(False)
            continue

        rng = _hist_range(t_finite, g_finite)
        if rng is None:
            ax.set_visible(False)
            continue

        bins = np.linspace(rng[0], rng[1], n_bins + 1)
        unit = RQ_UNITS.get(rq_name, "")
        xlabel = f"{RQ_DISPLAY_NAMES.get(rq_name, rq_name)}" + (f" [{unit}]" if unit else "")

        # True (original) — plotted first so it shows through
        ax.hist(
            t_finite, bins=bins, density=True,
            color=color_true, alpha=0.55,
            label=label_true,
            linewidth=0.5, edgecolor=color_true,
        )
        # Generated — plotted on top, slightly more transparent
        ax.hist(
            g_finite, bins=bins, density=True,
            color=color_gen, alpha=0.55,
            label=label_gen,
            linewidth=0.5, edgecolor=color_gen,
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Probability Density")
        ax.legend(loc="upper right", fontsize=8)

        # KS statistic as subtitle
        from scipy.stats import ks_2samp
        ks_stat, _ = ks_2samp(t_finite, g_finite)
        ax.set_title(f"KS = {ks_stat:.3f}", fontsize=9)

    # Hide unused axes
    for idx in range(n_rqs, n_rows * N_COLS):
        row, col = divmod(idx, N_COLS)
        axes[row][col].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Plot RQ distributional overlays")
    parser.add_argument("--n-samples",  type=int, default=500,           help="Number of events to process")
    parser.add_argument("--batch-size", type=int, default=16,            help="Inference batch size")
    parser.add_argument("--n-bins",     type=int, default=50,            help="Number of histogram bins")
    parser.add_argument("--output-dir", type=str, default="rq_dist_plots", help="Output directory")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    cfg = default_config
    cfg.encoder.hidden_dim = max(cfg.encoder.hidden_dim, cfg.encoder.latent_dim)
    cfg.conditioning.cond_proj_dim = max(cfg.conditioning.cond_proj_dim, cfg.encoder.latent_dim)

    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device : {device}")
    print(f"Samples: {args.n_samples}")

    # ------------------------------------------------------------------ #
    #  Load AE
    # ------------------------------------------------------------------ #
    has_ae = False
    ae_ctx = None
    try:
        ae_ctx = AEContext.build(cfg, for_training=False, verbose=False, use_ms_data=True)
        ae_ckpt = ae_ctx.latest_checkpoint()
        if ae_ckpt is not None:
            ae_ctx.load_checkpoint(ae_ckpt, load_optim=False)
            ae_ctx.encoder = ae_ctx.ema_encoder if ae_ctx.ema_encoder is not None else ae_ctx.encoder
            ae_ctx.decoder = ae_ctx.ema_decoder if ae_ctx.ema_decoder is not None else ae_ctx.decoder
            ae_ctx.encoder.eval()
            ae_ctx.decoder.eval()
            has_ae = True
            print(f"AE     : {ae_ckpt}")
        else:
            print("AE     : no checkpoint found — skipping")
    except Exception as e:
        print(f"AE     : could not load — {e}")

    # ------------------------------------------------------------------ #
    #  Load DiffAE
    # ------------------------------------------------------------------ #
    has_diffae = False
    diffae_ctx = None
    try:
        diffae_ctx = DiffAEContext.build(cfg, for_training=False, verbose=False, use_ms_data=True)
        diffae_ckpt = diffae_ctx.latest_checkpoint()
        if diffae_ckpt is not None:
            diffae_ctx.load_checkpoint(diffae_ckpt, load_optim=False)
            diffae_ctx.encoder    = diffae_ctx.ema_encoder    if diffae_ctx.ema_encoder    is not None else diffae_ctx.encoder
            diffae_ctx.decoder    = diffae_ctx.ema_decoder    if diffae_ctx.ema_decoder    is not None else diffae_ctx.decoder
            diffae_ctx.latent_proj = diffae_ctx.ema_latent_proj if diffae_ctx.ema_latent_proj is not None else diffae_ctx.latent_proj
            diffae_ctx.encoder.eval()
            diffae_ctx.decoder.eval()
            diffae_ctx.latent_proj.eval()
            has_diffae = True
            print(f"DiffAE : {diffae_ckpt}")
        else:
            print("DiffAE : no checkpoint found — skipping")
    except Exception as e:
        print(f"DiffAE : could not load — {e}")

    if not has_ae and not has_diffae:
        print("ERROR: No models found. Train AE or DiffAE first.")
        return

    # Use whichever context is available for the data loader
    ref_ctx = ae_ctx if has_ae else diffae_ctx
    assert ref_ctx is not None  # guaranteed by the checks above

    loader     = ref_ctx.loader
    n_channels = ref_ctx.n_channels
    n_time     = ref_ctx.n_time_points

    # ------------------------------------------------------------------ #
    #  Collect reconstructions
    # ------------------------------------------------------------------ #
    all_raw        = []
    all_ae_rec     = []
    all_diffae_rec = []

    n_batches = (args.n_samples + args.batch_size - 1) // args.batch_size
    collected = 0

    pbar = tqdm(range(n_batches), desc="Reconstructing", ncols=90)
    for _ in pbar:
        B = min(args.batch_size, args.n_samples - collected)
        if B <= 0:
            break

        wf_col, *_ = loader.get_batch(B)          # (B, N, 1)  raw (not normalised)
        raw_np = wf_col[:, :, 0]                  # (B, N)
        all_raw.append(raw_np)

        if has_ae:
            assert ae_ctx is not None
            wf_norm = ae_ctx.data_stats.normalize(wf_col)
            x_ae    = torch.from_numpy(wf_norm.astype(np.float32)).to(device)
            rec     = reconstruct_ae(
                encoder  = ae_ctx.encoder,
                decoder  = ae_ctx.decoder,
                A_sparse = ae_ctx.A_sparse,
                pos      = ae_ctx.pos,
                x_ref    = x_ae,
            )
            rec_np = ae_ctx.data_stats.denormalize(rec.cpu().numpy())  # (B, 1, N)
            rec_np = np.clip(rec_np, 0, None)
            all_ae_rec.append(rec_np[:, 0, :])

        if has_diffae:
            assert diffae_ctx is not None
            wf_norm    = diffae_ctx.data_stats.normalize(wf_col)
            x_diffae   = torch.from_numpy(wf_norm.astype(np.float32)).to(device)
            rec        = sample_diffae(
                encoder     = diffae_ctx.encoder,
                decoder     = diffae_ctx.decoder,
                latent_proj = diffae_ctx.latent_proj,
                schedule    = diffae_ctx.schedule,
                A_sparse    = diffae_ctx.A_sparse,
                pos         = diffae_ctx.pos,
                time_dim    = cfg.conditioning.time_dim,
                x_ref       = x_diffae,
                parametrization = cfg.diffusion.parametrization,
                pbar        = False,
            )
            rec_np = diffae_ctx.data_stats.denormalize(rec.cpu().numpy())  # (B, 1, N)
            rec_np = np.clip(rec_np, 0, None)
            all_diffae_rec.append(rec_np[:, 0, :])

        collected += B
        pbar.set_postfix(n=collected)

    raw = np.concatenate(all_raw, axis=0)
    print(f"\nComputing RQs for {raw.shape[0]} events…")

    rq_true = collect_rqs(raw, n_channels, n_time)

    # ------------------------------------------------------------------ #
    #  Plot DiffAE comparison
    # ------------------------------------------------------------------ #
    if has_diffae:
        diffae_rec = np.concatenate(all_diffae_rec, axis=0)
        rq_diffae  = collect_rqs(diffae_rec, n_channels, n_time)
        plot_distributions(
            rq_true    = rq_true,
            rq_gen     = rq_diffae,
            label_true = "Original",
            label_gen  = "DiffAE",
            color_true = COLORS["truth"],
            color_gen  = COLORS["diffae"],
            output_path = os.path.join(args.output_dir, "diffae_rq_distributions.png"),
            n_bins     = args.n_bins,
            title      = "RQ Distributions — Original vs DiffAE Reconstructions",
        )

    # ------------------------------------------------------------------ #
    #  Plot AE comparison
    # ------------------------------------------------------------------ #
    if has_ae:
        ae_rec  = np.concatenate(all_ae_rec, axis=0)
        rq_ae   = collect_rqs(ae_rec, n_channels, n_time)
        plot_distributions(
            rq_true    = rq_true,
            rq_gen     = rq_ae,
            label_true = "Original",
            label_gen  = "AE",
            color_true = COLORS["truth"],
            color_gen  = COLORS["ae"],
            output_path = os.path.join(args.output_dir, "ae_rq_distributions.png"),
            n_bins     = args.n_bins,
            title      = "RQ Distributions — Original vs AE Reconstructions",
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
