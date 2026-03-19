"""
Plot RQ distributional overlays comparing original events to DiffAE and AE reconstructions.

Generates two figures:
  - diffae_rq_distributions.png : original vs DiffAE per-RQ histogram overlays
  - ae_rq_distributions.png     : original vs AE per-RQ histogram overlays

Usage:
    python plot_rq_distributions.py
    python plot_rq_distributions.py --n-samples 1000 --batch-size 16
    python plot_rq_distributions.py --output-dir results/rq_dist --n-bins 60
    python plot_rq_distributions.py --latent-dim 128
"""

import os
import glob
import argparse
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import ks_2samp

from plot_style import apply_style, COLORS
from config import default_config, get_config
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
    "width_10_90":    "Width 10–90",
    "std_dev":        "Std Dev",
}

N_COLS = 4


# -------------------------------------------------------------------------
# Config auto-detection from checkpoint keys
# -------------------------------------------------------------------------

def _latest_ckpt(directory: str, prefix: str) -> Optional[str]:
    """Return the checkpoint with the highest epoch number in *directory*."""
    pattern = os.path.join(directory, f"{prefix}_epoch_*.pt")
    files = glob.glob(pattern)
    if not files:
        return None
    def _epoch(p: str) -> int:
        try:
            return int(os.path.basename(p).split("_epoch_")[1].replace(".pt", ""))
        except Exception:
            return -1
    return max(files, key=_epoch)


def _infer_ae_cfg(ckpt_path: str) -> Tuple[str, int]:
    """
    Peek at an AE checkpoint and return (encoder_type, latent_dim).

    Encoder-type detection:
      - keys starting with 'mlp.'  → 'mlp'
      - keys starting with 'backbone.' → 'cnn'  (Conv1DEncoder)
    latent_dim: from the final linear layer of the encoder.
    """
    chk = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    enc = chk.get("ema_encoder") or chk["encoder"]
    keys = list(enc.keys())

    if any(k.startswith("mlp.") for k in keys):
        encoder_type = "mlp"
    else:
        encoder_type = "cnn"

    # latent_dim = output dim of the last linear layer
    # Conv1DEncoder: 'to_latent.weight'   shape [latent_dim, ...]
    # MLPEncoder:    last 'mlp.X.weight'  shape [latent_dim, ...]
    linear_keys = [k for k in keys if k.endswith(".weight") and len(enc[k].shape) == 2]
    latent_dim = int(enc[linear_keys[-1]].shape[0])

    return encoder_type, latent_dim


def _infer_diffae_cfg(ckpt_path: str) -> Tuple[str, int, int]:
    """
    Peek at a DiffAE checkpoint and return (encoder_type, latent_dim, cond_proj_dim).

    latent_proj architecture:
        Linear(latent_dim,  cond_proj_dim * 2)  →  key '0.weight' shape [cond_proj_dim*2, latent_dim]
        SiLU
        Linear(cond_proj_dim*2, cond_proj_dim)  →  key '2.weight' shape [cond_proj_dim, cond_proj_dim*2]
    """
    chk = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # latent_proj
    lp = chk.get("ema_latent_proj") or chk["latent_proj"]
    w0 = lp["0.weight"]                      # shape [cond_proj_dim*2, latent_dim]
    latent_dim   = int(w0.shape[1])
    cond_proj_dim = int(w0.shape[0]) // 2

    # encoder type
    enc = chk.get("ema_encoder") or chk["encoder"]
    enc_keys = list(enc.keys())
    if any(k.startswith("mlp.") for k in enc_keys):
        encoder_type = "mlp"
    else:
        encoder_type = "cnn"

    return encoder_type, latent_dim, cond_proj_dim


# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------

def _hist_range(true_vals: np.ndarray, gen_vals: np.ndarray, clip_pct: float = 99.5):
    """Shared histogram range, clipping outliers."""
    finite = np.concatenate([true_vals, gen_vals])
    finite = finite[np.isfinite(finite)]
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
    Grid of histogram overlays (one subplot per RQ).
    Y-axis is probability density; histograms are semi-transparent fills.
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

        t_finite = rq_true[rq_name]
        g_finite = rq_gen[rq_name]
        t_finite = t_finite[np.isfinite(t_finite)]
        g_finite = g_finite[np.isfinite(g_finite)]

        if len(t_finite) < 2 or len(g_finite) < 2:
            ax.set_visible(False)
            continue

        rng = _hist_range(t_finite, g_finite)
        if rng is None:
            ax.set_visible(False)
            continue

        bins  = np.linspace(rng[0], rng[1], n_bins + 1)
        unit  = RQ_UNITS.get(rq_name, "")
        xlabel = RQ_DISPLAY_NAMES.get(rq_name, rq_name) + (f" [{unit}]" if unit else "")

        ax.hist(t_finite, bins=bins, density=True,
                color=color_true, alpha=0.55, linewidth=0.5, edgecolor=color_true,
                label=label_true)
        ax.hist(g_finite, bins=bins, density=True,
                color=color_gen,  alpha=0.55, linewidth=0.5, edgecolor=color_gen,
                label=label_gen)

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Probability Density")
        ax.legend(loc="upper right", fontsize=8)

        ks_stat, _ = ks_2samp(t_finite, g_finite)
        ax.set_title(f"KS = {ks_stat:.3f}", fontsize=9)

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


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Plot RQ distributional overlays")
    parser.add_argument("--n-samples",   type=int,   default=500,           help="Number of events")
    parser.add_argument("--batch-size",  type=int,   default=16,            help="Inference batch size")
    parser.add_argument("--n-bins",      type=int,   default=50,            help="Histogram bins")
    parser.add_argument("--output-dir",  type=str,   default="rq_dist_plots")
    parser.add_argument("--latent-dim",  type=int,   default=64,
                        help="Latent dim used to locate checkpoint subdirs (e.g. 64 → ae_z64/diffae_z64)")
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    cfg = default_config
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device : {device}")
    print(f"Samples: {args.n_samples}")

    ckpt_root = cfg.paths.checkpoint_dir  # "checkpoints"

    # ------------------------------------------------------------------ #
    #  Load AE — peek at checkpoint to infer architecture before building
    # ------------------------------------------------------------------ #
    has_ae = False
    ae_ctx = None

    ae_dir  = os.path.join(ckpt_root, cfg.paths.ae_subdir.format(latent_dim=args.latent_dim))
    ae_ckpt = _latest_ckpt(ae_dir, "ae")
    if ae_ckpt is None:
        print(f"AE     : no checkpoint found in {ae_dir} — skipping")
    else:
        try:
            enc_type, latent_dim = _infer_ae_cfg(ae_ckpt)
            ae_cfg = get_config(encoder_type=enc_type, latent_dim=latent_dim)
            ae_cfg.encoder.hidden_dim = max(ae_cfg.encoder.hidden_dim, latent_dim)
            print(f"AE     : {ae_ckpt}  (encoder={enc_type}, z={latent_dim})")

            ae_ctx = AEContext.build(ae_cfg, for_training=False, verbose=False, use_ms_data=True)
            ae_ctx.load_checkpoint(ae_ckpt, load_optim=False)
            ae_ctx.encoder = ae_ctx.ema_encoder if ae_ctx.ema_encoder is not None else ae_ctx.encoder
            ae_ctx.decoder = ae_ctx.ema_decoder if ae_ctx.ema_decoder is not None else ae_ctx.decoder
            ae_ctx.encoder.eval()
            ae_ctx.decoder.eval()
            has_ae = True
        except Exception as e:
            print(f"AE     : could not load — {e}")

    # ------------------------------------------------------------------ #
    #  Load DiffAE — same pattern
    # ------------------------------------------------------------------ #
    has_diffae = False
    diffae_ctx = None

    diffae_dir  = os.path.join(ckpt_root, cfg.paths.diffae_subdir.format(latent_dim=args.latent_dim))
    diffae_ckpt = _latest_ckpt(diffae_dir, "diffae")
    if diffae_ckpt is None:
        print(f"DiffAE : no checkpoint found in {diffae_dir} — skipping")
    else:
        try:
            enc_type, latent_dim, cond_proj_dim = _infer_diffae_cfg(diffae_ckpt)
            diffae_cfg = get_config(encoder_type=enc_type, latent_dim=latent_dim)
            diffae_cfg.encoder.hidden_dim    = max(diffae_cfg.encoder.hidden_dim, latent_dim)
            diffae_cfg.conditioning.cond_proj_dim = cond_proj_dim
            print(f"DiffAE : {diffae_ckpt}  (encoder={enc_type}, z={latent_dim}, cond={cond_proj_dim})")

            diffae_ctx = DiffAEContext.build(diffae_cfg, for_training=False, verbose=False, use_ms_data=True)
            diffae_ctx.load_checkpoint(diffae_ckpt, load_optim=False)
            diffae_ctx.encoder     = diffae_ctx.ema_encoder     if diffae_ctx.ema_encoder     is not None else diffae_ctx.encoder
            diffae_ctx.decoder     = diffae_ctx.ema_decoder     if diffae_ctx.ema_decoder     is not None else diffae_ctx.decoder
            diffae_ctx.latent_proj = diffae_ctx.ema_latent_proj if diffae_ctx.ema_latent_proj is not None else diffae_ctx.latent_proj
            diffae_ctx.encoder.eval()
            diffae_ctx.decoder.eval()
            diffae_ctx.latent_proj.eval()
            has_diffae = True
        except Exception as e:
            print(f"DiffAE : could not load — {e}")

    if not has_ae and not has_diffae:
        print("ERROR: No models loaded. Check checkpoint paths or --latent-dim.")
        return

    ref_ctx = ae_ctx if has_ae else diffae_ctx
    assert ref_ctx is not None
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

        wf_col, *_ = loader.get_batch(B)       # (B, N, 1) raw
        all_raw.append(wf_col[:, :, 0])        # (B, N)

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
            rec_np = ae_ctx.data_stats.denormalize(rec.cpu().numpy())   # (B, 1, N)
            all_ae_rec.append(np.clip(rec_np[:, 0, :], 0, None))

        if has_diffae:
            assert diffae_ctx is not None
            wf_norm  = diffae_ctx.data_stats.normalize(wf_col)
            x_diffae = torch.from_numpy(wf_norm.astype(np.float32)).to(device)
            rec      = sample_diffae(
                encoder         = diffae_ctx.encoder,
                decoder         = diffae_ctx.decoder,
                latent_proj     = diffae_ctx.latent_proj,
                schedule        = diffae_ctx.schedule,
                A_sparse        = diffae_ctx.A_sparse,
                pos             = diffae_ctx.pos,
                time_dim        = diffae_cfg.conditioning.time_dim,
                x_ref           = x_diffae,
                parametrization = diffae_cfg.diffusion.parametrization,
                pbar            = False,
            )
            rec_np = diffae_ctx.data_stats.denormalize(rec.cpu().numpy())  # (B, 1, N)
            all_diffae_rec.append(np.clip(rec_np[:, 0, :], 0, None))

        collected += B
        pbar.set_postfix(n=collected)

    raw = np.concatenate(all_raw, axis=0)
    print(f"\nComputing RQs for {raw.shape[0]} events…")
    rq_true = collect_rqs(raw, n_channels, n_time)

    # ------------------------------------------------------------------ #
    #  Plots
    # ------------------------------------------------------------------ #
    if has_diffae:
        diffae_rec = np.concatenate(all_diffae_rec, axis=0)
        rq_diffae  = collect_rqs(diffae_rec, n_channels, n_time)
        plot_distributions(
            rq_true     = rq_true,
            rq_gen      = rq_diffae,
            label_true  = "Original",
            label_gen   = "DiffAE",
            color_true  = COLORS["truth"],
            color_gen   = COLORS["diffae"],
            output_path = os.path.join(args.output_dir, "diffae_rq_distributions.png"),
            n_bins      = args.n_bins,
            title       = "RQ Distributions — Original vs DiffAE Reconstructions",
        )

    if has_ae:
        ae_rec = np.concatenate(all_ae_rec, axis=0)
        rq_ae  = collect_rqs(ae_rec, n_channels, n_time)
        plot_distributions(
            rq_true     = rq_true,
            rq_gen      = rq_ae,
            label_true  = "Original",
            label_gen   = "AE",
            color_true  = COLORS["truth"],
            color_gen   = COLORS["ae"],
            output_path = os.path.join(args.output_dir, "ae_rq_distributions.png"),
            n_bins      = args.n_bins,
            title       = "RQ Distributions — Original vs AE Reconstructions",
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
