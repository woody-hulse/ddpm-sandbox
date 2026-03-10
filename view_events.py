"""
View and compress/reconstruct individual events.

Usage:
    python view_events.py --view                          # plot 8 random MS events
    python view_events.py --view --n 16 --seed 7          # 16 events, specific seed
    python view_events.py --compress 3 --model ae         # reconstruct event 3 with AE
    python view_events.py --compress 3 --model diffae     # reconstruct event 3 with DiffAE
    python view_events.py --compress 3 --model both       # side-by-side AE + DiffAE
"""
import os
import argparse
from typing import Optional

import numpy as np
import h5py
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config, default_config, get_config
from ae import AEContext, reconstruct_ae
from diffae import DiffAEContext, sample_diffae, apply_lopsided_augmentation
from compare_rqs import wf_to_z_profile


def load_events(cfg: Config, indices: np.ndarray) -> np.ndarray:
    """Load specific MS events by generating them from a seeded batcher.

    Returns (len(indices), N, 1) array of raw waveforms.
    """
    ctx = AEContext.build(cfg, for_training=False, verbose=False)
    batch_np, cond, *_ = ctx.loader.get_batch(len(indices))
    return batch_np, cond, ctx


def load_ae(cfg: Config):
    cfg.conditioning.cond_proj_dim = max(cfg.conditioning.cond_proj_dim, cfg.encoder.latent_dim)
    ctx = AEContext.build(cfg, for_training=True, verbose=False)
    ckpt = ctx.latest_checkpoint()
    if ckpt is None:
        raise FileNotFoundError(f"No AE checkpoint found in {ctx.checkpoint_dir}")
    ctx.load_checkpoint(ckpt, load_optim=False)
    enc = ctx.ema_encoder if ctx.ema_encoder is not None else ctx.encoder
    dec = ctx.ema_decoder if ctx.ema_decoder is not None else ctx.decoder
    enc.eval()
    dec.eval()
    print(f"AE loaded from {ckpt}")
    return ctx, enc, dec


def load_diffae(cfg: Config):
    cfg.conditioning.cond_proj_dim = max(cfg.conditioning.cond_proj_dim, cfg.encoder.latent_dim)
    ctx = DiffAEContext.build(cfg, for_training=True, verbose=False)
    ckpt = ctx.latest_checkpoint()
    if ckpt is None:
        raise FileNotFoundError(f"No DiffAE checkpoint found in {ctx.checkpoint_dir}")
    ctx.load_checkpoint(ckpt, load_optim=False)
    enc = ctx.ema_encoder if ctx.ema_encoder is not None else ctx.encoder
    dec = ctx.ema_decoder if ctx.ema_decoder is not None else ctx.decoder
    lp = ctx.ema_latent_proj if ctx.ema_latent_proj is not None else ctx.latent_proj
    enc.eval()
    dec.eval()
    lp.eval()
    print(f"DiffAE loaded from {ckpt}")
    return ctx, enc, dec, lp


@torch.no_grad()
def reconstruct_with_ae(ctx, enc, dec, wf_raw: np.ndarray) -> np.ndarray:
    """Reconstruct raw waveforms with AE. Input/output: (B, N, 1) unnormalized."""
    wf_norm = ctx.data_stats.normalize(wf_raw)
    x = torch.from_numpy(wf_norm.astype(np.float32)).to(ctx.device)
    rec = reconstruct_ae(enc, dec, ctx.A_sparse, ctx.pos, x)
    rec_np = ctx.data_stats.denormalize(rec.cpu().numpy())
    return np.clip(rec_np, 0, None)[:, 0, :]


@torch.no_grad()
def reconstruct_with_diffae(ctx, enc, dec, lp, cfg, wf_raw: np.ndarray) -> np.ndarray:
    """Reconstruct raw waveforms with DiffAE. Input/output: (B, N, 1) unnormalized."""
    wf_norm = ctx.data_stats.normalize(wf_raw)
    x = torch.from_numpy(wf_norm.astype(np.float32)).to(ctx.device)
    rec = sample_diffae(
        enc, dec, lp, ctx.schedule, ctx.A_sparse, ctx.pos,
        cfg.conditioning.time_dim, x,
        parametrization=cfg.diffusion.parametrization,
    )
    rec_np = ctx.data_stats.denormalize(rec.cpu().numpy())
    return np.clip(rec_np, 0, None)[:, 0, :]


def cmd_view(args):
    cfg = default_config
    np.random.seed(args.seed)
    ctx = AEContext.build(cfg, for_training=False, verbose=False)
    batch_np, cond, *_ = ctx.loader.get_batch(args.n)
    if args.lopsided:
        batch_np = apply_lopsided_augmentation(batch_np, frac=1.0, sigma=args.lopsided_sigma)
    n_channels = ctx.n_channels
    n_time = ctx.n_time_points

    cols = 4
    rows = (args.n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.5 * rows), squeeze=False)
    time_axis = np.arange(n_time)

    for i in range(args.n):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        z = wf_to_z_profile(batch_np[i, :, 0], n_channels, n_time)
        ax.plot(time_axis, z, color="black", linewidth=1.2)
        ax.fill_between(time_axis, z, alpha=0.1, color="black")
        ax.set_title(f"Event {i}", fontsize=9)
        ax.set_ylim(0, max(z.max() * 1.15, 1))
        if r == rows - 1:
            ax.set_xlabel("Time bin")
        if c == 0:
            ax.set_ylabel("Amplitude")

    for i in range(args.n, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].set_visible(False)

    title = f"Random MS Events (seed={args.seed})"
    if args.lopsided:
        title += f"  [lopsided σ={args.lopsided_sigma}]"
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    suffix = f"_lopsided_s{args.lopsided_sigma}" if args.lopsided else ""
    out = os.path.join(args.output_dir, f"view_events{suffix}.png")
    os.makedirs(args.output_dir, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def cmd_compress(args):
    cfg = get_config(latent_dim=args.latent_dim)
    np.random.seed(args.seed)
    ctx_ref = AEContext.build(cfg, for_training=False, verbose=False)
    n_needed = args.compress + 1
    batch_np, cond, *_ = ctx_ref.loader.get_batch(n_needed)
    n_channels = ctx_ref.n_channels
    n_time = ctx_ref.n_time_points
    time_axis = np.arange(n_time)

    if args.lopsided:
        batch_np = apply_lopsided_augmentation(batch_np, frac=1.0, sigma=args.lopsided_sigma)

    idx = args.compress
    wf_single = batch_np[idx: idx + 1]
    raw_flat = batch_np[idx, :, 0]
    z_raw = wf_to_z_profile(raw_flat, n_channels, n_time)

    models_to_plot = []
    model_arg = args.model.lower()

    if model_arg in ("ae", "both"):
        try:
            ae_ctx, ae_enc, ae_dec = load_ae(cfg)
            ae_rec = reconstruct_with_ae(ae_ctx, ae_enc, ae_dec, wf_single)
            z_ae = wf_to_z_profile(ae_rec[0], n_channels, n_time)
            models_to_plot.append(("AE", z_ae, "#1f77b4"))
        except Exception as e:
            print(f"AE failed: {e}")

    if model_arg in ("diffae", "both"):
        try:
            dae_ctx, dae_enc, dae_dec, dae_lp = load_diffae(cfg)
            dae_rec = reconstruct_with_diffae(dae_ctx, dae_enc, dae_dec, dae_lp, cfg, wf_single)
            z_dae = wf_to_z_profile(dae_rec[0], n_channels, n_time)
            models_to_plot.append(("DiffAE", z_dae, "#ff7f0e"))
        except Exception as e:
            print(f"DiffAE failed: {e}")

    if not models_to_plot:
        print("No models loaded successfully.")
        return

    n_cols = 1 + len(models_to_plot)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 3.5), squeeze=False)
    y_max = max(z_raw.max() * 1.15, 1)

    ax = axes[0, 0]
    ax.plot(time_axis, z_raw, color="black", linewidth=1.5)
    ax.fill_between(time_axis, z_raw, alpha=0.12, color="black")
    ax.set_title("Raw", fontweight="bold")
    ax.set_xlabel("Time bin")
    ax.set_ylabel("Amplitude")
    ax.set_ylim(0, y_max)

    for col, (name, z_rec, color) in enumerate(models_to_plot, start=1):
        ax = axes[0, col]
        ax.plot(time_axis, z_raw, color="black", linewidth=0.8, alpha=0.3, label="Raw")
        ax.plot(time_axis, z_rec, color=color, linewidth=1.5, label=name)
        ax.fill_between(time_axis, z_rec, alpha=0.15, color=color)
        ax.set_title(f"{name} (z={cfg.encoder.latent_dim})", fontweight="bold")
        ax.set_xlabel("Time bin")
        ax.set_ylim(0, y_max)
        ax.legend(fontsize=8, loc="upper right")

    title = f"Event {idx}  (seed={args.seed}, latent_dim={cfg.encoder.latent_dim})"
    if args.lopsided:
        title += f"  [lopsided σ={args.lopsided_sigma}]"
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    suffix = f"_lopsided_s{args.lopsided_sigma}" if args.lopsided else ""
    out = os.path.join(args.output_dir, f"compress_event{idx}_{model_arg}_z{cfg.encoder.latent_dim}{suffix}.png")
    os.makedirs(args.output_dir, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser(description="View or compress/reconstruct events")
    parser.add_argument("--view", action="store_true", help="Plot random MS events")
    parser.add_argument("--compress", type=int, default=None, metavar="IDX",
                        help="Compress and reconstruct event at this index")
    parser.add_argument("--model", type=str, default="both", choices=["ae", "diffae", "both"],
                        help="Model to use for --compress")
    parser.add_argument("--latent-dim", type=int, default=None,
                        help="Latent dim (default: from config)")
    parser.add_argument("--n", type=int, default=8, help="Number of events for --view")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducible events")
    parser.add_argument("--lopsided", action="store_true",
                        help="Gaussian-blur the first half of each event")
    parser.add_argument("--lopsided-sigma", type=float, default=3.0,
                        help="Gaussian kernel sigma for --lopsided (default: 3.0)")
    parser.add_argument("--output-dir", type=str, default="event_plots")
    args = parser.parse_args()

    if args.latent_dim is None:
        args.latent_dim = default_config.encoder.latent_dim

    if not args.view and args.compress is None:
        parser.error("Specify --view or --compress IDX")

    if args.view:
        cmd_view(args)

    if args.compress is not None:
        cmd_compress(args)


if __name__ == "__main__":
    main()
