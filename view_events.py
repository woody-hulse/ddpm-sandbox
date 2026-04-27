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
from lz_data_loader import TritiumSSDataLoader


def apply_lopsided_to_profile(profile: np.ndarray, sigma: float) -> np.ndarray:
    """Blur only the first half of a 1D summed event profile."""
    from scipy.ndimage import gaussian_filter1d

    out = np.asarray(profile, dtype=np.float32).copy()
    half = out.shape[0] // 2
    out[:half] = gaussian_filter1d(out[:half], sigma=sigma)
    return out


def load_events(cfg: Config, indices: np.ndarray) -> np.ndarray:
    """Load specific MS events by generating them from a seeded batcher.

    Returns (len(indices), N, 1) array of raw waveforms.
    """
    ctx = AEContext.build(cfg, for_training=False, verbose=False)
    batch_np, cond, *_ = ctx.loader.get_batch(len(indices))
    return batch_np, cond, ctx


def load_ae(cfg: Config):
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
    ctx = DiffAEContext.build(cfg, for_training=True, verbose=False)
    ckpt = ctx.latest_checkpoint()
    if ckpt is None:
        raise FileNotFoundError(f"No DiffAE checkpoint found in {ctx.checkpoint_dir}")
    ctx.load_checkpoint(ckpt, load_optim=False)
    enc = ctx.ema_encoder if ctx.ema_encoder is not None else ctx.encoder
    dec = ctx.ema_decoder if ctx.ema_decoder is not None else ctx.decoder
    enc.eval()
    dec.eval()
    print(f"DiffAE loaded from {ckpt}")
    return ctx, enc, dec


@torch.no_grad()
def reconstruct_with_ae(ctx, enc, dec, wf_raw: np.ndarray) -> np.ndarray:
    """Reconstruct raw waveforms with AE. Input/output: (B, N, 1) unnormalized."""
    wf_norm = ctx.data_stats.normalize(wf_raw)
    x = torch.from_numpy(wf_norm.astype(np.float32)).to(ctx.device)
    rec = reconstruct_ae(enc, dec, ctx.A_sparse, ctx.pos, x)
    rec_np = ctx.data_stats.denormalize(rec.cpu().numpy())
    return np.clip(rec_np, 0, None)[:, 0, :]


@torch.no_grad()
def reconstruct_with_diffae(ctx, enc, dec, cfg, wf_raw: np.ndarray) -> np.ndarray:
    """Reconstruct raw waveforms with DiffAE. Input/output: (B, N, 1) unnormalized."""
    wf_norm = ctx.data_stats.normalize(wf_raw)
    x = torch.from_numpy(wf_norm.astype(np.float32)).to(ctx.device)
    rec = sample_diffae(
        enc, dec, ctx.schedule, ctx.A_sparse, ctx.pos,
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
    n_channels = ctx.n_channels
    n_time = ctx.n_time_points

    n_plot = min(args.n, 3)
    fig, axes = plt.subplots(1, n_plot, figsize=(5.0 * n_plot, 3.6), squeeze=False, sharey=True)
    z_axis = np.linspace(0.0, 1.0, n_time)
    z_profiles = []
    for i in range(n_plot):
        z = wf_to_z_profile(batch_np[i, :, 0], n_channels, n_time)
        if args.lopsided:
            z = apply_lopsided_to_profile(z, sigma=args.lopsided_sigma)
        z_profiles.append(z)
    y_max = max(max(z.max() for z in z_profiles) * 1.25, 1.0)

    for i in range(n_plot):
        ax = axes[0, i]
        z = z_profiles[i]
        ax.plot(z_axis, z, linewidth=2.0)
        ax.set_title(f"Event {i + 1}", fontsize=11)
        ax.set_ylim(0, y_max)
        ax.set_xlabel("z")
        if i == 0:
            ax.set_ylabel("intensity")
        ax.grid(False)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

    title = "Sampled lopsided MS events" if args.lopsided else "MS events"
    fig.suptitle(title, fontsize=14)
    fig.subplots_adjust(top=0.82, wspace=0.14)
    suffix = f"_lopsided_s{args.lopsided_sigma}" if args.lopsided else ""
    out = os.path.join(args.output_dir, f"view_events{suffix}.png")
    os.makedirs(args.output_dir, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def cmd_view_ss(args):
    cfg = default_config
    np.random.seed(args.seed)
    loader = TritiumSSDataLoader(cfg.paths.tritium_h5, cfg.paths.channel_positions)
    idx = np.random.randint(0, loader.n_samples, size=args.n)
    uniq_idx, inverse = np.unique(idx, return_inverse=True)
    import h5py
    with h5py.File(cfg.paths.tritium_h5, 'r') as f:
        wf = f['waveforms'][uniq_idx][inverse]   # (n, n_channels, n_time)
        xc = f['xc'][uniq_idx][inverse].astype(np.float32)
        yc = f['yc'][uniq_idx][inverse].astype(np.float32)
        dt = f['dt'][uniq_idx][inverse].astype(np.float32)
    pmt_pos = loader.channel_positions           # (n_channels, 2)
    n_time = loader.n_time_points

    fig, axes = plt.subplots(args.n, 2, figsize=(12, 3.5 * args.n), squeeze=False)

    for i in range(args.n):
        ax_xy, ax_t = axes[i]

        charge = wf[i].sum(axis=1)              # (n_channels,)
        sc = ax_xy.scatter(
            pmt_pos[:, 0], pmt_pos[:, 1],
            c=charge, cmap="viridis", s=80, edgecolors="k", linewidths=0.3,
        )
        ax_xy.scatter([xc[i]], [yc[i]], marker="x", color="red", s=80,
                      linewidths=1.5, label=f"({xc[i]:.1f}, {yc[i]:.1f}) cm")
        ax_xy.set_title(f"Event {i+1}  PMT hit map", fontsize=9)
        ax_xy.set_xlabel("x (cm)")
        ax_xy.set_ylabel("y (cm)")
        ax_xy.set_aspect("equal")
        ax_xy.legend(fontsize=7)
        plt.colorbar(sc, ax=ax_xy, label="Charge (AU)")

        summed = wf[i].sum(axis=0)              # (n_time,)
        t_axis = np.arange(n_time)
        ax_t.plot(t_axis, summed, linewidth=0.8, color="steelblue")
        ax_t.axvline(dt[i], color="red", linestyle="--", linewidth=0.8,
                     label=f"dt={dt[i]:.0f} bins")
        ax_t.set_title(f"Event {i+1}  summed waveform", fontsize=9)
        ax_t.set_xlabel("Time bin")
        ax_t.set_ylabel("Amplitude (AU)")
        ax_t.legend(fontsize=7)

    fig.suptitle(f"Tritium SS events  (seed={args.seed})", fontweight="bold")
    fig.tight_layout()
    out = os.path.join(args.output_dir, f"view_ss_events_seed{args.seed}.png")
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
            dae_ctx, dae_enc, dae_dec = load_diffae(cfg)
            dae_rec = reconstruct_with_diffae(dae_ctx, dae_enc, dae_dec, cfg, wf_single)
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
    parser.add_argument("--ss", action="store_true", help="Plot SS events from the tritium dataset (PMT hit map + summed waveform)")
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

    if not args.view and not args.ss and args.compress is None:
        parser.error("Specify --view, --ss, or --compress IDX")

    if args.view:
        cmd_view(args)

    if args.ss:
        cmd_view_ss(args)

    if args.compress is not None:
        cmd_compress(args)


if __name__ == "__main__":
    main()
