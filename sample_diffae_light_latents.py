"""
Sample new synthetic events from a trained DiffAE Light model by drawing fresh
latents and running the diffusion decoder.

This script:
1. Loads a trained `diffae_light` checkpoint.
2. Fits a simple diagonal Gaussian latent prior, either from saved encoded
   latents or by encoding a batch of real events.
3. Samples new latent vectors.
4. Runs the reverse diffusion decoder conditioned on those latents.
5. Saves the sampled waveforms plus a quick-look figure.

Example:
    python sample_diffae_light_latents.py --latent-dim 64 --n-samples 6 --seed 42
"""
import argparse
import os
from typing import Optional, Tuple

import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from config import get_config
from diffae_light import (
    DiffAELightContext,
    _encode_with_context,
    sample_from_latent_diffae_light,
)
from plot_style import COLORS, apply_style
from plot_real_event_3d import plot_waveform_3d_scatter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-dim", type=int, default=64, help="Latent size used by the trained DiffAE Light checkpoint.")
    parser.add_argument("--n-samples", type=int, default=4, help="Number of synthetic events to generate.")
    parser.add_argument("--prior", choices=("empirical", "standard"), default="empirical",
                        help="Latent prior: fit a diagonal Gaussian from encoded data, or use N(0, I).")
    parser.add_argument("--prior-samples", type=int, default=512,
                        help="Number of real events to encode when fitting the empirical latent prior.")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for latent fitting and sampling.")
    parser.add_argument("--latent-temperature", type=float, default=1.0,
                        help="Multiplier applied to the latent standard deviation before sampling.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Optional explicit checkpoint path. Defaults to the latest diffae_light checkpoint for this latent size.")
    parser.add_argument("--latents-h5", type=str, default=None,
                        help="Optional HDF5 file containing a `latents` dataset to use for the empirical prior.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for sampled waveforms and figures. Defaults to plots/diffae_light_z*/latent_samples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default=None, help="Torch device override, e.g. `cpu` or `cuda`.")
    parser.add_argument("--tritium-h5", type=str, default=None, help="Override the waveform dataset path from config.")
    parser.add_argument("--channel-positions", type=str, default=None, help="Override the PMT positions path from config.")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Override the root checkpoint directory from config.")
    parser.add_argument("--plot-dir", type=str, default=None, help="Override the root plot directory from config.")
    parser.add_argument("--parametrization", choices=("v", "eps"), default=None,
                        help="Override the diffusion parametrization stored in config.")
    parser.add_argument("--use-ss-data", action="store_true",
                        help="Build the context with SS data instead of the default online MS batcher.")
    parser.add_argument("--pbar", action="store_true", help="Show the diffusion sampling progress bar.")
    return parser.parse_args()


def build_cfg(args: argparse.Namespace):
    cfg = get_config(latent_dim=args.latent_dim)
    if args.device is not None:
        cfg.device = args.device
    if args.parametrization is not None:
        cfg.diffusion.parametrization = args.parametrization
    if args.tritium_h5 is not None:
        cfg.paths.tritium_h5 = args.tritium_h5
    if args.channel_positions is not None:
        cfg.paths.channel_positions = args.channel_positions
    if args.checkpoint_dir is not None:
        cfg.paths.checkpoint_dir = args.checkpoint_dir
    if args.plot_dir is not None:
        cfg.paths.plot_dir = args.plot_dir
    return cfg


def find_checkpoint(ctx: DiffAELightContext, explicit_path: Optional[str]) -> str:
    checkpoint_path = explicit_path or ctx.latest_checkpoint()
    if checkpoint_path is None:
        raise FileNotFoundError(
            f"No DiffAE Light checkpoint found in {ctx.checkpoint_dir}. "
            "Pass --checkpoint explicitly or train the model first."
        )
    return checkpoint_path


def inference_modules(ctx: DiffAELightContext) -> Tuple[torch.nn.Module, torch.nn.Module]:
    encoder = ctx.ema_encoder if ctx.ema_encoder is not None else ctx.encoder
    decoder = ctx.ema_decoder if ctx.ema_decoder is not None else ctx.decoder
    encoder.eval()
    decoder.eval()
    return encoder, decoder


def load_latents_from_h5(path: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        if "latents" not in f:
            raise KeyError(f"{path} does not contain a `latents` dataset.")
        latents = f["latents"][:]
    latents = np.asarray(latents, dtype=np.float32)
    if latents.ndim != 2:
        raise ValueError(f"Expected latents to have shape (N, D), got {latents.shape}.")
    return latents


@torch.no_grad()
def encode_latents_from_loader(
    ctx: DiffAELightContext,
    encoder: torch.nn.Module,
    n_samples: int,
    batch_size: int,
) -> np.ndarray:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive when fitting the empirical latent prior from the loader.")
    latents = []
    remaining = n_samples
    while remaining > 0:
        current_batch = min(batch_size, remaining)
        wf_batch, *_ = ctx.loader.get_batch(current_batch)
        wf_norm = ctx.data_stats.normalize(wf_batch).astype(np.float32)
        x_ref = torch.from_numpy(wf_norm).to(ctx.device)
        x_flat = x_ref.reshape(current_batch * ctx.n_nodes, 1)
        z, _, _ = _encode_with_context(ctx, x_flat, current_batch, encoder=encoder)
        latents.append(z.cpu().numpy())
        remaining -= current_batch
    return np.concatenate(latents, axis=0)


def fit_latent_prior(
    args: argparse.Namespace,
    ctx: DiffAELightContext,
    encoder: torch.nn.Module,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if args.prior == "standard":
        mean = np.zeros(ctx.cfg.encoder.latent_dim, dtype=np.float32)
        std = np.ones(ctx.cfg.encoder.latent_dim, dtype=np.float32)
        return mean, std, np.empty((0, ctx.cfg.encoder.latent_dim), dtype=np.float32)

    latents_path = args.latents_h5
    if latents_path is None:
        candidate = os.path.join(ctx.checkpoint_dir, ctx.cfg.paths.diffae_latents_file)
        if os.path.exists(candidate):
            latents_path = candidate

    if latents_path is not None:
        print(f"Fitting empirical latent prior from {latents_path}")
        latents = load_latents_from_h5(latents_path)
        if latents.shape[1] != ctx.cfg.encoder.latent_dim:
            raise ValueError(
                f"Latent file dimension mismatch: expected {ctx.cfg.encoder.latent_dim}, got {latents.shape[1]}."
            )
        if args.prior_samples > 0 and latents.shape[0] > args.prior_samples:
            latents = latents[:args.prior_samples]
    else:
        print(f"Fitting empirical latent prior from {args.prior_samples} freshly encoded events")
        latents = encode_latents_from_loader(ctx, encoder, args.prior_samples, args.batch_size)

    mean = latents.mean(axis=0).astype(np.float32)
    std = latents.std(axis=0).astype(np.float32)
    std = np.clip(std, 1e-6, None)
    return mean, std, latents


def sample_latents(
    n_samples: int,
    mean: np.ndarray,
    std: np.ndarray,
    temperature: float,
    device: torch.device,
) -> torch.Tensor:
    z = np.random.randn(n_samples, mean.shape[0]).astype(np.float32)
    z = mean[None, :] + temperature * std[None, :] * z
    return torch.from_numpy(z).to(device)


def reshape_waveforms(flat_waveforms: np.ndarray, n_channels: int, n_time_points: int) -> np.ndarray:
    return flat_waveforms.reshape(flat_waveforms.shape[0], n_channels, n_time_points, order="F")


def z_profile(waveform: np.ndarray) -> np.ndarray:
    return waveform.sum(axis=0)


def save_visualization(
    waveforms: np.ndarray,
    channel_positions: np.ndarray,
    output_path: str,
) -> None:
    apply_style()
    n_samples = waveforms.shape[0]
    fig, axes = plt.subplots(n_samples, 2, figsize=(11, 3.4 * n_samples), squeeze=False)

    for idx in range(n_samples):
        charge = waveforms[idx].sum(axis=1)
        trace = z_profile(waveforms[idx])

        ax_xy = axes[idx, 0]
        sc = ax_xy.scatter(
            channel_positions[:, 0],
            channel_positions[:, 1],
            c=charge,
            cmap="viridis",
            s=72,
            edgecolors="k",
            linewidths=0.25,
        )
        ax_xy.set_aspect("equal")
        ax_xy.set_title(f"Sample {idx} charge map")
        ax_xy.set_xlabel("x (cm)")
        ax_xy.set_ylabel("y (cm)")
        fig.colorbar(sc, ax=ax_xy, fraction=0.046, pad=0.04, label="Integrated charge")

        ax_t = axes[idx, 1]
        ax_t.plot(np.arange(trace.shape[0]), trace, color=COLORS["diffae"], linewidth=1.4)
        ax_t.fill_between(np.arange(trace.shape[0]), trace, color=COLORS["diffae"], alpha=0.12)
        ax_t.set_title(f"Sample {idx} summed waveform")
        ax_t.set_xlabel("Time bin")
        ax_t.set_ylabel("Amplitude")

    fig.suptitle("DiffAE Light samples from random latent draws", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_visualization_3d(
    waveforms: np.ndarray,
    channel_positions: np.ndarray,
    output_path: str,
    ns_per_bin: float,
    threshold_quantile: float = 0.95,
    min_amplitude: float | None = None,
) -> None:
    apply_style()
    n_samples = waveforms.shape[0]
    n_cols = min(3, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(4.15 * n_cols + 0.7, 4.55 * n_rows))
    axes = [fig.add_subplot(n_rows, n_cols, idx + 1, projection="3d") for idx in range(n_samples)]

    vmax = max(float(np.max(wf)) for wf in waveforms)
    norm = Normalize(vmin=0.0, vmax=max(vmax, 1e-8))
    cmap = plt.get_cmap("viridis")
    scatter = None

    for idx, ax in enumerate(axes):
        panel = plot_waveform_3d_scatter(
            ax,
            waveforms[idx],
            channel_positions,
            ns_per_bin=ns_per_bin,
            threshold_quantile=threshold_quantile,
            min_amplitude=min_amplitude,
            norm=norm,
            cmap=cmap,
            title=f"Sample {idx + 1}",
        )
        scatter = panel["scatter"]

    fig.subplots_adjust(left=0.03, right=0.91, bottom=0.08, top=0.90, wspace=0.02, hspace=0.18)
    cax = fig.add_axes([0.925, 0.20, 0.013, 0.58])
    cbar = fig.colorbar(scatter, cax=cax)
    cbar.ax.set_ylabel("Amplitude (AU)")
    fig.suptitle("DiffAE Light samples from random latent draws", fontweight="bold", y=0.965)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = build_cfg(args)
    ctx = DiffAELightContext.build(
        cfg,
        for_training=True,
        verbose=True,
        use_ms_data=not args.use_ss_data,
    )

    checkpoint_path = find_checkpoint(ctx, args.checkpoint)
    print(f"Loading checkpoint: {checkpoint_path}")
    ctx.load_checkpoint(checkpoint_path, load_optim=False)

    encoder, decoder = inference_modules(ctx)
    prior_mean, prior_std, fitted_latents = fit_latent_prior(args, ctx, encoder)
    sampled_latents = sample_latents(
        args.n_samples,
        prior_mean,
        prior_std,
        args.latent_temperature,
        ctx.device,
    )

    samples = sample_from_latent_diffae_light(
        ctx,
        sampled_latents,
        decoder=decoder,
        parametrization=args.parametrization,
        pbar=args.pbar,
    )
    samples_denorm = ctx.data_stats.denormalize(samples.cpu().numpy())
    samples_denorm = np.clip(samples_denorm, 0.0, None)
    flat_waveforms = samples_denorm[:, 0, :]
    waveforms = reshape_waveforms(flat_waveforms, ctx.n_channels, ctx.n_time_points)

    output_dir = args.output_dir or os.path.join(ctx.plot_dir, "latent_samples")
    os.makedirs(output_dir, exist_ok=True)
    stem = f"diffae_light_latent_samples_seed{args.seed}"
    figure_path = os.path.join(output_dir, f"{stem}.png")
    figure_3d_path = os.path.join(output_dir, f"{stem}_3d.png")
    array_path = os.path.join(output_dir, f"{stem}.npz")

    save_visualization(waveforms, ctx.loader.channel_positions, figure_path)
    save_visualization_3d(
        waveforms,
        ctx.loader.channel_positions,
        figure_3d_path,
        ns_per_bin=ctx.cfg.ms_data.ns_per_bin,
    )
    np.savez_compressed(
        array_path,
        waveforms=waveforms.astype(np.float32),
        waveforms_flat=flat_waveforms.astype(np.float32),
        latents=sampled_latents.cpu().numpy().astype(np.float32),
        latent_prior_mean=prior_mean.astype(np.float32),
        latent_prior_std=prior_std.astype(np.float32),
        fitted_latents=fitted_latents.astype(np.float32),
        seed=np.int64(args.seed),
        latent_temperature=np.float32(args.latent_temperature),
    )

    print(f"Saved samples to {array_path}")
    print(f"Saved quick-look figure to {figure_path}")
    print(f"Saved 3D quick-look figure to {figure_3d_path}")


if __name__ == "__main__":
    main()
