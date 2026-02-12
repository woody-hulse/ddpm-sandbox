"""
Compare Reduced Quantities (RQs) between raw data and AE/DiffAE reconstructions.

Computes waveform shape metrics (rise time, fall time, FWHM, etc.) on raw
and reconstructed events, then plots per-RQ correlation and residual panels.

Usage:
    python compare_rqs.py --n-samples 500
    python compare_rqs.py --n-samples 1000 --batch-size 32
"""
import os
import sys
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from tqdm import tqdm

from config import Config, default_config, get_config
from lz_data_loader import OnlineMSBatcher
from ae import AEContext, reconstruct_ae
from diffae import DiffAEContext, DiffAEDataStats, sample_diffae


def compute_rqs(z_profile: np.ndarray) -> Optional[Dict[str, float]]:
    """Compute reduced quantities from a z-profile (summed over channels)."""
    if z_profile.sum() == 0 or z_profile.max() == 0:
        return None

    z_profile = np.abs(z_profile)
    time_axis = np.arange(len(z_profile))

    peak_idx = int(np.argmax(z_profile))
    peak_amplitude = float(z_profile[peak_idx])
    peak_time = float(time_axis[peak_idx])
    total_integral = float(np.trapz(z_profile, time_axis))

    t10 = 0.1 * peak_amplitude
    t90 = 0.9 * peak_amplitude

    rise_10_idx = np.where(z_profile[:peak_idx] >= t10)[0]
    rise_10_t = float(time_axis[rise_10_idx[0]]) if len(rise_10_idx) > 0 else 0.0

    rise_90_idx = np.where(z_profile[:peak_idx] >= t90)[0]
    rise_90_t = float(time_axis[rise_90_idx[0]]) if len(rise_90_idx) > 0 else peak_time

    rise_time = rise_90_t - rise_10_t

    fall_90_idx = np.where(z_profile[peak_idx:] <= t90)[0]
    fall_90_t = float(time_axis[peak_idx + fall_90_idx[0]]) if len(fall_90_idx) > 0 else peak_time

    fall_10_idx = np.where(z_profile[peak_idx:] <= t10)[0]
    fall_10_t = float(time_axis[peak_idx + fall_10_idx[0]]) if len(fall_10_idx) > 0 else float(time_axis[-1])

    fall_time = fall_10_t - fall_90_t

    half_max = 0.5 * peak_amplitude
    left_idx = np.where(z_profile[:peak_idx] >= half_max)[0]
    left_t = float(time_axis[left_idx[0]]) if len(left_idx) > 0 else 0.0

    right_idx = np.where(z_profile[peak_idx:] <= half_max)[0]
    right_t = float(time_axis[peak_idx + right_idx[0]]) if len(right_idx) > 0 else float(time_axis[-1])

    fwhm = max(0.0, right_t - left_t)
    width_10_90 = max(0.0, (fall_10_t) - rise_10_t)

    std_dev = float(np.sqrt(np.average((time_axis - peak_time) ** 2, weights=z_profile + 1e-8)))

    return {
        'peak_amplitude': peak_amplitude,
        'peak_time': peak_time,
        'total_integral': total_integral,
        'rise_time': rise_time,
        'fall_time': fall_time,
        'fwhm': fwhm,
        'width_10_90': width_10_90,
        'std_dev': std_dev,
    }


def wf_to_z_profile(wf_flat: np.ndarray, n_channels: int, n_time: int) -> np.ndarray:
    """Convert flat waveform (N,) to z-profile (T,) by summing over channels."""
    return wf_flat.reshape(n_channels, n_time, order='F').sum(axis=0)


def collect_rqs(
    waveforms: np.ndarray,
    n_channels: int,
    n_time: int,
) -> Dict[str, np.ndarray]:
    """Compute RQs for a batch of flat waveforms (B, N).
    Returns dict of rq_name -> array(B,). NaN for failed events."""
    n = waveforms.shape[0]
    rq_names = ['peak_amplitude', 'peak_time', 'total_integral',
                'rise_time', 'fall_time', 'fwhm', 'width_10_90', 'std_dev']
    out = {k: np.full(n, np.nan) for k in rq_names}

    for i in range(n):
        z = wf_to_z_profile(waveforms[i], n_channels, n_time)
        rqs = compute_rqs(z)
        if rqs is not None:
            for k in rq_names:
                out[k][i] = rqs[k]
    return out


RQ_UNITS = {
    'peak_amplitude': 'a.u.',
    'peak_time': 'bins',
    'total_integral': 'a.u.',
    'rise_time': 'bins',
    'fall_time': 'bins',
    'fwhm': 'bins',
    'width_10_90': 'bins',
    'std_dev': 'bins',
}


MODEL_PALETTE = ['#6C9BC4', '#E8907E', '#7DBCB0', '#9B8DC4', '#D4A96A', '#C47D7D']
PLOT_DPI = 300


def _color_map(names):
    return {n: MODEL_PALETTE[i % len(MODEL_PALETTE)] for i, n in enumerate(names)}


def plot_rq_comparison(
    rq_true: Dict[str, np.ndarray],
    rq_models: Dict[str, Dict[str, np.ndarray]],
    output_dir: str,
):
    """
    Plot per-RQ hexbin + residual panels, one figure per RQ.
    Layout matches aux dmu plots: top row hexbin, bottom row residual histogram.
    """
    os.makedirs(output_dir, exist_ok=True)

    model_names = list(rq_models.keys())
    n_models = len(model_names)
    cmap = _color_map(model_names)

    rq_names = list(rq_true.keys())

    for rq_name in rq_names:
        true_vals = rq_true[rq_name]
        unit = RQ_UNITS.get(rq_name, '')
        display_name = rq_name.replace('_', ' ').title()

        fig, axes = plt.subplots(2, n_models, figsize=(4.5 * n_models, 7), squeeze=False)

        all_vals = [true_vals]
        for mname in model_names:
            all_vals.append(rq_models[mname][rq_name])
        all_concat = np.concatenate(all_vals)
        valid = np.isfinite(all_concat)
        if valid.sum() < 2:
            plt.close(fig)
            continue
        margin = 0.05 * (np.nanmax(all_concat[valid]) - np.nanmin(all_concat[valid]) + 1e-8)
        lims = [np.nanmin(all_concat[valid]) - margin, np.nanmax(all_concat[valid]) + margin]

        all_residuals_rq = []
        for mname in model_names:
            pred_vals = rq_models[mname][rq_name]
            mask = np.isfinite(true_vals) & np.isfinite(pred_vals)
            all_residuals_rq.append(pred_vals[mask] - true_vals[mask])
        resid_concat = np.concatenate(all_residuals_rq) if all_residuals_rq else np.array([0.0])
        resid_xlim = max(abs(resid_concat.min()), abs(resid_concat.max())) * 1.05 if len(resid_concat) > 0 else 1.0
        resid_bins = np.linspace(-resid_xlim, resid_xlim, 51)

        for i, mname in enumerate(model_names):
            pred_vals = rq_models[mname][rq_name]
            mask = np.isfinite(true_vals) & np.isfinite(pred_vals)
            t = true_vals[mask]
            p = pred_vals[mask]

            if len(t) < 5:
                axes[0, i].text(0.5, 0.5, 'Not enough data', transform=axes[0, i].transAxes,
                                ha='center', va='center')
                continue

            residuals = p - t
            mae = float(np.mean(np.abs(residuals)))
            r2 = float(1 - np.sum(residuals ** 2) / (np.sum((t - t.mean()) ** 2) + 1e-12))

            ax_hex = axes[0, i]
            ax_hex.hexbin(t, p, gridsize=30, cmap='Blues', mincnt=1)
            ax_hex.plot(lims, lims, 'r--', alpha=0.7, linewidth=1.5)
            ax_hex.set_xlim(lims)
            ax_hex.set_ylim(lims)
            ax_hex.set_xlabel(f'True ({unit})')
            ax_hex.set_ylabel(f'Reconstructed ({unit})')
            ax_hex.set_title(f'{mname}\nMAE={mae:.2f}, R²={r2:.3f}')
            ax_hex.set_aspect('equal', adjustable='box')

            ax_res = axes[1, i]
            ax_res.hist(residuals, bins=resid_bins, color=cmap[mname], alpha=0.7, edgecolor='white', linewidth=0.5)
            ax_res.axvline(0, color='r', linestyle='--', linewidth=1.5)
            mu_r = float(np.mean(residuals))
            sigma_r = float(np.std(residuals))
            ax_res.axvline(mu_r, color='black', linestyle='-', linewidth=1.5,
                           label=f'μ={mu_r:.2f}')
            ax_res.set_xlabel(f'Residual ({unit})')
            ax_res.set_ylabel('Count')
            ax_res.set_title(f'σ={sigma_r:.2f}')
            ax_res.set_xlim(-resid_xlim, resid_xlim)
            ax_res.legend(fontsize=8)

        fig.suptitle(display_name, fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f'rq_{rq_name}.png'), dpi=PLOT_DPI, bbox_inches='tight')
        plt.close(fig)

    fig_vio, axes_vio = plt.subplots(2, 4, figsize=(18, 8))
    axes_flat = axes_vio.flatten()

    for idx, rq_name in enumerate(rq_names):
        if idx >= len(axes_flat):
            break
        ax = axes_flat[idx]
        display_name = rq_name.replace('_', ' ').title()
        unit = RQ_UNITS.get(rq_name, '')

        data_list = []
        labels = []
        mae_vals = []
        for mname in model_names:
            pred = rq_models[mname][rq_name]
            residuals = pred - rq_true[rq_name]
            valid_mask = np.isfinite(residuals)
            abs_err = np.abs(residuals[valid_mask])
            data_list.append(abs_err)
            labels.append(mname)
            mae_vals.append(float(np.mean(abs_err)) if len(abs_err) > 0 else 0.0)

        if any(len(d) > 0 for d in data_list):
            positions = np.arange(len(data_list))
            vp = ax.violinplot(data_list, positions=positions, widths=0.7,
                               showmeans=True, showmedians=True)
            for j, body in enumerate(vp['bodies']):
                body.set_facecolor(cmap[model_names[j]])
                body.set_edgecolor(cmap[model_names[j]])
                body.set_alpha(0.6)
            for partname in ('cmeans', 'cmedians', 'cmins', 'cmaxes', 'cbars'):
                if partname in vp:
                    vp[partname].set_edgecolor('#333333')
                    vp[partname].set_linewidth(1.0)
            ax.set_xticks(positions)
            ax.set_xticklabels(labels)

            for j, d in enumerate(data_list):
                if len(d) > 0:
                    violin_max = float(d.max())
                    ax.annotate(f'MAE={mae_vals[j]:.2f}', (j, violin_max),
                                textcoords="offset points", xytext=(0, 8),
                                ha='center', fontsize=7)

        ax.set_title(display_name, fontsize=10, fontweight='bold')
        ax.set_ylabel(f'|Error| ({unit})')
        ax.grid(axis='y', alpha=0.3)

    for idx in range(len(rq_names), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.suptitle('Absolute Error Distribution per RQ', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_vio.savefig(os.path.join(output_dir, 'rq_error_summary.png'), dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig_vio)

    print(f"\nPlots saved to {output_dir}/")
    for rq_name in rq_names:
        print(f"  - rq_{rq_name}.png")
    print(f"  - rq_error_summary.png")


def plot_example_reconstructions(
    raw: np.ndarray,
    ae_rec: Optional[np.ndarray],
    diffae_rec: Optional[np.ndarray],
    n_channels: int,
    n_time: int,
    output_dir: str,
    n_examples: int = 6,
    seed: int = 42,
):
    """Plot example z-profiles and 2D waveform heatmaps for sanity checking."""
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_total = raw.shape[0]
    indices = rng.choice(n_total, size=min(n_examples, n_total), replace=False)
    indices = np.sort(indices)

    models = {}
    if ae_rec is not None:
        models['AE'] = ae_rec
    if diffae_rec is not None:
        models['DiffAE'] = diffae_rec
    n_cols = 1 + len(models)

    time_axis = np.arange(n_time)

    fig, axes = plt.subplots(len(indices), n_cols, figsize=(4 * n_cols, 2.8 * len(indices)),
                             squeeze=False)

    for row, idx in enumerate(indices):
        z_raw = wf_to_z_profile(raw[idx], n_channels, n_time)

        ax = axes[row, 0]
        ax.plot(time_axis, z_raw, color='black', linewidth=1.5)
        ax.fill_between(time_axis, z_raw, alpha=0.15, color='black')
        if row == 0:
            ax.set_title('Raw', fontweight='bold')
        ax.set_ylabel(f'#{idx}', fontweight='bold', rotation=0, labelpad=30)
        if row == len(indices) - 1:
            ax.set_xlabel('Time bin')
        y_max = z_raw.max() * 1.15
        ax.set_ylim(0, max(y_max, 1))

        model_cmap = _color_map(list(models.keys()))
        for col_offset, (mname, rec_data) in enumerate(models.items(), start=1):
            z_rec = wf_to_z_profile(rec_data[idx], n_channels, n_time)
            mc = model_cmap[mname]

            ax = axes[row, col_offset]
            ax.plot(time_axis, z_raw, color='black', linewidth=1, alpha=0.4, label='Raw')
            ax.plot(time_axis, z_rec, color=mc, linewidth=1.5, label=mname)
            ax.fill_between(time_axis, z_rec, alpha=0.2, color=mc)
            if row == 0:
                ax.set_title(mname, fontweight='bold')
            if row == len(indices) - 1:
                ax.set_xlabel('Time bin')
            ax.set_ylim(0, max(y_max, 1))
            ax.legend(fontsize=7, loc='upper right')

    plt.suptitle('Example Z-Profile Reconstructions', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'example_z_profiles.png'), dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)

    fig2, axes2 = plt.subplots(len(indices), n_cols, figsize=(4 * n_cols, 2.2 * len(indices)),
                                squeeze=False)

    for row, idx in enumerate(indices):
        wf_2d_raw = raw[idx].reshape(n_channels, n_time, order='F')
        vmax = wf_2d_raw.max()

        ax = axes2[row, 0]
        ax.imshow(wf_2d_raw, aspect='auto', origin='lower', cmap='viridis', vmin=0, vmax=vmax)
        if row == 0:
            ax.set_title('Raw', fontweight='bold')
        ax.set_ylabel(f'#{idx}', fontweight='bold', rotation=0, labelpad=30)
        if row == len(indices) - 1:
            ax.set_xlabel('Time bin')

        for col_offset, (mname, rec_data) in enumerate(models.items(), start=1):
            wf_2d_rec = rec_data[idx].reshape(n_channels, n_time, order='F')
            ax = axes2[row, col_offset]
            ax.imshow(wf_2d_rec, aspect='auto', origin='lower', cmap='viridis', vmin=0, vmax=vmax)
            if row == 0:
                ax.set_title(mname, fontweight='bold')
            if row == len(indices) - 1:
                ax.set_xlabel('Time bin')

    plt.suptitle('Example 2D Waveform Reconstructions (channel × time)', fontsize=14,
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    fig2.savefig(os.path.join(output_dir, 'example_heatmaps.png'), dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig2)

    print(f"  - example_z_profiles.png ({len(indices)} examples)")
    print(f"  - example_heatmaps.png ({len(indices)} examples)")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description='Compare RQs between raw data and AE/DiffAE reconstructions',
    )
    parser.add_argument('--n-samples', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--output-dir', type=str, default='rq_results')
    parser.add_argument('--n-examples', type=int, default=6,
                        help='Number of example reconstructions to plot')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    cfg = default_config
    cfg.conditioning.cond_proj_dim = max(cfg.conditioning.cond_proj_dim, cfg.encoder.latent_dim)
    device = torch.device(cfg.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Device: {device}")
    print(f"Samples: {args.n_samples}")

    has_ae = False
    has_diffae = False

    try:
        ae_ctx = AEContext.build(cfg, for_training=True, verbose=False, use_ms_data=True)
        ae_ckpt = ae_ctx.latest_checkpoint()
        if ae_ckpt is not None:
            ae_ctx.load_checkpoint(ae_ckpt, load_optim=False)
            ae_ctx.encoder = ae_ctx.ema_encoder if ae_ctx.ema_encoder is not None else ae_ctx.encoder
            ae_ctx.decoder = ae_ctx.ema_decoder if ae_ctx.ema_decoder is not None else ae_ctx.decoder
            ae_ctx.encoder.eval()
            ae_ctx.decoder.eval()
            has_ae = True
            print(f"AE loaded from {ae_ckpt} (EMA weights)")
        else:
            print("No AE checkpoint found, skipping AE.")
    except Exception as e:
        print(f"Could not load AE: {e}")

    try:
        diffae_ctx = DiffAEContext.build(cfg, for_training=True, verbose=False, use_ms_data=True)
        diffae_ckpt = diffae_ctx.latest_checkpoint()
        if diffae_ckpt is not None:
            diffae_ctx.load_checkpoint(diffae_ckpt, load_optim=False)
            diffae_ctx.encoder = diffae_ctx.ema_encoder if diffae_ctx.ema_encoder is not None else diffae_ctx.encoder
            diffae_ctx.decoder = diffae_ctx.ema_decoder if diffae_ctx.ema_decoder is not None else diffae_ctx.decoder
            diffae_ctx.latent_proj = diffae_ctx.ema_latent_proj if diffae_ctx.ema_latent_proj is not None else diffae_ctx.latent_proj
            diffae_ctx.encoder.eval()
            diffae_ctx.decoder.eval()
            diffae_ctx.latent_proj.eval()
            has_diffae = True
            print(f"DiffAE loaded from {diffae_ckpt} (EMA weights)")
        else:
            print("No DiffAE checkpoint found, skipping DiffAE.")
    except Exception as e:
        print(f"Could not load DiffAE: {e}")

    if not has_ae and not has_diffae:
        print("ERROR: No models found. Train AE or DiffAE first.")
        return

    ref_ctx = ae_ctx if has_ae else diffae_ctx
    loader = ref_ctx.loader
    n_channels = ref_ctx.n_channels
    n_time = ref_ctx.n_time_points
    n_nodes = ref_ctx.n_nodes

    all_raw = []
    all_ae_rec = []
    all_diffae_rec = []

    n_batches = (args.n_samples + args.batch_size - 1) // args.batch_size
    collected = 0

    pbar = tqdm(range(n_batches), desc='Reconstructing', ncols=100)
    for _ in pbar:
        remaining = args.n_samples - collected
        B = min(args.batch_size, remaining)
        if B <= 0:
            break

        wf_col, _ = loader.get_batch(B)
        raw_denorm = wf_col[:, :, 0]
        all_raw.append(raw_denorm)

        if has_ae:
            wf_norm_ae = ae_ctx.data_stats.normalize(wf_col)
            x_ae = torch.from_numpy(wf_norm_ae.astype(np.float32)).to(device)
            rec = reconstruct_ae(
                encoder=ae_ctx.encoder,
                decoder=ae_ctx.decoder,
                A_sparse=ae_ctx.A_sparse,
                pos=ae_ctx.pos,
                x_ref=x_ae,
            )
            rec_np = ae_ctx.data_stats.denormalize(rec.cpu().numpy())
            rec_np = np.clip(rec_np, 0, None)
            all_ae_rec.append(rec_np[:, 0, :])

        if has_diffae:
            wf_norm_diffae = diffae_ctx.data_stats.normalize(wf_col)
            x_diffae = torch.from_numpy(wf_norm_diffae.astype(np.float32)).to(device)
            rec = sample_diffae(
                encoder=diffae_ctx.encoder,
                decoder=diffae_ctx.decoder,
                latent_proj=diffae_ctx.latent_proj,
                schedule=diffae_ctx.schedule,
                A_sparse=diffae_ctx.A_sparse,
                pos=diffae_ctx.pos,
                time_dim=cfg.conditioning.time_dim,
                x_ref=x_diffae,
                parametrization=cfg.diffusion.parametrization,
                pbar=False,
            )
            rec_np = diffae_ctx.data_stats.denormalize(rec.cpu().numpy())
            rec_np = np.clip(rec_np, 0, None)
            all_diffae_rec.append(rec_np[:, 0, :])

        collected += B
        pbar.set_postfix(n=collected)

    raw = np.concatenate(all_raw, axis=0)
    print(f"\nComputing RQs for {raw.shape[0]} events...")

    rq_true = collect_rqs(raw, n_channels, n_time)

    rq_models = {}
    if has_ae:
        ae_rec = np.concatenate(all_ae_rec, axis=0)
        rq_models['AE'] = collect_rqs(ae_rec, n_channels, n_time)
        print("  AE RQs computed")
    if has_diffae:
        diffae_rec = np.concatenate(all_diffae_rec, axis=0)
        rq_models['DiffAE'] = collect_rqs(diffae_rec, n_channels, n_time)
        print("  DiffAE RQs computed")

    print("\n" + "=" * 60)
    print("RQ Summary (MAE | R²)")
    print("=" * 60)
    for rq_name in rq_true.keys():
        t = rq_true[rq_name]
        line = f"  {rq_name:20s}"
        for mname in rq_models:
            p = rq_models[mname][rq_name]
            mask = np.isfinite(t) & np.isfinite(p)
            if mask.sum() > 1:
                res = p[mask] - t[mask]
                mae = float(np.mean(np.abs(res)))
                r2 = float(1 - np.sum(res ** 2) / (np.sum((t[mask] - t[mask].mean()) ** 2) + 1e-12))
                line += f"  | {mname}: MAE={mae:.3f}, R²={r2:.3f}"
            else:
                line += f"  | {mname}: N/A"
        print(line)

    plot_rq_comparison(rq_true, rq_models, args.output_dir)

    plot_example_reconstructions(
        raw=raw,
        ae_rec=ae_rec if has_ae else None,
        diffae_rec=diffae_rec if has_diffae else None,
        n_channels=n_channels,
        n_time=n_time,
        output_dir=args.output_dir,
        n_examples=args.n_examples,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
