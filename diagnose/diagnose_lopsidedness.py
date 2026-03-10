"""
Diagnose why DiffAE fails to reconstruct lopsided events.

Runs three inference-only experiments on an existing checkpoint:
  A) Partial noise reconstruction sweep  (tests H2)
  B) Conditioning swap                   (tests H3)
  C) Per-timestep loss analysis          (tests H1)

Usage:
    python diagnose_lopsidedness.py [--n_events 200] [--out_dir lopsided_diagnostics]
"""
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from config import default_config, Config
from diffae import (
    DiffAEContext,
    DiffAEDataStats,
    apply_lopsided_augmentation,
    sample_diffae,
    sample_diffae_partial,
    sample_from_latent,
)
from diffusion.schedule import build_cosine_schedule, sinusoidal_embedding


# ---------------------------------------------------------------------------
# Lopsidedness metric
# ---------------------------------------------------------------------------

def hf_energy(waveform: np.ndarray) -> float:
    """High-frequency energy via mean squared first-difference."""
    d = np.diff(waveform)
    return float(np.mean(d ** 2))


def lopsidedness_ratio(waveform: np.ndarray) -> float:
    """log(HF_left / HF_right).  Negative = left is smoother."""
    half = len(waveform) // 2
    left = hf_energy(waveform[:half])
    right = hf_energy(waveform[half:])
    return float(np.log(left / max(right, 1e-12)))


def lopsidedness_correlation(inputs: np.ndarray, outputs: np.ndarray):
    """Pearson r between input and output lopsidedness ratios.

    Args:
        inputs:  (B, N) raw waveforms
        outputs: (B, N) reconstructed waveforms

    Returns:
        r, p-value, input_ratios, output_ratios
    """
    lr_in = np.array([lopsidedness_ratio(w) for w in inputs])
    lr_out = np.array([lopsidedness_ratio(w) for w in outputs])
    if np.std(lr_in) < 1e-8 or np.std(lr_out) < 1e-8:
        return 0.0, 1.0, lr_in, lr_out
    r, p = pearsonr(lr_in, lr_out)
    return r, p, lr_in, lr_out


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def prepare_lopsided_batch(ctx: DiffAEContext, cfg: Config, n: int):
    """Return (batch_np, labels) with exactly n/2 left-smooth and n/2 right-smooth.

    labels: 0 = left-smooth, 1 = right-smooth.
    batch_np: (n, N, 1) raw (un-normalised) waveforms after augmentation.
    """
    from scipy.ndimage import gaussian_filter1d

    loader = ctx.loader
    half_n = n // 2
    sigma = cfg.training.lopsided_sigma

    left_collected, right_collected = [], []
    attempts = 0
    while len(left_collected) < half_n or len(right_collected) < half_n:
        bs = min(256, n * 4)
        batch_np, _, idx = loader.get_batch(bs)
        N = batch_np.shape[1]
        half = N // 2
        for i in range(bs):
            if int(idx[i]) % 2 == 0 and len(left_collected) < half_n:
                wf = batch_np[i].copy()
                wf[:half, 0] = gaussian_filter1d(wf[:half, 0], sigma=sigma)
                left_collected.append(wf)
            elif int(idx[i]) % 2 == 1 and len(right_collected) < half_n:
                wf = batch_np[i].copy()
                wf[half:, 0] = gaussian_filter1d(wf[half:, 0], sigma=sigma)
                right_collected.append(wf)
        attempts += 1
        if attempts > 200:
            break

    left_arr = np.stack(left_collected[:half_n])
    right_arr = np.stack(right_collected[:half_n])
    batch_np = np.concatenate([left_arr, right_arr], axis=0)
    labels = np.array([0] * half_n + [1] * half_n)
    return batch_np, labels


# ---------------------------------------------------------------------------
# Experiment A: Partial noise reconstruction sweep
# ---------------------------------------------------------------------------

def experiment_partial_noise(ctx, cfg, batch_np, labels, out_dir):
    """Sweep t_start and measure how lopsidedness preservation degrades."""
    print("\n=== Experiment A: Partial Noise Reconstruction Sweep ===")
    data_stats = ctx.data_stats
    device = next(ctx.encoder.parameters()).device
    schedule = ctx.schedule

    batch_norm = data_stats.normalize(batch_np)
    x_ref = torch.from_numpy(batch_norm.astype(np.float32)).to(device)

    T = schedule['betas'].shape[0]
    t_starts = [0, 5, 10, 25, 50, 100, 150, 200, T - 1]

    correlations = []
    all_ratios_out = {}

    wf_input = batch_np[:, :, 0]
    lr_in = np.array([lopsidedness_ratio(w) for w in wf_input])

    for t_s in t_starts:
        if t_s == 0:
            rec_np = batch_np.copy()
        else:
            rec = sample_diffae_partial(
                encoder=ctx.encoder,
                decoder=ctx.decoder,
                latent_proj=ctx.latent_proj,
                schedule=schedule,
                A_sparse=ctx.A_sparse,
                pos=ctx.pos,
                time_dim=cfg.conditioning.time_dim,
                x_ref=x_ref,
                t_start=t_s,
                parametrization=cfg.diffusion.parametrization,
            )
            rec_np = data_stats.denormalize(rec.cpu().numpy())
            rec_np = np.clip(rec_np, 0, None)

        wf_out = rec_np[:, 0, :] if rec_np.ndim == 3 and rec_np.shape[1] == 1 else rec_np[:, :, 0]
        lr_out = np.array([lopsidedness_ratio(w) for w in wf_out])
        if np.std(lr_in) > 1e-8 and np.std(lr_out) > 1e-8:
            r, _ = pearsonr(lr_in, lr_out)
        else:
            r = 0.0
        correlations.append(r)
        all_ratios_out[t_s] = lr_out
        print(f"  t_start={t_s:>4d}  corr(lopsidedness)={r:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(t_starts, correlations, 'o-', linewidth=2)
    axes[0].set_xlabel('t_start (noise level)')
    axes[0].set_ylabel('Pearson r (lopsidedness)')
    axes[0].set_title('Exp A: Lopsidedness preservation vs noise level')
    axes[0].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_ylim(-0.3, 1.05)
    axes[0].grid(True, alpha=0.3)

    for t_s in [0, 25, 100, T - 1]:
        if t_s in all_ratios_out:
            axes[1].scatter(lr_in, all_ratios_out[t_s], alpha=0.4, s=15,
                            label=f't_start={t_s}')
    mn, mx = lr_in.min(), lr_in.max()
    axes[1].plot([mn, mx], [mn, mx], 'k--', alpha=0.3)
    axes[1].set_xlabel('Input lopsidedness ratio')
    axes[1].set_ylabel('Output lopsidedness ratio')
    axes[1].set_title('Input vs output lopsidedness')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'exp_a_partial_noise.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_dir}/exp_a_partial_noise.png")

    return t_starts, correlations


# ---------------------------------------------------------------------------
# Experiment B: Conditioning swap
# ---------------------------------------------------------------------------

def experiment_conditioning_swap(ctx, cfg, batch_np, labels, out_dir):
    """Encode left/right events, swap z, decode, measure which z the output follows."""
    print("\n=== Experiment B: Conditioning Swap ===")
    data_stats = ctx.data_stats
    device = next(ctx.encoder.parameters()).device
    schedule = ctx.schedule

    half_n = len(labels) // 2
    left_np = batch_np[:half_n]
    right_np = batch_np[half_n:]

    left_norm = data_stats.normalize(left_np)
    right_norm = data_stats.normalize(right_np)

    x_left = torch.from_numpy(left_norm.astype(np.float32)).to(device)
    x_right = torch.from_numpy(right_norm.astype(np.float32)).to(device)

    N = x_left.shape[1]
    A, pos = ctx.A_sparse, ctx.pos

    z_left, _, _ = ctx.encoder(x_left.view(-1, 1), A, pos, batch_size=half_n)
    z_right, _, _ = ctx.encoder(x_right.view(-1, 1), A, pos, batch_size=half_n)

    n_nodes = ctx.n_nodes

    rec_left_zleft = sample_from_latent(
        ctx.decoder, ctx.latent_proj, schedule, A, pos,
        cfg.conditioning.time_dim, z_left, n_nodes, cfg.diffusion.parametrization,
    )
    rec_left_zright = sample_from_latent(
        ctx.decoder, ctx.latent_proj, schedule, A, pos,
        cfg.conditioning.time_dim, z_right, n_nodes, cfg.diffusion.parametrization,
    )

    rec_ll_np = np.clip(data_stats.denormalize(rec_left_zleft.cpu().numpy()), 0, None)
    rec_lr_np = np.clip(data_stats.denormalize(rec_left_zright.cpu().numpy()), 0, None)

    wf_left = left_np[:, :, 0]
    wf_right = right_np[:, :, 0]
    wf_rec_ll = rec_ll_np[:, 0, :]
    wf_rec_lr = rec_lr_np[:, 0, :]

    lr_left_input = np.array([lopsidedness_ratio(w) for w in wf_left])
    lr_right_input = np.array([lopsidedness_ratio(w) for w in wf_right])
    lr_rec_own_z = np.array([lopsidedness_ratio(w) for w in wf_rec_ll])
    lr_rec_swapped_z = np.array([lopsidedness_ratio(w) for w in wf_rec_lr])

    mean_lr_left = np.mean(lr_left_input)
    mean_lr_right = np.mean(lr_right_input)
    mean_rec_own = np.mean(lr_rec_own_z)
    mean_rec_swap = np.mean(lr_rec_swapped_z)

    print(f"  Mean lopsidedness ratio:")
    print(f"    Left-smooth input:          {mean_lr_left:.4f}")
    print(f"    Right-smooth input:         {mean_lr_right:.4f}")
    print(f"    Reconstructed (own z):      {mean_rec_own:.4f}")
    print(f"    Reconstructed (swapped z):  {mean_rec_swap:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    bins = np.linspace(-3, 3, 40)
    axes[0].hist(lr_left_input, bins=bins, alpha=0.5, label='Left-smooth input', density=True)
    axes[0].hist(lr_right_input, bins=bins, alpha=0.5, label='Right-smooth input', density=True)
    axes[0].set_xlabel('Lopsidedness ratio')
    axes[0].set_title('Input distributions')
    axes[0].legend(fontsize=8)

    axes[1].hist(lr_rec_own_z, bins=bins, alpha=0.5, label='Rec (own z_left)', density=True)
    axes[1].hist(lr_rec_swapped_z, bins=bins, alpha=0.5, label='Rec (swapped z_right)', density=True)
    axes[1].set_xlabel('Lopsidedness ratio')
    axes[1].set_title('Reconstructions from left-smooth events')
    axes[1].legend(fontsize=8)

    categories = ['Left input', 'Right input', 'Rec own z', 'Rec swapped z']
    means = [mean_lr_left, mean_lr_right, mean_rec_own, mean_rec_swap]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    axes[2].bar(categories, means, color=colors)
    axes[2].set_ylabel('Mean lopsidedness ratio')
    axes[2].set_title('Mean lopsidedness by condition')
    axes[2].axhline(0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'exp_b_cond_swap.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_dir}/exp_b_cond_swap.png")

    return {
        'mean_left_input': mean_lr_left,
        'mean_right_input': mean_lr_right,
        'mean_rec_own': mean_rec_own,
        'mean_rec_swap': mean_rec_swap,
    }


# ---------------------------------------------------------------------------
# Experiment C: Per-timestep loss & P2 weight analysis
# ---------------------------------------------------------------------------

def experiment_p2_analysis(ctx, cfg, batch_np, labels, out_dir):
    """Compute per-timestep MSE on lopsided events and overlay P2 weights."""
    print("\n=== Experiment C: Per-Timestep Loss & P2 Weight Analysis ===")
    data_stats = ctx.data_stats
    device = next(ctx.encoder.parameters()).device
    schedule = ctx.schedule

    batch_norm = data_stats.normalize(batch_np)
    x0 = torch.from_numpy(batch_norm.astype(np.float32)).to(device)
    B, N_nodes, C = x0.shape
    A, pos = ctx.A_sparse, ctx.pos

    x0_flat = x0.view(B * N_nodes, C)
    z, _, _ = ctx.encoder(x0_flat, A, pos, batch_size=B)
    cond_base = ctx.latent_proj(z)

    T = schedule['betas'].shape[0]
    snr_all = schedule['snr'].cpu().numpy()
    p2_gamma = cfg.diffusion.p2_gamma
    p2_k = cfg.diffusion.p2_k
    p2_weights = (p2_k + snr_all) ** (-p2_gamma) if p2_gamma > 0 else np.ones(T)

    mse_per_t = np.zeros(T)
    mse_lopsided_per_t = np.zeros(T)

    n_repeats = 1
    t_sample = list(range(0, T, 5))
    for rep in range(n_repeats):
        for t_val in t_sample:
            t_tensor = torch.full((B,), t_val, device=device, dtype=torch.long)
            t_emb = sinusoidal_embedding(t_tensor, cfg.conditioning.time_dim)
            cond_full = torch.cat([cond_base, t_emb], dim=-1)

            sqrt_ab = schedule['sqrt_alphas_cumprod'][t_val].view(1, 1, 1)
            sqrt_om = schedule['sqrt_one_minus_alphas_cumprod'][t_val].view(1, 1, 1)
            noise = torch.randn_like(x0)
            x_t = sqrt_ab * x0 + sqrt_om * noise

            x_t_flat = x_t.view(B * N_nodes, C)
            pred_flat = ctx.decoder(x_t_flat, A, cond_full, pos, batch_size=B)
            pred = pred_flat.view(B, N_nodes, C)

            if cfg.diffusion.parametrization == "v":
                target = sqrt_ab * noise - sqrt_om * x0
            else:
                target = noise

            mse = F.mse_loss(pred, target, reduction='none').mean(dim=(1, 2))
            mse_per_t[t_val] += mse.mean().item()

            half = N_nodes // 2
            left_mse = F.mse_loss(pred[:, :half], target[:, :half], reduction='none').mean(dim=(1, 2))
            right_mse = F.mse_loss(pred[:, half:], target[:, half:], reduction='none').mean(dim=(1, 2))
            lopsided_mse_diff = (left_mse - right_mse).abs().mean().item()
            mse_lopsided_per_t[t_val] += lopsided_mse_diff

    mse_sampled = np.array([mse_per_t[t] for t in t_sample]) / n_repeats
    asym_sampled = np.array([mse_lopsided_per_t[t] for t in t_sample]) / n_repeats
    p2_sampled = np.array([p2_weights[t] for t in t_sample])
    weighted_mse_sampled = mse_sampled * p2_sampled

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].semilogy(range(T), snr_all)
    axes[0, 0].set_xlabel('Timestep t')
    axes[0, 0].set_ylabel('SNR')
    axes[0, 0].set_title('Signal-to-Noise Ratio vs timestep')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(range(T), p2_weights, label=f'P2 weight (gamma={p2_gamma})')
    axes[0, 1].set_xlabel('Timestep t')
    axes[0, 1].set_ylabel('P2 weight')
    axes[0, 1].set_title('P2 loss weight vs timestep')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(t_sample, mse_sampled, label='Raw MSE', alpha=0.7)
    axes[1, 0].plot(t_sample, weighted_mse_sampled, label='P2-weighted MSE', alpha=0.7)
    axes[1, 0].set_xlabel('Timestep t')
    axes[1, 0].set_ylabel('MSE')
    axes[1, 0].set_title('Per-timestep MSE (raw vs P2-weighted)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(t_sample, asym_sampled, color='red', alpha=0.7)
    axes[1, 1].set_xlabel('Timestep t')
    axes[1, 1].set_ylabel('|MSE_left - MSE_right|')
    axes[1, 1].set_title('Left-right MSE asymmetry vs timestep')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'exp_c_p2_analysis.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_dir}/exp_c_p2_analysis.png")

    low_t_idx = [i for i, t in enumerate(t_sample) if t < 50]
    high_t_idx = [i for i, t in enumerate(t_sample) if t >= 200]
    low_t_weight = np.mean(p2_sampled[low_t_idx]) if low_t_idx else 0
    high_t_weight = np.mean(p2_sampled[high_t_idx]) if high_t_idx else 0
    asymmetry_low_t = np.mean(asym_sampled[low_t_idx]) if low_t_idx else 0
    asymmetry_high_t = np.mean(asym_sampled[high_t_idx]) if high_t_idx else 0
    print(f"  P2 weight  — low noise (t<50): {low_t_weight:.4f},  high noise (t>200): {high_t_weight:.4f}")
    print(f"  L-R asymmetry — low noise (t<50): {asymmetry_low_t:.6f},  high noise (t>200): {asymmetry_high_t:.6f}")

    return {
        'snr': snr_all,
        'p2_weights': p2_weights,
        't_sample': t_sample,
        'mse_sampled': mse_sampled,
        'asym_sampled': asym_sampled,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Diagnose DiffAE lopsidedness reconstruction")
    parser.add_argument('--n_events', type=int, default=200)
    parser.add_argument('--out_dir', type=str, default='lopsided_diagnostics')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = default_config

    print("Building DiffAE context and loading checkpoint...")
    ctx = DiffAEContext.build(cfg, for_training=False, verbose=True)
    ckpt = ctx.latest_checkpoint()
    if ckpt is None:
        print("ERROR: No checkpoint found. Train a model first.")
        return

    chk = torch.load(ckpt, map_location=ctx.device, weights_only=False)
    if "ema_encoder" in chk:
        ctx.encoder.load_state_dict(chk["ema_encoder"], strict=False)
    else:
        ctx.encoder.load_state_dict(chk["encoder"], strict=False)
    if "ema_decoder" in chk:
        ctx.decoder.load_state_dict(chk["ema_decoder"])
    else:
        ctx.decoder.load_state_dict(chk["decoder"])
    if "ema_latent_proj" in chk:
        ctx.latent_proj.load_state_dict(chk["ema_latent_proj"])
    else:
        ctx.latent_proj.load_state_dict(chk["latent_proj"])
    if "data_stats" in chk:
        ctx.data_stats.mean = chk["data_stats"]["mean"]
        ctx.data_stats.std = chk["data_stats"]["std"]
    epoch = int(chk.get("epoch", 0))
    print(f"Loaded checkpoint: {ckpt}  (epoch {epoch})")

    ctx.encoder.eval()
    ctx.decoder.eval()
    ctx.latent_proj.eval()

    print(f"\nPreparing {args.n_events} lopsided events ({args.n_events // 2} left, {args.n_events // 2} right)...")
    batch_np, labels = prepare_lopsided_batch(ctx, cfg, args.n_events)
    print(f"  batch shape: {batch_np.shape}, labels: {np.bincount(labels)}")

    wf_input = batch_np[:, :, 0]
    r_input, _, lr_in, _ = lopsidedness_correlation(wf_input, wf_input)
    print(f"  Input lopsidedness ratio — left mean: {np.mean(lr_in[labels == 0]):.4f}, "
          f"right mean: {np.mean(lr_in[labels == 1]):.4f}")

    with torch.no_grad():
        results_a = experiment_partial_noise(ctx, cfg, batch_np, labels, args.out_dir)
        results_b = experiment_conditioning_swap(ctx, cfg, batch_np, labels, args.out_dir)
        results_c = experiment_p2_analysis(ctx, cfg, batch_np, labels, args.out_dir)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY OF FINDINGS")
    print("=" * 60)

    t_starts, corrs = results_a
    print(f"\nH1 (P2 weighting):")
    pw = results_c['p2_weights']
    ts = results_c['t_sample']
    asym = results_c['asym_sampled']
    low_idx = [i for i, t in enumerate(ts) if t < 50]
    high_idx = [i for i, t in enumerate(ts) if t >= 200]
    p2_low = np.mean([pw[t] for t in ts if t < 50]) if any(t < 50 for t in ts) else 0
    p2_high = np.mean([pw[t] for t in ts if t >= 200]) if any(t >= 200 for t in ts) else 0
    print(f"  P2 weight ratio (high-noise / low-noise): {p2_high / max(p2_low, 1e-8):.1f}x")
    asym_low = np.mean(asym[low_idx]) if low_idx else 0
    asym_high = np.mean(asym[high_idx]) if high_idx else 0
    print(f"  L/R asymmetry is {'stronger at low noise' if asym_low > asym_high else 'comparable across noise levels'}")
    if p2_high / max(p2_low, 1e-8) > 5 and asym_low > asym_high:
        print(f"  --> SUPPORTED: P2 weighting suppresses fine-detail gradients by {p2_high / max(p2_low, 1e-8):.0f}x")
    else:
        print(f"  --> INCONCLUSIVE")

    print(f"\nH2 (Reverse process convergence):")
    full_corr = corrs[-1] if len(corrs) > 0 else 0
    low_corr = corrs[3] if len(corrs) > 3 else 0
    print(f"  Corr at t_start=full ({t_starts[-1]}): {full_corr:.4f}")
    print(f"  Corr at t_start={t_starts[3]}: {low_corr:.4f}")
    if low_corr > 0.5 and full_corr < 0.3:
        print(f"  --> SUPPORTED: Lopsidedness preserved at low noise, lost at high noise")
    elif low_corr < 0.3:
        print(f"  --> SUPPORTED (STRONG): Lopsidedness lost even at low noise — decoder cannot represent it")
    else:
        print(f"  --> NOT SUPPORTED: Lopsidedness preserved through reverse process")

    print(f"\nH3 (Conditioning carries no lopsidedness):")
    swap = results_b
    own_bias = swap['mean_rec_own']
    swap_bias = swap['mean_rec_swap']
    print(f"  Rec with own z:     mean ratio = {own_bias:.4f}")
    print(f"  Rec with swapped z: mean ratio = {swap_bias:.4f}")
    if abs(own_bias - swap_bias) < 0.1:
        print(f"  --> SUPPORTED: Swapping z has negligible effect (diff={abs(own_bias - swap_bias):.4f})")
    else:
        print(f"  --> NOT SUPPORTED: z swap causes shift of {abs(own_bias - swap_bias):.4f}")

    print(f"\nH4 (Signal too weak):")
    input_spread = np.mean(lr_in[labels == 0]) - np.mean(lr_in[labels == 1])
    print(f"  Input left-right spread: {abs(input_spread):.4f}")
    if abs(input_spread) < 0.5:
        print(f"  --> SUPPORTED: Lopsidedness signal is weak (spread < 0.5)")
    else:
        print(f"  --> NOT SUPPORTED: Lopsidedness signal is clear (spread = {abs(input_spread):.4f})")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
