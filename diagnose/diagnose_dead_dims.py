"""
Diagnose why 56/64 latent dims are dead in the trained DiffAE.

Tests:
  1. Encoder output statistics per dim
  2. Latent proj Jacobian: which z dims actually influence cond_base?
  3. FiLM Jacobian: which cond dims influence decoder gamma/beta?
  4. End-to-end Jacobian: d(decoder_output)/d(z) per dim
  5. Decoder sensitivity: how much does the decoder actually use z vs x_t?
  6. Training ablation: does the decoder learn to ignore z early?
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d

from config import default_config
from diffae import DiffAEContext, apply_lopsided_augmentation
from diffusion.schedule import sinusoidal_embedding
from probe_lopsidedness import get_lopsided_batch, encode_batch
import copy


def load_model_mlp(cfg):
    """Load checkpoint with MLP encoder (matches saved weights)."""
    cfg = copy.deepcopy(cfg)
    cfg.encoder.encoder_type = "mlp"
    from probe_lopsidedness import load_model
    return load_model(cfg)


def test_encoder_output_stats(ctx, cfg):
    """Check what the encoder actually outputs per dimension."""
    print("\n" + "=" * 60)
    print("TEST 1: Encoder Output Statistics")
    print("=" * 60)

    loader = ctx.loader
    all_z = []
    for _ in range(20):
        batch_np, _, idx = loader.get_batch(64)
        z = encode_batch(ctx, batch_np)
        all_z.append(z.numpy())
    z_all = np.concatenate(all_z, axis=0)

    var = z_all.var(axis=0)
    mean = z_all.mean(axis=0)
    sorted_idx = np.argsort(var)[::-1]

    print(f"  N={z_all.shape[0]}, D={z_all.shape[1]}")
    print(f"  Dims with var > 0.1:    {(var > 0.1).sum()}")
    print(f"  Dims with var > 0.01:   {(var > 0.01).sum()}")
    print(f"  Dims with var > 0.001:  {(var > 0.001).sum()}")
    print(f"  Dims with var < 1e-4:   {(var < 1e-4).sum()}")
    print(f"\n  Top 10 dims by variance:")
    for i in range(10):
        d = sorted_idx[i]
        print(f"    dim {d:2d}: var={var[d]:.6f}, mean={mean[d]:.4f}")
    print(f"\n  Bottom 10 dims by variance:")
    for i in range(10):
        d = sorted_idx[-(i + 1)]
        print(f"    dim {d:2d}: var={var[d]:.8f}, mean={mean[d]:.4f}")

    return z_all, var


@torch.no_grad()
def test_latent_proj_jacobian(ctx):
    """Measure sensitivity of latent_proj output to each z dimension."""
    print("\n" + "=" * 60)
    print("TEST 2: Latent Proj Jacobian (which z dims affect cond?)")
    print("=" * 60)

    latent_proj = ctx.latent_proj
    latent_dim = 64
    eps = 0.01

    z_base = torch.zeros(1, latent_dim)
    cond_base = latent_proj(z_base)

    sensitivities = []
    for d in range(latent_dim):
        z_pert = z_base.clone()
        z_pert[0, d] = eps
        cond_pert = latent_proj(z_pert)
        diff = (cond_pert - cond_base).abs().sum().item() / eps
        sensitivities.append(diff)

    sensitivities = np.array(sensitivities)
    sorted_idx = np.argsort(sensitivities)[::-1]

    print(f"  Mean sensitivity: {sensitivities.mean():.4f}")
    print(f"  Max sensitivity:  {sensitivities.max():.4f}")
    print(f"  Min sensitivity:  {sensitivities.min():.4f}")
    print(f"  Ratio max/min:    {sensitivities.max()/max(sensitivities.min(), 1e-8):.1f}")
    print(f"\n  Top 5 most sensitive z dims:")
    for i in range(5):
        print(f"    dim {sorted_idx[i]:2d}: sensitivity={sensitivities[sorted_idx[i]]:.4f}")
    print(f"  Bottom 5 least sensitive z dims:")
    for i in range(5):
        print(f"    dim {sorted_idx[-(i+1)]:2d}: sensitivity={sensitivities[sorted_idx[-(i+1)]]:.6f}")

    return sensitivities


@torch.no_grad()
def test_film_sensitivity(ctx, cfg):
    """Measure how much FiLM gammas/betas change when z varies."""
    print("\n" + "=" * 60)
    print("TEST 3: FiLM Output Sensitivity to z")
    print("=" * 60)

    decoder = ctx.decoder
    latent_proj = ctx.latent_proj
    latent_dim = 64

    z_base = torch.zeros(1, latent_dim)
    cond_base = latent_proj(z_base)
    t_emb = sinusoidal_embedding(torch.tensor([50]), cfg.conditioning.time_dim)
    cond_full_base = torch.cat([cond_base, t_emb], dim=-1)

    gamma_base, beta_base = decoder.film(cond_full_base, batch_size=1)

    gamma_diffs = []
    beta_diffs = []
    for d in range(latent_dim):
        z_pert = z_base.clone()
        z_pert[0, d] = 0.1
        cond_pert = latent_proj(z_pert)
        cond_full_pert = torch.cat([cond_pert, t_emb], dim=-1)
        gamma_pert, beta_pert = decoder.film(cond_full_pert, batch_size=1)
        gamma_diffs.append((gamma_pert - gamma_base).abs().mean().item())
        beta_diffs.append((beta_pert - beta_base).abs().mean().item())

    gamma_diffs = np.array(gamma_diffs)
    beta_diffs = np.array(beta_diffs)

    print(f"  Gamma sensitivity — mean: {gamma_diffs.mean():.6f}, max: {gamma_diffs.max():.6f}, min: {gamma_diffs.min():.8f}")
    print(f"  Beta sensitivity  — mean: {beta_diffs.mean():.6f}, max: {beta_diffs.max():.6f}, min: {beta_diffs.min():.8f}")
    print(f"  Gamma range ratio: {gamma_diffs.max()/max(gamma_diffs.min(), 1e-10):.1f}x")
    print(f"  Beta range ratio:  {beta_diffs.max()/max(beta_diffs.min(), 1e-10):.1f}x")

    print(f"\n  Baseline gamma values: mean={gamma_base.mean().item():.4f}, std={gamma_base.std().item():.6f}")
    print(f"  Baseline beta values:  mean={beta_base.mean().item():.4f}, std={beta_base.std().item():.6f}")

    return gamma_diffs, beta_diffs


def test_end_to_end_jacobian(ctx, cfg, batch_np):
    """Compute d(output)/d(z) for each z dimension using real inputs."""
    print("\n" + "=" * 60)
    print("TEST 4: End-to-End Jacobian d(output)/d(z)")
    print("=" * 60)

    device = next(ctx.encoder.parameters()).device
    data_stats = ctx.data_stats
    schedule = ctx.schedule
    n_nodes = ctx.n_nodes

    batch_norm = data_stats.normalize(batch_np[:8])
    x0 = torch.from_numpy(batch_norm.astype(np.float32)).to(device)
    B, N, C = x0.shape

    x0_flat = x0.view(B * N, C)
    with torch.no_grad():
        z, _, _ = ctx.encoder(x0_flat, ctx.A_sparse, ctx.pos, batch_size=B)

    t = torch.full((B,), 50, device=device, dtype=torch.long)
    t_emb = sinusoidal_embedding(t, cfg.conditioning.time_dim)

    sqrt_ab = schedule['sqrt_alphas_cumprod'][50]
    sqrt_om = schedule['sqrt_one_minus_alphas_cumprod'][50]
    noise = torch.randn_like(x0)
    x_t = sqrt_ab * x0 + sqrt_om * noise
    x_t_flat = x_t.view(B * N, C).detach()

    z_var = z.detach().clone().requires_grad_(True)
    cond_base = ctx.latent_proj(z_var)
    cond_full = torch.cat([cond_base, t_emb], dim=-1)
    pred = ctx.decoder(x_t_flat, ctx.A_sparse, cond_full, ctx.pos, batch_size=B)
    pred_2d = pred.view(B, N, C)

    output_energy = pred_2d.pow(2).sum()
    output_energy.backward()

    grad_z = z_var.grad.abs()
    per_dim_sensitivity = grad_z.mean(dim=0).detach().numpy()

    sorted_idx = np.argsort(per_dim_sensitivity)[::-1]
    print(f"  Mean |d(output)/d(z)|: {per_dim_sensitivity.mean():.6f}")
    print(f"  Max:                   {per_dim_sensitivity.max():.6f}")
    print(f"  Min:                   {per_dim_sensitivity.min():.8f}")
    print(f"  Ratio max/min:         {per_dim_sensitivity.max()/max(per_dim_sensitivity.min(), 1e-10):.1f}x")
    print(f"\n  Top 10:")
    for i in range(10):
        d = sorted_idx[i]
        print(f"    dim {d:2d}: {per_dim_sensitivity[d]:.6f}")
    print(f"  Bottom 5:")
    for i in range(5):
        d = sorted_idx[-(i + 1)]
        print(f"    dim {d:2d}: {per_dim_sensitivity[d]:.8f}")

    return per_dim_sensitivity


@torch.no_grad()
def test_z_vs_xt_contribution(ctx, cfg, batch_np):
    """How much does decoder output depend on z vs on x_t?"""
    print("\n" + "=" * 60)
    print("TEST 5: z vs x_t Contribution to Decoder Output")
    print("=" * 60)

    device = next(ctx.encoder.parameters()).device
    data_stats = ctx.data_stats
    schedule = ctx.schedule
    n_nodes = ctx.n_nodes

    batch_norm = data_stats.normalize(batch_np[:16])
    x0 = torch.from_numpy(batch_norm.astype(np.float32)).to(device)
    B, N, C = x0.shape
    x0_flat = x0.view(B * N, C)

    z_real, _, _ = ctx.encoder(x0_flat, ctx.A_sparse, ctx.pos, batch_size=B)
    z_zero = torch.zeros_like(z_real)
    z_rand = torch.randn_like(z_real) * z_real.std()

    cond_real = ctx.latent_proj(z_real)
    cond_zero = ctx.latent_proj(z_zero)
    cond_rand = ctx.latent_proj(z_rand)

    for t_val in [10, 50, 100, 200]:
        t_tensor = torch.full((B,), t_val, device=device, dtype=torch.long)
        t_emb = sinusoidal_embedding(t_tensor, cfg.conditioning.time_dim)

        sqrt_ab = schedule['sqrt_alphas_cumprod'][t_val]
        sqrt_om = schedule['sqrt_one_minus_alphas_cumprod'][t_val]
        noise = torch.randn_like(x0)
        x_t = sqrt_ab * x0 + sqrt_om * noise
        x_t_flat = x_t.view(B * N, C)

        pred_real = ctx.decoder(x_t_flat, ctx.A_sparse, torch.cat([cond_real, t_emb], dim=-1), ctx.pos, batch_size=B)
        pred_zero = ctx.decoder(x_t_flat, ctx.A_sparse, torch.cat([cond_zero, t_emb], dim=-1), ctx.pos, batch_size=B)
        pred_rand = ctx.decoder(x_t_flat, ctx.A_sparse, torch.cat([cond_rand, t_emb], dim=-1), ctx.pos, batch_size=B)

        diff_zero = (pred_real - pred_zero).pow(2).mean().item()
        diff_rand = (pred_real - pred_rand).pow(2).mean().item()
        pred_mag = pred_real.pow(2).mean().item()

        print(f"  t={t_val:3d}: |pred|²={pred_mag:.4f}  "
              f"|pred_real - pred_zero|²={diff_zero:.4f} ({diff_zero/pred_mag*100:.1f}%)  "
              f"|pred_real - pred_rand|²={diff_rand:.4f} ({diff_rand/pred_mag*100:.1f}%)")


@torch.no_grad()
def test_latent_proj_rank(ctx):
    """Check effective rank of latent_proj weight matrices."""
    print("\n" + "=" * 60)
    print("TEST 6: Latent Proj Weight Matrix Analysis")
    print("=" * 60)

    for i, m in enumerate(ctx.latent_proj):
        if isinstance(m, nn.Linear):
            W = m.weight.data
            U, S, V = torch.svd(W)
            S_norm = S / S[0]
            rank_90 = (S_norm.cumsum(0) / S_norm.sum() < 0.9).sum().item() + 1
            rank_99 = (S_norm.cumsum(0) / S_norm.sum() < 0.99).sum().item() + 1
            cond_num = S[0].item() / max(S[-1].item(), 1e-10)

            print(f"  Layer {i}: {W.shape[0]}x{W.shape[1]}")
            print(f"    Singular values — top5: {S[:5].tolist()}")
            print(f"    Bottom 5: {S[-5:].tolist()}")
            print(f"    Rank for 90% energy: {rank_90}/{min(W.shape)}")
            print(f"    Rank for 99% energy: {rank_99}/{min(W.shape)}")
            print(f"    Condition number:     {cond_num:.1f}")


@torch.no_grad()
def test_cond_mlp_analysis(ctx, cfg):
    """Check how much the decoder's internal cond_mlp collapses conditioning."""
    print("\n" + "=" * 60)
    print("TEST 7: Decoder cond_mlp Bottleneck Analysis")
    print("=" * 60)

    decoder = ctx.decoder

    for i, m in enumerate(decoder.cond_mlp):
        if isinstance(m, nn.Linear):
            W = m.weight.data
            U, S, V = torch.svd(W)
            S_norm = S / S[0]
            rank_90 = (S_norm.cumsum(0) / S_norm.sum() < 0.9).sum().item() + 1
            rank_99 = (S_norm.cumsum(0) / S_norm.sum() < 0.99).sum().item() + 1
            print(f"  cond_mlp layer {i}: {W.shape[0]}x{W.shape[1]}")
            print(f"    SV top5: {S[:5].tolist()}")
            print(f"    SV bot5: {S[-5:].tolist()}")
            print(f"    Rank for 90%/99%: {rank_90}/{rank_99} of {min(W.shape)}")

    for i, m in enumerate(decoder.film.mlp):
        if isinstance(m, nn.Linear):
            W = m.weight.data
            U, S, V = torch.svd(W)
            S_norm = S / S[0]
            rank_90 = (S_norm.cumsum(0) / S_norm.sum() < 0.9).sum().item() + 1
            rank_99 = (S_norm.cumsum(0) / S_norm.sum() < 0.99).sum().item() + 1
            print(f"  film.mlp layer {i}: {W.shape[0]}x{W.shape[1]}")
            print(f"    SV top5: {S[:5].tolist()}")
            print(f"    SV bot5: {S[-5:].tolist()}")
            print(f"    Rank for 90%/99%: {rank_90}/{rank_99} of {min(W.shape)}")


def main():
    cfg = default_config
    print("Loading model...")
    ctx = load_model_mlp(cfg)

    print("Generating test data...")
    batch_np, labels, _ = get_lopsided_batch(ctx, cfg, 200)

    z_all, z_var = test_encoder_output_stats(ctx, cfg)
    proj_sens = test_latent_proj_jacobian(ctx)
    gamma_sens, beta_sens = test_film_sensitivity(ctx, cfg)
    e2e_sens = test_end_to_end_jacobian(ctx, cfg, batch_np)
    test_z_vs_xt_contribution(ctx, cfg, batch_np)
    test_latent_proj_rank(ctx)
    test_cond_mlp_analysis(ctx, cfg)

    print("\n" + "=" * 60)
    print("CROSS-TEST CORRELATION")
    print("=" * 60)
    from scipy.stats import spearmanr
    corr_var_e2e, _ = spearmanr(z_var, e2e_sens)
    corr_proj_e2e, _ = spearmanr(proj_sens, e2e_sens)
    corr_var_proj, _ = spearmanr(z_var, proj_sens)
    print(f"  z_var vs e2e_sensitivity:  rho={corr_var_e2e:.3f}")
    print(f"  proj_sens vs e2e_sens:     rho={corr_proj_e2e:.3f}")
    print(f"  z_var vs proj_sens:        rho={corr_var_proj:.3f}")

    if corr_var_proj > 0.7:
        print("\n  --> CONCLUSION: Dead dims are driven by latent_proj selectivity")
        print("     (latent_proj ignores certain z dims)")
    elif corr_var_e2e > 0.7 and corr_proj_e2e < 0.3:
        print("\n  --> CONCLUSION: Dead dims are driven by decoder ignoring cond dims")
    elif corr_var_e2e < 0.3:
        print("\n  --> CONCLUSION: Encoder collapses dims independently of decoder")
        print("     (dims die before any downstream effect)")
    else:
        print("\n  --> Multiple factors contributing to dead dims")


if __name__ == "__main__":
    main()
