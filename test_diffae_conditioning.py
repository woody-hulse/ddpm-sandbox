"""
Sanity checks for DiffAE conditioning.
Tests whether the latent representation is actually affecting the output.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass

from diffae import DiffAEContext, GraphEncoder
from diffusion.schedule import sinusoidal_embedding
from config import default_config


def test_encoder_sensitivity():
    """Test 1: Check if encoder produces different latents for different inputs."""
    print("\n" + "=" * 60)
    print("TEST 1: Encoder sensitivity to input")
    print("=" * 60)
    
    cfg = default_config
    ctx = DiffAEContext.build(cfg, for_training=False, verbose=False)
    
    latest_ckpt = ctx.latest_checkpoint()
    if latest_ckpt:
        print(f"Loading checkpoint: {latest_ckpt}")
        ctx.load_checkpoint(latest_ckpt, load_optim=False)
    else:
        print("No checkpoint found, using random weights")
    
    ctx.encoder.eval()
    
    with torch.no_grad():
        batch_np1, _ = ctx.loader.get_batch(4)
        batch_np2, _ = ctx.loader.get_batch(4)
        
        x1 = torch.from_numpy(ctx.data_stats.normalize(batch_np1).astype(np.float32)).to(ctx.device)
        x2 = torch.from_numpy(ctx.data_stats.normalize(batch_np2).astype(np.float32)).to(ctx.device)
        
        x1_flat = x1.view(4 * ctx.n_nodes, 1)
        x2_flat = x2.view(4 * ctx.n_nodes, 1)
        
        z1, _, _ = ctx.encoder(x1_flat, ctx.A_sparse, ctx.pos, batch_size=4)
        z2, _, _ = ctx.encoder(x2_flat, ctx.A_sparse, ctx.pos, batch_size=4)
        
        print(f"\nLatent stats for batch 1:")
        print(f"  Shape: {z1.shape}")
        print(f"  Mean: {z1.mean().item():.6f}, Std: {z1.std().item():.6f}")
        print(f"  Min: {z1.min().item():.6f}, Max: {z1.max().item():.6f}")
        
        print(f"\nLatent stats for batch 2:")
        print(f"  Mean: {z2.mean().item():.6f}, Std: {z2.std().item():.6f}")
        print(f"  Min: {z2.min().item():.6f}, Max: {z2.max().item():.6f}")
        
        z1_norm = z1 / (z1.norm(dim=1, keepdim=True) + 1e-8)
        z2_norm = z2 / (z2.norm(dim=1, keepdim=True) + 1e-8)
        
        within_batch1_sim = (z1_norm @ z1_norm.T).fill_diagonal_(0).abs().mean().item()
        within_batch2_sim = (z2_norm @ z2_norm.T).fill_diagonal_(0).abs().mean().item()
        across_batch_sim = (z1_norm @ z2_norm.T).abs().mean().item()
        
        print(f"\nCosine similarity (lower = more diverse):")
        print(f"  Within batch 1: {within_batch1_sim:.6f}")
        print(f"  Within batch 2: {within_batch2_sim:.6f}")
        print(f"  Across batches: {across_batch_sim:.6f}")
        
        z_diff = (z1 - z2).abs().mean().item()
        print(f"\nMean absolute difference between batch latents: {z_diff:.6f}")
        
        if z1.std().item() < 0.01:
            print("\n*** WARNING: Latent std is very small - encoder may have collapsed! ***")
        if within_batch1_sim > 0.99:
            print("\n*** WARNING: Latents are nearly identical within batch - encoder collapsed! ***")
    
    return z1, z2


def test_decoder_conditioning_sensitivity():
    """Test 2: Check if decoder output changes when conditioning changes."""
    print("\n" + "=" * 60)
    print("TEST 2: Decoder sensitivity to conditioning")
    print("=" * 60)
    
    cfg = default_config
    ctx = DiffAEContext.build(cfg, for_training=False, verbose=False)
    
    latest_ckpt = ctx.latest_checkpoint()
    if latest_ckpt:
        ctx.load_checkpoint(latest_ckpt, load_optim=False)
    
    ctx.decoder.eval()
    ctx.latent_proj.eval()
    
    with torch.no_grad():
        B = 4
        x_noise = torch.randn(B * ctx.n_nodes, 1, device=ctx.device)
        
        z_random1 = torch.randn(B, cfg.encoder.latent_dim, device=ctx.device)
        z_random2 = torch.randn(B, cfg.encoder.latent_dim, device=ctx.device)
        z_zeros = torch.zeros(B, cfg.encoder.latent_dim, device=ctx.device)
        
        t = torch.full((B,), 50, device=ctx.device, dtype=torch.long)
        t_emb = sinusoidal_embedding(t, cfg.conditioning.time_dim)
        
        cond1 = torch.cat([ctx.latent_proj(z_random1), t_emb], dim=-1)
        cond2 = torch.cat([ctx.latent_proj(z_random2), t_emb], dim=-1)
        cond_zeros = torch.cat([ctx.latent_proj(z_zeros), t_emb], dim=-1)
        
        out1 = ctx.decoder(x_noise, ctx.A_sparse, cond1, ctx.pos, batch_size=B)
        out2 = ctx.decoder(x_noise, ctx.A_sparse, cond2, ctx.pos, batch_size=B)
        out_zeros = ctx.decoder(x_noise, ctx.A_sparse, cond_zeros, ctx.pos, batch_size=B)
        
        diff_12 = (out1 - out2).abs().mean().item()
        diff_1z = (out1 - out_zeros).abs().mean().item()
        diff_2z = (out2 - out_zeros).abs().mean().item()
        
        print(f"\nDecoder output differences:")
        print(f"  Between random latents 1 & 2: {diff_12:.6f}")
        print(f"  Between random 1 & zeros: {diff_1z:.6f}")
        print(f"  Between random 2 & zeros: {diff_2z:.6f}")
        
        out1_std = out1.std().item()
        out2_std = out2.std().item()
        print(f"\nDecoder output stats:")
        print(f"  Output 1 std: {out1_std:.6f}")
        print(f"  Output 2 std: {out2_std:.6f}")
        
        if diff_12 < 1e-4:
            print("\n*** WARNING: Decoder output doesn't change with different conditioning! ***")
            print("*** This suggests conditioning is not being used properly! ***")
        else:
            print("\n[OK] Decoder output varies with conditioning")
    
    return diff_12


def test_gradient_flow():
    """Test 3: Check if gradients flow back through the encoder."""
    print("\n" + "=" * 60)
    print("TEST 3: Gradient flow through encoder")
    print("=" * 60)
    
    cfg = default_config
    ctx = DiffAEContext.build(cfg, for_training=True, verbose=False)
    
    ctx.encoder.train()
    ctx.decoder.train()
    ctx.latent_proj.train()
    
    B = 2
    batch_np, _ = ctx.loader.get_batch(B)
    x0 = torch.from_numpy(ctx.data_stats.normalize(batch_np).astype(np.float32)).to(ctx.device)
    x0_flat = x0.view(B * ctx.n_nodes, 1)
    
    z, _, _ = ctx.encoder(x0_flat, ctx.A_sparse, ctx.pos, batch_size=B)
    cond_base = ctx.latent_proj(z)
    
    print(f"\nLatent z requires_grad: {z.requires_grad}")
    print(f"Latent z shape: {z.shape}")
    print(f"Latent z stats: mean={z.mean().item():.4f}, std={z.std().item():.4f}")
    
    t = torch.randint(0, cfg.diffusion.timesteps, (B,), device=ctx.device, dtype=torch.long)
    t_emb = sinusoidal_embedding(t, cfg.conditioning.time_dim)
    cond_full = torch.cat([cond_base, t_emb], dim=-1)
    
    sqrt_ab = ctx.schedule['sqrt_alphas_cumprod'][t].view(B, 1, 1)
    sqrt_om = ctx.schedule['sqrt_one_minus_alphas_cumprod'][t].view(B, 1, 1)
    noise = torch.randn_like(x0)
    x_t = sqrt_ab * x0 + sqrt_om * noise
    x_t_flat = x_t.view(B * ctx.n_nodes, 1)
    
    pred_flat = ctx.decoder(x_t_flat, ctx.A_sparse, cond_full, ctx.pos, batch_size=B)
    pred = pred_flat.view(B, ctx.n_nodes, 1)
    
    if cfg.diffusion.parametrization == "v":
        target = sqrt_ab * noise - sqrt_om * x0
    else:
        target = noise
    
    loss = F.mse_loss(pred, target)
    loss.backward()
    
    encoder_grad_norm = sum(p.grad.norm().item() for p in ctx.encoder.parameters() if p.grad is not None)
    decoder_grad_norm = sum(p.grad.norm().item() for p in ctx.decoder.parameters() if p.grad is not None)
    latent_proj_grad_norm = sum(p.grad.norm().item() for p in ctx.latent_proj.parameters() if p.grad is not None)
    
    encoder_params_with_grad = sum(1 for p in ctx.encoder.parameters() if p.grad is not None)
    encoder_total_params = sum(1 for p in ctx.encoder.parameters())
    
    print(f"\nGradient norms:")
    print(f"  Encoder: {encoder_grad_norm:.6f} ({encoder_params_with_grad}/{encoder_total_params} params with grad)")
    print(f"  Decoder: {decoder_grad_norm:.6f}")
    print(f"  Latent proj: {latent_proj_grad_norm:.6f}")
    
    to_latent = ctx.encoder.to_latent if hasattr(ctx.encoder, 'to_latent') else None
    if to_latent is not None:
        to_latent_grad = to_latent.weight.grad
        if to_latent_grad is not None:
            print(f"  Encoder to_latent.weight grad norm: {to_latent_grad.norm().item():.6f}")
        else:
            print("  Encoder to_latent.weight grad: None")
    
    if encoder_grad_norm < 1e-8:
        print("\n*** WARNING: No gradients flowing through encoder! ***")
    elif encoder_grad_norm < decoder_grad_norm * 0.001:
        print("\n*** WARNING: Encoder gradients are much smaller than decoder! ***")
    else:
        print("\n[OK] Gradients are flowing through encoder")
    
    return encoder_grad_norm, decoder_grad_norm


def test_latent_proj_output():
    """Test 4: Check the latent projection output."""
    print("\n" + "=" * 60)
    print("TEST 4: Latent projection analysis")
    print("=" * 60)
    
    cfg = default_config
    ctx = DiffAEContext.build(cfg, for_training=False, verbose=False)
    
    latest_ckpt = ctx.latest_checkpoint()
    if latest_ckpt:
        ctx.load_checkpoint(latest_ckpt, load_optim=False)
    
    with torch.no_grad():
        batch_np, _ = ctx.loader.get_batch(8)
        x = torch.from_numpy(ctx.data_stats.normalize(batch_np).astype(np.float32)).to(ctx.device)
        x_flat = x.view(8 * ctx.n_nodes, 1)
        
        z, _, _ = ctx.encoder(x_flat, ctx.A_sparse, ctx.pos, batch_size=8)
        cond_proj = ctx.latent_proj(z)
        
        print(f"\nLatent z:")
        print(f"  Shape: {z.shape}")
        print(f"  Mean: {z.mean().item():.6f}, Std: {z.std().item():.6f}")
        print(f"  Per-dim std: {z.std(dim=0).mean().item():.6f}")
        
        print(f"\nProjected conditioning:")
        print(f"  Shape: {cond_proj.shape}")
        print(f"  Mean: {cond_proj.mean().item():.6f}, Std: {cond_proj.std().item():.6f}")
        print(f"  Per-dim std: {cond_proj.std(dim=0).mean().item():.6f}")
        
        t_emb = sinusoidal_embedding(torch.zeros(8, dtype=torch.long, device=ctx.device), cfg.conditioning.time_dim)
        print(f"\nTime embedding (t=0):")
        print(f"  Shape: {t_emb.shape}")
        print(f"  Mean: {t_emb.mean().item():.6f}, Std: {t_emb.std().item():.6f}")
        
        cond_full = torch.cat([cond_proj, t_emb], dim=-1)
        print(f"\nFull conditioning (cond_proj + t_emb):")
        print(f"  Shape: {cond_full.shape}")
        print(f"  Latent portion std: {cond_full[:, :cfg.conditioning.cond_proj_dim].std().item():.6f}")
        print(f"  Time portion std: {cond_full[:, cfg.conditioning.cond_proj_dim:].std().item():.6f}")
        
        if cond_proj.std().item() < t_emb.std().item() * 0.1:
            print("\n*** WARNING: Latent projection has much lower variance than time embedding! ***")
            print("*** The decoder might be ignoring the latent and just using time! ***")
    
    return z, cond_proj


def test_reconstruction_correlation():
    """Test 5: Check if reconstruction correlates with input."""
    print("\n" + "=" * 60)
    print("TEST 5: Reconstruction correlation with input")
    print("=" * 60)
    
    from diffae import sample_diffae
    
    cfg = default_config
    ctx = DiffAEContext.build(cfg, for_training=False, verbose=False)
    
    latest_ckpt = ctx.latest_checkpoint()
    if latest_ckpt:
        ctx.load_checkpoint(latest_ckpt, load_optim=False)
    else:
        print("No checkpoint found, skipping this test")
        return
    
    ctx.encoder.eval()
    ctx.decoder.eval()
    
    with torch.no_grad():
        batch_np, _ = ctx.loader.get_batch(4)
        batch_np_norm = ctx.data_stats.normalize(batch_np)
        x_ref = torch.from_numpy(batch_np_norm.astype(np.float32)).to(ctx.device)
        
        print("Sampling reconstructions (this may take a moment)...")
        samples = sample_diffae(
            encoder=ctx.encoder,
            decoder=ctx.decoder,
            latent_proj=ctx.latent_proj,
            schedule=ctx.schedule,
            A_sparse=ctx.A_sparse,
            pos=ctx.pos,
            time_dim=cfg.conditioning.time_dim,
            x_ref=x_ref,
            parametrization=cfg.diffusion.parametrization,
            pbar=True
        )
        
        samples_np = samples.cpu().numpy()[:, 0, :]  # (B, N)
        inputs_np = batch_np_norm[:, :, 0]  # (B, N)
        
        correlations = []
        for i in range(4):
            corr = np.corrcoef(inputs_np[i], samples_np[i])[0, 1]
            correlations.append(corr)
            print(f"  Sample {i}: correlation = {corr:.4f}")
        
        print(f"\nMean correlation: {np.mean(correlations):.4f}")
        
        if np.mean(correlations) < 0.1:
            print("\n*** WARNING: Very low correlation between input and reconstruction! ***")
            print("*** The model is not learning to reconstruct the input! ***")
        elif np.mean(correlations) < 0.5:
            print("\n*** Note: Moderate correlation - model may need more training ***")
        else:
            print("\n[OK] Good correlation between input and reconstruction")


def test_conditioning_ablation():
    """Test 6: Compare output with real latent vs random latent."""
    print("\n" + "=" * 60)
    print("TEST 6: Conditioning ablation (real vs random latent)")
    print("=" * 60)
    
    cfg = default_config
    ctx = DiffAEContext.build(cfg, for_training=False, verbose=False)
    
    latest_ckpt = ctx.latest_checkpoint()
    if latest_ckpt:
        ctx.load_checkpoint(latest_ckpt, load_optim=False)
    
    ctx.encoder.eval()
    ctx.decoder.eval()
    ctx.latent_proj.eval()
    
    with torch.no_grad():
        B = 4
        batch_np, _ = ctx.loader.get_batch(B)
        x0 = torch.from_numpy(ctx.data_stats.normalize(batch_np).astype(np.float32)).to(ctx.device)
        x0_flat = x0.view(B * ctx.n_nodes, 1)
        
        z_real, _, _ = ctx.encoder(x0_flat, ctx.A_sparse, ctx.pos, batch_size=B)
        
        z_random = torch.randn_like(z_real)
        z_random = z_random * z_real.std() + z_real.mean()
        
        t = torch.full((B,), 10, device=ctx.device, dtype=torch.long)
        t_emb = sinusoidal_embedding(t, cfg.conditioning.time_dim)
        sqrt_ab = ctx.schedule['sqrt_alphas_cumprod'][t].view(B, 1, 1)
        sqrt_om = ctx.schedule['sqrt_one_minus_alphas_cumprod'][t].view(B, 1, 1)
        
        noise = torch.randn_like(x0)
        x_t = sqrt_ab * x0 + sqrt_om * noise
        x_t_flat = x_t.view(B * ctx.n_nodes, 1)
        
        cond_real = torch.cat([ctx.latent_proj(z_real), t_emb], dim=-1)
        cond_random = torch.cat([ctx.latent_proj(z_random), t_emb], dim=-1)
        
        pred_real = ctx.decoder(x_t_flat, ctx.A_sparse, cond_real, ctx.pos, batch_size=B)
        pred_random = ctx.decoder(x_t_flat, ctx.A_sparse, cond_random, ctx.pos, batch_size=B)
        
        pred_real = pred_real.view(B, ctx.n_nodes, 1)
        pred_random = pred_random.view(B, ctx.n_nodes, 1)
        
        if cfg.diffusion.parametrization == "v":
            x0_pred_real = sqrt_ab * x_t - sqrt_om * pred_real
            x0_pred_random = sqrt_ab * x_t - sqrt_om * pred_random
        else:
            x0_pred_real = (x_t - sqrt_om * pred_real) / sqrt_ab
            x0_pred_random = (x_t - sqrt_om * pred_random) / sqrt_ab
        
        mse_real = F.mse_loss(x0_pred_real, x0).item()
        mse_random = F.mse_loss(x0_pred_random, x0).item()
        
        print(f"\nReconstruction MSE at t=10:")
        print(f"  With real encoder latent: {mse_real:.6f}")
        print(f"  With random latent: {mse_random:.6f}")
        print(f"  Ratio (real/random): {mse_real/mse_random:.4f}")
        
        if mse_real >= mse_random * 0.95:
            print("\n*** WARNING: Real latent doesn't help reconstruction! ***")
            print("*** The encoder output is not being used effectively! ***")
        else:
            improvement = (1 - mse_real/mse_random) * 100
            print(f"\n[OK] Real latent improves reconstruction by {improvement:.1f}%")


def run_all_tests():
    """Run all diagnostic tests."""
    print("\n" + "=" * 60)
    print("DiffAE CONDITIONING DIAGNOSTIC TESTS")
    print("=" * 60)
    
    test_encoder_sensitivity()
    test_decoder_conditioning_sensitivity()
    test_gradient_flow()
    test_latent_proj_output()
    test_conditioning_ablation()
    
    print("\n" + "=" * 60)
    print("OPTIONAL: Full reconstruction test (slow)")
    print("=" * 60)
    response = input("Run full reconstruction test? (y/n): ")
    if response.lower() == 'y':
        test_reconstruction_correlation()
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
