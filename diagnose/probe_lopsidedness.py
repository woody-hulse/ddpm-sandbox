"""
Post-hoc investigation of lopsidedness encoding in DiffAE.

This script does NOT train the DiffAE. It loads a frozen checkpoint and runs:
  1. Linear probe: train a small classifier on frozen z to detect lopsidedness
  2. Loss contribution: measure how much lopsidedness affects the diffusion MSE
  3. Conditioning sensitivity: perturb z and measure output change
  4. Per-node analysis: check if the decoder uses position info to vary noise

Usage:
    python probe_lopsidedness.py
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d

from config import default_config
from diffae import DiffAEContext, apply_lopsided_augmentation, sample_diffae
from ae import AEContext, reconstruct_ae
from diffusion.schedule import sinusoidal_embedding


def load_diffae_model(cfg):
    ctx = DiffAEContext.build(cfg, for_training=False, verbose=False)
    ckpt_path = ctx.latest_checkpoint()
    if ckpt_path is None:
        raise FileNotFoundError("No DiffAE checkpoint found")
    chk = torch.load(ckpt_path, map_location=ctx.device, weights_only=False)
    for key, target in [("ema_encoder", ctx.encoder), ("ema_decoder", ctx.decoder),
                        ("ema_latent_proj", ctx.latent_proj)]:
        if key in chk:
            target.load_state_dict(chk[key], strict=False)
        else:
            target.load_state_dict(chk[key.replace("ema_", "")], strict=False)
    if "data_stats" in chk:
        ctx.data_stats.mean = chk["data_stats"]["mean"]
        ctx.data_stats.std = chk["data_stats"]["std"]
    epoch = int(chk.get("epoch", 0))
    ctx.encoder.eval()
    ctx.decoder.eval()
    ctx.latent_proj.eval()
    print(f"Loaded epoch {epoch} from {ckpt_path}")
    return ctx


def load_ae_model(cfg):
    ctx = AEContext.build(cfg, for_training=False, verbose=False)
    ckpt_path = ctx.latest_checkpoint()
    if ckpt_path is None:
        raise FileNotFoundError("No AE checkpoint found")
    chk = torch.load(ckpt_path, map_location=ctx.device, weights_only=False)
    for key, target in [("ema_encoder", ctx.encoder), ("ema_decoder", ctx.decoder)]:
        if key in chk:
            target.load_state_dict(chk[key], strict=False)
        else:
            target.load_state_dict(chk[key.replace("ema_", "")], strict=False)
    if "data_stats" in chk:
        ctx.data_stats.mean = chk["data_stats"]["mean"]
        ctx.data_stats.std = chk["data_stats"]["std"]
    epoch = int(chk.get("epoch", 0))
    ctx.encoder.eval()
    ctx.decoder.eval()
    print(f"Loaded AE epoch {epoch} from {ckpt_path}")
    return ctx


def get_lopsided_batch(ctx, cfg, n, sigma=None):
    """Get n events with known lopsidedness labels. Returns (batch_np, labels, indices)."""
    if sigma is None:
        sigma = cfg.training.lopsided_sigma
    loader = ctx.loader
    left, right = [], []
    left_idx, right_idx = [], []
    while len(left) < n // 2 or len(right) < n // 2:
        batch_np, _, idx = loader.get_batch(256)
        N = batch_np.shape[1]
        half = N // 2
        for i in range(batch_np.shape[0]):
            if int(idx[i]) % 2 == 0 and len(left) < n // 2:
                wf = batch_np[i].copy()
                wf[:half, 0] = gaussian_filter1d(wf[:half, 0], sigma=sigma)
                left.append(wf)
                left_idx.append(idx[i])
            elif int(idx[i]) % 2 == 1 and len(right) < n // 2:
                wf = batch_np[i].copy()
                wf[half:, 0] = gaussian_filter1d(wf[half:, 0], sigma=sigma)
                right.append(wf)
                right_idx.append(idx[i])
    batch_np = np.concatenate([np.stack(left), np.stack(right)], axis=0)
    labels = np.array([0] * (n // 2) + [1] * (n // 2))
    indices = np.array(left_idx + right_idx)
    return batch_np, labels, indices


def _extract_z(enc_out):
    """Handle encoder API differences across DiffAE/AE variants."""
    if isinstance(enc_out, tuple):
        return enc_out[0]
    return enc_out


@torch.no_grad()
def encode_batch(ctx, batch_np, batch_size=32):
    """Encode events to latent z. Returns z (N, latent_dim)."""
    data_stats = ctx.data_stats
    device = next(ctx.encoder.parameters()).device
    batch_norm = data_stats.normalize(batch_np)
    all_z = []
    for i in range(0, len(batch_norm), batch_size):
        chunk = batch_norm[i:i + batch_size]
        x = torch.from_numpy(chunk.astype(np.float32)).to(device)
        x_flat = x.view(x.shape[0] * x.shape[1], 1)
        enc_out = ctx.encoder(x_flat, ctx.A_sparse, ctx.pos, batch_size=x.shape[0])
        z = _extract_z(enc_out)
        all_z.append(z.cpu())
    return torch.cat(all_z, dim=0)


# =========================================================================
# Test 1: Linear probe on frozen latents
# =========================================================================
def test_linear_probe(z_train, labels_train, z_test, labels_test, model_name="Model", epochs=2000):
    """Train a linear classifier on frozen z to predict lopsidedness."""
    print("\n" + "=" * 60)
    print(f"TEST 1: Linear Probe on Frozen Latents ({model_name})")
    print("=" * 60)

    in_dim = z_train.shape[1]
    probe = nn.Linear(in_dim, 2)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)

    y_train = torch.tensor(labels_train, dtype=torch.long)
    y_test = torch.tensor(labels_test, dtype=torch.long)

    for ep in range(epochs):
        logits = probe(z_train)
        loss = F.cross_entropy(logits, y_train)
        if (ep + 1) % max(1, epochs // 4) == 0:
            print(f'Epoch {ep + 1}, Loss: {loss.item():.4f}')
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        train_acc = (probe(z_train).argmax(1) == y_train).float().mean().item()
        test_acc = (probe(z_test).argmax(1) == y_test).float().mean().item()

    print(f"  Train accuracy: {train_acc:.4f}")
    print(f"  Test accuracy:  {test_acc:.4f}")
    print(f"  (Chance = 0.50)")

    w = probe.weight.data
    importance = (w[0] - w[1]).abs()
    top_dims = importance.argsort(descending=True)[:5]
    print(f"  Top 5 discriminative latent dims: {top_dims.tolist()}")
    print(f"  Their importance:                 {importance[top_dims].tolist()}")

    return test_acc


def finetune_ae(ctx, cfg, epochs: int, steps_per_epoch: int, batch_size: int, lr: float):
    """Optional light fine-tuning of AE with the same online sampling pattern as train_ae."""
    print("\n" + "=" * 60)
    print(f"Fine-tuning AE for {epochs} epochs x {steps_per_epoch} steps (batch={batch_size}, lr={lr:g})")
    print("=" * 60)

    optimizer = torch.optim.AdamW(
        list(ctx.encoder.parameters()) + list(ctx.decoder.parameters()),
        lr=lr, betas=(0.9, 0.999), weight_decay=cfg.training.weight_decay
    )

    ctx.encoder.train()
    ctx.decoder.train()
    for ep in range(epochs):
        loss_acc = 0.0
        for step in range(steps_per_epoch):
            batch_np, _, sample_idx = ctx.loader.get_batch(batch_size)
            if cfg.training.lopsided_aug:
                batch_np = apply_lopsided_augmentation(
                    batch_np, frac=cfg.training.lopsided_frac, sigma=cfg.training.lopsided_sigma,
                    sample_indices=sample_idx
                )
            x0 = torch.from_numpy(ctx.data_stats.normalize(batch_np).astype(np.float32)).to(ctx.device)
            B, N, C = x0.shape
            z, _ = ctx.encoder(x0.view(B * N, C), ctx.A_sparse, ctx.pos, batch_size=B)
            rec = ctx.decoder(z, ctx.A_sparse, ctx.pos, batch_size=B).view(B, N, C)
            loss = F.mse_loss(rec, x0, reduction='mean')

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(ctx.encoder.parameters()) + list(ctx.decoder.parameters()), max_norm=cfg.training.grad_clip)
            optimizer.step()
            loss_acc += float(loss.item())
        print(f"  AE finetune epoch {ep + 1}/{epochs}: loss={loss_acc / max(steps_per_epoch, 1):.6f}")

    ctx.encoder.eval()
    ctx.decoder.eval()


def finetune_diffae(ctx, cfg, epochs: int, steps_per_epoch: int, batch_size: int, lr: float):
    """Optional light fine-tuning of DiffAE with the same online sampling pattern as train_diffae."""
    print("\n" + "=" * 60)
    print(f"Fine-tuning DiffAE for {epochs} epochs x {steps_per_epoch} steps (batch={batch_size}, lr={lr:g})")
    print("=" * 60)

    optimizer = torch.optim.AdamW(
        list(ctx.encoder.parameters()) + list(ctx.decoder.parameters()) + list(ctx.latent_proj.parameters()),
        lr=lr, betas=(0.9, 0.999), weight_decay=cfg.training.weight_decay
    )

    schedule = ctx.schedule
    ctx.encoder.train()
    ctx.decoder.train()
    ctx.latent_proj.train()
    for ep in range(epochs):
        loss_acc = 0.0
        kl_acc = 0.0
        for step in range(steps_per_epoch):
            batch_np, _, sample_idx = ctx.loader.get_batch(batch_size)
            if cfg.training.lopsided_aug:
                batch_np = apply_lopsided_augmentation(
                    batch_np, frac=cfg.training.lopsided_frac, sigma=cfg.training.lopsided_sigma,
                    sample_indices=sample_idx
                )
            x0 = torch.from_numpy(ctx.data_stats.normalize(batch_np).astype(np.float32)).to(ctx.device)
            B, N, C = x0.shape

            z, mu, logvar = ctx.encoder(x0.view(B * N, C), ctx.A_sparse, ctx.pos, batch_size=B)
            cond_base = ctx.latent_proj(z)

            t_min = int(getattr(cfg.diffusion, 't_min_frac', 0.0) * cfg.diffusion.timesteps)
            t = torch.randint(t_min, cfg.diffusion.timesteps, (B,), device=ctx.device, dtype=torch.long)
            t_emb = sinusoidal_embedding(t, cfg.conditioning.time_dim)
            cond_full = torch.cat([cond_base, t_emb], dim=-1)

            sqrt_ab = schedule['sqrt_alphas_cumprod'][t].view(B, 1, 1)
            sqrt_om = schedule['sqrt_one_minus_alphas_cumprod'][t].view(B, 1, 1)
            snr_t = schedule['snr'][t].view(B)

            noise = torch.randn_like(x0)
            x_t = sqrt_ab * x0 + sqrt_om * noise
            pred = ctx.decoder(x_t.view(B * N, C), ctx.A_sparse, cond_full, ctx.pos, batch_size=B).view(B, N, C)

            if cfg.diffusion.parametrization == "eps":
                target = noise
            else:
                target = sqrt_ab * noise - sqrt_om * x0

            mse_per_sample = F.mse_loss(pred, target, reduction='none').mean(dim=(1, 2))
            if cfg.diffusion.p2_gamma > 0.0:
                weight = torch.pow(cfg.diffusion.p2_k + snr_t, -cfg.diffusion.p2_gamma)
                mse_per_sample = mse_per_sample * weight
            loss = mse_per_sample.mean()

            if cfg.encoder.use_stochastic and mu is not None and logvar is not None:
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = loss + cfg.encoder.kl_weight * kl_loss
                kl_acc += float(kl_loss.item())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(ctx.encoder.parameters()) + list(ctx.decoder.parameters()) + list(ctx.latent_proj.parameters()),
                max_norm=cfg.training.grad_clip
            )
            optimizer.step()
            loss_acc += float(loss.item())

        msg = f"  DiffAE finetune epoch {ep + 1}/{epochs}: loss={loss_acc / max(steps_per_epoch, 1):.6f}"
        if cfg.encoder.use_stochastic:
            msg += f", kl={kl_acc / max(steps_per_epoch, 1):.6f}"
        print(msg)

    ctx.encoder.eval()
    ctx.decoder.eval()
    ctx.latent_proj.eval()


@torch.no_grad()
def save_autoencoding_plots(diffae_ctx, ae_ctx, cfg, out_dir: str, n_events: int = 4):
    """Generate side-by-side raw vs AE/DiffAE reconstruction plots."""
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    batch_np, _, sample_idx = diffae_ctx.loader.get_batch(n_events)
    if cfg.training.lopsided_aug:
        batch_np = apply_lopsided_augmentation(
            batch_np.copy(), frac=cfg.training.lopsided_frac, sigma=cfg.training.lopsided_sigma,
            sample_indices=sample_idx
        )
    raw = batch_np[:, :, 0]

    x_diffae = torch.from_numpy(diffae_ctx.data_stats.normalize(batch_np).astype(np.float32)).to(diffae_ctx.device)
    rec_diffae = sample_diffae(
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
    ).cpu().numpy()
    rec_diffae = diffae_ctx.data_stats.denormalize(rec_diffae)[:, 0, :]

    x_ae = torch.from_numpy(ae_ctx.data_stats.normalize(batch_np).astype(np.float32)).to(ae_ctx.device)
    rec_ae = reconstruct_ae(
        encoder=ae_ctx.encoder,
        decoder=ae_ctx.decoder,
        A_sparse=ae_ctx.A_sparse,
        pos=ae_ctx.pos,
        x_ref=x_ae,
    ).cpu().numpy()
    rec_ae = ae_ctx.data_stats.denormalize(rec_ae)[:, 0, :]

    fig, axes = plt.subplots(n_events, 2, figsize=(14, 3.2 * n_events), sharex=True)
    if n_events == 1:
        axes = np.array([axes])
    for i in range(n_events):
        axes[i, 0].plot(raw[i], color='black', linewidth=1.0, label='Raw')
        axes[i, 0].plot(rec_ae[i], color='#1f77b4', linewidth=1.0, alpha=0.9, label='AE recon')
        axes[i, 0].set_title(f'AE Reconstruction (event {i})')
        axes[i, 0].grid(alpha=0.25, linewidth=0.4)

        axes[i, 1].plot(raw[i], color='black', linewidth=1.0, label='Raw')
        axes[i, 1].plot(rec_diffae[i], color='#ff7f0e', linewidth=1.0, alpha=0.9, label='DiffAE recon')
        axes[i, 1].set_title(f'DiffAE Reconstruction (event {i})')
        axes[i, 1].grid(alpha=0.25, linewidth=0.4)

        if i == 0:
            axes[i, 0].legend(loc='upper right', fontsize=8)
            axes[i, 1].legend(loc='upper right', fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "autoencoding_comparison.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved autoencoding plot: {out_path}")


# =========================================================================
# Test 2: Loss contribution of lopsidedness
# =========================================================================
@torch.no_grad()
def test_loss_contribution(ctx, cfg, batch_np, labels):
    """Measure MSE difference between lopsided-correct and lopsided-reversed targets."""
    print("\n" + "=" * 60)
    print("TEST 2: Loss Contribution of Lopsidedness")
    print("=" * 60)

    half = batch_np.shape[1] // 2
    sigma = cfg.training.lopsided_sigma

    mse_with_correct = []
    mse_with_reversed = []

    for i in range(len(batch_np)):
        wf = batch_np[i, :, 0].copy()
        wf_reversed = wf.copy()
        if labels[i] == 0:
            wf_reversed[:half] = wf[half:][::-1].copy()
            wf_reversed[half:] = gaussian_filter1d(wf_reversed[half:], sigma=sigma)
            wf_reversed[:half] = wf[:half]
        else:
            tmp = wf.copy()
            wf_reversed[half:] = wf[:half][::-1].copy()
            wf_reversed[:half] = gaussian_filter1d(wf_reversed[:half], sigma=sigma)
            wf_reversed[half:] = wf[half:]

        mse_correct = float(np.mean((wf - wf) ** 2))
        mse_reversed = float(np.mean((wf - wf_reversed) ** 2))
        mse_with_correct.append(mse_correct)
        mse_with_reversed.append(mse_reversed)

    avg_mse_diff = np.mean(mse_with_reversed)
    typical_mse = np.mean(batch_np[:, :, 0] ** 2)

    print(f"  Average MSE from reversing lopsidedness: {avg_mse_diff:.4f}")
    print(f"  Typical event energy (mean x^2):         {typical_mse:.4f}")
    print(f"  Lopsidedness fraction of total energy:    {avg_mse_diff / max(typical_mse, 1e-8) * 100:.2f}%")

    hf_diffs = []
    for i in range(len(batch_np)):
        wf = batch_np[i, :, 0]
        left_hf = np.mean(np.diff(wf[:half]) ** 2)
        right_hf = np.mean(np.diff(wf[half:]) ** 2)
        hf_diffs.append(abs(left_hf - right_hf))

    print(f"  Mean |HF_left - HF_right|:               {np.mean(hf_diffs):.4f}")

    return avg_mse_diff, typical_mse


# =========================================================================
# Test 3: Conditioning sensitivity
# =========================================================================
@torch.no_grad()
def test_conditioning_sensitivity(ctx, cfg, batch_np):
    """Measure how much perturbing z changes the decoder output at each timestep."""
    print("\n" + "=" * 60)
    print("TEST 3: Conditioning Sensitivity (how much does z matter?)")
    print("=" * 60)

    device = next(ctx.encoder.parameters()).device
    schedule = ctx.schedule
    data_stats = ctx.data_stats
    T = schedule['betas'].shape[0]

    batch_norm = data_stats.normalize(batch_np[:16])
    x0 = torch.from_numpy(batch_norm.astype(np.float32)).to(device)
    B, N, C = x0.shape
    x0_flat = x0.view(B * N, C)

    z, _, _ = ctx.encoder(x0_flat, ctx.A_sparse, ctx.pos, batch_size=B)
    cond_base = ctx.latent_proj(z)

    z_random = torch.randn_like(z)
    cond_random = ctx.latent_proj(z_random)

    t_vals = [0, 10, 25, 50, 100, 150, 200, 249]
    print(f"  {'t':>5s}  {'pred_diff':>12s}  {'pred_mag':>12s}  {'ratio':>10s}")

    for t_val in t_vals:
        t_tensor = torch.full((B,), t_val, device=device, dtype=torch.long)
        t_emb = sinusoidal_embedding(t_tensor, cfg.conditioning.time_dim)

        sqrt_ab = schedule['sqrt_alphas_cumprod'][t_val]
        sqrt_om = schedule['sqrt_one_minus_alphas_cumprod'][t_val]
        noise = torch.randn_like(x0)
        x_t = sqrt_ab * x0 + sqrt_om * noise
        x_t_flat = x_t.view(B * N, C)

        cond_real = torch.cat([cond_base, t_emb], dim=-1)
        cond_rand = torch.cat([cond_random, t_emb], dim=-1)

        pred_real = ctx.decoder(x_t_flat, ctx.A_sparse, cond_real, ctx.pos, batch_size=B)
        pred_rand = ctx.decoder(x_t_flat, ctx.A_sparse, cond_rand, ctx.pos, batch_size=B)

        diff = (pred_real - pred_rand).pow(2).mean().item()
        mag = pred_real.pow(2).mean().item()
        ratio = diff / max(mag, 1e-8)

        print(f"  {t_val:>5d}  {diff:>12.6f}  {mag:>12.6f}  {ratio:>10.4f}")

    return


# =========================================================================
# Test 4: Per-node output variance analysis
# =========================================================================
@torch.no_grad()
def test_per_node_variance(ctx, cfg, batch_np, labels):
    """Check if the decoder produces different noise levels for left vs right nodes."""
    print("\n" + "=" * 60)
    print("TEST 4: Per-Node Output Variance (does decoder vary noise by position?)")
    print("=" * 60)

    device = next(ctx.encoder.parameters()).device
    schedule = ctx.schedule
    data_stats = ctx.data_stats

    batch_norm = data_stats.normalize(batch_np[:32])
    x0 = torch.from_numpy(batch_norm.astype(np.float32)).to(device)
    B, N, C = x0.shape
    half = N // 2
    x0_flat = x0.view(B * N, C)

    z, _, _ = ctx.encoder(x0_flat, ctx.A_sparse, ctx.pos, batch_size=B)
    cond_base = ctx.latent_proj(z)

    for t_val in [10, 50, 100]:
        t_tensor = torch.full((B,), t_val, device=device, dtype=torch.long)
        t_emb = sinusoidal_embedding(t_tensor, cfg.conditioning.time_dim)
        cond_full = torch.cat([cond_base, t_emb], dim=-1)

        preds = []
        for _ in range(5):
            noise = torch.randn_like(x0)
            x_t = schedule['sqrt_alphas_cumprod'][t_val] * x0 + schedule['sqrt_one_minus_alphas_cumprod'][t_val] * noise
            x_t_flat = x_t.view(B * N, C)
            pred = ctx.decoder(x_t_flat, ctx.A_sparse, cond_full, ctx.pos, batch_size=B)
            preds.append(pred.view(B, N, C))

        preds = torch.stack(preds, dim=0)
        var_per_node = preds.var(dim=0).mean(dim=(0, 2))

        var_left = var_per_node[:half].mean().item()
        var_right = var_per_node[half:].mean().item()

        left_labels = labels[:32]
        left_smooth_mask = left_labels == 0
        right_smooth_mask = left_labels == 1

        if left_smooth_mask.sum() > 0:
            ls_pred_var_left = preds[:, left_smooth_mask].var(dim=0)[:, :half].mean().item()
            ls_pred_var_right = preds[:, left_smooth_mask].var(dim=0)[:, half:].mean().item()
        else:
            ls_pred_var_left = ls_pred_var_right = 0

        print(f"  t={t_val}: var_left={var_left:.6f}, var_right={var_right:.6f}, "
              f"ratio={var_left / max(var_right, 1e-8):.4f}")
        if left_smooth_mask.sum() > 0:
            print(f"         Left-smooth events: var_left={ls_pred_var_left:.6f}, "
                  f"var_right={ls_pred_var_right:.6f}, ratio={ls_pred_var_left / max(ls_pred_var_right, 1e-8):.4f}")


# =========================================================================
# Test 5: MSE decomposition by timestep - how much does lopsidedness contribute?
# =========================================================================
@torch.no_grad()
def test_mse_decomposition(ctx, cfg, batch_np, labels):
    """At each timestep, measure how much of the loss is from lopsided vs non-lopsided structure."""
    print("\n" + "=" * 60)
    print("TEST 5: MSE Decomposition by Timestep")
    print("=" * 60)

    device = next(ctx.encoder.parameters()).device
    schedule = ctx.schedule
    data_stats = ctx.data_stats

    batch_norm = data_stats.normalize(batch_np[:32])
    x0 = torch.from_numpy(batch_norm.astype(np.float32)).to(device)
    B, N, C = x0.shape
    half = N // 2
    x0_flat = x0.view(B * N, C)

    z, _, _ = ctx.encoder(x0_flat, ctx.A_sparse, ctx.pos, batch_size=B)
    cond_base = ctx.latent_proj(z)

    left_mask = torch.tensor(labels[:32] == 0, dtype=torch.bool)

    print(f"  {'t':>5s}  {'total_mse':>10s}  {'smooth_half':>12s}  {'rough_half':>11s}  {'smooth/rough':>13s}")

    for t_val in [5, 10, 25, 50, 100, 200, 249]:
        t_tensor = torch.full((B,), t_val, device=device, dtype=torch.long)
        t_emb = sinusoidal_embedding(t_tensor, cfg.conditioning.time_dim)
        cond_full = torch.cat([cond_base, t_emb], dim=-1)

        sqrt_ab = schedule['sqrt_alphas_cumprod'][t_val]
        sqrt_om = schedule['sqrt_one_minus_alphas_cumprod'][t_val]
        noise = torch.randn_like(x0)
        x_t = sqrt_ab * x0 + sqrt_om * noise
        x_t_flat = x_t.view(B * N, C)

        pred = ctx.decoder(x_t_flat, ctx.A_sparse, cond_full, ctx.pos, batch_size=B).view(B, N, C)

        if cfg.diffusion.parametrization == "v":
            target = sqrt_ab * noise - sqrt_om * x0
        else:
            target = noise

        mse_total = F.mse_loss(pred, target).item()

        smooth_mse_list, rough_mse_list = [], []
        for i in range(B):
            if left_mask[i]:
                smooth_mse_list.append(F.mse_loss(pred[i, :half], target[i, :half]).item())
                rough_mse_list.append(F.mse_loss(pred[i, half:], target[i, half:]).item())
            else:
                smooth_mse_list.append(F.mse_loss(pred[i, half:], target[i, half:]).item())
                rough_mse_list.append(F.mse_loss(pred[i, :half], target[i, :half]).item())

        smooth_mse = np.mean(smooth_mse_list)
        rough_mse = np.mean(rough_mse_list)
        ratio = smooth_mse / max(rough_mse, 1e-8)

        print(f"  {t_val:>5d}  {mse_total:>10.6f}  {smooth_mse:>12.6f}  {rough_mse:>11.6f}  {ratio:>13.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Probe lopsidedness encoding in DiffAE vs AE")
    parser.add_argument("--n-total", type=int, default=5000, help="Total labeled lopsided events for probe task")
    parser.add_argument("--probe-epochs", type=int, default=2000, help="Linear probe training epochs")
    parser.add_argument("--finetune", action="store_true", help="Fine-tune both AE and DiffAE before probing")
    parser.add_argument("--finetune-epochs", type=int, default=2, help="Fine-tuning epochs when --finetune is set")
    parser.add_argument("--finetune-steps-per-epoch", type=int, default=64, help="Fine-tuning steps per epoch")
    parser.add_argument("--finetune-batch-size", type=int, default=8, help="Fine-tuning batch size")
    parser.add_argument("--finetune-lr", type=float, default=None, help="Fine-tuning LR (default: config training.lr)")
    parser.add_argument("--plot-events", type=int, default=4, help="Events to plot for AE/DiffAE reconstructions")
    parser.add_argument("--plot-dir", type=str, default="probe_outputs", help="Directory for generated plots")
    parser.add_argument("--no-autoencoding-plots", action="store_true", help="Disable reconstruction plot generation")
    parser.add_argument("--skip-diffae-diagnostics", action="store_true", help="Skip DiffAE-only tests 2-5")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = default_config
    print("Loading DiffAE model...")
    diffae_ctx = load_diffae_model(cfg)
    print("Loading AE model...")
    ae_ctx = load_ae_model(cfg)

    if args.finetune:
        lr = cfg.training.lr if args.finetune_lr is None else args.finetune_lr
        finetune_diffae(
            diffae_ctx, cfg,
            epochs=args.finetune_epochs,
            steps_per_epoch=args.finetune_steps_per_epoch,
            batch_size=args.finetune_batch_size,
            lr=lr,
        )
        finetune_ae(
            ae_ctx, cfg,
            epochs=args.finetune_epochs,
            steps_per_epoch=args.finetune_steps_per_epoch,
            batch_size=args.finetune_batch_size,
            lr=lr,
        )

    if not args.no_autoencoding_plots:
        save_autoencoding_plots(diffae_ctx, ae_ctx, cfg, out_dir=args.plot_dir, n_events=args.plot_events)

    print("Generating lopsided events...")
    n_total = args.n_total
    batch_np, labels, indices = get_lopsided_batch(diffae_ctx, cfg, n_total)
    print(f"  Generated {n_total} events: {(labels == 0).sum()} left-smooth, {(labels == 1).sum()} right-smooth")

    perm = np.random.RandomState(42).permutation(n_total)
    labels_shuffled = labels[perm]
    split = n_total * 3 // 4
    l_train, l_test = labels_shuffled[:split], labels_shuffled[split:]

    print("Encoding events with DiffAE...")
    z_diffae = encode_batch(diffae_ctx, batch_np)[perm]
    z_train_d, z_test_d = z_diffae[:split], z_diffae[split:]
    probe_acc_diffae = test_linear_probe(z_train_d, l_train, z_test_d, l_test, model_name="DiffAE", epochs=args.probe_epochs)

    print("Encoding events with AE...")
    z_ae = encode_batch(ae_ctx, batch_np)[perm]
    z_train_a, z_test_a = z_ae[:split], z_ae[split:]
    probe_acc_ae = test_linear_probe(z_train_a, l_train, z_test_a, l_test, model_name="AE", epochs=args.probe_epochs)

    # Keep existing DiffAE-specific diagnostics for deeper analysis.
    if not args.skip_diffae_diagnostics:
        test_loss_contribution(diffae_ctx, cfg, batch_np, labels)
        test_conditioning_sensitivity(diffae_ctx, cfg, batch_np)
        test_per_node_variance(diffae_ctx, cfg, batch_np, labels)
        test_mse_decomposition(diffae_ctx, cfg, batch_np, labels)

    print("\n" + "=" * 60)
    print("OVERALL SUMMARY (Lopsided Linear Probe)")
    print("=" * 60)
    print(f"  DiffAE linear probe accuracy: {probe_acc_diffae:.4f} (chance=0.50)")
    print(f"  AE linear probe accuracy:     {probe_acc_ae:.4f} (chance=0.50)")
    print(f"  Difference (DiffAE - AE):     {probe_acc_diffae - probe_acc_ae:+.4f}")


if __name__ == "__main__":
    main()
