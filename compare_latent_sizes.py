"""
Compare AE vs DiffAE auxiliary-task performance across latent sizes.

Pipeline:
  1. For each latent_dim in [2, 4, 8, 16, 32, 64, 128]:
       a. Train AE  (early stopping, patience=50, smart weight loading)
       b. Train DiffAE (early stopping, patience=50, smart weight loading)
       c. Encode datasets with both models
  2. Train baseline MLP on raw waveforms (once)
  3. For each encoded dataset, train aux MLPs (randomised train/test split)
  4. Plot MAE vs latent size (log2 x-axis, error bars, paper-ready)

Usage:
    python compare_latent_sizes.py
    python compare_latent_sizes.py --latent-dims 2 4 8 16 32 64 128
    python compare_latent_sizes.py --skip-training  # use existing checkpoints only
"""
import os
import sys
import glob
import argparse
import json
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config, default_config, get_config
from ae import AEContext, reconstruct_ae, save_encoded_dataset as ae_save_encoded
from diffae import DiffAEContext, sample_diffae, save_encoded_dataset as diffae_save_encoded
from compare_rqs import wf_to_z_profile
from aux import (
    EncodedMSDataset, OnlineMSDataset, MLP,
    train_aux_mlp_on_latents, evaluate_aux_mlp_on_latents,
    train_baseline_mlp, evaluate_baseline_mlp,
)


# ---------------------------------------------------------------------------
# Early-stopping training wrappers
# ---------------------------------------------------------------------------

def train_ae_early_stop(
    cfg: Config,
    patience: int = 50,
    min_epochs: int = 100,
    verbose: bool = True,
) -> str:
    """Train AE with early stopping on epoch loss. Returns path to best checkpoint."""
    ctx = AEContext.build(cfg, for_training=True, verbose=verbose)

    start_epoch = 0
    if cfg.resume:
        last = ctx.latest_checkpoint()
        if last is not None:
            try:
                start_epoch = ctx.load_checkpoint(last) + 1
                if verbose:
                    print(f"Loaded AE checkpoint: {last} (epoch {start_epoch - 1})")
            except Exception:
                pass

        if start_epoch == 0:
            best_ckpt_cross = ctx.find_best_checkpoint()
            if best_ckpt_cross is not None:
                try:
                    _, is_full = ctx.load_checkpoint_partial(best_ckpt_cross, verbose=verbose)
                    if verbose:
                        tag = "full" if is_full else "partial"
                        print(f"Loaded {tag} weights from {best_ckpt_cross}")
                except Exception as e:
                    if verbose:
                        print(f"Partial load failed: {e}")

    for g in ctx.optim.param_groups:
        g["lr"] = cfg.training.lr

    B = cfg.training.batch_size
    best_loss = float("inf")
    wait = 0
    best_epoch = start_epoch
    best_ckpt_path = None
    max_epoch = cfg.training.epochs

    epoch = start_epoch
    while epoch < max_epoch:
        ctx.encoder.train()
        ctx.decoder.train()
        epoch_loss = 0.0
        n_steps = cfg.training.steps_per_epoch

        pbar = tqdm(
            range(n_steps),
            desc=f"AE z{cfg.encoder.latent_dim} Epoch {epoch+1}",
            ncols=120, file=sys.stdout, disable=not verbose,
        )
        for step in pbar:
            batch_np, _ = ctx.loader.get_batch(B)
            batch_np = ctx.data_stats.normalize(batch_np)
            x0 = torch.from_numpy(batch_np.astype(np.float32)).to(ctx.device)
            x0_flat = x0.view(B * ctx.n_nodes, 1)

            z, _ = ctx.encoder(x0_flat, ctx.A_sparse, ctx.pos, batch_size=B)
            rec_flat = ctx.decoder(z, ctx.A_sparse, ctx.pos, batch_size=B)
            rec = rec_flat.view(B, ctx.n_nodes, 1)
            loss = F.mse_loss(rec, x0)

            if torch.isnan(loss) or torch.isinf(loss):
                ctx.optim.zero_grad(set_to_none=True)
                continue

            epoch_loss += loss.item()
            ctx.optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(ctx.encoder.parameters()) + list(ctx.decoder.parameters()),
                max_norm=cfg.training.grad_clip,
            )
            ctx.optim.step()

            with torch.no_grad():
                for p_ema, p in zip(ctx.ema_encoder.parameters(), ctx.encoder.parameters()):
                    p_ema.data.mul_(cfg.training.ema_decay).add_(p.data, alpha=1.0 - cfg.training.ema_decay)
                for p_ema, p in zip(ctx.ema_decoder.parameters(), ctx.decoder.parameters()):
                    p_ema.data.mul_(cfg.training.ema_decay).add_(p.data, alpha=1.0 - cfg.training.ema_decay)

            pbar.set_postfix(loss=epoch_loss / (step + 1))

        avg_loss = epoch_loss / n_steps

        if (epoch + 1) % cfg.training.checkpoint_every == 0:
            ctx.save_checkpoint(epoch)

        if avg_loss < best_loss:
            best_loss = avg_loss
            wait = 0
            best_epoch = epoch
            best_ckpt_path = ctx.save_checkpoint(epoch)
        else:
            wait += 1

        epochs_trained = epoch - start_epoch + 1
        if wait >= patience and epochs_trained >= patience:
            if verbose:
                print(f"  Early stop at epoch {epoch+1} (best={best_epoch+1}, loss={best_loss:.6f}, trained {epochs_trained} epochs)")
            break

        epoch += 1

    ctx.save_checkpoint(epoch)

    if best_ckpt_path is None:
        best_ckpt_path = ctx.save_checkpoint(best_epoch)

    if verbose:
        print(f"  AE z{cfg.encoder.latent_dim} best epoch {best_epoch+1}, loss {best_loss:.6f}")

    return best_ckpt_path


def train_diffae_early_stop(
    cfg: Config,
    patience: int = 50,
    min_epochs: int = 100,
    verbose: bool = True,
) -> str:
    """Train DiffAE with early stopping. Returns path to best checkpoint."""
    from diffusion.schedule import sinusoidal_embedding

    ctx = DiffAEContext.build(cfg, for_training=True, verbose=verbose)

    start_epoch = 0
    if cfg.resume:
        last = ctx.latest_checkpoint()
        if last is not None:
            try:
                start_epoch = ctx.load_checkpoint(last) + 1
                if verbose:
                    print(f"Loaded DiffAE checkpoint: {last} (epoch {start_epoch - 1})")
            except Exception:
                pass

        if start_epoch == 0:
            best_ckpt_cross = ctx.find_best_checkpoint()
            if best_ckpt_cross is not None:
                try:
                    _, is_full = ctx.load_checkpoint_partial(best_ckpt_cross, verbose=verbose)
                    if verbose:
                        tag = "full" if is_full else "partial"
                        print(f"Loaded {tag} weights from {best_ckpt_cross}")
                except Exception as e:
                    if verbose:
                        print(f"Partial load failed: {e}")

    for g in ctx.optim.param_groups:
        g["lr"] = cfg.training.lr

    B = cfg.training.batch_size
    use_regressive = cfg.encoder.use_regressive_head and ctx.regressive_decoder is not None
    if verbose:
        print(f"  regressive_head={use_regressive} (weight={cfg.encoder.regressive_head_weight}), "
              f"stochastic={cfg.encoder.use_stochastic}, encoder_type={cfg.encoder.encoder_type}")
    best_loss = float("inf")
    wait = 0
    best_epoch = start_epoch
    best_ckpt_path = None
    schedule = ctx.schedule
    max_epoch = cfg.training.epochs

    epoch = start_epoch
    while epoch < max_epoch:
        ctx.encoder.train()
        ctx.decoder.train()
        ctx.latent_proj.train()
        if use_regressive and ctx.regressive_decoder is not None:
            ctx.regressive_decoder.train()

        epoch_loss = 0.0
        epoch_reg = 0.0
        epoch_kl = 0.0
        n_steps = cfg.training.steps_per_epoch

        pbar = tqdm(
            range(n_steps),
            desc=f"DiffAE z{cfg.encoder.latent_dim} Epoch {epoch+1}",
            ncols=140, file=sys.stdout, disable=not verbose,
        )
        for step in pbar:
            batch_np, _ = ctx.loader.get_batch(B)
            batch_np = ctx.data_stats.normalize(batch_np)
            x0 = torch.from_numpy(batch_np.astype(np.float32)).to(ctx.device)
            x0_flat = x0.view(B * ctx.n_nodes, 1)

            z, mu, logvar = ctx.encoder(x0_flat, ctx.A_sparse, ctx.pos, batch_size=B)
            cond_base = ctx.latent_proj(z)

            t = torch.randint(0, cfg.diffusion.timesteps, (B,), device=ctx.device, dtype=torch.long)
            t_emb = sinusoidal_embedding(t, cfg.conditioning.time_dim)
            cond_full = torch.cat([cond_base, t_emb], dim=-1)

            sqrt_ab = schedule["sqrt_alphas_cumprod"][t].view(B, 1, 1)
            sqrt_om = schedule["sqrt_one_minus_alphas_cumprod"][t].view(B, 1, 1)
            snr_t = schedule["snr"][t].view(B)

            noise = torch.randn_like(x0)
            x_t = sqrt_ab * x0 + sqrt_om * noise
            x_t_flat = x_t.view(B * ctx.n_nodes, 1)

            pred_flat = ctx.decoder(x_t_flat, ctx.A_sparse, cond_full, ctx.pos, batch_size=B)
            pred = pred_flat.view(B, ctx.n_nodes, 1)

            if cfg.diffusion.parametrization == "v":
                target = sqrt_ab * noise - sqrt_om * x0
            else:
                target = noise

            mse_per_sample = F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2))
            if cfg.diffusion.p2_gamma > 0.0:
                weight = torch.pow(cfg.diffusion.p2_k + snr_t, -cfg.diffusion.p2_gamma)
                mse_per_sample = mse_per_sample * weight
            loss = mse_per_sample.mean()

            if use_regressive and ctx.regressive_decoder is not None:
                reg_flat = ctx.regressive_decoder(z, ctx.A_sparse, ctx.pos, batch_size=B)
                reg_pred = reg_flat.view(B, ctx.n_nodes, 1)
                reg_loss = F.mse_loss(reg_pred, x0)
                loss = loss + cfg.encoder.regressive_head_weight * reg_loss
                epoch_reg += reg_loss.item()

            if cfg.encoder.use_stochastic and mu is not None and logvar is not None:
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = loss + cfg.encoder.kl_weight * kl_loss
                epoch_kl += kl_loss.item()

            if torch.isnan(loss) or torch.isinf(loss):
                ctx.optim.zero_grad(set_to_none=True)
                continue

            epoch_loss += loss.item()
            ctx.optim.zero_grad(set_to_none=True)
            loss.backward()

            clip_params = (
                list(ctx.encoder.parameters())
                + list(ctx.decoder.parameters())
                + list(ctx.latent_proj.parameters())
            )
            if use_regressive and ctx.regressive_decoder is not None:
                clip_params += list(ctx.regressive_decoder.parameters())
            torch.nn.utils.clip_grad_norm_(clip_params, max_norm=cfg.training.grad_clip)
            ctx.optim.step()

            with torch.no_grad():
                for p_ema, p in zip(ctx.ema_encoder.parameters(), ctx.encoder.parameters()):
                    p_ema.data.mul_(cfg.training.ema_decay).add_(p.data, alpha=1.0 - cfg.training.ema_decay)
                for p_ema, p in zip(ctx.ema_decoder.parameters(), ctx.decoder.parameters()):
                    p_ema.data.mul_(cfg.training.ema_decay).add_(p.data, alpha=1.0 - cfg.training.ema_decay)
                for p_ema, p in zip(ctx.ema_latent_proj.parameters(), ctx.latent_proj.parameters()):
                    p_ema.data.mul_(cfg.training.ema_decay).add_(p.data, alpha=1.0 - cfg.training.ema_decay)
                if use_regressive and ctx.ema_regressive_decoder is not None:
                    for p_ema, p in zip(ctx.ema_regressive_decoder.parameters(), ctx.regressive_decoder.parameters()):
                        p_ema.data.mul_(cfg.training.ema_decay).add_(p.data, alpha=1.0 - cfg.training.ema_decay)

            postfix = {"loss": epoch_loss / (step + 1)}
            if use_regressive:
                postfix["reg"] = epoch_reg / (step + 1)
            if cfg.encoder.use_stochastic:
                postfix["kl"] = epoch_kl / (step + 1)
            pbar.set_postfix(**postfix)

        avg_loss = epoch_loss / n_steps

        if (epoch + 1) % cfg.training.checkpoint_every == 0:
            ctx.save_checkpoint(epoch)

        if avg_loss < best_loss:
            best_loss = avg_loss
            wait = 0
            best_epoch = epoch
            best_ckpt_path = ctx.save_checkpoint(epoch)
        else:
            wait += 1

        epochs_trained = epoch - start_epoch + 1
        if wait >= patience and epochs_trained >= patience:
            if verbose:
                print(f"  Early stop at epoch {epoch+1} (best={best_epoch+1}, loss={best_loss:.6f}, trained {epochs_trained} epochs)")
            break

        epoch += 1

    ctx.save_checkpoint(epoch)

    if best_ckpt_path is None:
        best_ckpt_path = ctx.save_checkpoint(best_epoch)

    if verbose:
        print(f"  DiffAE z{cfg.encoder.latent_dim} best epoch {best_epoch+1}, loss {best_loss:.6f}")

    return best_ckpt_path


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def encode_ae_dataset(cfg: Config, ckpt_path: str, n_samples: int = 100_000, verbose: bool = True) -> str:
    """Load trained AE and encode a dataset, returning path to h5 file."""
    ctx = AEContext.build(cfg, for_training=True, verbose=False)
    ctx.load_checkpoint(ckpt_path, load_optim=False)
    encoder = ctx.ema_encoder if ctx.ema_encoder is not None else ctx.encoder
    encoder.eval()

    out_path = os.path.join(ctx.checkpoint_dir, "ae_encoded_ms_latents.h5")
    ae_save_encoded(ctx, out_path, encoder=encoder, batch_size=128, n_samples=n_samples, verbose=verbose)
    return out_path


def encode_diffae_dataset(cfg: Config, ckpt_path: str, n_samples: int = 100_000, verbose: bool = True) -> str:
    """Load trained DiffAE and encode a dataset, returning path to h5 file."""
    ctx = DiffAEContext.build(cfg, for_training=True, verbose=False)
    ctx.load_checkpoint(ckpt_path, load_optim=False)
    encoder = ctx.ema_encoder if ctx.ema_encoder is not None else ctx.encoder
    encoder.eval()

    out_path = os.path.join(ctx.checkpoint_dir, "encoded_ms_latents.h5")
    diffae_save_encoded(ctx, out_path, encoder=encoder, batch_size=128, n_samples=n_samples, verbose=verbose)
    return out_path


# ---------------------------------------------------------------------------
# Aux evaluation on encoded latents
# ---------------------------------------------------------------------------

def run_aux_on_encoded(
    h5_path: str,
    device: torch.device,
    n_trials: int = 3,
    aux_epochs: int = 20,
    batch_size: int = 512,
    lr: float = 1e-3,
    seed_base: int = 0,
) -> Tuple[List[float], List[float]]:
    """Train aux MLP on pre-encoded latents. Returns (mae_list, rmse_list) over trials."""
    dataset = EncodedMSDataset(h5_path)
    latent_dim = dataset.latent_dim

    mae_list, rmse_list = [], []
    for trial in range(n_trials):
        seed = seed_base + trial
        n_total = len(dataset)
        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)
        n_test = n_total - n_train - n_val

        train_set, val_set, test_set = random_split(
            dataset, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(seed),
        )
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

        mlp = MLP(in_dim=latent_dim, hidden_dims=[128, 64], out_dim=1, dropout=0.1).to(device)
        train_aux_mlp_on_latents(mlp, train_loader, val_loader, device, epochs=aux_epochs, lr=lr)
        mae, rmse, _, _ = evaluate_aux_mlp_on_latents(mlp, test_loader, device)
        mae_list.append(mae)
        rmse_list.append(rmse)
        print(f"    trial {trial+1}/{n_trials}: MAE={mae:.2f}, RMSE={rmse:.2f}")

    return mae_list, rmse_list


def run_baseline_on_raw(
    cfg: Config,
    device: torch.device,
    n_events: int = 50_000,
    n_trials: int = 3,
    aux_epochs: int = 50,
    batch_size: int = 512,
    lr: float = 1e-3,
    seed_base: int = 1000,
) -> Tuple[List[float], List[float]]:
    """Train baseline MLP on raw flattened waveforms. Returns (mae_list, rmse_list)."""
    dataset = OnlineMSDataset(
        ss_h5_path=cfg.paths.tritium_h5,
        n_events=n_events,
        ms_config=cfg.ms_data,
        seed=42,
    )
    mae_list, rmse_list = [], []
    for trial in range(n_trials):
        seed = seed_base + trial
        n_total = len(dataset)
        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)
        n_test = n_total - n_train - n_val

        train_set, val_set, test_set = random_split(
            dataset, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(seed),
        )
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

        mlp = MLP(in_dim=dataset.input_dim, hidden_dims=[256, 128, 64], out_dim=1, dropout=0.1).to(device)
        train_baseline_mlp(mlp, train_loader, val_loader, device, epochs=aux_epochs, lr=lr)
        mae, rmse, _, _ = evaluate_baseline_mlp(mlp, test_loader, device)
        mae_list.append(mae)
        rmse_list.append(rmse)
        print(f"    baseline trial {trial+1}/{n_trials}: MAE={mae:.2f}, RMSE={rmse:.2f}")

    return mae_list, rmse_list


# ---------------------------------------------------------------------------
# Locate existing encoded latents or checkpoints
# ---------------------------------------------------------------------------

def find_ae_checkpoint(cfg: Config, latent_dim: int) -> Optional[str]:
    subdir = cfg.paths.ae_subdir.format(latent_dim=latent_dim)
    ckpt_dir = os.path.join(cfg.paths.checkpoint_dir, subdir)
    files = sorted(glob.glob(os.path.join(ckpt_dir, "ae_epoch_*.pt")))
    return files[-1] if files else None


def find_diffae_checkpoint(cfg: Config, latent_dim: int) -> Optional[str]:
    subdir = cfg.paths.diffae_subdir.format(latent_dim=latent_dim)
    ckpt_dir = os.path.join(cfg.paths.checkpoint_dir, subdir)
    files = sorted(glob.glob(os.path.join(ckpt_dir, "diffae_epoch_*.pt")))
    return files[-1] if files else None


def find_ae_latents(cfg: Config, latent_dim: int) -> Optional[str]:
    subdir = cfg.paths.ae_subdir.format(latent_dim=latent_dim)
    path = os.path.join(cfg.paths.checkpoint_dir, subdir, "ae_encoded_ms_latents.h5")
    return path if os.path.exists(path) else None


def find_diffae_latents(cfg: Config, latent_dim: int) -> Optional[str]:
    subdir = cfg.paths.diffae_subdir.format(latent_dim=latent_dim)
    path = os.path.join(cfg.paths.checkpoint_dir, subdir, "encoded_ms_latents.h5")
    return path if os.path.exists(path) else None


# ---------------------------------------------------------------------------
# Z-profile reconstruction examples
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_z_profiles(
    latent_dims: List[int],
    output_dir: str,
    n_examples: int = 4,
    seed: int = 42,
):
    """Generate a z-profile comparison figure for each latent size.

    For each latent_dim, loads the AE and DiffAE checkpoints, reconstructs
    a fixed set of reference events, and plots raw vs AE vs DiffAE z-profiles.
    """
    os.makedirs(output_dir, exist_ok=True)

    base_cfg = get_config(latent_dim=latent_dims[0])
    ref_ctx = AEContext.build(base_cfg, for_training=False, verbose=False)
    np.random.seed(seed)
    batch_np, _ = ref_ctx.loader.get_batch(n_examples)
    n_channels = ref_ctx.n_channels
    n_time = ref_ctx.n_time_points
    time_axis = np.arange(n_time)

    for ldim in latent_dims:
        cfg = get_config(latent_dim=ldim)
        cfg.conditioning.cond_proj_dim = max(cfg.conditioning.cond_proj_dim, ldim)

        ae_ckpt = find_ae_checkpoint(cfg, ldim)
        dae_ckpt = find_diffae_checkpoint(cfg, ldim)

        ae_rec_np = None
        dae_rec_np = None

        if ae_ckpt is not None:
            try:
                ctx = AEContext.build(cfg, for_training=True, verbose=False)
                ctx.load_checkpoint(ae_ckpt, load_optim=False)
                enc = ctx.ema_encoder if ctx.ema_encoder is not None else ctx.encoder
                dec = ctx.ema_decoder if ctx.ema_decoder is not None else ctx.decoder
                enc.eval(); dec.eval()
                wf_norm = ctx.data_stats.normalize(batch_np)
                x_ref = torch.from_numpy(wf_norm.astype(np.float32)).to(ctx.device)
                rec = reconstruct_ae(enc, dec, ctx.A_sparse, ctx.pos, x_ref)
                ae_rec_np = ctx.data_stats.denormalize(rec.cpu().numpy())
                ae_rec_np = np.clip(ae_rec_np, 0, None)[:, 0, :]
            except Exception as e:
                print(f"  [z-profile] AE z{ldim} failed: {e}")

        if dae_ckpt is not None:
            try:
                ctx = DiffAEContext.build(cfg, for_training=True, verbose=False)
                ctx.load_checkpoint(dae_ckpt, load_optim=False)
                enc = ctx.ema_encoder if ctx.ema_encoder is not None else ctx.encoder
                dec = ctx.ema_decoder if ctx.ema_decoder is not None else ctx.decoder
                lp = ctx.ema_latent_proj if ctx.ema_latent_proj is not None else ctx.latent_proj
                enc.eval(); dec.eval(); lp.eval()
                wf_norm = ctx.data_stats.normalize(batch_np)
                x_ref = torch.from_numpy(wf_norm.astype(np.float32)).to(ctx.device)
                rec = sample_diffae(
                    enc, dec, lp, ctx.schedule, ctx.A_sparse, ctx.pos,
                    cfg.conditioning.time_dim, x_ref,
                    parametrization=cfg.diffusion.parametrization,
                )
                dae_rec_np = ctx.data_stats.denormalize(rec.cpu().numpy())
                dae_rec_np = np.clip(dae_rec_np, 0, None)[:, 0, :]
            except Exception as e:
                print(f"  [z-profile] DiffAE z{ldim} failed: {e}")

        models = {}
        if ae_rec_np is not None:
            models["AE"] = ae_rec_np
        if dae_rec_np is not None:
            models["DiffAE"] = dae_rec_np
        if not models:
            continue

        n_cols = 1 + len(models)
        fig, axes = plt.subplots(n_examples, n_cols,
                                 figsize=(4 * n_cols, 2.5 * n_examples),
                                 squeeze=False)

        for row in range(n_examples):
            raw_flat = batch_np[row, :, 0]
            z_raw = wf_to_z_profile(raw_flat, n_channels, n_time)
            y_max = max(z_raw.max() * 1.15, 1)

            ax = axes[row, 0]
            ax.plot(time_axis, z_raw, color="black", linewidth=1.5)
            ax.fill_between(time_axis, z_raw, alpha=0.12, color="black")
            if row == 0:
                ax.set_title("Raw", fontweight="bold")
            ax.set_ylabel(f"#{row+1}", fontweight="bold", rotation=0, labelpad=25)
            ax.set_ylim(0, y_max)
            if row == n_examples - 1:
                ax.set_xlabel("Time bin")

            colors = {"AE": "#1f77b4", "DiffAE": "#ff7f0e"}
            for col_off, (mname, rec_data) in enumerate(models.items(), start=1):
                z_rec = wf_to_z_profile(rec_data[row], n_channels, n_time)
                ax = axes[row, col_off]
                ax.plot(time_axis, z_raw, color="black", linewidth=0.8, alpha=0.35, label="Raw")
                ax.plot(time_axis, z_rec, color=colors.get(mname, "C2"), linewidth=1.5, label=mname)
                ax.fill_between(time_axis, z_rec, alpha=0.15, color=colors.get(mname, "C2"))
                if row == 0:
                    ax.set_title(mname, fontweight="bold")
                ax.set_ylim(0, y_max)
                ax.legend(fontsize=7, loc="upper right")
                if row == n_examples - 1:
                    ax.set_xlabel("Time bin")

        fig.suptitle(f"Z-Profile Reconstructions  —  latent dim = {ldim}",
                     fontsize=13, fontweight="bold", y=1.01)
        fig.tight_layout()
        path = os.path.join(output_dir, f"z_profiles_z{ldim}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Paper-ready plot
# ---------------------------------------------------------------------------

def plot_comparison(
    latent_dims: List[int],
    ae_results: Dict[int, Tuple[float, float]],
    diffae_results: Dict[int, Tuple[float, float]],
    baseline_mean: float,
    baseline_std: float,
    output_path: str,
):
    """
    Produce a paper-ready line plot of MAE vs latent dimension.

    Args:
        latent_dims: sorted list of latent sizes tested
        ae_results: {latent_dim: (mae_mean, mae_std)}
        diffae_results: {latent_dim: (mae_mean, mae_std)}
        baseline_mean / baseline_std: from raw-waveform MLP
        output_path: where to save the figure
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # Baseline
    ax.axhline(baseline_mean, color="gray", linestyle=":", linewidth=1.8, label="Baseline (raw)", zorder=1)
    ax.axhspan(
        baseline_mean - baseline_std, baseline_mean + baseline_std,
        color="gray", alpha=0.15, zorder=0,
    )

    # AE line
    ae_dims = sorted([d for d in latent_dims if d in ae_results])
    if ae_dims:
        ae_x = [np.log2(d) for d in ae_dims]
        ae_mean = [ae_results[d][0] for d in ae_dims]
        ae_err = [ae_results[d][1] for d in ae_dims]
        ax.errorbar(
            ae_x, ae_mean, yerr=ae_err,
            fmt="o-", color="#1f77b4", capsize=4, capthick=1.5,
            linewidth=2, markersize=6, label="AE", zorder=3,
        )

    # DiffAE line
    dae_dims = sorted([d for d in latent_dims if d in diffae_results])
    if dae_dims:
        dae_x = [np.log2(d) for d in dae_dims]
        dae_mean = [diffae_results[d][0] for d in dae_dims]
        dae_err = [diffae_results[d][1] for d in dae_dims]
        ax.errorbar(
            dae_x, dae_mean, yerr=dae_err,
            fmt="s-", color="#ff7f0e", capsize=4, capthick=1.5,
            linewidth=2, markersize=6, label="DiffAE", zorder=3,
        )

    all_dims = sorted(set(ae_dims) | set(dae_dims))
    if all_dims:
        tick_pos = [np.log2(d) for d in all_dims]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels([str(d) for d in all_dims])

    ax.set_xlabel(r"Latent dimension")
    ax.set_ylabel(r"MAE (ns)")
    ax.legend(loc="best", frameon=True, edgecolor="0.8", fancybox=False)
    ax.grid(True, which="both", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"\nPlot saved to {output_path}")


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare AE vs DiffAE across latent sizes")
    parser.add_argument("--latent-dims", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128])
    parser.add_argument("--patience", type=int, default=100, help="Early-stopping patience (epochs)")
    parser.add_argument("--min-epochs", type=int, default=100, help="Minimum epochs before early stop")
    parser.add_argument("--n-encode", type=int, default=100_000, help="Samples to encode for aux task")
    parser.add_argument("--n-trials", type=int, default=3, help="Aux MLP trials per latent size")
    parser.add_argument("--aux-epochs", type=int, default=50, help="Aux MLP training epochs")
    parser.add_argument("--baseline-events", type=int, default=50_000)
    parser.add_argument("--baseline-epochs", type=int, default=50)
    parser.add_argument("--skip-training", action="store_true", help="Skip model training, use existing ckpts")
    parser.add_argument("--skip-encoding", action="store_true", help="Skip encoding, use existing h5")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline MLP")
    parser.add_argument("--skip-z-profiles", action="store_true", help="Skip z-profile reconstruction plots")
    parser.add_argument("--n-z-examples", type=int, default=4, help="Number of example events for z-profile plots")
    parser.add_argument("--output-dir", type=str, default="latent_comparison")
    parser.add_argument("--results-json", type=str, default=None, help="Path to pre-saved results JSON to skip everything and just plot")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    latent_dims = sorted(args.latent_dims)
    print(f"Latent sizes: {latent_dims}")

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "results.json")

    # If a pre-saved results JSON is given, just plot and exit
    if args.results_json and os.path.exists(args.results_json):
        print(f"Loading results from {args.results_json}")
        with open(args.results_json) as f:
            saved = json.load(f)
        ae_results = {int(k): tuple(v) for k, v in saved["ae"].items()}
        diffae_results = {int(k): tuple(v) for k, v in saved["diffae"].items()}
        baseline_mean = saved["baseline_mean"]
        baseline_std = saved["baseline_std"]
        plot_comparison(
            latent_dims, ae_results, diffae_results,
            baseline_mean, baseline_std,
            os.path.join(args.output_dir, "latent_comparison_mae.pdf"),
        )
        return

    ae_results: Dict[int, Tuple[float, float]] = {}
    diffae_results: Dict[int, Tuple[float, float]] = {}

    # ---------------------------------------------------------------
    # Phase 1: Train AE / DiffAE for each latent size
    # ---------------------------------------------------------------
    for ldim in latent_dims:
        print(f"\n{'='*60}")
        print(f" Latent dimension = {ldim}  (log2 = {np.log2(ldim):.1f})")
        print(f"{'='*60}")

        cfg = get_config(latent_dim=ldim)
        cfg.conditioning.cond_proj_dim = max(cfg.conditioning.cond_proj_dim, ldim)
        cfg.resume = True
        cfg.visualize = False

        # --- AE ---
        if not args.skip_training:
            existing = find_ae_checkpoint(cfg, ldim)
            if existing:
                print(f"  [AE z{ldim}] Existing checkpoint: {existing} — continuing training...")
            else:
                print(f"  [AE z{ldim}] No checkpoint found — training from scratch...")
            ae_ckpt = train_ae_early_stop(cfg, patience=args.patience, min_epochs=args.min_epochs)
        else:
            ae_ckpt = find_ae_checkpoint(cfg, ldim)
            if ae_ckpt:
                print(f"  [AE z{ldim}] Using existing checkpoint: {ae_ckpt}")
            else:
                print(f"  [AE z{ldim}] No checkpoint and --skip-training set; skipping")

        ae_h5 = find_ae_latents(cfg, ldim)
        if (ae_h5 is None or not args.skip_encoding) and ae_ckpt is not None:
            print(f"  [AE z{ldim}] Encoding {args.n_encode} samples...")
            ae_h5 = encode_ae_dataset(cfg, ae_ckpt, n_samples=args.n_encode)
        if ae_h5 is not None:
            print(f"  [AE z{ldim}] Latents: {ae_h5}")

        # --- DiffAE ---
        if not args.skip_training:
            existing = find_diffae_checkpoint(cfg, ldim)
            if existing:
                print(f"  [DiffAE z{ldim}] Existing checkpoint: {existing} — continuing training...")
            else:
                print(f"  [DiffAE z{ldim}] No checkpoint found — training from scratch...")
            dae_ckpt = train_diffae_early_stop(cfg, patience=args.patience, min_epochs=args.min_epochs)
        else:
            dae_ckpt = find_diffae_checkpoint(cfg, ldim)
            if dae_ckpt:
                print(f"  [DiffAE z{ldim}] Using existing checkpoint: {dae_ckpt}")
            else:
                print(f"  [DiffAE z{ldim}] No checkpoint and --skip-training set; skipping")

        dae_h5 = find_diffae_latents(cfg, ldim)
        if (dae_h5 is None or not args.skip_encoding) and dae_ckpt is not None:
            print(f"  [DiffAE z{ldim}] Encoding {args.n_encode} samples...")
            dae_h5 = encode_diffae_dataset(cfg, dae_ckpt, n_samples=args.n_encode)
        if dae_h5 is not None:
            print(f"  [DiffAE z{ldim}] Latents: {dae_h5}")

    # ---------------------------------------------------------------
    # Phase 2: Aux-task evaluation
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print(" Phase 2: Auxiliary-task evaluation")
    print(f"{'='*60}")

    for ldim in latent_dims:
        cfg = get_config(latent_dim=ldim)

        ae_h5 = find_ae_latents(cfg, ldim)
        if ae_h5 is not None:
            print(f"\n  [AE z{ldim}] Running {args.n_trials} aux trials...")
            mae_list, rmse_list = run_aux_on_encoded(
                ae_h5, device, n_trials=args.n_trials, aux_epochs=args.aux_epochs,
            )
            ae_results[ldim] = (float(np.mean(mae_list)), float(np.std(mae_list)))
            print(f"  [AE z{ldim}] MAE = {ae_results[ldim][0]:.2f} ± {ae_results[ldim][1]:.2f}")
        else:
            print(f"  [AE z{ldim}] No latent file found; skipping aux eval")

        dae_h5 = find_diffae_latents(cfg, ldim)
        if dae_h5 is not None:
            print(f"\n  [DiffAE z{ldim}] Running {args.n_trials} aux trials...")
            mae_list, rmse_list = run_aux_on_encoded(
                dae_h5, device, n_trials=args.n_trials, aux_epochs=args.aux_epochs,
            )
            diffae_results[ldim] = (float(np.mean(mae_list)), float(np.std(mae_list)))
            print(f"  [DiffAE z{ldim}] MAE = {diffae_results[ldim][0]:.2f} ± {diffae_results[ldim][1]:.2f}")
        else:
            print(f"  [DiffAE z{ldim}] No latent file found; skipping aux eval")

    # ---------------------------------------------------------------
    # Phase 3: Baseline
    # ---------------------------------------------------------------
    baseline_mean, baseline_std = 0.0, 0.0
    if not args.skip_baseline:
        print(f"\n{'='*60}")
        print(" Phase 3: Baseline MLP on raw waveforms")
        print(f"{'='*60}")
        cfg = default_config
        mae_list, rmse_list = run_baseline_on_raw(
            cfg, device,
            n_events=args.baseline_events,
            n_trials=args.n_trials,
            aux_epochs=args.baseline_epochs,
        )
        baseline_mean = float(np.mean(mae_list))
        baseline_std = float(np.std(mae_list))
        print(f"  Baseline MAE = {baseline_mean:.2f} ± {baseline_std:.2f}")

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------
    saved = {
        "ae": {str(k): list(v) for k, v in ae_results.items()},
        "diffae": {str(k): list(v) for k, v in diffae_results.items()},
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
    }
    with open(results_path, "w") as f:
        json.dump(saved, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ---------------------------------------------------------------
    # Phase 4: Plot
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print(" Phase 4: Plotting")
    print(f"{'='*60}")

    plot_comparison(
        latent_dims, ae_results, diffae_results,
        baseline_mean, baseline_std,
        os.path.join(args.output_dir, "latent_comparison_mae.pdf"),
    )

    # ---------------------------------------------------------------
    # Phase 5: Z-profile reconstruction examples
    # ---------------------------------------------------------------
    if not args.skip_z_profiles:
        print(f"\n{'='*60}")
        print(" Phase 5: Z-profile reconstruction plots")
        print(f"{'='*60}")
        generate_z_profiles(
            latent_dims,
            output_dir=os.path.join(args.output_dir, "z_profiles"),
            n_examples=args.n_z_examples,
        )

    # Summary table
    print(f"\n{'='*60}")
    print(" Summary")
    print(f"{'='*60}")
    print(f"  Baseline:  MAE = {baseline_mean:.2f} ± {baseline_std:.2f} ns")
    print(f"  {'z':>5s}  {'AE MAE':>14s}  {'DiffAE MAE':>14s}")
    print(f"  {'-'*5}  {'-'*14}  {'-'*14}")
    for d in latent_dims:
        ae_str = f"{ae_results[d][0]:.2f}±{ae_results[d][1]:.2f}" if d in ae_results else "—"
        dae_str = f"{diffae_results[d][0]:.2f}±{diffae_results[d][1]:.2f}" if d in diffae_results else "—"
        print(f"  {d:5d}  {ae_str:>14s}  {dae_str:>14s}")


if __name__ == "__main__":
    main()
