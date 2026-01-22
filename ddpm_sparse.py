"""
DDPM training on sparse 3D graphs for tritium detector simulation.
"""
import os
import sys
import glob
from typing import Optional, Tuple
from matplotlib import pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy

from data import Graph, visualize_event, visualize_event_z, SparseGraph
from lz_data_loader import TritiumSSDataLoader
from config import Config, default_config, print_config

from models.graph_unet import GraphDDPMUNet
from diffusion.schedule import build_cosine_schedule, sinusoidal_embedding
from utils.visualization import build_xy_adjacency_radius


@torch.no_grad()
def sample_core(
    core: nn.Module,
    schedule: dict,
    A_sparse: torch.Tensor,
    cond_proj: torch.Tensor,
    pos: torch.Tensor,
    time_dim: int,
    shape: Tuple[int, int, int],
    parametrization: str = 'v',
    pbar: bool = False
) -> torch.Tensor:
    """DDPM sampling loop for graph diffusion."""
    B, C, N = shape
    device = cond_proj.device
    x = torch.randn(shape, device=device)
    T = schedule['betas'].shape[0]
    
    for i in tqdm(reversed(range(T)), desc="Sampling", disable=not pbar, total=T, ncols=150):
        t_tensor = torch.full((B,), i, device=device, dtype=torch.long)
        betas_t = schedule['betas'][i].view(1, 1, 1)
        sqrt_one_minus_ab_t = schedule['sqrt_one_minus_alphas_cumprod'][i].view(1, 1, 1)
        alpha_bar_t = schedule['alphas_cumprod'][i].view(1, 1, 1)
        alpha_bar_prev_t = schedule['alphas_cumprod_prev'][i].view(1, 1, 1)

        eps_list = []
        for b in range(B):
            xb = x[b, 0].unsqueeze(1)
            t_emb = sinusoidal_embedding(t_tensor[b:b+1], time_dim).squeeze(0)
            cond_full_b = torch.cat([cond_proj[b], t_emb], dim=-1)
            pred_b = core(xb, A_sparse, cond_full_b, pos)
            eps_list.append(pred_b.t())
        pred = torch.stack(eps_list, dim=0)

        if parametrization == 'eps':
            eps_theta = pred
            x0_pred = (x - sqrt_one_minus_ab_t * eps_theta) / torch.clamp(torch.sqrt(alpha_bar_t), min=1e-8)
        elif parametrization == 'v':
            a = torch.clamp(torch.sqrt(alpha_bar_t), min=1e-8)
            b_coef = torch.clamp(torch.sqrt(1.0 - alpha_bar_t), min=1e-8)
            x0_pred = a * x - b_coef * pred
        else:
            raise ValueError("parametrization must be 'eps' or 'v'")

        coef1 = betas_t * torch.sqrt(torch.clamp(alpha_bar_prev_t, min=1e-12)) / torch.clamp(1.0 - alpha_bar_t, min=1e-12)
        coef2 = torch.clamp(1.0 - alpha_bar_prev_t, min=0.0) * torch.sqrt(torch.clamp(1.0 - betas_t, min=1e-12)) / torch.clamp(1.0 - alpha_bar_t, min=1e-12)
        mean = coef1 * x0_pred + coef2 * x
        x = mean
    
    return x


def standardize_batch(x: np.ndarray, mean: float = 0., std: float = 0.09) -> np.ndarray:
    """Standardize batch to zero mean and unit variance."""
    eps = 1e-8
    return (x - mean) / (std + eps)


def visualize_event_3d(G: SparseGraph, event: np.ndarray, ax=None, colorbar: bool = False):
    """Visualize event with z axis as time."""
    x = G.positions_xyz[:, 0].cpu().numpy()
    y = G.positions_xyz[:, 1].cpu().numpy()
    z_pos = G.positions_xyz[:, 2].cpu().numpy()

    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

    s = np.clip(event * 10.0, 0.1, 100)
    mask = event >= 1.0
    
    if mask.sum() > 0:
        scatter = ax.scatter(
            x[mask], y[mask], z_pos[mask], 
            c=event[mask], s=s[mask], 
            cmap='viridis', alpha=0.5
        )
        if colorbar:
            plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (time)')
    return ax


def train(cfg: Config = default_config):
    """Main training function using configuration."""
    print_config(cfg)
    
    device_t = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"\nUsing device: {device_t}")

    # Load data
    tr = TritiumSSDataLoader(cfg.paths.tritium_h5, cfg.paths.channel_positions)
    channel_positions = tr.channel_positions
    graph = tr.load_adjacency_sparse(
        z_sep=cfg.graph.z_sep, 
        radius=cfg.graph.radius, 
        z_hops=cfg.graph.z_hops
    )
    A_sparse = graph.adjacency.to(device_t)
    pos = graph.positions_xyz.to(device_t)
    n_channels = tr.n_channels
    n_time_points = tr.n_time_points
    n_nodes = n_channels * n_time_points

    print(f"Graph: {n_nodes} nodes, {A_sparse._nnz()} edges")

    # Build schedule
    schedule = build_cosine_schedule(cfg.diffusion.timesteps, device_t)

    # Build models
    cond_proj = nn.Sequential(
        nn.Linear(cfg.conditioning.cond_in_dim, 64), 
        nn.SiLU(), 
        nn.Linear(64, cfg.conditioning.cond_proj_dim)
    ).to(device_t)
    
    core = GraphDDPMUNet(
        in_dim=cfg.model.in_dim,
        cond_dim=cfg.conditioning.cond_proj_dim + cfg.conditioning.time_dim,
        hidden_dim=cfg.model.hidden_dim,
        depth=cfg.model.depth,
        blocks_per_stage=cfg.model.blocks_per_stage,
        pool_ratio=cfg.model.pool_ratio,
        out_dim=cfg.model.out_dim,
        dropout=cfg.model.dropout,
        pos_dim=cfg.model.pos_dim,
    ).to(device_t)

    # Optimizer
    optim = torch.optim.AdamW(
        list(core.parameters()) + list(cond_proj.parameters()),
        lr=cfg.training.lr,
        betas=(0.9, 0.999),
        weight_decay=cfg.training.weight_decay,
    )

    os.makedirs(cfg.paths.checkpoint_dir, exist_ok=True)

    # Checkpoint functions
    def save_checkpoint(epoch_idx: int):
        state = {
            "core": core.state_dict(),
            "cond_proj": cond_proj.state_dict(),
            "optim": optim.state_dict(),
            "epoch": epoch_idx,
        }
        path = os.path.join(cfg.paths.checkpoint_dir, f"ckpt_epoch_{epoch_idx:04d}.pt")
        torch.save(state, path)
        return path

    def latest_checkpoint() -> Optional[str]:
        files = sorted(glob.glob(os.path.join(cfg.paths.checkpoint_dir, "ckpt_epoch_*.pt")))
        return files[-1] if files else None

    def load_checkpoint(path: str) -> int:
        chk = torch.load(path, map_location=device_t)
        core.load_state_dict(chk["core"])
        cond_proj.load_state_dict(chk["cond_proj"])
        optim.load_state_dict(chk["optim"])
        return int(chk.get("epoch", 0))

    # Resume
    start_epoch = 0
    if cfg.resume:
        last = latest_checkpoint()
        if last is not None:
            try:
                start_epoch = load_checkpoint(last) + 1
                print(f"Resumed from epoch {start_epoch}")
            except Exception as e:
                print(f"Could not resume: {e}")
                start_epoch = 0

    # EMA model
    ema_core = deepcopy(core).to(device_t)
    for g in optim.param_groups:
        g["lr"] = cfg.training.lr

    # Training loop
    for epoch in range(start_epoch, cfg.training.epochs):
        core.train()
        epoch_loss = 0.0
        pbar = tqdm(range(cfg.training.steps_per_epoch), desc=f"Epoch {epoch+1}/{cfg.training.epochs}", ncols=120, file=sys.stdout)
        
        for step in pbar:
            batch_np, batch_cond = tr.get_batch(cfg.training.batch_size)
            batch_cond = torch.from_numpy(batch_cond.astype(np.float32)).to(device_t)
            batch_np = standardize_batch(batch_np)
            x0 = torch.from_numpy(batch_np.astype(np.float32)).to(device_t).reshape(cfg.training.batch_size, -1, 1)
            cond_base = cond_proj(batch_cond)
            t = torch.randint(0, cfg.diffusion.timesteps, (cfg.training.batch_size,), device=device_t, dtype=torch.long)

            loss_total = 0.0
            for b in range(cfg.training.batch_size):
                tb = t[b]
                sqrt_ab = schedule['sqrt_alphas_cumprod'][tb]
                sqrt_om = schedule['sqrt_one_minus_alphas_cumprod'][tb]
                snr_tb = schedule['snr'][tb]

                noise = torch.randn_like(x0[b])
                x_t = sqrt_ab * x0[b] + sqrt_om * noise

                t_emb = sinusoidal_embedding(tb.view(1), cfg.conditioning.time_dim).squeeze(0)
                cond_full = torch.cat([cond_base[b], t_emb], dim=-1)
                pred = core(x_t, A_sparse, cond_full, pos)

                if cfg.diffusion.parametrization == "eps":
                    target = noise
                elif cfg.diffusion.parametrization == "v":
                    target = sqrt_ab * noise - sqrt_om * x0[b]
                else:
                    raise ValueError("parametrization must be 'eps' or 'v'")

                mse = F.mse_loss(pred, target)
                if cfg.diffusion.p2_gamma > 0.0:
                    weight = torch.pow(cfg.diffusion.p2_k + snr_tb, -cfg.diffusion.p2_gamma)
                    mse = mse * weight

                loss_total = loss_total + mse

            loss = loss_total / float(cfg.training.batch_size)
            epoch_loss += float(loss.item())

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(core.parameters()) + list(cond_proj.parameters()), 
                max_norm=cfg.training.grad_clip
            )
            optim.step()

            # EMA update
            with torch.no_grad():
                for p_ema, p in zip(ema_core.parameters(), core.parameters()):
                    p_ema.data.mul_(cfg.training.ema_decay).add_(p.data, alpha=1.0 - cfg.training.ema_decay)

            pbar.set_postfix(loss=epoch_loss / (step + 1))

        # Checkpointing
        if (epoch + 1) % cfg.training.checkpoint_every == 0:
            save_checkpoint(epoch)

        # Visualization
        if cfg.visualize and (epoch % cfg.training.visualize_every == 0 or epoch == cfg.training.epochs - 1):
            core.eval()
            with torch.no_grad():
                b_vis = min(cfg.training.batch_size, 4)
                batch_np, batch_cond = tr.get_batch(b_vis)
                batch_cond = torch.from_numpy(batch_cond.astype(np.float32)).to(device_t)
                batch_np = standardize_batch(batch_np)
                cond_vis = cond_proj(batch_cond)
                samples = sample_core(
                    core=ema_core,
                    schedule=schedule,
                    A_sparse=A_sparse,
                    cond_proj=cond_vis,
                    time_dim=cfg.conditioning.time_dim,
                    shape=(b_vis, 1, n_nodes),
                    parametrization=cfg.diffusion.parametrization,
                    pos=pos,
                ).clamp_min_(0.0)

            plots_dir = f"{cfg.paths.plot_dir}/epoch_{epoch}"
            os.makedirs(plots_dir, exist_ok=True)
            
            for idx in range(samples.shape[0]):
                rec_int = samples[idx, 0].detach().cpu().numpy()
                true_int = batch_np[idx, :, 0]

                rec_xy = rec_int.reshape(n_channels, n_time_points, order='F').sum(axis=1)
                true_xy = true_int.reshape(n_channels, n_time_points, order='F').sum(axis=1)

                rec_z = rec_int.reshape(n_channels, n_time_points, order='F')
                true_z = true_int.reshape(n_channels, n_time_points, order='F')

                adj2d = build_xy_adjacency_radius(channel_positions, radius=cfg.graph.radius)
                Gxy = Graph(adjacency=adj2d, positions_xy=channel_positions, positions_z=np.zeros(n_channels, dtype=np.float32))
                
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                visualize_event(Gxy, true_xy, None, ax=axes[0])
                axes[0].set_title("Ground truth")
                visualize_event(Gxy, rec_xy, None, ax=axes[1])
                axes[1].set_title("DDPM sample")
                plt.tight_layout()
                fig.savefig(f"{plots_dir}/event_{idx}_xy.png")
                plt.close(fig)

                fig, axes = plt.subplots(1, 2, figsize=(10, 3))
                visualize_event_z(Graph(adjacency=None, positions_xy=channel_positions, positions_z=np.concatenate([range(n_time_points) for i in range(n_channels)])), true_z, None, ax=axes[0])
                axes[0].set_title("Ground truth")
                visualize_event_z(Graph(adjacency=None, positions_xy=channel_positions, positions_z=np.concatenate([range(n_time_points) for i in range(n_channels)])), rec_z, None, ax=axes[1])
                axes[1].set_title("DDPM sample")
                plt.tight_layout()
                fig.savefig(f"{plots_dir}/event_{idx}_z.png")
                plt.close(fig)

                # 3D scatter plot
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection="3d")
                visualize_event_3d(graph, true_int, ax=ax, colorbar=True)
                fig.savefig(f"{plots_dir}/event_{idx}_true_3d.png", dpi=300)
                plt.close(fig)

                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection="3d")
                visualize_event_3d(graph, rec_int, ax=ax, colorbar=True)
                fig.savefig(f"{plots_dir}/event_{idx}_rec_3d.png", dpi=300)
                plt.close(fig)


if __name__ == "__main__":
    train(default_config)
