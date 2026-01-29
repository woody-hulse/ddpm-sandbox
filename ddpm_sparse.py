"""
DDPM training on sparse 3D graphs for tritium detector simulation.
"""
import os
import sys
import glob
from dataclasses import dataclass, field
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


@dataclass
class ModelContext:
    """Holds all model components and data for training/inference."""
    cfg: Config
    device: torch.device
    loader: TritiumSSDataLoader
    graph: SparseGraph
    A_sparse: torch.Tensor
    pos: torch.Tensor
    n_channels: int
    n_time_points: int
    n_nodes: int
    data_stats: 'DataStats'
    schedule: dict
    core: nn.Module
    cond_proj: nn.Module
    ema_core: Optional[nn.Module] = None
    optim: Optional[torch.optim.Optimizer] = None

    @classmethod
    def build(cls, cfg: Config, for_training: bool = True, verbose: bool = True) -> 'ModelContext':
        """Build model context from configuration."""
        device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if verbose:
            print(f"Using device: {device}")

        loader = TritiumSSDataLoader(cfg.paths.tritium_h5, cfg.paths.channel_positions)
        graph = loader.load_adjacency_sparse(
            z_sep=cfg.graph.z_sep,
            radius=cfg.graph.radius,
            z_hops=cfg.graph.z_hops
        )
        A_sparse = graph.adjacency.to(device)
        pos = graph.positions_xyz.to(device)
        n_channels = loader.n_channels
        n_time_points = loader.n_time_points
        n_nodes = n_channels * n_time_points

        if verbose:
            print(f"Graph: {n_nodes} nodes, {A_sparse._nnz()} edges")
            print("Computing data statistics...")

        data_stats = DataStats.from_loader(loader, n_samples=1000, batch_size=32)
        if verbose:
            print(f"Data mean: {data_stats.mean:.4f}, std: {data_stats.std:.4f}")

        schedule = build_cosine_schedule(cfg.diffusion.timesteps, device)

        cond_proj = nn.Sequential(
            nn.Linear(cfg.conditioning.cond_in_dim, 128),
            nn.SiLU(),
            nn.Linear(128, cfg.conditioning.cond_proj_dim)
        ).to(device)

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
        ).to(device)

        ema_core = None
        optim = None
        if for_training:
            ema_core = deepcopy(core).to(device)
            optim = torch.optim.AdamW(
                list(core.parameters()) + list(cond_proj.parameters()),
                lr=cfg.training.lr,
                betas=(0.9, 0.999),
                weight_decay=cfg.training.weight_decay,
            )
            os.makedirs(cfg.paths.checkpoint_dir, exist_ok=True)

        if verbose:
            n_params = sum(p.numel() for p in core.parameters() if p.requires_grad)
            n_params += sum(p.numel() for p in cond_proj.parameters() if p.requires_grad)
            print(f"Total trainable parameters: {n_params:,}")

        return cls(
            cfg=cfg,
            device=device,
            loader=loader,
            graph=graph,
            A_sparse=A_sparse,
            pos=pos,
            n_channels=n_channels,
            n_time_points=n_time_points,
            n_nodes=n_nodes,
            data_stats=data_stats,
            schedule=schedule,
            core=core,
            cond_proj=cond_proj,
            ema_core=ema_core,
            optim=optim,
        )

    def latest_checkpoint(self) -> Optional[str]:
        files = sorted(glob.glob(os.path.join(self.cfg.paths.checkpoint_dir, "ckpt_epoch_*.pt")))
        return files[-1] if files else None

    def save_checkpoint(self, epoch: int) -> str:
        state = {
            "core": self.core.state_dict(),
            "ema_core": self.ema_core.state_dict() if self.ema_core else self.core.state_dict(),
            "cond_proj": self.cond_proj.state_dict(),
            "optim": self.optim.state_dict() if self.optim else None,
            "epoch": epoch,
            "data_stats": {"mean": self.data_stats.mean, "std": self.data_stats.std},
        }
        path = os.path.join(self.cfg.paths.checkpoint_dir, f"ckpt_epoch_{epoch:04d}.pt")
        torch.save(state, path)
        return path

    def load_checkpoint(self, path: str, load_optim: bool = True) -> int:
        chk = torch.load(path, map_location=self.device)
        self.core.load_state_dict(chk["core"])
        if self.ema_core is not None:
            if "ema_core" in chk:
                self.ema_core.load_state_dict(chk["ema_core"])
            else:
                self.ema_core.load_state_dict(chk["core"])
        self.cond_proj.load_state_dict(chk["cond_proj"])
        if load_optim and self.optim is not None and chk.get("optim"):
            self.optim.load_state_dict(chk["optim"])
        if "data_stats" in chk:
            self.data_stats.mean = chk["data_stats"]["mean"]
            self.data_stats.std = chk["data_stats"]["std"]
        return int(chk.get("epoch", 0))


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
    """DDPM sampling loop for graph diffusion with batched processing."""
    B, C, N = shape
    device = cond_proj.device
    x = torch.randn((B, N, C), device=device)  # (B, N, C)
    T = schedule['betas'].shape[0]
    
    for i in tqdm(reversed(range(T)), desc="Sampling", disable=not pbar, total=T, ncols=150):
        betas_t = schedule['betas'][i]
        sqrt_one_minus_ab_t = schedule['sqrt_one_minus_alphas_cumprod'][i]
        alpha_bar_t = schedule['alphas_cumprod'][i]
        alpha_bar_prev_t = schedule['alphas_cumprod_prev'][i]
        
        t_emb = sinusoidal_embedding(torch.tensor([i], device=device), time_dim)  # (1, time_dim)
        t_emb_batch = t_emb.expand(B, -1)  # (B, time_dim)
        cond_full = torch.cat([cond_proj, t_emb_batch], dim=-1)  # (B, cond_proj_dim + time_dim)
        
        x_flat = x.view(B * N, C)  # (B*N, C)
        pred_flat = core(x_flat, A_sparse, cond_full, pos, batch_size=B)  # (B*N, C)
        pred = pred_flat.view(B, N, C)  # (B, N, C)

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
        
        if i > 0:
            posterior_var = schedule['posterior_variance'][i]
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(posterior_var) * noise
        else:
            x = mean
    
    out = x.permute(0, 2, 1)  # (B, C, N)

    # Perform 1d Gaussian convolution over time axis
    kernel = torch.tensor([0.25, 0.5, 0.25], device=device, dtype=torch.float32).view(1, 1, 3)
    B, C, N = out.shape
    out = F.conv1d(
        out.reshape(B * C, 1, N), weight=kernel, padding=1, stride=1, groups=1
    ).reshape(B, C, N)

    return out


class DataStats:
    """Compute and store data statistics for normalization."""
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std
    
    @classmethod
    def from_loader(cls, loader, n_samples: int = 1000, batch_size: int = 32) -> 'DataStats':
        """Compute statistics from data loader samples."""
        all_data = []
        samples_collected = 0
        while samples_collected < n_samples:
            batch_np, _ = loader.get_batch(min(batch_size, n_samples - samples_collected))
            all_data.append(batch_np.flatten())
            samples_collected += batch_np.shape[0]
        
        all_data = np.concatenate(all_data)
        return cls(mean=float(np.mean(all_data)), std=float(np.std(all_data)))
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std + 1e-8)
    
    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


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
    ax.set_zlabel('t')
    ax.set_box_aspect([1, 1, 3])
    return ax


def train(cfg: Config = default_config):
    """Main training function using configuration."""
    print_config(cfg)
    
    ctx = ModelContext.build(cfg, for_training=True, verbose=True)
    
    # Aliases for readability
    device_t = ctx.device
    core = ctx.core
    cond_proj = ctx.cond_proj
    ema_core = ctx.ema_core
    optim = ctx.optim
    schedule = ctx.schedule
    data_stats = ctx.data_stats
    A_sparse = ctx.A_sparse
    pos = ctx.pos
    n_nodes = ctx.n_nodes
    n_channels = ctx.n_channels
    n_time_points = ctx.n_time_points
    graph = ctx.graph
    tr = ctx.loader
    channel_positions = tr.channel_positions
    
    # Verify requires_grad is set
    n_grad = sum(1 for p in core.parameters() if p.requires_grad)
    n_total = sum(1 for p in core.parameters())
    print(f"Parameters with requires_grad: {n_grad}/{n_total}")

    # Resume
    start_epoch = 0
    if cfg.resume:
        last = ctx.latest_checkpoint()
        if last is not None:
            try:
                start_epoch = ctx.load_checkpoint(last) + 1
                print(f"Resumed from epoch {start_epoch}")
            except Exception as e:
                print(f"Could not resume: {e}")
                start_epoch = 0

    for g in optim.param_groups:
        g["lr"] = cfg.training.lr

    # Quick gradient test
    print("\nRunning quick gradient test...")
    core.train()
    test_B = 2
    test_x = torch.randn(test_B * n_nodes, 1, device=device_t)
    test_cond = torch.randn(test_B, cfg.conditioning.cond_proj_dim + cfg.conditioning.time_dim, device=device_t)
    test_out = core(test_x, A_sparse, test_cond, pos, batch_size=test_B)
    test_loss = test_out.sum()
    test_loss.backward()
    test_grad_norm = sum(p.grad.norm().item() for p in core.parameters() if p.grad is not None)
    print(f"Quick test gradient norm: {test_grad_norm:.4f}")
    if test_grad_norm == 0:
        print("*** ERROR: No gradients in quick test! Model is broken. ***")
    else:
        print("Quick test passed - gradients are flowing\n")
    optim.zero_grad()

    B = cfg.training.batch_size

    # Training loop
    for epoch in range(start_epoch, cfg.training.epochs):
        core.train()
        epoch_loss = 0.0
        loss_by_t_range = {'low': 0.0, 'mid': 0.0, 'high': 0.0, 'count_low': 0, 'count_mid': 0, 'count_high': 0}
        pbar = tqdm(range(cfg.training.steps_per_epoch), desc=f"Epoch {epoch+1}/{cfg.training.epochs}", ncols=120, file=sys.stdout)
        
        for step in pbar:
            batch_np, batch_cond = tr.get_batch(B)
            batch_cond = torch.from_numpy(batch_cond.astype(np.float32)).to(device_t)
            batch_np = data_stats.normalize(batch_np)
            
            x0 = torch.from_numpy(batch_np.astype(np.float32)).to(device_t)  # (B, N, 1)
            x0_flat = x0.view(B * n_nodes, 1)  # (B*N, 1)
            
            cond_base = cond_proj(batch_cond)  # (B, cond_proj_dim)
            
            t = torch.randint(0, cfg.diffusion.timesteps, (B,), device=device_t, dtype=torch.long)
            t_emb = sinusoidal_embedding(t, cfg.conditioning.time_dim)  # (B, time_dim)
            cond_full = torch.cat([cond_base, t_emb], dim=-1)  # (B, cond_proj_dim + time_dim)

            sqrt_ab = schedule['sqrt_alphas_cumprod'][t].view(B, 1, 1)
            sqrt_om = schedule['sqrt_one_minus_alphas_cumprod'][t].view(B, 1, 1)
            snr_t = schedule['snr'][t].view(B)

            noise = torch.randn_like(x0)  # (B, N, 1)
            x_t = sqrt_ab * x0 + sqrt_om * noise  # (B, N, 1)
            x_t_flat = x_t.view(B * n_nodes, 1)  # (B*N, 1)

            pred_flat = core(x_t_flat, A_sparse, cond_full, pos, batch_size=B)  # (B*N, 1)
            pred = pred_flat.view(B, n_nodes, 1)  # (B, N, 1)
            
            # Monitor prediction scale (every 10 epochs, first step)
            if step == 0 and epoch % 10 == 0:
                # Check input stats
                print(f"  Input stats: x_t mean={x_t.mean().item():.4f}, std={x_t.std().item():.4f}")
                print(f"  x0 stats: mean={x0.mean().item():.4f}, std={x0.std().item():.4f}")
                pred_mean = pred.mean().item()
                pred_std = pred.std().item()
                target_std = (sqrt_ab * noise - sqrt_om * x0).std().item() if cfg.diffusion.parametrization == "v" else noise.std().item()
                
                # Check if output layer weights are changing
                out_w_norm = core.out_proj.weight.norm().item()
                in_w_norm = core.in_proj.weight.norm().item()
                print(f"  Pred stats: mean={pred_mean:.4f}, std={pred_std:.4f}, target_std={target_std:.4f}")
                print(f"  Weight norms: out_proj={out_w_norm:.6f}, in_proj={in_w_norm:.4f}")
                print(f"  pred.requires_grad={pred.requires_grad}")

            if cfg.diffusion.parametrization == "eps":
                target = noise
            elif cfg.diffusion.parametrization == "v":
                target = sqrt_ab * noise - sqrt_om * x0
            else:
                raise ValueError("parametrization must be 'eps' or 'v'")

            mse_per_sample = F.mse_loss(pred, target, reduction='none').mean(dim=(1, 2))  # (B,)
            
            if cfg.diffusion.p2_gamma > 0.0:
                weight = torch.pow(cfg.diffusion.p2_k + snr_t, -cfg.diffusion.p2_gamma)
                mse_per_sample = mse_per_sample * weight

            loss = mse_per_sample.mean()
            
            # NaN detection
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  WARNING: NaN/Inf loss detected at step {step}! Skipping...")
                print(f"    pred stats: min={pred.min().item():.4f}, max={pred.max().item():.4f}")
                print(f"    target stats: min={target.min().item():.4f}, max={target.max().item():.4f}")
                optim.zero_grad(set_to_none=True)
                continue
            
            epoch_loss += float(loss.item())
            
            # Track loss by timestep range
            T_max = cfg.diffusion.timesteps
            for bi in range(B):
                ti = t[bi].item()
                li = mse_per_sample[bi].item()
                if ti < T_max // 3:
                    loss_by_t_range['low'] += li
                    loss_by_t_range['count_low'] += 1
                elif ti < 2 * T_max // 3:
                    loss_by_t_range['mid'] += li
                    loss_by_t_range['count_mid'] += 1
                else:
                    loss_by_t_range['high'] += li
                    loss_by_t_range['count_high'] += 1

            optim.zero_grad(set_to_none=True)
            loss.backward()
            
            # Debug: Check if gradients exist
            if step == 0 and epoch % 10 == 0:
                out_grad = core.out_proj.weight.grad
                in_grad = core.in_proj.weight.grad
                out_grad_norm = out_grad.norm().item() if out_grad is not None else 0.0
                in_grad_norm = in_grad.norm().item() if in_grad is not None else 0.0
                
                # Check layer gradients
                conv_grad_norm = 0.0
                for stage in core.enc_stages:
                    for blk in stage.blocks:
                        if blk.lin1.weight.grad is not None:
                            conv_grad_norm += blk.lin1.weight.grad.norm().item()
                        if blk.lin2.weight.grad is not None:
                            conv_grad_norm += blk.lin2.weight.grad.norm().item()
                
                # Check total gradient norm
                total_grad_norm = sum(p.grad.norm().item() for p in core.parameters() if p.grad is not None)
                n_with_grad = sum(1 for p in core.parameters() if p.grad is not None)
                
                print(f"  Gradient norms: out_proj={out_grad_norm:.6f}, in_proj={in_grad_norm:.6f}, conv_sum={conv_grad_norm:.6f}")
                print(f"  Total grad norm={total_grad_norm:.6f}, params with grad={n_with_grad}")
            
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

        # Print loss by timestep range
        if epoch % 10 == 0:
            avg_low = loss_by_t_range['low'] / max(loss_by_t_range['count_low'], 1)
            avg_mid = loss_by_t_range['mid'] / max(loss_by_t_range['count_mid'], 1)
            avg_high = loss_by_t_range['high'] / max(loss_by_t_range['count_high'], 1)
            print(f"  Loss by t: low(0-{cfg.diffusion.timesteps//3})={avg_low:.4f}, "
                  f"mid={avg_mid:.4f}, high({2*cfg.diffusion.timesteps//3}-{cfg.diffusion.timesteps})={avg_high:.4f}")

        # Quick validation: check model prediction at low noise level
        if epoch % cfg.training.visualize_every == 0:
            core.eval()
            with torch.no_grad():
                val_np, val_cond = tr.get_batch(4)
                val_cond_t = torch.from_numpy(val_cond.astype(np.float32)).to(device_t)
                val_np_norm = data_stats.normalize(val_np)
                x0_val = torch.from_numpy(val_np_norm.astype(np.float32)).to(device_t)
                
                t_low = torch.full((4,), 10, device=device_t, dtype=torch.long)
                sqrt_ab_low = schedule['sqrt_alphas_cumprod'][t_low].view(4, 1, 1)
                sqrt_om_low = schedule['sqrt_one_minus_alphas_cumprod'][t_low].view(4, 1, 1)
                
                noise_val = torch.randn_like(x0_val)
                x_t_val = sqrt_ab_low * x0_val + sqrt_om_low * noise_val
                
                cond_val = cond_proj(val_cond_t)
                t_emb_val = sinusoidal_embedding(t_low, cfg.conditioning.time_dim)
                cond_full_val = torch.cat([cond_val, t_emb_val], dim=-1)
                
                pred_val = core(x_t_val.view(4 * n_nodes, 1), A_sparse, cond_full_val, pos, batch_size=4)
                pred_val = pred_val.view(4, n_nodes, 1)
                
                if cfg.diffusion.parametrization == "v":
                    x0_pred_val = sqrt_ab_low * x_t_val - sqrt_om_low * pred_val
                else:
                    x0_pred_val = (x_t_val - sqrt_om_low * pred_val) / sqrt_ab_low
                
                recon_mse = F.mse_loss(x0_pred_val, x0_val).item()
                
                # Also check prediction variance vs target variance
                pred_std = pred_val.std().item()
                if cfg.diffusion.parametrization == "v":
                    target_val = sqrt_ab_low * noise_val - sqrt_om_low * x0_val
                else:
                    target_val = noise_val
                target_std = target_val.std().item()
                
                print(f"  Validation (t=10): x0 recon MSE={recon_mse:.6f}, pred_std={pred_std:.4f}, target_std={target_std:.4f}")
            core.train()

        # Checkpointing
        if (epoch + 1) % cfg.training.checkpoint_every == 0:
            ctx.save_checkpoint(epoch)

        # Visualization
        if cfg.visualize and (epoch % cfg.training.visualize_every == 0 or epoch == cfg.training.epochs - 1):
            ema_core.eval()
            with torch.no_grad():
                b_vis = min(cfg.training.batch_size, 4)
                batch_np, batch_cond = tr.get_batch(b_vis)
                batch_cond_t = torch.from_numpy(batch_cond.astype(np.float32)).to(device_t)
                cond_vis = cond_proj(batch_cond_t)
                samples = sample_core(
                    core=ema_core,
                    schedule=schedule,
                    A_sparse=A_sparse,
                    cond_proj=cond_vis,
                    time_dim=cfg.conditioning.time_dim,
                    shape=(b_vis, 1, n_nodes),
                    parametrization=cfg.diffusion.parametrization,
                    pos=pos,
                )
                samples_denorm = data_stats.denormalize(samples.cpu().numpy())
                samples_denorm = np.clip(samples_denorm, 0, None)
                
                # Diagnostic: compare statistics
                true_data = batch_np[:, :, 0]  # (B, N)
                gen_data = samples_denorm[:, 0, :]  # (B, N)
                print(f"\n  [Vis] True data - mean: {true_data.mean():.4f}, std: {true_data.std():.4f}, "
                      f"min: {true_data.min():.4f}, max: {true_data.max():.4f}")
                print(f"  [Vis] Gen data  - mean: {gen_data.mean():.4f}, std: {gen_data.std():.4f}, "
                      f"min: {gen_data.min():.4f}, max: {gen_data.max():.4f}")

            plots_dir = f"{cfg.paths.plot_dir}/epoch_{epoch}"
            os.makedirs(plots_dir, exist_ok=True)
            
            for idx in range(samples.shape[0]):
                rec_int = samples_denorm[idx, 0]
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


def generate_samples(cfg: Config):
    """Generate samples from a trained model."""
    ctx = ModelContext.build(cfg, for_training=False, verbose=True)
    
    # Aliases for readability
    core = ctx.core
    cond_proj = ctx.cond_proj
    schedule = ctx.schedule
    data_stats = ctx.data_stats
    A_sparse = ctx.A_sparse
    pos = ctx.pos
    n_nodes = ctx.n_nodes
    n_channels = ctx.n_channels
    n_time_points = ctx.n_time_points
    graph = ctx.graph
    tr = ctx.loader
    channel_positions = tr.channel_positions
    
    # Load checkpoint
    latest_ckpt = ctx.latest_checkpoint()
    if latest_ckpt is None:
        raise FileNotFoundError(f"No checkpoints found in {cfg.paths.checkpoint_dir}")
    print(f"Loading checkpoint: {latest_ckpt}")
    ctx.load_checkpoint(latest_ckpt, load_optim=False)
    core.eval()

    with torch.no_grad():
        b_vis = min(cfg.training.batch_size, 4)
        batch_np, batch_cond = tr.get_batch(b_vis)
        batch_cond_t = torch.from_numpy(batch_cond.astype(np.float32)).to(ctx.device)
        cond_vis = cond_proj(batch_cond_t)
        samples = sample_core(
            core=core,
            schedule=schedule,
            A_sparse=A_sparse,
            cond_proj=cond_vis,
            time_dim=cfg.conditioning.time_dim,
            shape=(b_vis, 1, n_nodes),
            parametrization=cfg.diffusion.parametrization,
            pos=pos,
            pbar=True,
        )
        samples_denorm = data_stats.denormalize(samples.cpu().numpy())
        samples_denorm = np.clip(samples_denorm, 0, None)

    for i in range(samples_denorm.shape[0]):
        rec_int = samples_denorm[i, 0]
        true_int = batch_np[i, :, 0]
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
        fig.savefig(f"{cfg.paths.plot_dir}/event_{i}_xy.png")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        visualize_event_z(Graph(adjacency=None, positions_xy=channel_positions, positions_z=np.concatenate([range(n_time_points) for i in range(n_channels)])), true_z, None, ax=axes[0])
        axes[0].set_title("Ground truth")
        visualize_event_z(Graph(adjacency=None, positions_xy=channel_positions, positions_z=np.concatenate([range(n_time_points) for i in range(n_channels)])), rec_z, None, ax=axes[1])
        axes[1].set_title("DDPM sample")
        plt.tight_layout()
        fig.savefig(f"{cfg.paths.plot_dir}/event_{i}_z.png")
        plt.close(fig)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        visualize_event_3d(graph, true_int, ax=ax, colorbar=True)
        fig.savefig(f"{cfg.paths.plot_dir}/event_{i}_true_3d.png", dpi=300)
        plt.close(fig)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        visualize_event_3d(graph, rec_int, ax=ax, colorbar=True)
        fig.savefig(f"{cfg.paths.plot_dir}/event_{i}_rec_3d.png", dpi=300)
        plt.close(fig)

    return samples_denorm

if __name__ == "__main__":
    train(default_config)
    # generate_samples(default_config)