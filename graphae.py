"""
Graph autoencoder with the current GraphAE encoder and a mirrored graph decoder.

Design notes:
- Reuses the existing GraphAEEncoder from ae.py (same latent interface).
- Decoder mirrors encoder hierarchy using recorded TopK keep indices.
- Decoding blocks use transpose message passing (A^T); for undirected graphs
  this is equivalent to A, but keeps the decoder formally "transpose-style".
- No encoder feature skip connections are used, so reconstruction must flow
  through the latent bottleneck plus deterministic unpool indices.
"""

import glob
import math
import os
import sys
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm

from ae import GraphAEEncoder
from config import Config, default_config, get_config, print_config
from data import Graph, SparseGraph, visualize_event, visualize_event_z
from diffae import DiffAEDataStats, apply_lopsided_augmentation
from lz_data_loader import OnlineMSBatcher, TritiumSSDataLoader
from models.graph_unet import _unpool_like, build_block_diagonal_adj
from utils.sparse_ops import subgraph_coo, to_binary, to_coalesced_coo
from utils.visualization import build_xy_adjacency_radius


class GraphTransposeDecoderBlock(nn.Module):
    """Residual decoder block using transpose graph aggregation."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0, eps_init: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.eps = nn.Parameter(torch.tensor(eps_init))

        nn.init.xavier_uniform_(self.lin1.weight, gain=1.0)
        nn.init.zeros_(self.lin1.bias)
        nn.init.xavier_uniform_(self.lin2.weight, gain=0.5)
        nn.init.zeros_(self.lin2.bias)

    def forward(self, x: torch.Tensor, adj_t: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        neighbor_sum = torch.sparse.mm(adj_t, h)
        h = (1 + self.eps) * h + neighbor_sum
        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)
        h = self.dropout(h)
        return x + h


class GraphTransposeDecoderStage(nn.Module):
    def __init__(self, hidden_dim: int, blocks_per_stage: int = 2, dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [GraphTransposeDecoderBlock(hidden_dim, dropout=dropout) for _ in range(blocks_per_stage)]
        )

    def forward(self, x: torch.Tensor, adj_t: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, adj_t)
        return x


class GraphTransposeDecoder(nn.Module):
    """
    Mirror decoder for GraphAEEncoder outputs.

    Expects encoder pooling indices:
        pool_indices[d] = (keep_idx, N_prev_total, nodes_per_graph_prev)
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_nodes: int,
        depth: int = 4,
        blocks_per_stage: int = 2,
        pool_ratio: float = 0.5,
        dropout: float = 0.0,
        pos_dim: int = 3,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_nodes = n_nodes
        self.depth = depth
        self.pool_ratio = pool_ratio

        deepest_nodes = n_nodes
        for _ in range(depth):
            deepest_nodes = max(1, int(math.ceil(deepest_nodes * pool_ratio)))
        self.deepest_nodes = deepest_nodes

        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * deepest_nodes),
        )

        self.bottleneck = GraphTransposeDecoderStage(
            hidden_dim=hidden_dim, blocks_per_stage=blocks_per_stage, dropout=dropout
        )
        self.stages = nn.ModuleList(
            [
                GraphTransposeDecoderStage(hidden_dim, blocks_per_stage=blocks_per_stage, dropout=dropout)
                for _ in range(depth)
            ]
        )

        self.pos_mlp = nn.Sequential(
            nn.Linear(pos_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self._cached_block_adj = {}
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.latent_proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.pos_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=1.0)
        nn.init.zeros_(self.out_proj.bias)

    def _get_block_adj(self, adj: torch.Tensor, batch_size: int) -> torch.Tensor:
        key = (adj.device, adj.size(), adj._nnz(), batch_size)
        if key in self._cached_block_adj:
            return self._cached_block_adj[key]
        adj_binary = to_binary(to_coalesced_coo(adj))
        block_adj = build_block_diagonal_adj(adj_binary, batch_size)
        self._cached_block_adj[key] = block_adj
        return block_adj

    @staticmethod
    def _transpose_adj(adj: torch.Tensor) -> torch.Tensor:
        return to_coalesced_coo(adj.transpose(0, 1))

    def _build_adj_pyramid(
        self,
        adj0: torch.Tensor,
        pool_indices: List[Tuple[torch.Tensor, int, int]],
    ) -> List[torch.Tensor]:
        adjs = [adj0]
        adj_cur = adj0
        for keep_idx, _, _ in pool_indices:
            K = int(keep_idx.numel())
            adj_cur = to_binary(subgraph_coo(adj_cur, keep_idx, K))
            adjs.append(adj_cur)
        return adjs

    def forward(
        self,
        z: torch.Tensor,
        adj: torch.Tensor,
        pos: torch.Tensor,
        pool_indices: List[Tuple[torch.Tensor, int, int]],
        batch_size: int = 1,
    ) -> torch.Tensor:
        B = batch_size
        N_single = adj.size(0)
        if len(pool_indices) != self.depth:
            raise ValueError(f"Expected {self.depth} pool levels, got {len(pool_indices)}")

        adj0 = self._get_block_adj(adj, B).to(device=z.device, dtype=z.dtype)
        adjs = self._build_adj_pyramid(adj0, pool_indices)

        h = self.latent_proj(z).view(B * self.deepest_nodes, self.hidden_dim)
        h = self.bottleneck(h, self._transpose_adj(adjs[-1]))

        for d in reversed(range(self.depth)):
            keep_idx, N_prev, _ = pool_indices[d]
            h = _unpool_like(h, keep_idx, N_prev)
            h = self.stages[d](h, self._transpose_adj(adjs[d]))

        pos_tiled = pos.repeat(B, 1)
        pos_norm = (pos_tiled - pos_tiled.mean(dim=0, keepdim=True)) / (
            pos_tiled.std(dim=0, keepdim=True) + 1e-8
        )
        h = h + self.pos_mlp(pos_norm.to(z.dtype))
        h = F.silu(self.out_norm(h))
        out = self.out_proj(h)
        return out


class GraphAutoencoder(nn.Module):
    """Graph autoencoder using existing GraphAEEncoder + transpose-style graph decoder."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        latent_dim: int,
        n_nodes: int,
        depth: int = 4,
        blocks_per_stage: int = 2,
        pool_ratio: float = 0.5,
        dropout: float = 0.0,
        pos_dim: int = 3,
        out_dim: int = 1,
    ):
        super().__init__()
        self.encoder = GraphAEEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            depth=depth,
            blocks_per_stage=blocks_per_stage,
            pool_ratio=pool_ratio,
            dropout=dropout,
            pos_dim=pos_dim,
        )
        self.decoder = GraphTransposeDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            n_nodes=n_nodes,
            depth=depth,
            blocks_per_stage=blocks_per_stage,
            pool_ratio=pool_ratio,
            dropout=dropout,
            pos_dim=pos_dim,
        )

    def encode(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        pos: torch.Tensor,
        batch_size: int = 1,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, int, int]]]:
        return self.encoder(x, adj, pos, batch_size=batch_size)

    def decode(
        self,
        z: torch.Tensor,
        adj: torch.Tensor,
        pos: torch.Tensor,
        pool_indices: List[Tuple[torch.Tensor, int, int]],
        batch_size: int = 1,
    ) -> torch.Tensor:
        return self.decoder(z, adj, pos, pool_indices, batch_size=batch_size)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        pos: torch.Tensor,
        batch_size: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z, pool_indices = self.encoder(x, adj, pos, batch_size=batch_size)
        rec = self.decoder(z, adj, pos, pool_indices, batch_size=batch_size)
        return rec, z


DataLoaderType = Union[TritiumSSDataLoader, OnlineMSBatcher]


@dataclass
class GraphAEContext:
    cfg: Config
    device: torch.device
    loader: DataLoaderType
    graph: SparseGraph
    A_sparse: torch.Tensor
    pos: torch.Tensor
    n_channels: int
    n_time_points: int
    n_nodes: int
    data_stats: DiffAEDataStats
    model: GraphAutoencoder
    checkpoint_dir: str = ""
    plot_dir: str = ""
    ema_model: Optional[GraphAutoencoder] = None
    optim: Optional[torch.optim.Optimizer] = None
    use_ms_data: bool = False

    @classmethod
    def build(
        cls,
        cfg: Config,
        for_training: bool = True,
        verbose: bool = True,
        use_ms_data: bool = True,
    ) -> "GraphAEContext":
        device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if verbose:
            print(f"Using device: {device}")

        if use_ms_data:
            loader = OnlineMSBatcher(
                cfg.paths.tritium_h5,
                cfg.paths.channel_positions,
                delta_min=cfg.ms_data.delta_min,
                delta_max=cfg.ms_data.delta_max,
                ns_per_bin=cfg.ms_data.ns_per_bin,
                seed=cfg.ms_data.seed,
            )
            if verbose:
                print(f"Using online MS data: delta=[{cfg.ms_data.delta_min}, {cfg.ms_data.delta_max}] bins")
        else:
            loader = TritiumSSDataLoader(cfg.paths.tritium_h5, cfg.paths.channel_positions)

        graph = loader.load_adjacency_sparse(
            z_sep=cfg.graph.z_sep,
            radius=cfg.graph.radius,
            z_hops=cfg.graph.z_hops,
        )
        A_sparse = graph.adjacency.to(device)
        pos = graph.positions_xyz.to(device)
        n_channels = loader.n_channels
        n_time_points = loader.n_time_points
        n_nodes = n_channels * n_time_points

        if verbose:
            print(f"Graph: {n_nodes} nodes, {A_sparse._nnz()} edges")
            print("Computing data statistics...")
        data_stats = DiffAEDataStats.from_loader(loader, n_samples=1000, batch_size=32)
        if verbose:
            print(f"Data mean: {data_stats.mean:.4f}, std: {data_stats.std:.4f}")

        model = GraphAutoencoder(
            in_dim=cfg.model.in_dim,
            hidden_dim=cfg.encoder.hidden_dim,
            latent_dim=cfg.encoder.latent_dim,
            n_nodes=n_nodes,
            depth=cfg.encoder.depth,
            blocks_per_stage=cfg.encoder.blocks_per_stage,
            pool_ratio=cfg.encoder.pool_ratio,
            dropout=cfg.encoder.dropout,
            pos_dim=cfg.model.pos_dim,
            out_dim=cfg.model.out_dim,
        ).to(device)

        subdir = cfg.paths.graph_ae_subdir.format(latent_dim=cfg.encoder.latent_dim)
        checkpoint_dir = os.path.join(cfg.paths.checkpoint_dir, subdir)
        plot_dir = os.path.join(cfg.paths.plot_dir, subdir)
        ema_model = None
        optim = None
        if for_training:
            ema_model = deepcopy(model).to(device)
            optim = torch.optim.AdamW(
                list(model.parameters()),
                lr=cfg.training.lr,
                betas=(0.9, 0.999),
                weight_decay=cfg.training.weight_decay,
            )
            os.makedirs(checkpoint_dir, exist_ok=True)

        if verbose:
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"GraphAE parameters: {n_params:,}")

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
            model=model,
            checkpoint_dir=checkpoint_dir,
            plot_dir=plot_dir,
            ema_model=ema_model,
            optim=optim,
            use_ms_data=use_ms_data,
        )

    def latest_checkpoint(self) -> Optional[str]:
        files = glob.glob(os.path.join(self.checkpoint_dir, "graphae_epoch_*.pt"))
        if not files:
            return None

        def _epoch_num(path: str) -> int:
            base = os.path.basename(path)
            stem = os.path.splitext(base)[0]
            try:
                return int(stem.split("_")[-1])
            except (ValueError, IndexError):
                return -1

        return max(files, key=_epoch_num)

    def save_checkpoint(self, epoch: int) -> str:
        state = {
            "model": self.model.state_dict(),
            "ema_model": self.ema_model.state_dict() if self.ema_model is not None else self.model.state_dict(),
            "optim": self.optim.state_dict() if self.optim is not None else None,
            "epoch": epoch,
            "data_stats": {"mean": self.data_stats.mean, "std": self.data_stats.std},
        }
        path = os.path.join(self.checkpoint_dir, f"graphae_epoch_{epoch:04d}.pt")
        torch.save(state, path)
        return path

    def load_checkpoint(self, path: str, load_optim: bool = True) -> int:
        chk = torch.load(path, map_location=self.device)
        self.model.load_state_dict(chk["model"])
        if self.ema_model is not None and "ema_model" in chk:
            self.ema_model.load_state_dict(chk["ema_model"])
        if load_optim and self.optim is not None and chk.get("optim"):
            self.optim.load_state_dict(chk["optim"])
        if "data_stats" in chk:
            self.data_stats.mean = chk["data_stats"]["mean"]
            self.data_stats.std = chk["data_stats"]["std"]
        return int(chk.get("epoch", 0))


@torch.no_grad()
def reconstruct_graphae(
    model: GraphAutoencoder,
    A_sparse: torch.Tensor,
    pos: torch.Tensor,
    x_ref: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct events with GraphAE. Returns (B, 1, N)."""
    B, N, C = x_ref.shape
    rec_flat, _ = model(x_ref.view(B * N, C), A_sparse, pos, batch_size=B)
    return rec_flat.view(B, N, C).permute(0, 2, 1)


def _save_training_visualizations(
    ctx: GraphAEContext,
    cfg: Config,
    epoch: int,
    batch_np_raw: np.ndarray,
    rec_denorm: np.ndarray,
) -> None:
    n_channels = ctx.n_channels
    n_time_points = ctx.n_time_points
    channel_positions = ctx.loader.channel_positions
    plots_dir = f"{ctx.plot_dir}/epoch_{epoch}"
    os.makedirs(plots_dir, exist_ok=True)

    adj2d = build_xy_adjacency_radius(channel_positions, radius=cfg.graph.radius)
    gxy = Graph(
        adjacency=adj2d,
        positions_xy=channel_positions,
        positions_z=np.zeros(n_channels, dtype=np.float32),
    )
    gz = Graph(
        adjacency=np.eye(n_channels, dtype=np.float32),
        positions_xy=channel_positions,
        positions_z=np.arange(n_time_points, dtype=np.float32),
    )

    b_vis = rec_denorm.shape[0]
    for idx in range(b_vis):
        rec_int = rec_denorm[idx, :, 0]
        true_int = batch_np_raw[idx, :, 0]

        rec_xy = rec_int.reshape(n_channels, n_time_points, order="F").sum(axis=1)
        true_xy = true_int.reshape(n_channels, n_time_points, order="F").sum(axis=1)
        rec_z = rec_int.reshape(n_channels, n_time_points, order="F")
        true_z = true_int.reshape(n_channels, n_time_points, order="F")

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        visualize_event(gxy, true_xy, None, ax=axes[0])
        axes[0].set_title("Ground truth")
        visualize_event(gxy, rec_xy, None, ax=axes[1])
        axes[1].set_title("GraphAE reconstruction")
        plt.tight_layout()
        fig.savefig(f"{plots_dir}/event_{idx}_xy.png")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        visualize_event_z(gz, true_z, None, ax=axes[0])
        axes[0].set_title("Ground truth z-profile")
        visualize_event_z(gz, rec_z, None, ax=axes[1])
        axes[1].set_title("GraphAE z-profile")
        plt.tight_layout()
        fig.savefig(f"{plots_dir}/event_{idx}_z.png")
        plt.close(fig)

        # Temporal cross-sections for strongest channels.
        top_k = min(4, n_channels)
        top_channels = np.argsort(true_xy)[-top_k:][::-1]
        fig, axes = plt.subplots(top_k, 1, figsize=(10, 2.2 * top_k), sharex=True)
        if top_k == 1:
            axes = [axes]
        t_axis = np.arange(n_time_points)
        for ax, ch in zip(axes, top_channels):
            ax.plot(t_axis, true_z[ch], color="black", linewidth=1.2, label="truth")
            ax.plot(t_axis, rec_z[ch], color="#d62728", linewidth=1.0, alpha=0.9, label="recon")
            ax.set_ylabel(f"ch {ch}")
            ax.grid(alpha=0.25, linewidth=0.4)
        axes[0].legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("time bin")
        fig.suptitle("Cross-section comparison (top channels)")
        plt.tight_layout()
        fig.savefig(f"{plots_dir}/event_{idx}_cross_sections.png")
        plt.close(fig)


def train_graphae(cfg: Config = default_config):
    """Train GraphAE with visualization outputs similar to AE/DiffAE scripts."""
    print("=" * 50)
    print("GraphAE Training")
    print("=" * 50)
    print_config(cfg, include_encoder=True)

    ctx = GraphAEContext.build(cfg, for_training=True, verbose=True)
    model = ctx.model
    ema_model = ctx.ema_model
    optim = ctx.optim
    tr = ctx.loader
    data_stats = ctx.data_stats
    A_sparse = ctx.A_sparse
    pos = ctx.pos
    n_nodes = ctx.n_nodes
    B = cfg.training.batch_size

    start_epoch = 0
    if cfg.resume:
        last = ctx.latest_checkpoint()
        if last is not None:
            start_epoch = ctx.load_checkpoint(last) + 1
            print(f"Resumed from epoch {start_epoch}")

    for g in optim.param_groups:
        g["lr"] = cfg.training.lr

    if cfg.training.lopsided_aug:
        print(f"  Lopsided augmentation ON: frac={cfg.training.lopsided_frac}, sigma={cfg.training.lopsided_sigma}")

    for epoch in range(start_epoch, cfg.training.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_z_std = 0.0
        pbar = tqdm(
            range(cfg.training.steps_per_epoch),
            desc=f"Epoch {epoch+1}/{cfg.training.epochs}",
            ncols=120,
            file=sys.stdout,
        )

        for step in pbar:
            batch_np, _, sample_idx = tr.get_batch(B)
            if cfg.training.lopsided_aug:
                batch_np = apply_lopsided_augmentation(
                    batch_np,
                    frac=cfg.training.lopsided_frac,
                    sigma=cfg.training.lopsided_sigma,
                    sample_indices=sample_idx,
                )
            batch_np = data_stats.normalize(batch_np)

            x0 = torch.from_numpy(batch_np.astype(np.float32)).to(ctx.device)
            x0_flat = x0.view(B * n_nodes, 1)

            rec_flat, z = model(x0_flat, A_sparse, pos, batch_size=B)
            rec = rec_flat.view(B, n_nodes, 1)
            loss = F.mse_loss(rec, x0, reduction="mean")
            if torch.isnan(loss) or torch.isinf(loss):
                optim.zero_grad(set_to_none=True)
                continue

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=cfg.training.grad_clip)
            optim.step()

            with torch.no_grad():
                for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                    p_ema.data.mul_(cfg.training.ema_decay).add_(p.data, alpha=1.0 - cfg.training.ema_decay)

            epoch_loss += float(loss.item())
            epoch_z_std += float(z.std().item())
            pbar.set_postfix(
                loss=epoch_loss / (step + 1),
                z_std=epoch_z_std / (step + 1),
            )

        if (epoch + 1) % cfg.training.checkpoint_every == 0:
            ctx.save_checkpoint(epoch)

        if cfg.visualize and (epoch % cfg.training.visualize_every == 0 or epoch == cfg.training.epochs - 1):
            ema_model.eval()
            with torch.no_grad():
                b_vis = min(cfg.training.batch_size, 4)
                batch_np_raw, _, sample_idx = tr.get_batch(b_vis)
                if cfg.training.lopsided_aug:
                    batch_np_raw = apply_lopsided_augmentation(
                        batch_np_raw,
                        frac=cfg.training.lopsided_frac,
                        sigma=cfg.training.lopsided_sigma,
                        sample_indices=sample_idx,
                    )
                x_ref = torch.from_numpy(data_stats.normalize(batch_np_raw).astype(np.float32)).to(ctx.device)
                rec_flat, _ = ema_model(x_ref.view(b_vis * n_nodes, 1), A_sparse, pos, batch_size=b_vis)
                rec = rec_flat.view(b_vis, n_nodes, 1).cpu().numpy()
                rec_denorm = np.clip(data_stats.denormalize(rec), 0, None)
                _save_training_visualizations(ctx, cfg, epoch, batch_np_raw, rec_denorm)

                true_data = batch_np_raw[:, :, 0]
                gen_data = rec_denorm[:, :, 0]
                print(f"\n  [Vis] True data - mean: {true_data.mean():.4f}, std: {true_data.std():.4f}")
                print(f"  [Vis] Gen data  - mean: {gen_data.mean():.4f}, std: {gen_data.std():.4f}")


if __name__ == "__main__":
    cfg = get_config(epochs=20_000)
    train_graphae(cfg)
