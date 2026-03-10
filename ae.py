"""
Graph AE: Autoencoder with Graph Encoder/Decoder.

Uses a graph encoder to map events to latent representations,
and a graph decoder for direct reconstruction.
"""
import os
import sys
import glob
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List
from copy import deepcopy

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm

from typing import Union
from data import Graph, visualize_event, visualize_event_z, SparseGraph
from lz_data_loader import TritiumSSDataLoader, OnlineMSBatcher
from config import Config, default_config, print_config

DataLoaderType = Union[TritiumSSDataLoader, OnlineMSBatcher]

from diffae import (
    GraphEncoder, GraphEncoderStage, GraphEncoderBlock,
    DiffAEDataStats, visualize_event_3d, apply_lopsided_augmentation,
)
from models.graph_unet import (
    TopKPool, build_block_diagonal_adj, _unpool_like
)
from utils.sparse_ops import to_coalesced_coo, subgraph_coo, to_binary
from utils.visualization import build_xy_adjacency_radius


class GraphDecoderBlock(nn.Module):
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

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        neighbor_sum = torch.sparse.mm(adj, h)
        h = (1 + self.eps) * h + neighbor_sum
        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)
        h = self.dropout(h)
        return x + h


class GraphDecoderStage(nn.Module):
    def __init__(self, hidden_dim: int, blocks_per_stage: int = 2, dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            GraphDecoderBlock(hidden_dim, dropout=dropout)
            for _ in range(blocks_per_stage)
        ])

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, adj)
        return x


class GraphDecoder(nn.Module):
    """
    Graph decoder that maps latent representations back to node features.
    Uses hierarchical unpooling to reconstruct the full graph.
    """
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_nodes: int,
        depth: int = 3,
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

        # Compute sizes at each level
        self.level_sizes = [n_nodes]
        current_size = n_nodes
        for _ in range(depth):
            current_size = max(1, int(math.ceil(current_size * pool_ratio)))
            self.level_sizes.append(current_size)
        self.level_sizes = self.level_sizes[::-1]  # smallest to largest

        # Project latent to initial hidden state
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * self.level_sizes[0]),
        )

        self.pos_mlp = nn.Sequential(
            nn.Linear(pos_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Upsampling layers
        self.upsample_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(depth)
        ])

        # Decoder stages (after each upsampling)
        self.stages = nn.ModuleList([
            GraphDecoderStage(hidden_dim, blocks_per_stage, dropout)
            for _ in range(depth)
        ])

        self.final_stage = GraphDecoderStage(hidden_dim, blocks_per_stage, dropout)

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
        for layer in self.upsample_layers:
            nn.init.xavier_uniform_(layer.weight, gain=1.0)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=1.0)
        nn.init.zeros_(self.out_proj.bias)

    def _get_block_adj(self, adj: torch.Tensor, batch_size: int) -> torch.Tensor:
        key = (adj.device, adj.size(), adj._nnz(), batch_size)
        if key in self._cached_block_adj:
            return self._cached_block_adj[key]
        adj_binary = to_binary(adj)
        block_adj = build_block_diagonal_adj(adj_binary, batch_size)
        self._cached_block_adj[key] = block_adj
        return block_adj

    def forward(
        self,
        z: torch.Tensor,
        adj: torch.Tensor,
        pos: torch.Tensor,
        encoder_pool_indices: List[Tuple[torch.Tensor, int, int]],
        batch_size: int = 1
    ) -> torch.Tensor:
        """
        Decode latent representations to node features.
        
        Args:
            z: Latent representation (B, latent_dim)
            adj: Single graph adjacency (N, N)
            pos: Node positions (N, pos_dim)
            encoder_pool_indices: List of (keep_idx, N_prev, npg_prev) from encoder
            batch_size: Number of graphs in batch
            
        Returns:
            Reconstructed node features (B*N, out_dim)
        """
        B = batch_size
        initial_size = self.level_sizes[0]

        # Project latent to initial nodes
        h = self.latent_proj(z)  # (B, hidden_dim * initial_size)
        h = h.view(B * initial_size, self.hidden_dim)

        # Build adjacency for smallest graph
        adj0 = self._get_block_adj(adj, B).to(device=z.device, dtype=z.dtype)

        # Process through decoder stages with upsampling
        for d in range(self.depth):
            keep_idx, N_prev, npg_prev = encoder_pool_indices[self.depth - 1 - d]
            
            # Upsample
            h_up = _unpool_like(h, keep_idx, N_prev)
            h = self.upsample_layers[d](h_up)
            
            # Get adjacency at this level
            adj_level = subgraph_coo(adj0, torch.arange(N_prev, device=z.device), N_prev)
            adj_level = to_binary(adj_level)
            
            # Process with graph convolutions
            h = self.stages[d](h, adj_level)

        # Add position embeddings at final resolution
        pos_tiled = pos.repeat(B, 1)
        pos_normalized = (pos_tiled - pos_tiled.mean(dim=0, keepdim=True)) / (pos_tiled.std(dim=0, keepdim=True) + 1e-8)
        pos_emb = self.pos_mlp(pos_normalized.to(z.dtype))
        h = h + pos_emb

        # Final processing
        h = self.final_stage(h, adj0)
        h = F.silu(self.out_norm(h))
        out = self.out_proj(h)

        return out


class Conv1DEncoder(nn.Module):
    """
    1D CNN encoder for AE: processes waveforms with convolutions to capture
    local texture features. Same interface as MLPEncoder/GraphAEEncoder.
    """
    def __init__(
        self,
        in_dim: int,
        latent_dim: int,
        n_nodes: int,
        dropout: float = 0.0,
        channels: tuple = (32, 64, 128),
        kernel_size: int = 7,
        pool_size: int = 4,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.n_nodes = n_nodes
        conv_layers = []
        ch_in = in_dim
        for ch_out in channels:
            conv_layers.extend([
                nn.Conv1d(ch_in, ch_out, kernel_size, padding=kernel_size // 2),
                nn.SiLU(),
                nn.AvgPool1d(pool_size),
                nn.Dropout(dropout),
            ])
            ch_in = ch_out
        conv_layers.append(nn.AdaptiveAvgPool1d(4))
        conv_layers.append(nn.Flatten())
        self.backbone = nn.Sequential(*conv_layers)
        self.to_latent = nn.Linear(channels[-1] * 4, latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        pos: torch.Tensor,
        batch_size: int = 1,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, int, int]]]:
        B = batch_size
        N = x.shape[0] // B
        x_2d = x.view(B, N, self.in_dim).permute(0, 2, 1)
        h = self.backbone(x_2d)
        z = self.to_latent(h)
        return z, []


class MLPEncoder(nn.Module):
    """
    MLP encoder: flattens input and maps to latent. No graph structure used.
    Same interface as GraphAEEncoder for drop-in use.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        latent_dim: int,
        n_nodes: int,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_nodes = n_nodes

        layers = []
        input_size = n_nodes * in_dim
        n_hidden = num_layers - 1
        if n_hidden <= 0:
            layer_sizes = []
        else:
            ratio = (latent_dim / input_size) ** (1.0 / n_hidden)
            layer_sizes = [max(latent_dim, int(round(input_size * ratio ** i))) for i in range(1, n_hidden + 1)]

        in_d = input_size
        for out_d in layer_sizes:
            layers.append(nn.Linear(in_d, out_d))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            in_d = out_d
        layers.append(nn.Linear(in_d, latent_dim))
        self.mlp = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        pos: torch.Tensor,
        batch_size: int = 1,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, int, int]]]:
        B = batch_size
        x_flat = x.view(B, -1)
        z = self.mlp(x_flat)
        return z, []


class GraphAEEncoder(nn.Module):
    """
    Graph encoder that returns latent z and pooling indices for the decoder.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        latent_dim: int,
        depth: int = 3,
        blocks_per_stage: int = 2,
        pool_ratio: float = 0.5,
        dropout: float = 0.0,
        pos_dim: int = 3,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.depth = depth
        self.pool_ratio = pool_ratio

        self.in_proj = nn.Linear(in_dim, hidden_dim)
        
        self.pos_mlp = nn.Sequential(
            nn.Linear(pos_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.stages = nn.ModuleList([
            GraphEncoderStage(hidden_dim, blocks_per_stage, dropout)
            for _ in range(depth)
        ])
        self.pools = nn.ModuleList([
            TopKPool(hidden_dim, ratio=pool_ratio)
            for _ in range(depth)
        ])

        self.final_stage = GraphEncoderStage(hidden_dim, blocks_per_stage, dropout)

        self.to_latent = nn.Linear(hidden_dim * 2, latent_dim)

        self._cached_block_adj = {}
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj.weight, gain=1.0)
        nn.init.zeros_(self.in_proj.bias)
        for m in self.pos_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.to_latent.weight, gain=0.5)
        nn.init.zeros_(self.to_latent.bias)

    def _get_block_adj(self, adj: torch.Tensor, batch_size: int) -> torch.Tensor:
        key = (adj.device, adj.size(), adj._nnz(), batch_size)
        if key in self._cached_block_adj:
            return self._cached_block_adj[key]
        adj_binary = to_binary(adj)
        block_adj = build_block_diagonal_adj(adj_binary, batch_size)
        self._cached_block_adj[key] = block_adj
        return block_adj

    def forward(
        self, 
        x: torch.Tensor, 
        adj: torch.Tensor, 
        pos: torch.Tensor,
        batch_size: int = 1
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, int, int]]]:
        """
        Encode input events to latent representations.

        Returns:
            z: Latent representation (B, latent_dim)
            pool_indices: List of (keep_idx, N_prev, npg_prev) for decoder
        """
        N_single = adj.size(0)
        total_nodes = x.size(0)
        assert total_nodes == batch_size * N_single

        adj0 = self._get_block_adj(adj, batch_size).to(device=x.device, dtype=x.dtype)

        h = self.in_proj(x)
        pos_tiled = pos.repeat(batch_size, 1)
        pos_normalized = (pos_tiled - pos_tiled.mean(dim=0, keepdim=True)) / (pos_tiled.std(dim=0, keepdim=True) + 1e-8)
        pos_emb = self.pos_mlp(pos_normalized.to(x.dtype))
        h = h + pos_emb

        nodes_per_graph = N_single
        h_cur, adj_cur = h, adj0

        pool_indices = []
        for d in range(self.depth):
            h_cur = self.stages[d](h_cur, adj_cur)
            h_pool, keep_idx, k_per_graph = self.pools[d](h_cur, nodes_per_graph)
            
            pool_indices.append((keep_idx, h_cur.size(0), nodes_per_graph))
            
            K = h_pool.size(0)
            adj_next = to_binary(subgraph_coo(adj_cur, keep_idx, K))
            h_cur, adj_cur, nodes_per_graph = h_pool, adj_next, k_per_graph

        h_cur = self.final_stage(h_cur, adj_cur)

        h_graph = h_cur.view(batch_size, nodes_per_graph, self.hidden_dim)
        h_mean = h_graph.mean(dim=1)
        h_std = h_graph.std(dim=1) + 1e-6
        h_agg = torch.cat([h_mean, h_std], dim=-1)
        z = self.to_latent(h_agg)
        return z, pool_indices


class SimpleGraphDecoder(nn.Module):
    """
    Simple MLP-based decoder that broadcasts latent to all nodes,
    then refines with graph message passing.
    """
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_nodes: int,
        depth: int = 3,
        blocks_per_stage: int = 2,
        dropout: float = 0.0,
        pos_dim: int = 3,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_nodes = n_nodes
        self.depth = depth

        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.pos_mlp = nn.Sequential(
            nn.Linear(pos_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.stages = nn.ModuleList([
            GraphDecoderStage(hidden_dim, blocks_per_stage, dropout)
            for _ in range(depth)
        ])

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
        adj_binary = to_binary(adj)
        block_adj = build_block_diagonal_adj(adj_binary, batch_size)
        self._cached_block_adj[key] = block_adj
        return block_adj

    def forward(
        self,
        z: torch.Tensor,
        adj: torch.Tensor,
        pos: torch.Tensor,
        batch_size: int = 1
    ) -> torch.Tensor:
        """
        Decode latent representations to node features.
        
        Args:
            z: Latent representation (B, latent_dim)
            adj: Single graph adjacency (N, N)
            pos: Node positions (N, pos_dim)
            batch_size: Number of graphs in batch
            
        Returns:
            Reconstructed node features (B*N, out_dim)
        """
        N = self.n_nodes
        B = batch_size
        
        adj0 = self._get_block_adj(adj, B).to(device=z.device, dtype=z.dtype)

        # Project latent and broadcast to all nodes
        z_proj = self.latent_proj(z)  # (B, hidden_dim)
        h = z_proj.unsqueeze(1).expand(B, N, -1).reshape(B * N, self.hidden_dim)

        # Add position embeddings
        pos_tiled = pos.repeat(B, 1)
        pos_normalized = (pos_tiled - pos_tiled.mean(dim=0, keepdim=True)) / (pos_tiled.std(dim=0, keepdim=True) + 1e-8)
        pos_emb = self.pos_mlp(pos_normalized.to(z.dtype))
        h = h + pos_emb

        # Graph message passing stages
        for stage in self.stages:
            h = stage(h, adj0)

        h = F.silu(self.out_norm(h))
        out = self.out_proj(h)

        return out


class MLPDecoder(nn.Module):
    """
    MLP decoder: latent z -> MLP -> (B*N, out_dim). No graph or position.
    Same interface as SimpleGraphDecoder for drop-in use.
    """
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_nodes: int,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_nodes = n_nodes

        layers = []
        in_d = latent_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            in_d = hidden_dim
        layers.append(nn.Linear(in_d, n_nodes * out_dim))
        self.mlp = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        z: torch.Tensor,
        adj: torch.Tensor,
        pos: torch.Tensor,
        batch_size: int = 1,
    ) -> torch.Tensor:
        B = batch_size
        N, C = self.n_nodes, self.out_dim
        out = self.mlp(z)
        return out.view(B * N, C)


@dataclass
class AEContext:
    """Holds all model components for AE training/inference."""
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
    encoder: nn.Module
    decoder: nn.Module
    checkpoint_dir: str = ""
    plot_dir: str = ""
    ema_encoder: Optional[nn.Module] = None
    ema_decoder: Optional[nn.Module] = None
    optim: Optional[torch.optim.Optimizer] = None
    use_ms_data: bool = False

    @classmethod
    def build(cls, cfg: Config, for_training: bool = True, verbose: bool = True, use_ms_data: bool = True) -> 'AEContext':
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

        data_stats = DiffAEDataStats.from_loader(loader, n_samples=1000, batch_size=32)
        if verbose:
            print(f"Data mean: {data_stats.mean:.4f}, std: {data_stats.std:.4f}")

        encoder_type = (getattr(cfg.encoder, "encoder_type", "graph") or "graph").lower()
        if encoder_type == "cnn":
            encoder = Conv1DEncoder(
                in_dim=cfg.model.in_dim,
                latent_dim=cfg.encoder.latent_dim,
                n_nodes=n_nodes,
                dropout=cfg.encoder.dropout,
            ).to(device)
        elif encoder_type == "mlp":
            encoder = MLPEncoder(
                in_dim=cfg.model.in_dim,
                hidden_dim=cfg.encoder.hidden_dim,
                latent_dim=cfg.encoder.latent_dim,
                n_nodes=n_nodes,
                num_layers=getattr(cfg.encoder, "mlp_encoder_layers", 3),
                dropout=cfg.encoder.dropout,
            ).to(device)
        else:
            encoder = GraphAEEncoder(
                in_dim=cfg.model.in_dim,
                hidden_dim=cfg.encoder.hidden_dim,
                latent_dim=cfg.encoder.latent_dim,
                depth=cfg.encoder.depth,
                blocks_per_stage=cfg.encoder.blocks_per_stage,
                pool_ratio=cfg.encoder.pool_ratio,
                dropout=cfg.encoder.dropout,
                pos_dim=cfg.model.pos_dim,
            ).to(device)

        decoder_type = (getattr(cfg.encoder, "decoder_type", "graph") or "graph").lower()
        if decoder_type == "mlp":
            decoder = MLPDecoder(
                latent_dim=cfg.encoder.latent_dim,
                hidden_dim=cfg.encoder.hidden_dim,
                out_dim=cfg.model.out_dim,
                n_nodes=n_nodes,
                num_layers=getattr(cfg.encoder, "mlp_decoder_layers", 3),
                dropout=cfg.encoder.dropout,
            ).to(device)
        else:
            decoder = SimpleGraphDecoder(
                latent_dim=cfg.encoder.latent_dim,
                hidden_dim=cfg.encoder.hidden_dim,
                out_dim=cfg.model.out_dim,
                n_nodes=n_nodes,
                depth=cfg.encoder.depth,
                blocks_per_stage=cfg.encoder.blocks_per_stage,
                dropout=cfg.encoder.dropout,
                pos_dim=cfg.model.pos_dim,
            ).to(device)

        ema_encoder = None
        ema_decoder = None
        optim = None
        subdir = cfg.paths.ae_subdir.format(latent_dim=cfg.encoder.latent_dim)
        checkpoint_dir = os.path.join(cfg.paths.checkpoint_dir, subdir)
        plot_dir = os.path.join(cfg.paths.plot_dir, subdir)

        if for_training:
            ema_encoder = deepcopy(encoder).to(device)
            ema_decoder = deepcopy(decoder).to(device)
            all_params = list(encoder.parameters()) + list(decoder.parameters())
            optim = torch.optim.AdamW(
                all_params,
                lr=cfg.training.lr,
                betas=(0.9, 0.999),
                weight_decay=cfg.training.weight_decay,
            )
            os.makedirs(checkpoint_dir, exist_ok=True)

        if verbose:
            n_enc = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
            n_dec = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
            print(f"Encoder parameters: {n_enc:,}")
            print(f"Decoder parameters: {n_dec:,}")
            print(f"Total trainable parameters: {n_enc + n_dec:,}")

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
            encoder=encoder,
            decoder=decoder,
            checkpoint_dir=checkpoint_dir,
            plot_dir=plot_dir,
            ema_encoder=ema_encoder,
            ema_decoder=ema_decoder,
            optim=optim,
            use_ms_data=use_ms_data,
        )

    def latest_checkpoint(self) -> Optional[str]:
        files = glob.glob(os.path.join(self.checkpoint_dir, "ae_epoch_*.pt"))
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
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "ema_encoder": self.ema_encoder.state_dict() if self.ema_encoder else self.encoder.state_dict(),
            "ema_decoder": self.ema_decoder.state_dict() if self.ema_decoder else self.decoder.state_dict(),
            "optim": self.optim.state_dict() if self.optim else None,
            "epoch": epoch,
            "data_stats": {"mean": self.data_stats.mean, "std": self.data_stats.std},
        }
        path = os.path.join(self.checkpoint_dir, f"ae_epoch_{epoch:04d}.pt")
        torch.save(state, path)
        return path

    def load_checkpoint(self, path: str, load_optim: bool = True) -> int:
        chk = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(chk["encoder"])
        self.decoder.load_state_dict(chk["decoder"])
        if self.ema_encoder is not None and "ema_encoder" in chk:
            self.ema_encoder.load_state_dict(chk["ema_encoder"])
        if self.ema_decoder is not None and "ema_decoder" in chk:
            self.ema_decoder.load_state_dict(chk["ema_decoder"])
        if load_optim and self.optim is not None and chk.get("optim"):
            self.optim.load_state_dict(chk["optim"])
        if "data_stats" in chk:
            self.data_stats.mean = chk["data_stats"]["mean"]
            self.data_stats.std = chk["data_stats"]["std"]
        return int(chk.get("epoch", 0))

    def find_best_checkpoint(self) -> Optional[str]:
        """Find the best available checkpoint, preferring same latent_dim but falling back to others."""
        same_latent = self.latest_checkpoint()
        if same_latent is not None:
            return same_latent
        
        parent_dir = os.path.dirname(self.checkpoint_dir)
        if not os.path.isdir(parent_dir):
            return None
        
        all_ckpts = []
        for subdir in os.listdir(parent_dir):
            if subdir.startswith("ae_z"):
                subdir_path = os.path.join(parent_dir, subdir)
                ckpt_files = sorted(glob.glob(os.path.join(subdir_path, "ae_epoch_*.pt")))
                if ckpt_files:
                    all_ckpts.append(ckpt_files[-1])
        
        if all_ckpts:
            return max(all_ckpts, key=os.path.getmtime)
        return None

    def load_checkpoint_partial(self, path: str, verbose: bool = True) -> Tuple[int, bool]:
        """
        Load checkpoint with partial weight loading for different latent sizes.
        
        Loads all compatible weights from checkpoint and keeps latent-dependent
        layers freshly initialized if sizes don't match.
        
        Returns:
            (epoch, is_full_load): epoch from checkpoint, True if all weights loaded
        """
        chk = torch.load(path, map_location=self.device)
        
        encoder_latent_keys = {'to_latent.weight', 'to_latent.bias', 'to_mu.weight', 'to_mu.bias', 'to_logvar.weight', 'to_logvar.bias'}
        
        decoder_latent_keys = {'latent_proj.0.weight', 'latent_proj.0.bias'}
        
        def load_partial(model: nn.Module, state_dict: dict, skip_keys: set, name: str) -> List[str]:
            """Load state dict, skipping keys with mismatched sizes."""
            model_dict = model.state_dict()
            loaded_keys = []
            skipped_keys = []
            
            for key, value in state_dict.items():
                if key in skip_keys:
                    skipped_keys.append(key)
                    continue
                if key in model_dict:
                    if model_dict[key].shape == value.shape:
                        model_dict[key] = value
                        loaded_keys.append(key)
                    else:
                        skipped_keys.append(key)
            
            model.load_state_dict(model_dict)
            if verbose and skipped_keys:
                print(f"  {name}: loaded {len(loaded_keys)}/{len(state_dict)} keys, skipped: {skipped_keys}")
            return skipped_keys
        
        all_skipped = []
        
        all_skipped.extend(load_partial(self.encoder, chk["encoder"], encoder_latent_keys, "encoder"))
        all_skipped.extend(load_partial(self.decoder, chk["decoder"], decoder_latent_keys, "decoder"))
        
        if self.ema_encoder is not None and "ema_encoder" in chk:
            load_partial(self.ema_encoder, chk["ema_encoder"], encoder_latent_keys, "ema_encoder")
        if self.ema_decoder is not None and "ema_decoder" in chk:
            load_partial(self.ema_decoder, chk["ema_decoder"], decoder_latent_keys, "ema_decoder")
        
        if "data_stats" in chk:
            self.data_stats.mean = chk["data_stats"]["mean"]
            self.data_stats.std = chk["data_stats"]["std"]
        
        is_full_load = len(all_skipped) == 0
        epoch = int(chk.get("epoch", 0))
        return epoch, is_full_load


@torch.no_grad()
def save_encoded_dataset(
    ctx: AEContext,
    output_path: str,
    encoder: Optional[nn.Module] = None,
    batch_size: int = 32,
    n_samples: int = 10000,
    verbose: bool = True,
) -> str:
    """Encode MS events and save latent vectors with delta_mu to h5.
    
    This is designed for the aux task: it encodes MS events and saves the
    latents along with delta_mu targets so that aux training only needs to
    load pre-computed embeddings (no encoder calls during MLP training).
    """
    if encoder is None:
        encoder = ctx.ema_encoder if ctx.ema_encoder is not None else ctx.encoder
    encoder.eval()
    
    latent_dim = ctx.cfg.encoder.latent_dim
    
    all_latents = []
    all_delta_mu = []
    all_delta_bins = []
    all_xc1 = []
    all_yc1 = []
    all_xc2 = []
    all_yc2 = []
    
    n_batches = (n_samples + batch_size - 1) // batch_size
    pbar = tqdm(range(n_batches), desc="Encoding MS dataset", disable=not verbose, ncols=120)
    
    samples_encoded = 0
    for batch_idx in pbar:
        remaining = n_samples - samples_encoded
        actual_batch_size = min(batch_size, remaining)
        if actual_batch_size <= 0:
            break
        
        wf_col, cond, *_ = ctx.loader.get_batch(actual_batch_size)
        
        if ctx.use_ms_data:
            xc1 = cond[:, 0]
            yc1 = cond[:, 1]
            xc2 = cond[:, 2]
            yc2 = cond[:, 3]
            delta_mu = cond[:, 4]
            delta_bins = cond[:, 5]
            all_xc1.append(xc1)
            all_yc1.append(yc1)
            all_xc2.append(xc2)
            all_yc2.append(yc2)
            all_delta_mu.append(delta_mu)
            all_delta_bins.append(delta_bins)
        
        wf_norm = ctx.data_stats.normalize(wf_col)
        x = torch.from_numpy(wf_norm).to(ctx.device)
        x_flat = x.view(actual_batch_size * ctx.n_nodes, 1)
        
        z, _ = encoder(x_flat, ctx.A_sparse, ctx.pos, batch_size=actual_batch_size)
        all_latents.append(z.cpu().numpy())
        samples_encoded += actual_batch_size
    
    latents = np.concatenate(all_latents, axis=0)
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('latents', data=latents, dtype=np.float32)
        
        if all_delta_mu:
            f.create_dataset('delta_mu', data=np.concatenate(all_delta_mu), dtype=np.float32)
            f.create_dataset('delta_bins', data=np.concatenate(all_delta_bins), dtype=np.float32)
            f.create_dataset('xc1', data=np.concatenate(all_xc1), dtype=np.float32)
            f.create_dataset('yc1', data=np.concatenate(all_yc1), dtype=np.float32)
            f.create_dataset('xc2', data=np.concatenate(all_xc2), dtype=np.float32)
            f.create_dataset('yc2', data=np.concatenate(all_yc2), dtype=np.float32)
        
        f.attrs['latent_dim'] = latent_dim
        f.attrs['n_samples'] = samples_encoded
        f.attrs['data_mean'] = ctx.data_stats.mean
        f.attrs['data_std'] = ctx.data_stats.std
        f.attrs['is_ms_data'] = ctx.use_ms_data
    
    if verbose:
        print(f"Saved encoded MS dataset to {output_path}: {samples_encoded} samples, latent_dim={latent_dim}")
    
    return output_path


@torch.no_grad()
def reconstruct_ae(
    encoder: nn.Module,
    decoder: nn.Module,
    A_sparse: torch.Tensor,
    pos: torch.Tensor,
    x_ref: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruct events using AE encoder-decoder.

    Args:
        encoder: AE encoder
        decoder: AE decoder
        A_sparse: Graph adjacency
        pos: Node positions
        x_ref: Reference events to reconstruct (B, N, 1)

    Returns:
        Reconstructed samples (B, 1, N)
    """
    B, N, C = x_ref.shape
    
    x_ref_flat = x_ref.view(B * N, C)
    z, _ = encoder(x_ref_flat, A_sparse, pos, batch_size=B)
    
    rec_flat = decoder(z, A_sparse, pos, batch_size=B)
    rec = rec_flat.view(B, N, C).permute(0, 2, 1)  # (B, C, N)
    
    return rec


@torch.no_grad()
def sample_from_latent(
    decoder: nn.Module,
    A_sparse: torch.Tensor,
    pos: torch.Tensor,
    z: torch.Tensor,
    n_nodes: int,
) -> torch.Tensor:
    """Sample from a given latent representation."""
    B = z.shape[0]
    
    rec_flat = decoder(z, A_sparse, pos, batch_size=B)
    rec = rec_flat.view(B, n_nodes, 1).permute(0, 2, 1)  # (B, 1, N)
    
    return rec


def train_ae(cfg: Config = default_config):
    """Main AE training function."""
    print("=" * 50)
    print("Graph AE Training")
    print("=" * 50)
    print_config(cfg, include_encoder=True)
    
    ctx = AEContext.build(cfg, for_training=True, verbose=True)
    
    device_t = ctx.device
    encoder = ctx.encoder
    decoder = ctx.decoder
    ema_encoder = ctx.ema_encoder
    ema_decoder = ctx.ema_decoder
    optim = ctx.optim
    data_stats = ctx.data_stats
    A_sparse = ctx.A_sparse
    pos = ctx.pos
    n_nodes = ctx.n_nodes
    n_channels = ctx.n_channels
    n_time_points = ctx.n_time_points
    graph = ctx.graph
    tr = ctx.loader
    channel_positions = tr.channel_positions

    start_epoch = 0
    if cfg.resume:
        last = ctx.latest_checkpoint()
        if last is not None:
            try:
                start_epoch = ctx.load_checkpoint(last) + 1
                print(f"Resumed from epoch {start_epoch}")
            except Exception as e:
                print(f"Could not resume exact checkpoint: {e}")
                start_epoch = 0
        
        if start_epoch == 0:
            best_ckpt = ctx.find_best_checkpoint()
            if best_ckpt is not None:
                print(f"Attempting partial load from: {best_ckpt}")
                try:
                    _, is_full = ctx.load_checkpoint_partial(best_ckpt, verbose=True)
                    if is_full:
                        print("Full checkpoint loaded (different directory)")
                    else:
                        print("Partial weights loaded - latent layers freshly initialized")
                except Exception as e:
                    print(f"Could not load partial checkpoint: {e}")

    for g in optim.param_groups:
        g["lr"] = cfg.training.lr

    B = cfg.training.batch_size
    
    global_step = start_epoch * cfg.training.steps_per_epoch
    encoded_output_path = os.path.join(ctx.checkpoint_dir, "ae_encoded_ms_latents.h5")

    if cfg.training.lopsided_aug:
        print(f"  Lopsided augmentation ON: frac={cfg.training.lopsided_frac}, sigma={cfg.training.lopsided_sigma}")

    for epoch in range(start_epoch, cfg.training.epochs):
        encoder.train()
        decoder.train()
        epoch_loss = 0.0
        pbar = tqdm(range(cfg.training.steps_per_epoch), desc=f"Epoch {epoch+1}/{cfg.training.epochs}", ncols=120, file=sys.stdout)
        
        for step in pbar:
            batch_np, _, sample_idx = tr.get_batch(B)
            if cfg.training.lopsided_aug:
                batch_np = apply_lopsided_augmentation(
                    batch_np, frac=cfg.training.lopsided_frac, sigma=cfg.training.lopsided_sigma,
                    sample_indices=sample_idx)
            batch_np = data_stats.normalize(batch_np)
            
            x0 = torch.from_numpy(batch_np.astype(np.float32)).to(device_t)  # (B, N, 1)
            x0_flat = x0.view(B * n_nodes, 1)
            
            z, _ = encoder(x0_flat, A_sparse, pos, batch_size=B)
            rec_flat = decoder(z, A_sparse, pos, batch_size=B)
            rec = rec_flat.view(B, n_nodes, 1)
            loss = F.mse_loss(rec, x0, reduction='mean')

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  WARNING: NaN/Inf loss at step {step}! Skipping...")
                optim.zero_grad(set_to_none=True)
                continue
            
            epoch_loss += float(loss.item())

            optim.zero_grad(set_to_none=True)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()),
                max_norm=cfg.training.grad_clip
            )
            optim.step()

            with torch.no_grad():
                for p_ema, p in zip(ema_encoder.parameters(), encoder.parameters()):
                    p_ema.data.mul_(cfg.training.ema_decay).add_(p.data, alpha=1.0 - cfg.training.ema_decay)
                for p_ema, p in zip(ema_decoder.parameters(), decoder.parameters()):
                    p_ema.data.mul_(cfg.training.ema_decay).add_(p.data, alpha=1.0 - cfg.training.ema_decay)

            global_step += 1

            pbar.set_postfix(loss=epoch_loss / (step + 1))

        if (epoch + 1) % cfg.training.checkpoint_every == 0:
            ctx.save_checkpoint(epoch)

        if cfg.training.encode_dataset_every > 0 and (epoch + 1) % cfg.training.encode_dataset_every == 0:
            ema_encoder.eval()
            save_encoded_dataset(ctx, encoded_output_path, encoder=ema_encoder, batch_size=B * 4, n_samples=cfg.training.encode_n_samples)
            encoder.train()

        if cfg.visualize and (epoch % cfg.training.visualize_every == 0 or epoch == cfg.training.epochs - 1):
            ema_encoder.eval()
            ema_decoder.eval()
            with torch.no_grad():
                b_vis = min(cfg.training.batch_size, 4)
                batch_np, _, sample_idx = tr.get_batch(b_vis)
                if cfg.training.lopsided_aug:
                    batch_np = apply_lopsided_augmentation(
                        batch_np, frac=cfg.training.lopsided_frac, sigma=cfg.training.lopsided_sigma,
                        sample_indices=sample_idx)
                batch_np_norm = data_stats.normalize(batch_np)
                x_ref = torch.from_numpy(batch_np_norm.astype(np.float32)).to(device_t)
                
                samples = reconstruct_ae(
                    encoder=ema_encoder,
                    decoder=ema_decoder,
                    A_sparse=A_sparse,
                    pos=pos,
                    x_ref=x_ref,
                )
                samples_denorm = data_stats.denormalize(samples.cpu().numpy())
                samples_denorm = np.clip(samples_denorm, 0, None)
                
                true_data = batch_np[:, :, 0]
                gen_data = samples_denorm[:, 0, :]
                print(f"\n  [Vis] True data - mean: {true_data.mean():.4f}, std: {true_data.std():.4f}")
                print(f"  [Vis] Gen data  - mean: {gen_data.mean():.4f}, std: {gen_data.std():.4f}")

            plots_dir = f"{ctx.plot_dir}/epoch_{epoch}"
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
                axes[1].set_title("AE reconstruction")
                plt.tight_layout()
                fig.savefig(f"{plots_dir}/event_{idx}_xy.png")
                plt.close(fig)

                fig, axes = plt.subplots(1, 2, figsize=(10, 3))
                visualize_event_z(Graph(adjacency=None, positions_xy=channel_positions, positions_z=np.concatenate([range(n_time_points) for i in range(n_channels)])), true_z, None, ax=axes[0])
                axes[0].set_title("Ground truth")
                visualize_event_z(Graph(adjacency=None, positions_xy=channel_positions, positions_z=np.concatenate([range(n_time_points) for i in range(n_channels)])), rec_z, None, ax=axes[1])
                axes[1].set_title("AE reconstruction")
                plt.tight_layout()
                fig.savefig(f"{plots_dir}/event_{idx}_z.png")
                plt.close(fig)


def interpolate_latents(cfg: Config = default_config, n_steps: int = 5):
    """Generate interpolations between two events in latent space."""
    ctx = AEContext.build(cfg, for_training=False, verbose=True)
    
    latest_ckpt = ctx.latest_checkpoint()
    if latest_ckpt is None:
        raise FileNotFoundError(f"No checkpoints found in {ctx.checkpoint_dir}")
    print(f"Loading checkpoint: {latest_ckpt}")
    ctx.load_checkpoint(latest_ckpt, load_optim=False)
    
    ctx.encoder.eval()
    ctx.decoder.eval()

    with torch.no_grad():
        batch_np, *_ = ctx.loader.get_batch(2)
        batch_np_norm = ctx.data_stats.normalize(batch_np)
        x_ref = torch.from_numpy(batch_np_norm.astype(np.float32)).to(ctx.device)
        
        x_ref_flat = x_ref.view(2 * ctx.n_nodes, 1)
        z, _ = ctx.encoder(x_ref_flat, ctx.A_sparse, ctx.pos, batch_size=2)
        z1, z2 = z[0], z[1]

        alphas = torch.linspace(0, 1, n_steps, device=ctx.device)
        z_interp = torch.stack([(1 - a) * z1 + a * z2 for a in alphas])

        samples = sample_from_latent(
            decoder=ctx.decoder,
            A_sparse=ctx.A_sparse,
            pos=ctx.pos,
            z=z_interp,
            n_nodes=ctx.n_nodes,
        )
        
        samples_denorm = ctx.data_stats.denormalize(samples.cpu().numpy())
        samples_denorm = np.clip(samples_denorm, 0, None)

    plots_dir = f"{ctx.plot_dir}/interpolation"
    os.makedirs(plots_dir, exist_ok=True)
    
    channel_positions = ctx.loader.channel_positions
    adj2d = build_xy_adjacency_radius(channel_positions, radius=cfg.graph.radius)
    
    fig, axes = plt.subplots(1, n_steps + 2, figsize=(3 * (n_steps + 2), 3))
    
    true1 = batch_np[0, :, 0].reshape(ctx.n_channels, ctx.n_time_points, order='F').sum(axis=1)
    true2 = batch_np[1, :, 0].reshape(ctx.n_channels, ctx.n_time_points, order='F').sum(axis=1)
    
    Gxy = Graph(adjacency=adj2d, positions_xy=channel_positions, positions_z=np.zeros(ctx.n_channels, dtype=np.float32))
    visualize_event(Gxy, true1, None, ax=axes[0])
    axes[0].set_title("Event A")
    
    for i in range(n_steps):
        interp_xy = samples_denorm[i, 0].reshape(ctx.n_channels, ctx.n_time_points, order='F').sum(axis=1)
        visualize_event(Gxy, interp_xy, None, ax=axes[i + 1])
        axes[i + 1].set_title(f"α={alphas[i].item():.2f}")
    
    visualize_event(Gxy, true2, None, ax=axes[-1])
    axes[-1].set_title("Event B")
    
    plt.tight_layout()
    fig.savefig(f"{plots_dir}/interpolation.png", dpi=150)
    plt.close(fig)
    print(f"Saved interpolation to {plots_dir}/interpolation.png")

    return samples_denorm


if __name__ == "__main__":
    train_ae(default_config)
