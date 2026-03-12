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

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ae import GraphAEEncoder
from models.graph_unet import _unpool_like, build_block_diagonal_adj
from utils.sparse_ops import subgraph_coo, to_binary, to_coalesced_coo


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
