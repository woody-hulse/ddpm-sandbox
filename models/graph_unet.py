import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.sparse_ops import gcn_norm, to_coalesced_coo, subgraph_coo


class SparseGraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor, adj_hat: torch.Tensor) -> torch.Tensor:
        z = torch.sparse.mm(adj_hat, x)
        return self.lin(z)


class FiLMFromCond(nn.Module):
    def __init__(self, cond_dim: int, hidden_dim: int, num_layers: int, width: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        out_dim = num_layers * 2 * hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, out_dim),
        )
        nn.init.xavier_uniform_(self.mlp[-1].weight, gain=0.01)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, cond: torch.Tensor, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        film = self.mlp(cond)  # (B, num_layers * 2 * hidden_dim)
        film = film.view(batch_size, self.num_layers, 2, self.hidden_dim)
        gamma, beta = film[:, :, 0, :], film[:, :, 1, :]  # (B, num_layers, hidden_dim)
        gamma = 1.0 + gamma
        return gamma, beta


class GraphResBlock(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.conv = SparseGraphConv(hidden_dim, hidden_dim, bias=True)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        nn.init.xavier_uniform_(self.cond_proj.weight, gain=0.1)
        nn.init.zeros_(self.cond_proj.bias)

    def forward(self, x: torch.Tensor, adj_hat: torch.Tensor, cond: torch.Tensor,
                gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        cond_signal = self.cond_proj(cond)
        h = h + cond_signal
        h = self.act(h)
        h = self.conv(h, adj_hat)
        h = self.norm2(h)
        h = h * gamma + beta
        h = self.dropout(h)
        return x + h


class TopKPool(nn.Module):
    def __init__(self, hidden_dim: int, ratio: float = 0.5):
        super().__init__()
        assert 0.0 < ratio <= 1.0
        self.ratio = ratio
        self.norm = nn.LayerNorm(hidden_dim)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.constant_(self.scorer[-1].bias, 1.0)
        nn.init.xavier_uniform_(self.scorer[-1].weight)

    def forward(self, x: torch.Tensor, nodes_per_graph: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        total_nodes = x.size(0)
        batch_size = total_nodes // nodes_per_graph
        k_per_graph = max(1, int(math.ceil(self.ratio * nodes_per_graph)))
        
        s = self.scorer(self.norm(x)).squeeze(-1)  # (total_nodes,)
        s = s.view(batch_size, nodes_per_graph)
        
        topk_local = torch.topk(s, k=k_per_graph, dim=1, largest=True, sorted=True)
        local_indices = topk_local.indices  # (B, k_per_graph)
        
        batch_offsets = torch.arange(batch_size, device=x.device).unsqueeze(1) * nodes_per_graph
        global_indices = (local_indices + batch_offsets).view(-1)
        
        x_pool = x[global_indices]
        return x_pool, global_indices, k_per_graph


def _unpool_like(x_small: torch.Tensor, keep_idx: torch.Tensor, N: int) -> torch.Tensor:
    H = x_small.size(1)
    x_big = x_small.new_zeros((N, H))
    x_big[keep_idx] = x_small
    return x_big


class GraphUNetStage(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int, blocks_per_stage: int = 1, dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList([GraphResBlock(hidden_dim, cond_dim=cond_dim, dropout=dropout)
                                     for _ in range(blocks_per_stage)])

    def forward(self, x: torch.Tensor, adj_hat: torch.Tensor, cond_expanded: torch.Tensor,
                gammas: torch.Tensor, betas: torch.Tensor, gamma_offset: int,
                batch_size: int) -> Tuple[torch.Tensor, int]:
        i = gamma_offset
        for blk in self.blocks:
            g = gammas[:, i, :]  # (B, hidden_dim)
            bt = betas[:, i, :]  # (B, hidden_dim)
            nodes_per_graph = x.size(0) // batch_size
            g_expanded = g.repeat_interleave(nodes_per_graph, dim=0)  # (B*N, hidden_dim)
            bt_expanded = bt.repeat_interleave(nodes_per_graph, dim=0)
            x = blk(x, adj_hat, cond_expanded, g_expanded, bt_expanded)
            i += 1
        return x, i


def build_block_diagonal_adj(adj: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Create block-diagonal adjacency for batched graph processing."""
    adj = to_coalesced_coo(adj)
    N = adj.size(0)
    indices = adj.indices()
    values = adj.values()
    
    all_indices = []
    all_values = []
    for b in range(batch_size):
        offset = b * N
        shifted_indices = indices + offset
        all_indices.append(shifted_indices)
        all_values.append(values)
    
    final_indices = torch.cat(all_indices, dim=1)
    final_values = torch.cat(all_values, dim=0)
    
    block_adj = torch.sparse_coo_tensor(
        final_indices, final_values, 
        size=(batch_size * N, batch_size * N),
        device=adj.device, dtype=adj.dtype
    ).coalesce()
    return block_adj


class GraphDDPMUNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        cond_dim: int,
        hidden_dim: int = 256,
        depth: int = 3,
        blocks_per_stage: int = 1,
        pool_ratio: float = 0.5,
        out_dim: Optional[int] = None,
        dropout: float = 0.0,
        cache_norm_top: bool = True,
        pos_dim: int = 3,
        pos_dropout: float = 0.0,
    ):
        super().__init__()
        assert 0 < pool_ratio <= 1.0
        self.in_dim = in_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.blocks_per_stage = blocks_per_stage
        self.pool_ratio = pool_ratio
        self.out_dim = in_dim if out_dim is None else out_dim

        self.pos_dim = int(pos_dim)
        self.pos_mlp = nn.Sequential(
            nn.Linear(self.pos_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pos_drop = nn.Dropout(pos_dropout)
        
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.total_stages = 2 * depth + 1
        self.total_blocks = self.total_stages * blocks_per_stage

        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.enc_stages = nn.ModuleList(
            [GraphUNetStage(hidden_dim, cond_dim=cond_dim, blocks_per_stage=blocks_per_stage, dropout=dropout) for _ in range(depth)]
        )
        self.pools = nn.ModuleList([TopKPool(hidden_dim, ratio=pool_ratio) for _ in range(depth)])

        self.bottleneck = GraphUNetStage(hidden_dim, cond_dim=cond_dim, blocks_per_stage=blocks_per_stage, dropout=dropout)

        self.dec_stages = nn.ModuleList(
            [GraphUNetStage(hidden_dim, cond_dim=cond_dim, blocks_per_stage=blocks_per_stage, dropout=dropout) for _ in range(depth)]
        )

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, self.out_dim)

        self.film = FiLMFromCond(cond_dim, hidden_dim, num_layers=self.total_blocks, width=max(512, 2 * hidden_dim))

        self.cache_norm_top = cache_norm_top
        self._cached_key = None
        self._cached_adj = None
        self._cached_block_adj = {}

        self.reset_parameters()

    def reset_parameters(self):
        def init_linear(m: nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(m.bias, -bound, bound)
        init_linear(self.in_proj)
        for st in list(self.enc_stages) + [self.bottleneck] + list(self.dec_stages):
            for b in st.blocks:
                init_linear(b.conv.lin)
        for m in self.pos_mlp:
            if isinstance(m, nn.Linear):
                init_linear(m)
        for m in self.cond_mlp:
            if isinstance(m, nn.Linear):
                init_linear(m)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.01)
        nn.init.zeros_(self.out_proj.bias)

    def _maybe_norm_top(self, adj: torch.Tensor) -> torch.Tensor:
        adj = to_coalesced_coo(adj)
        key = (adj.device, adj.size(), adj._nnz())
        if self.cache_norm_top and self._cached_key == key and self._cached_adj is not None:
            return self._cached_adj
        adj_hat = gcn_norm(adj, add_self_loops=True)
        if self.cache_norm_top:
            self._cached_key = key
            self._cached_adj = adj_hat
        return adj_hat

    def _get_block_adj(self, adj: torch.Tensor, batch_size: int) -> torch.Tensor:
        key = (adj.device, adj.size(), adj._nnz(), batch_size)
        if key in self._cached_block_adj:
            return self._cached_block_adj[key]
        adj_normed = self._maybe_norm_top(adj)
        block_adj = build_block_diagonal_adj(adj_normed, batch_size)
        self._cached_block_adj[key] = block_adj
        return block_adj

    def forward(self, x0: torch.Tensor, adj: torch.Tensor, cond: torch.Tensor, 
                pos: Optional[torch.Tensor] = None, batch_size: int = 1) -> torch.Tensor:
        """
        Forward pass with batched graph inputs.
        
        Args:
            x0: Node features (B*N, in_dim) stacked for all graphs
            adj: Single graph adjacency (N, N) - will be block-diagonalized
            cond: Conditioning vectors (B, cond_dim)
            pos: Node positions (N, pos_dim) for single graph - will be tiled
            batch_size: Number of graphs in batch
        """
        N_single = adj.size(0)
        total_nodes = x0.size(0)
        
        assert x0.dim() == 2 and x0.size(1) == self.in_dim
        assert total_nodes == batch_size * N_single, f"Expected {batch_size * N_single} nodes, got {total_nodes}"
        
        if pos is None:
            raise ValueError("per-node positions `pos` (N, pos_dim) are required but got None")
        if pos.dim() != 2 or pos.size(0) != N_single or pos.size(1) != self.pos_dim:
            raise ValueError(f"`pos` must be shape ({N_single},{self.pos_dim}), got {tuple(pos.shape)}")

        adj0 = self._get_block_adj(adj, batch_size).to(device=x0.device, dtype=x0.dtype)

        gammas, betas = self.film(cond, batch_size=batch_size)  # (B, num_layers, hidden_dim)
        g_ptr = 0

        h = self.in_proj(x0)
        
        pos_tiled = pos.repeat(batch_size, 1)
        pos_emb = self.pos_mlp(pos_tiled.to(x0.dtype))
        h = h + self.pos_drop(pos_emb)
        
        cond_emb = self.cond_mlp(cond)  # (B, hidden_dim)
        cond_expanded = cond.repeat_interleave(N_single, dim=0)  # (B*N, cond_dim)
        cond_emb_expanded = cond_emb.repeat_interleave(N_single, dim=0)  # (B*N, hidden_dim)
        h = h + cond_emb_expanded

        nodes_per_graph = N_single

        skips: List[Tuple[torch.Tensor, torch.Tensor, int, int]] = []
        adjs: List[torch.Tensor] = []
        h_cur, adj_cur = h, adj0

        for d in range(self.depth):
            cond_exp_cur = cond.repeat_interleave(nodes_per_graph, dim=0)
            h_cur, g_ptr = self.enc_stages[d](h_cur, adj_cur, cond_exp_cur, gammas, betas, g_ptr, batch_size)
            h_skip = h_cur
            h_pool, keep_idx, k_per_graph = self.pools[d](h_cur, nodes_per_graph)
            K = h_pool.size(0)
            adj_next = gcn_norm(subgraph_coo(adj_cur, keep_idx, K), add_self_loops=True)
            skips.append((h_skip, keep_idx, h_cur.size(0), nodes_per_graph))
            adjs.append(adj_cur)
            h_cur, adj_cur, nodes_per_graph = h_pool, adj_next, k_per_graph

        cond_exp_cur = cond.repeat_interleave(nodes_per_graph, dim=0)
        h_cur, g_ptr = self.bottleneck(h_cur, adj_cur, cond_exp_cur, gammas, betas, g_ptr, batch_size)

        for d in reversed(range(self.depth)):
            h_skip, keep_idx, N_prev, npg_prev = skips[d]
            h_up = _unpool_like(h_cur, keep_idx, N_prev)
            h_cur = h_up + h_skip
            adj_prev = adjs[d]
            nodes_per_graph = npg_prev
            cond_exp_cur = cond.repeat_interleave(nodes_per_graph, dim=0)
            h_cur, g_ptr = self.dec_stages[d](h_cur, adj_prev, cond_exp_cur, gammas, betas, g_ptr, batch_size)

        h_cur = F.silu(self.out_norm(h_cur))
        y = self.out_proj(h_cur)
        return y
