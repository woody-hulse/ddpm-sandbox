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
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        film = self.mlp(cond)
        film = film[0].view(self.num_layers, 2, self.hidden_dim)
        gamma, beta = film[:, 0, :], film[:, 1, :]
        gamma = 1.0 + gamma
        return gamma, beta


class GraphResBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.conv = SparseGraphConv(hidden_dim, hidden_dim, bias=True)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_hat: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.act(h)
        h = self.conv(h, adj_hat)
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
        nn.init.constant_(self.scorer[-1].bias, 0.0)
        nn.init.zeros_(self.scorer[-1].weight)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        N = x.size(0)
        k = max(1, int(math.ceil(self.ratio * N)))
        s = self.scorer(self.norm(x)).squeeze(-1)
        keep_idx = torch.topk(s, k=k, largest=True, sorted=True).indices
        x_pool = x[keep_idx]
        return x_pool, keep_idx


def _unpool_like(x_small: torch.Tensor, keep_idx: torch.Tensor, N: int) -> torch.Tensor:
    H = x_small.size(1)
    x_big = x_small.new_zeros((N, H))
    x_big[keep_idx] = x_small
    return x_big


class GraphUNetStage(nn.Module):
    def __init__(self, hidden_dim: int, blocks_per_stage: int = 1, dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList([GraphResBlock(hidden_dim, dropout=dropout)
                                     for _ in range(blocks_per_stage)])

    def forward(self, x: torch.Tensor, adj_hat: torch.Tensor,
                gammas: torch.Tensor, betas: torch.Tensor, gamma_offset: int) -> Tuple[torch.Tensor, int]:
        i = gamma_offset
        for b in self.blocks:
            g = gammas[i].unsqueeze(0)
            bt = betas[i].unsqueeze(0)
            x = b(x, adj_hat, g, bt)
            i += 1
        return x, i


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

        self.total_stages = 2 * depth + 1
        self.total_blocks = self.total_stages * blocks_per_stage

        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.enc_stages = nn.ModuleList(
            [GraphUNetStage(hidden_dim, blocks_per_stage, dropout=dropout) for _ in range(depth)]
        )
        self.pools = nn.ModuleList([TopKPool(hidden_dim, ratio=pool_ratio) for _ in range(depth)])

        self.bottleneck = GraphUNetStage(hidden_dim, blocks_per_stage, dropout=dropout)

        self.dec_stages = nn.ModuleList(
            [GraphUNetStage(hidden_dim, blocks_per_stage, dropout=dropout) for _ in range(depth)]
        )

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, self.out_dim)

        self.film = FiLMFromCond(cond_dim, hidden_dim, num_layers=self.total_blocks, width=max(512, 2 * hidden_dim))

        self.cache_norm_top = cache_norm_top
        self._cached_key = None
        self._cached_adj = None

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
        nn.init.zeros_(self.out_proj.weight)
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

    def forward(self, x0: torch.Tensor, adj: torch.Tensor, cond: torch.Tensor, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert x0.dim() == 2 and x0.size(1) == self.in_dim
        if pos is None:
            raise ValueError("per-node positions `pos` (N, pos_dim) are required but got None")
        if pos.dim() != 2 or pos.size(0) != x0.size(0) or pos.size(1) != self.pos_dim:
            raise ValueError(f"`pos` must be shape (N,{self.pos_dim}), got {tuple(pos.shape)}")

        adj0 = self._maybe_norm_top(adj).to(device=x0.device, dtype=x0.dtype)

        gammas, betas = self.film(cond)
        g_ptr = 0

        h = self.in_proj(x0)
        pos_emb = self.pos_mlp(pos.to(x0.dtype))
        h = h + self.pos_drop(pos_emb)

        N0 = h.size(0)

        skips: List[Tuple[torch.Tensor, torch.Tensor, int]] = []
        adjs: List[torch.Tensor] = []
        h_cur, adj_cur, N_cur = h, adj0, N0

        for d in range(self.depth):
            h_cur, g_ptr = self.enc_stages[d](h_cur, adj_cur, gammas, betas, g_ptr)
            h_skip = h_cur
            h_pool, keep_idx = self.pools[d](h_cur)
            K = h_pool.size(0)
            adj_next = gcn_norm(subgraph_coo(adj_cur, keep_idx, K), add_self_loops=True)
            skips.append((h_skip, keep_idx, N_cur))
            adjs.append(adj_cur)
            h_cur, adj_cur, N_cur = h_pool, adj_next, K

        h_cur, g_ptr = self.bottleneck(h_cur, adj_cur, gammas, betas, g_ptr)

        for d in reversed(range(self.depth)):
            h_skip, keep_idx, N_prev = skips[d]
            h_up = _unpool_like(h_cur, keep_idx, N_prev)
            h_cur = h_up + h_skip
            adj_prev = adjs[d]
            h_cur, g_ptr = self.dec_stages[d](h_cur, adj_prev, gammas, betas, g_ptr)

        h_cur = F.silu(self.out_norm(h_cur))
        y = self.out_proj(h_cur)
        return y
