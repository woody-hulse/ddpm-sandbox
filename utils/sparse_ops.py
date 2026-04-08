import torch

# Feature-dimension chunk size for scatter operations.
# At E=11.7M edges: each chunk uses E * _CHUNK_H * 4 bytes for h_chunk
# and E * _CHUNK_H * 8 bytes for the index → ~750 MB total at chunk=8.
_CHUNK_H = 8


def sparse_multi_agg(
    adj: torch.Tensor,
    h: torch.Tensor,
    edge_weight: "torch.Tensor | None" = None,
    edge_bias: "torch.Tensor | None" = None,
    pos_cur: "torch.Tensor | None" = None,
) -> torch.Tensor:
    """
    Compute mean, max, and std neighborhood aggregations over a sparse adjacency.

    Two paths depending on whether directional edge features are needed:

    • No edges (edge_weight is None or pos_cur is None):
        Uses torch.sparse.mm for mean + std — never materialises an (E, H) tensor.
        Max is computed via chunked scatter_reduce over _CHUNK_H features at a time.

    • With edges (edge_weight and pos_cur provided):
        Computes edge projections in _CHUNK_H-wide feature chunks to avoid
        ever holding the full (E, H) edge-embedding tensor in memory.
        edge_weight / edge_bias should be the .weight / .bias of the caller's
        nn.Linear(3, H) edge_proj layer.

    Runs in float32 internally for numerical stability, returns in h.dtype.

    Args:
        adj:         (N, N) sparse COO adjacency (block-diagonal for batched graphs)
        h:           (N, H) node features
        edge_weight: (H, 3) weight matrix of edge_proj Linear, or None
        edge_bias:   (H,)   bias of edge_proj Linear, or None
        pos_cur:     (N, 3) node positions for delta computation, or None
    Returns:
        (N, 3*H) concatenation of [mean_agg, max_agg, std_agg]
    """
    adj = adj.coalesce()
    rows = adj.indices()[0]
    cols = adj.indices()[1]
    N, H = h.size()
    dtype = h.dtype
    h_f = h.float() if h.dtype != torch.float32 else h

    # Degree per target node — (N, 1), used for mean and std
    deg = torch.zeros(N, 1, device=h.device, dtype=torch.float32)
    deg.scatter_add_(0, rows.unsqueeze(1), torch.ones(rows.size(0), 1, device=h.device))
    deg.clamp_(min=1.0)

    use_edges = (edge_weight is not None) and (pos_cur is not None)

    if not use_edges:
        # ── Memory-efficient path ──────────────────────────────────────────────
        # sparse.mm never materialises the (E, H) source-feature tensor.
        adj_f = adj.float() if adj.dtype != torch.float32 else adj
        sum_agg = torch.sparse.mm(adj_f, h_f)            # (N, H)
        mean_agg = sum_agg / deg

        sq_agg = torch.sparse.mm(adj_f, h_f ** 2)        # (N, H)  h²: (N,H), not (E,H)
        std_agg = ((sq_agg / deg) - mean_agg ** 2).clamp_(min=0).sqrt_()

        # Max: chunked scatter to bound peak memory to ~750 MB per chunk
        max_agg = torch.full((N, H), float('-inf'), device=h.device, dtype=torch.float32)
        for s in range(0, H, _CHUNK_H):
            e = min(s + _CHUNK_H, H)
            chunk = h_f[cols, s:e]                        # (E, cs)
            idx = rows.unsqueeze(1).expand(-1, e - s)     # (E, cs)
            max_agg[:, s:e].scatter_reduce_(0, idx, chunk, reduce='amax', include_self=True)
        max_agg = torch.where(max_agg.isinf(), torch.zeros_like(max_agg), max_agg)

    else:
        # ── Edge-modulated path (chunked) ─────────────────────────────────────
        # Compute (pos[cols] - pos[rows]) once: (E, 3) — tiny (~140 MB at E=11.7M)
        delta = (pos_cur[cols] - pos_cur[rows]).float()   # (E, 3)
        ew = edge_weight.float()                           # (H, 3)

        mean_agg = torch.zeros(N, H, device=h.device, dtype=torch.float32)
        max_agg  = torch.full((N, H), float('-inf'), device=h.device, dtype=torch.float32)
        sq_sum   = torch.zeros(N, H, device=h.device, dtype=torch.float32)

        for s in range(0, H, _CHUNK_H):
            e = min(s + _CHUNK_H, H)
            # Edge projection for this chunk: (E, cs) — never holds full (E, H)
            e_chunk = delta @ ew[s:e].T                   # (E, cs)
            if edge_bias is not None:
                e_chunk = e_chunk + edge_bias[s:e].float()
            h_chunk = h_f[cols, s:e] + e_chunk            # (E, cs)
            idx = rows.unsqueeze(1).expand(-1, e - s)     # (E, cs)
            mean_agg[:, s:e].scatter_add_(0, idx, h_chunk)
            max_agg[:, s:e].scatter_reduce_(0, idx, h_chunk, reduce='amax', include_self=True)
            sq_sum[:, s:e].scatter_add_(0, idx, h_chunk ** 2)

        mean_agg = mean_agg / deg
        max_agg = torch.where(max_agg.isinf(), torch.zeros_like(max_agg), max_agg)
        std_agg = ((sq_sum / deg) - mean_agg ** 2).clamp_(min=0).sqrt_()

    return torch.cat([mean_agg, max_agg, std_agg], dim=-1).to(dtype)


def to_coalesced_coo(adj: torch.Tensor) -> torch.Tensor:
    if adj.layout == torch.sparse_csr:
        adj = adj.to_sparse_coo()
    assert adj.is_sparse, "adj must be a torch sparse tensor (COO or CSR)."
    if not adj.is_coalesced():
        adj = adj.coalesce()
    return adj


def to_binary(adj: torch.Tensor) -> torch.Tensor:
    """Convert adjacency to binary (all edge values = 1), no self-loops added."""
    adj = to_coalesced_coo(adj)
    indices = adj.indices()
    N = adj.size(0)
    
    # Remove any existing self-loops for clean GIN aggregation
    row, col = indices[0], indices[1]
    non_diag = row != col
    indices = indices[:, non_diag]
    
    values = torch.ones(indices.size(1), device=adj.device, dtype=adj.dtype)
    
    adj_binary = torch.sparse_coo_tensor(
        indices, values, size=(N, N), device=adj.device, dtype=adj.dtype
    ).coalesce()
    return adj_binary


def add_self_loops_binary(adj: torch.Tensor) -> torch.Tensor:
    """Add self-loops to adjacency and ensure all values are 1 (binary)."""
    adj = to_coalesced_coo(adj)
    indices = adj.indices()
    N = adj.size(0)
    
    row, col = indices[0], indices[1]
    is_diag = (row == col)
    mask_diag_present = torch.zeros(N, dtype=torch.bool, device=indices.device)
    if is_diag.any():
        mask_diag_present[row[is_diag]] = True
    
    missing = (~mask_diag_present).nonzero(as_tuple=False).flatten()
    if missing.numel() > 0:
        add_idx = torch.stack([missing, missing], dim=0)
        indices = torch.cat([indices, add_idx], dim=1)
    
    values = torch.ones(indices.size(1), device=adj.device, dtype=adj.dtype)
    
    adj_binary = torch.sparse_coo_tensor(
        indices, values, size=(N, N), device=adj.device, dtype=adj.dtype
    ).coalesce()
    return adj_binary


def gcn_norm(adj: torch.Tensor, add_self_loops: bool = True) -> torch.Tensor:
    adj = to_coalesced_coo(adj)
    indices = adj.indices()
    values = adj.values()
    N = adj.size(0)

    if add_self_loops:
        row, col = indices[0], indices[1]
        diag_idx = torch.arange(N, device=indices.device, dtype=torch.long)
        is_diag = (row == col)
        mask_diag_present = torch.zeros(N, dtype=torch.bool, device=indices.device)
        if is_diag.any():
            mask_diag_present[row[is_diag]] = True

        missing = (~mask_diag_present).nonzero(as_tuple=False).flatten()
        if missing.numel() > 0:
            add_idx = torch.stack([missing, missing], dim=0)
            add_val = torch.ones(missing.numel(), device=values.device, dtype=values.dtype)
            indices = torch.cat([indices, add_idx], dim=1)
            values = torch.cat([values, add_val], dim=0)

        adj = torch.sparse_coo_tensor(indices, values, size=(N, N), device=adj.device, dtype=values.dtype).coalesce()

        idx = adj.indices()
        vals = adj.values()
        diag_mask = (idx[0] == idx[1])
        if diag_mask.any():
            vals = vals.clone()
            vals[diag_mask] = 1.0
            adj = torch.sparse_coo_tensor(idx, vals, size=(N, N), device=adj.device, dtype=vals.dtype).coalesce()

    idx = adj.indices()
    vals = adj.values()
    row, col = idx[0], idx[1]

    deg = torch.zeros(N, device=vals.device, dtype=vals.dtype)
    deg.index_add_(0, row, vals)
    deg_inv_sqrt = torch.pow(torch.clamp_min(deg, 1e-12), -0.5)

    norm_vals = deg_inv_sqrt[row] * vals * deg_inv_sqrt[col]
    norm_adj = torch.sparse_coo_tensor(idx, norm_vals, size=(N, N), device=adj.device, dtype=vals.dtype)
    return norm_adj.coalesce()


def subgraph_coo(adj_hat: torch.Tensor, keep_idx: torch.Tensor, newN: int) -> torch.Tensor:
    adj_hat = to_coalesced_coo(adj_hat)
    oldN = adj_hat.size(0)
    assert keep_idx.numel() == newN
    map_new = torch.full((oldN,), -1, device=keep_idx.device, dtype=torch.int64)
    map_new[keep_idx] = torch.arange(newN, device=keep_idx.device, dtype=torch.int64)

    idx = adj_hat.indices()
    vals = adj_hat.values()
    r, c = idx[0], idx[1]
    r_new = map_new[r]
    c_new = map_new[c]
    mask = (r_new >= 0) & (c_new >= 0)
    
    if mask.sum() == 0:
        eye_idx = torch.arange(newN, device=keep_idx.device)
        return torch.sparse_coo_tensor(
            torch.stack([eye_idx, eye_idx], dim=0),
            torch.ones(newN, device=vals.device, dtype=vals.dtype),
            size=(newN, newN),
            device=adj_hat.device
        ).coalesce()
    
    r_new = r_new[mask]
    c_new = c_new[mask]
    vals = vals[mask]
    sub = torch.sparse_coo_tensor(
        torch.stack([r_new, c_new], dim=0), vals, size=(newN, newN), device=adj_hat.device, dtype=vals.dtype
    ).coalesce()
    return sub
