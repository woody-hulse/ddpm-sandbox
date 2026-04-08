import torch


def sparse_multi_agg(
    adj: torch.Tensor,
    h: torch.Tensor,
    edge_emb: "torch.Tensor | None" = None,
) -> torch.Tensor:
    """
    Compute mean, max, and std neighborhood aggregations over a sparse adjacency.

    Optionally adds per-edge embeddings to source messages before aggregation,
    allowing directional / distance information to modulate each message.

    Runs in float32 internally for numerical stability, returns in h.dtype.

    Args:
        adj:      (N, N) sparse COO adjacency (block-diagonal for batched graphs)
        h:        (N, H) node features
        edge_emb: (E, H) optional per-edge embeddings added to source messages
    Returns:
        (N, 3*H) concatenation of [mean_agg, max_agg, std_agg]
    """
    adj = adj.coalesce()
    rows = adj.indices()[0]   # target nodes (aggregate INTO row i FROM col j)
    cols = adj.indices()[1]   # source nodes
    N, H = h.size()
    dtype = h.dtype
    h_f = h.float() if h.dtype != torch.float32 else h

    # Degree per target node
    deg = torch.zeros(N, 1, device=h.device, dtype=torch.float32)
    deg.scatter_add_(0, rows.unsqueeze(1), torch.ones(rows.size(0), 1, device=h.device))
    deg.clamp_(min=1.0)

    row_idx = rows.unsqueeze(1).expand(-1, H)   # (E, H)
    h_src = h_f[cols]                            # (E, H) source features

    # Add per-edge embeddings to source messages if provided
    if edge_emb is not None:
        h_src = h_src + (edge_emb.float() if edge_emb.dtype != torch.float32 else edge_emb)

    # Mean: sum / degree
    sum_agg = torch.zeros(N, H, device=h.device, dtype=torch.float32)
    sum_agg.scatter_add_(0, row_idx, h_src)
    mean_agg = sum_agg / deg

    # Max: amax over neighbors; isolated nodes default to 0
    max_agg = torch.full((N, H), float('-inf'), device=h.device, dtype=torch.float32)
    max_agg.scatter_reduce_(0, row_idx, h_src, reduce='amax', include_self=True)
    max_agg = torch.where(max_agg.isinf(), torch.zeros_like(max_agg), max_agg)

    # Std: sqrt(E[x²] - E[x]²)
    sq_agg = torch.zeros(N, H, device=h.device, dtype=torch.float32)
    sq_agg.scatter_add_(0, row_idx, h_src ** 2)
    std_agg = ((sq_agg / deg) - mean_agg ** 2).clamp_(min=0).sqrt_()

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
