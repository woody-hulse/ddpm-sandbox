import torch


def to_coalesced_coo(adj: torch.Tensor) -> torch.Tensor:
    if adj.layout == torch.sparse_csr:
        adj = adj.to_sparse_coo()
    assert adj.is_sparse, "adj must be a torch sparse tensor (COO or CSR)."
    if not adj.is_coalesced():
        adj = adj.coalesce()
    return adj


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


@torch.no_grad()
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
