"""Quick test to verify gradient flow through the model."""
import torch
import torch.nn as nn
from models.graph_unet import GraphDDPMUNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create model with same config as training
model = GraphDDPMUNet(
    in_dim=1,
    cond_dim=128,  # cond_proj_dim + time_dim = 64 + 64
    hidden_dim=128,
    depth=2,        # Reduced depth
    blocks_per_stage=2,
    pool_ratio=0.7, # Less aggressive pooling
).to(device)

# Create dummy data  
B, N = 4, 5000  # Smaller for faster testing
x = torch.randn(B * N, 1, device=device)

# Create adjacency with actual edges (k-nearest neighbors style)
# Each node connects to itself and ~10 neighbors
rows, cols = [torch.arange(N, device=device)], [torch.arange(N, device=device)]  # Self loops
for offset in [1, 2, 3, 4, 5, -1, -2, -3, -4, -5]:  # Connect to nearby nodes
    valid = (torch.arange(N, device=device) + offset >= 0) & (torch.arange(N, device=device) + offset < N)
    rows.append(torch.arange(N, device=device)[valid])
    cols.append((torch.arange(N, device=device) + offset)[valid])
rows = torch.cat(rows)
cols = torch.cat(cols)
adj = torch.sparse_coo_tensor(
    torch.stack([rows, cols]),
    torch.ones(rows.shape[0], device=device),
    size=(N, N)
).coalesce()

cond = torch.randn(B, 128, device=device)
pos = torch.randn(N, 3, device=device)

print(f"Input x shape: {x.shape}")
print(f"Adjacency shape: {adj.shape}, nnz: {adj._nnz()}")
print(f"Cond shape: {cond.shape}")
print(f"Pos shape: {pos.shape}")

# Forward pass
model.train()
out = model(x, adj, cond, pos, batch_size=B)
print(f"\nOutput shape: {out.shape}")
print(f"Output requires_grad: {out.requires_grad}")
print(f"Output mean: {out.mean().item():.4f}, std: {out.std().item():.4f}")

# Compute loss
target = torch.randn_like(out)
loss = nn.functional.mse_loss(out, target)
print(f"\nLoss: {loss.item():.4f}")
print(f"Loss requires_grad: {loss.requires_grad}")

# Backward pass
loss.backward()

# Check gradients
print("\n=== Gradient Check ===")
print(f"out_proj.weight.grad norm: {model.out_proj.weight.grad.norm().item() if model.out_proj.weight.grad is not None else 'None'}")
print(f"in_proj.weight.grad norm: {model.in_proj.weight.grad.norm().item() if model.in_proj.weight.grad is not None else 'None'}")

# Check conv layer gradients
for i, stage in enumerate(model.enc_stages):
    for j, blk in enumerate(stage.blocks):
        grad1 = blk.lin1.weight.grad
        grad2 = blk.lin2.weight.grad
        norm1 = grad1.norm().item() if grad1 is not None else 0.0
        norm2 = grad2.norm().item() if grad2 is not None else 0.0
        print(f"enc_stage[{i}].block[{j}].lin1 grad: {norm1:.6f}, lin2 grad: {norm2:.6f}")

# Summary
n_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.norm() > 0)
n_total = sum(1 for p in model.parameters())
total_grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
print(f"\nParams with non-zero grads: {n_with_grad}/{n_total}")
print(f"Total gradient norm: {total_grad_norm:.4f}")

if n_with_grad == 0:
    print("\n*** ERROR: No gradients flowing! ***")
elif n_with_grad < n_total:
    print(f"\n*** WARNING: Only {n_with_grad}/{n_total} params have gradients ***")
else:
    print("\n*** SUCCESS: All parameters have gradients ***")
