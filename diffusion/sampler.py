import torch
import torch.nn as nn
from typing import Tuple

from .schedule import sinusoidal_embedding


@torch.no_grad()
def sample_ddpm(
    core: nn.Module,
    schedule: dict,
    A_sparse: torch.Tensor,
    cond_proj: torch.Tensor,
    pos: torch.Tensor,
    time_dim: int,
    shape: Tuple[int, int, int],
    parametrization: str = 'v',
) -> torch.Tensor:
    """
    DDPM sampling loop for graph diffusion.
    
    Args:
        core: GraphDDPMUNet model
        schedule: Dictionary from build_cosine_schedule
        A_sparse: Sparse adjacency tensor (N, N)
        cond_proj: Projected conditioning (B, cond_proj_dim)
        pos: Node positions (N, pos_dim)
        time_dim: Dimension for time embedding
        shape: Output shape (B, C, N)
        parametrization: 'eps' or 'v'
    
    Returns:
        Sampled tensor of shape (B, C, N)
    """
    B, C, N = shape
    device = A_sparse.device
    dtype = A_sparse.dtype if A_sparse.dtype.is_floating_point else torch.float32
    
    x = torch.randn(B, N, C, device=device, dtype=dtype)
    T = len(schedule['betas'])
    t_tensor = torch.arange(T, device=device, dtype=torch.long)
    
    for step in reversed(range(T)):
        t = step
        alpha_bar = schedule['alphas_cumprod'][t]
        alpha_bar_prev = schedule['alphas_cumprod_prev'][t]
        alpha = schedule['alphas'][t]
        beta = schedule['betas'][t]
        sqrt_ab = schedule['sqrt_alphas_cumprod'][t]
        sqrt_om = schedule['sqrt_one_minus_alphas_cumprod'][t]
        
        for b in range(B):
            xb = x[b]
            t_emb = sinusoidal_embedding(t_tensor[t:t+1], time_dim).squeeze(0)
            cond_full_b = torch.cat([cond_proj[b], t_emb], dim=-1)
            pred_b = core(xb, A_sparse, cond_full_b, pos)
            
            if parametrization == 'eps':
                eps_theta = pred_b
                x0_pred = (xb - sqrt_om * eps_theta) / torch.clamp(sqrt_ab, min=1e-8)
            elif parametrization == 'v':
                v_theta = pred_b
                x0_pred = sqrt_ab * xb - sqrt_om * v_theta
                eps_theta = sqrt_om * xb + sqrt_ab * v_theta
            else:
                raise ValueError("parametrization must be 'eps' or 'v'")
            
            # Compute posterior mean
            coef1 = beta * torch.sqrt(alpha_bar_prev) / (1.0 - alpha_bar + 1e-8)
            coef2 = (1.0 - alpha_bar_prev) * torch.sqrt(alpha) / (1.0 - alpha_bar + 1e-8)
            mean = coef1 * x0_pred + coef2 * xb
            
            if step > 0:
                var = schedule['posterior_variance'][t]
                noise = torch.randn_like(xb)
                x[b] = mean + torch.sqrt(var) * noise
            else:
                x[b] = mean
    
    return x.permute(0, 2, 1)
