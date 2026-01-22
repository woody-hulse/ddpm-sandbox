import math
import torch
import torch.nn.functional as F


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    device = t.device
    half = dim // 2
    freq_exp = -(math.log(10000.0) / max(half - 1, 1))
    freqs = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * freq_exp)
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def build_cosine_schedule(timesteps: int, device: torch.device) -> dict:
    s = 0.008
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float32, device=device)
    f = torch.cos(((t / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
    f = f / torch.clamp_min(f[0], 1e-12)
    alpha_t = torch.clamp(f[1:] / torch.clamp_min(f[:-1], 1e-12), 1e-8, 0.9999)
    betas = torch.clamp(1.0 - alpha_t, 1e-8, 0.9999)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat(
        [torch.ones(1, dtype=torch.float32, device=device), alphas_cumprod[:-1]], dim=0
    )

    one_minus_ab = torch.clamp(1.0 - alphas_cumprod, min=1e-12)
    sqrt_ab = torch.sqrt(torch.clamp_min(alphas_cumprod, 1e-12))
    sqrt_one_minus_ab = torch.sqrt(one_minus_ab)
    posterior_variance = betas * torch.clamp(1.0 - alphas_cumprod_prev, min=1e-12) / one_minus_ab
    posterior_variance = torch.clamp(posterior_variance, min=1e-20)

    snr = alphas_cumprod / one_minus_ab

    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'alphas_cumprod_prev': alphas_cumprod_prev,
        'sqrt_alphas_cumprod': sqrt_ab,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_ab,
        'sqrt_recip_alphas': torch.sqrt(torch.clamp(1.0 / alphas, min=1e-12)),
        'posterior_variance': posterior_variance,
        'snr': snr,
    }
