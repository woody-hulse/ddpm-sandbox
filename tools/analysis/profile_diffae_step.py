import argparse
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function

from config import get_config
from diffae import DiffAEContext
from diffusion.schedule import sinusoidal_embedding


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _elapsed(device: torch.device, fn):
    _sync(device)
    t0 = time.perf_counter()
    out = fn()
    _sync(device)
    return out, time.perf_counter() - t0


def build_cfg(args: argparse.Namespace):
    cfg = get_config()
    cfg.device = args.device
    cfg.resume = False
    cfg.visualize = False
    cfg.training.epochs = 1
    cfg.training.encode_dataset_every = 0
    cfg.training.batch_size = args.batch_size
    cfg.training.use_amp = not args.no_amp
    cfg.training.lr = args.lr
    cfg.model.hidden_dim = args.model_hidden_dim
    cfg.model.depth = args.model_depth
    cfg.model.blocks_per_stage = args.model_blocks_per_stage
    cfg.model.pool_ratio = args.model_pool_ratio
    cfg.encoder.hidden_dim = args.encoder_hidden_dim
    cfg.encoder.depth = args.encoder_depth
    cfg.encoder.blocks_per_stage = args.encoder_blocks_per_stage
    cfg.encoder.pool_ratio = args.encoder_pool_ratio
    cfg.encoder.latent_dim = args.latent_dim
    cfg.encoder.use_regressive_head = args.use_regressive_head
    cfg.encoder.regressive_head_weight = args.regressive_head_weight
    cfg.graph.radius = args.radius
    cfg.graph.z_hops = args.z_hops
    cfg.diffusion.timesteps = args.timesteps
    return cfg


def run_profile(args: argparse.Namespace) -> None:
    cfg = build_cfg(args)
    ctx = DiffAEContext.build(cfg, for_training=True, verbose=True, use_ms_data=True)
    encoder = ctx.encoder
    decoder = ctx.decoder
    regressive_decoder = ctx.regressive_decoder
    optim = ctx.optim
    tr = ctx.loader
    data_stats = ctx.data_stats
    A_sparse = ctx.A_sparse
    pos = ctx.pos
    lpe = ctx.lpe
    schedule = ctx.schedule
    device = ctx.device
    n_nodes = ctx.n_nodes
    B = cfg.training.batch_size
    use_regressive = cfg.encoder.use_regressive_head and regressive_decoder is not None

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    amp_dtype = torch.bfloat16 if cfg.training.use_amp and device.type == "cuda" else torch.float32
    coarse = defaultdict(float)

    def run_step(profile_enabled: bool) -> float:
        with record_function("load_batch"):
            (batch_np, _, _), dt = _elapsed(device, lambda: tr.get_batch(B))
        coarse["load_batch"] += dt

        with record_function("normalize_batch"):
            batch_np = data_stats.normalize(batch_np)
            x0 = torch.from_numpy(batch_np.astype("float32")).to(device)
            x0_flat = x0.view(B * n_nodes, 1)

        optim.zero_grad(set_to_none=True)

        def do_encode():
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=cfg.training.use_amp and device.type == "cuda"):
                return encoder(x0_flat, A_sparse, pos, batch_size=B, lpe=lpe)

        with record_function("encoder_forward"):
            (z, mu, logvar), dt = _elapsed(device, do_encode)
        coarse["encoder_forward"] += dt

        t = torch.randint(0, cfg.diffusion.timesteps, (B,), device=device, dtype=torch.long)
        t_emb = sinusoidal_embedding(t, cfg.conditioning.time_dim)
        cond_full = torch.cat([z, t_emb], dim=-1)
        sqrt_ab = schedule["sqrt_alphas_cumprod"][t].view(B, 1, 1)
        sqrt_om = schedule["sqrt_one_minus_alphas_cumprod"][t].view(B, 1, 1)
        snr_t = schedule["snr"][t].view(B)
        noise = torch.randn_like(x0)
        x_t = sqrt_ab * x0 + sqrt_om * noise
        x_t_flat = x_t.view(B * n_nodes, 1)

        def do_decode():
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=cfg.training.use_amp and device.type == "cuda"):
                pred_flat = decoder(x_t_flat, A_sparse, cond_full, pos, batch_size=B)
                pred = pred_flat.view(B, n_nodes, 1)
                target = sqrt_ab * noise - sqrt_om * x0
                mse_per_sample = F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2))
                if cfg.diffusion.p2_gamma > 0.0:
                    weight = torch.pow(cfg.diffusion.p2_k + snr_t, -cfg.diffusion.p2_gamma)
                    mse_per_sample = mse_per_sample * weight
                loss = mse_per_sample.mean()
                if cfg.encoder.use_stochastic and mu is not None and logvar is not None:
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = loss + cfg.encoder.kl_weight * kl_loss
                return loss

        with record_function("diffusion_decoder_forward"):
            loss, dt = _elapsed(device, do_decode)
        coarse["diffusion_decoder_forward"] += dt

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite diffusion loss during profile run.")

        with record_function("diffusion_backward"):
            _, dt = _elapsed(device, loss.backward)
        coarse["diffusion_backward"] += dt

        if use_regressive:
            def do_regressive():
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=cfg.training.use_amp and device.type == "cuda"):
                    z_reg, _, _ = encoder(x0_flat, A_sparse, pos, batch_size=B, lpe=lpe)
                    reg_flat = regressive_decoder(z_reg, A_sparse, pos, batch_size=B)
                    reg_pred = reg_flat.view(B, n_nodes, 1)
                    reg_loss = F.mse_loss(reg_pred, x0, reduction="mean")
                    return cfg.encoder.regressive_head_weight * reg_loss

            with record_function("regressive_forward"):
                reg_term, dt = _elapsed(device, do_regressive)
            coarse["regressive_forward"] += dt

            if not torch.isfinite(reg_term):
                raise RuntimeError("Non-finite regressive loss during profile run.")

            with record_function("regressive_backward"):
                _, dt = _elapsed(device, reg_term.backward)
            coarse["regressive_backward"] += dt

        with record_function("optimizer_step"):
            _, dt = _elapsed(device, optim.step)
        coarse["optimizer_step"] += dt
        return float(loss.item())

    for i in range(args.warmup):
        loss = run_step(profile_enabled=False)
        print(f"warmup {i+1}/{args.warmup}: loss={loss:.4f}")

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=args.profile_memory,
        with_stack=False,
    ) as prof:
        for i in range(args.steps):
            loss = run_step(profile_enabled=True)
            prof.step()
            print(f"profile step {i+1}/{args.steps}: loss={loss:.4f}")

    total_profiled = sum(
        coarse[k]
        for k in [
            "load_batch",
            "encoder_forward",
            "diffusion_decoder_forward",
            "diffusion_backward",
            "regressive_forward",
            "regressive_backward",
            "optimizer_step",
        ]
    )
    print("\nCoarse wall times")
    for name, dt in sorted(coarse.items(), key=lambda kv: kv[1], reverse=True):
        pct = 100.0 * dt / max(total_profiled, 1e-12)
        print(f"{name:28s} {dt:8.3f}s  {pct:5.1f}%")

    sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    print(f"\nTop ops by {sort_key}")
    print(prof.key_averages().table(sort_by=sort_key, row_limit=args.top_ops))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile one or a few DiffAE training steps.")
    p.add_argument("--device", default=None)
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--top-ops", type=int, default=25)
    p.add_argument("--profile-memory", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--use-regressive-head", action="store_true")
    p.add_argument("--regressive-head-weight", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--model-hidden-dim", type=int, default=32)
    p.add_argument("--model-depth", type=int, default=3)
    p.add_argument("--model-blocks-per-stage", type=int, default=2)
    p.add_argument("--model-pool-ratio", type=float, default=0.5)
    p.add_argument("--encoder-hidden-dim", type=int, default=32)
    p.add_argument("--encoder-depth", type=int, default=4)
    p.add_argument("--encoder-blocks-per-stage", type=int, default=2)
    p.add_argument("--encoder-pool-ratio", type=float, default=0.5)
    p.add_argument("--radius", type=float, default=12.0)
    p.add_argument("--z-hops", type=int, default=3)
    p.add_argument("--timesteps", type=int, default=250)
    return p.parse_args()


if __name__ == "__main__":
    run_profile(parse_args())
