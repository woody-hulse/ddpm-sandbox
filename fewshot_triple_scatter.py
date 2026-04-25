#!/usr/bin/env python3
"""
Few-shot triple-scatter fine-tuning experiment for DiffAE Light.

This script is intentionally self-contained. It does not modify the main
training codepaths, and it writes all experiment artifacts to a dedicated
output directory.

Experiment flow:
1. Load a pretrained DiffAE Light checkpoint.
2. Build a fixed triple-scatter train/held-out split by co-adding three
   single-scatter events with the same random time-shift regime used for
   online double-scatter generation, plus one additional co-added scatter.
3. Evaluate held-out triple-scatter reconstructions before fine-tuning.
4. Fine-tune for a single pass over 8k triple-scatter events.
5. Re-evaluate on 2k held-out triple-scatter events and save metrics/plots.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.path.join(os.environ["XDG_CACHE_HOME"], "fontconfig"), exist_ok=True)

import matplotlib
from matplotlib import pyplot as plt
from tqdm import tqdm

matplotlib.use("Agg")

from config import Config, get_config
from diffae_light import (
    DiffAELightContext,
    _ema_update,
    _encode_with_context,
    sample_diffae_light,
)
from diffusion.schedule import sinusoidal_embedding
from lz_data_loader import TritiumSSDataLoader, shift_waveform_2d


@dataclass
class TripleScatterSpecs:
    idx1: np.ndarray
    idx2: np.ndarray
    idx3: np.ndarray
    delta2: np.ndarray
    delta3: np.ndarray

    def __len__(self) -> int:
        return int(self.idx1.shape[0])

    def subset(self, indices: np.ndarray) -> "TripleScatterSpecs":
        return TripleScatterSpecs(
            idx1=self.idx1[indices],
            idx2=self.idx2[indices],
            idx3=self.idx3[indices],
            delta2=self.delta2[indices],
            delta3=self.delta3[indices],
        )

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            idx1=self.idx1,
            idx2=self.idx2,
            idx3=self.idx3,
            delta2=self.delta2,
            delta3=self.delta3,
        )


class TripleScatterBatcher:
    """Fixed triple-scatter split backed by SS events on disk."""

    def __init__(
        self,
        h5_file_path: str,
        channel_positions_path: str,
        specs: TripleScatterSpecs,
        ns_per_bin: float,
    ):
        self.h5_file_path = h5_file_path
        self.channel_positions_path = channel_positions_path
        self.specs = specs
        self.ns_per_bin = float(ns_per_bin)

        self.ss_loader = TritiumSSDataLoader(h5_file_path, channel_positions_path)
        self.n_samples = len(specs)
        self.n_channels = self.ss_loader.n_channels
        self.n_time_points = self.ss_loader.n_time_points
        self.channel_positions = self.ss_loader.channel_positions

    def iter_indices(
        self,
        batch_size: int,
        *,
        shuffle: bool,
        seed: Optional[int] = None,
    ) -> Iterator[np.ndarray]:
        order = np.arange(self.n_samples, dtype=np.int64)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(order)
        for start in range(0, self.n_samples, batch_size):
            yield order[start:start + batch_size]

    def load_batch(
        self,
        batch_indices: Sequence[int],
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        batch_indices = np.asarray(batch_indices, dtype=np.int64)
        idx1 = self.specs.idx1[batch_indices]
        idx2 = self.specs.idx2[batch_indices]
        idx3 = self.specs.idx3[batch_indices]
        delta2 = self.specs.delta2[batch_indices]
        delta3 = self.specs.delta3[batch_indices]

        unique_ss = np.unique(np.concatenate([idx1, idx2, idx3]))
        with h5py.File(self.h5_file_path, "r") as f:
            wf_all = f["waveforms"][unique_ss]

        index_map = {int(ss_idx): i for i, ss_idx in enumerate(unique_ss)}
        batch_size = int(batch_indices.shape[0])
        waveforms = np.zeros(
            (batch_size, self.n_channels, self.n_time_points),
            dtype=np.float32,
        )

        for row in range(batch_size):
            wf1 = wf_all[index_map[int(idx1[row])]]
            wf2 = shift_waveform_2d(wf_all[index_map[int(idx2[row])]], int(delta2[row]))
            wf3 = shift_waveform_2d(wf_all[index_map[int(idx3[row])]], int(delta3[row]))
            waveforms[row] = wf1 + wf2 + wf3

        wf_col = np.transpose(waveforms, (0, 2, 1)).reshape(batch_size, -1, 1)
        meta = {
            "batch_indices": batch_indices,
            "idx1": idx1,
            "idx2": idx2,
            "idx3": idx3,
            "delta2": delta2,
            "delta3": delta3,
            "delta2_ns": delta2.astype(np.float32) * self.ns_per_bin,
            "delta3_ns": delta3.astype(np.float32) * self.ns_per_bin,
        }
        return wf_col.astype(np.float32, copy=False), meta


def _latest_checkpoint(root: Path, latent_dim: Optional[int]) -> Optional[Path]:
    search_dirs: List[Path]
    if latent_dim is not None:
        search_dirs = [root / f"diffae_light_z{latent_dim}"]
    else:
        search_dirs = sorted(root.glob("diffae_light_z*"))

    candidates: List[Path] = []
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        candidates.extend(directory.glob("diffae_light_epoch_*.pt"))

    if not candidates:
        return None

    def _epoch_key(path: Path) -> Tuple[int, float]:
        match = re.search(r"diffae_light_epoch_(\d+)\.pt$", path.name)
        epoch = int(match.group(1)) if match else -1
        return epoch, path.stat().st_mtime

    return max(candidates, key=_epoch_key)


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint:
        path = Path(args.checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    root = Path(args.checkpoint_root).expanduser().resolve()
    ckpt = _latest_checkpoint(root, args.latent_dim)
    if ckpt is None:
        latent_msg = "" if args.latent_dim is None else f" for latent_dim={args.latent_dim}"
        raise FileNotFoundError(
            f"No diffae_light checkpoint found under {root}{latent_msg}. "
            "Pass --checkpoint explicitly if the checkpoint lives elsewhere."
        )
    return ckpt.resolve()


def _count_indexed_modules(state_dict: Dict[str, torch.Tensor], prefix: str) -> int:
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.")
    indices = {int(m.group(1)) for key in state_dict for m in [pattern.match(key)] if m}
    return max(indices) + 1 if indices else 0


def _count_nested_modules(state_dict: Dict[str, torch.Tensor], prefix: str) -> int:
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.")
    indices = {int(m.group(1)) for key in state_dict for m in [pattern.match(key)] if m}
    return max(indices) + 1 if indices else 0


def _infer_encoder_type(state_dict: Dict[str, torch.Tensor]) -> str:
    keys = list(state_dict.keys())
    if any(key.startswith("stages.") for key in keys):
        return "graph"
    if any(key.startswith("backbone.") for key in keys):
        has_conv = any(state_dict[key].ndim == 3 for key in state_dict if key.startswith("backbone.") and key.endswith(".weight"))
        return "cnn" if has_conv else "mlp"
    if any(key.startswith("to_latent.") or key.startswith("to_mu.") for key in keys):
        return "mlp"
    raise ValueError("Could not infer encoder_type from checkpoint.")


def _infer_latent_dim(enc_state: Dict[str, torch.Tensor]) -> int:
    for key in ("to_latent.3.weight", "to_mu.3.weight", "to_latent.weight", "to_mu.weight"):
        if key in enc_state:
            return int(enc_state[key].shape[0])
    raise ValueError("Could not infer latent_dim from DiffAE Light checkpoint.")


def infer_config_from_checkpoint(ckpt_path: Path) -> Config:
    chk = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    enc_state = chk.get("ema_encoder") or chk["encoder"]
    dec_state = chk.get("ema_decoder") or chk["decoder"]

    cfg = get_config()
    cfg.resume = False
    cfg.visualize = False
    cfg.encoder.use_regressive_head = False

    encoder_type = _infer_encoder_type(enc_state)
    latent_dim = _infer_latent_dim(enc_state)

    cfg.encoder.latent_dim = latent_dim
    cfg.encoder.encoder_type = encoder_type
    cfg.encoder.use_stochastic = any(key.startswith("to_mu.") for key in enc_state)

    if encoder_type == "graph":
        cfg.encoder.hidden_dim = int(enc_state["in_proj.weight"].shape[0])
        cfg.encoder.depth = max(_count_indexed_modules(enc_state, "stages") - 1, 0)
        cfg.encoder.blocks_per_stage = max(_count_nested_modules(enc_state, "stages.0"), 1)
        head_key = "to_mu.1.weight" if cfg.encoder.use_stochastic else "to_latent.1.weight"
        if head_key in enc_state:
            cfg.encoder.latent_head_dim = int(enc_state[head_key].shape[0])
        if "lpe_proj.weight" in enc_state:
            cfg.graph.lpe_dim = int(enc_state["lpe_proj.weight"].shape[1])
        else:
            cfg.graph.lpe_dim = 0
        if "raw_anchor_pool.anchor_embed" in enc_state:
            cfg.encoder.latent_anchor_count = int(enc_state["raw_anchor_pool.anchor_embed"].shape[0])
            cfg.encoder.latent_anchor_value_dim = int(enc_state["raw_anchor_pool.value_proj.weight"].shape[0])
            anchor_out_dims = [
                int(value.shape[0])
                for key, value in enc_state.items()
                if key.endswith("out_proj.1.weight")
                and (key.startswith("raw_anchor_pool.") or key.startswith("anchor_pools."))
            ]
            cfg.encoder.latent_anchor_dim = int(sum(anchor_out_dims))
        else:
            cfg.encoder.latent_anchor_dim = 0
    elif encoder_type == "cnn":
        conv_keys = sorted(
            [
                key
                for key, value in enc_state.items()
                if key.startswith("backbone.") and key.endswith(".weight") and value.ndim == 3
            ],
            key=lambda key: int(key.split(".")[1]),
        )
        cfg.encoder.conv_channels = tuple(int(enc_state[key].shape[0]) for key in conv_keys)
    elif encoder_type == "mlp":
        linear_keys = sorted(
            [
                key
                for key, value in enc_state.items()
                if key.startswith("backbone.") and key.endswith(".weight") and value.ndim == 2
            ],
            key=lambda key: int(key.split(".")[1]),
        )
        if linear_keys:
            cfg.encoder.mlp_hidden_dim = int(enc_state[linear_keys[0]].shape[0])
            cfg.encoder.mlp_encoder_layers = len(linear_keys) + 1

    cfg.model.hidden_dim = int(dec_state["in_proj.weight"].shape[0])
    cfg.model.depth = max(_count_indexed_modules(dec_state, "down_stages") - 1, 0)
    cfg.model.blocks_per_stage = max(_count_nested_modules(dec_state, "down_stages.0"), 1)
    cond_key = "down_scale_cond.0.weight"
    if cond_key in dec_state:
        cond_dim = int(dec_state[cond_key].shape[1])
        cfg.conditioning.time_dim = max(cond_dim - cfg.encoder.latent_dim, 1)

    return cfg


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    if args.device is not None:
        cfg.device = args.device
    if args.latent_dim is not None:
        cfg.encoder.latent_dim = args.latent_dim
    if args.encoder_hidden_dim is not None:
        cfg.encoder.hidden_dim = args.encoder_hidden_dim
    if args.model_hidden_dim is not None:
        cfg.model.hidden_dim = args.model_hidden_dim
    if args.encoder_depth is not None:
        cfg.encoder.depth = args.encoder_depth
    if args.model_depth is not None:
        cfg.model.depth = args.model_depth
    if args.encoder_blocks_per_stage is not None:
        cfg.encoder.blocks_per_stage = args.encoder_blocks_per_stage
    if args.model_blocks_per_stage is not None:
        cfg.model.blocks_per_stage = args.model_blocks_per_stage
    if args.parametrization is not None:
        cfg.diffusion.parametrization = args.parametrization
    if args.timesteps is not None:
        cfg.diffusion.timesteps = args.timesteps
    if args.radius is not None:
        cfg.graph.radius = args.radius
    if args.z_hops is not None:
        cfg.graph.z_hops = args.z_hops
    if args.z_sep is not None:
        cfg.graph.z_sep = args.z_sep
    if args.lpe_dim is not None:
        cfg.graph.lpe_dim = args.lpe_dim
    if args.weighted_edges:
        cfg.graph.weighted_edges = True
    if args.delta_min is not None:
        cfg.ms_data.delta_min = args.delta_min
    if args.delta_max is not None:
        cfg.ms_data.delta_max = args.delta_max
    if args.ns_per_bin is not None:
        cfg.ms_data.ns_per_bin = args.ns_per_bin
    if args.tritium_h5 is not None:
        cfg.paths.tritium_h5 = args.tritium_h5
    if args.channel_positions is not None:
        cfg.paths.channel_positions = args.channel_positions

    cfg.training.lr = args.lr
    cfg.training.batch_size = args.batch_size
    cfg.training.use_amp = not args.disable_amp
    cfg.resume = False
    cfg.visualize = False
    cfg.encoder.use_regressive_head = False
    return cfg


def sample_triple_scatter_specs(
    n_events: int,
    n_ss: int,
    delta_min: int,
    delta_max: int,
    seed: int,
) -> TripleScatterSpecs:
    rng = np.random.default_rng(seed)
    idx1 = rng.integers(0, n_ss, size=n_events, dtype=np.int64)
    idx2 = rng.integers(0, n_ss, size=n_events, dtype=np.int64)
    mask = idx2 == idx1
    while mask.any():
        idx2[mask] = rng.integers(0, n_ss, size=int(mask.sum()), dtype=np.int64)
        mask = idx2 == idx1

    idx3 = rng.integers(0, n_ss, size=n_events, dtype=np.int64)
    mask = (idx3 == idx1) | (idx3 == idx2)
    while mask.any():
        idx3[mask] = rng.integers(0, n_ss, size=int(mask.sum()), dtype=np.int64)
        mask = (idx3 == idx1) | (idx3 == idx2)

    delta2 = rng.integers(delta_min, delta_max + 1, size=n_events, dtype=np.int32)
    delta3 = rng.integers(delta_min, delta_max + 1, size=n_events, dtype=np.int32)
    return TripleScatterSpecs(idx1=idx1, idx2=idx2, idx3=idx3, delta2=delta2, delta3=delta3)


def make_output_dir(args: argparse.Namespace, ckpt_path: Path) -> Path:
    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser().resolve()
    else:
        out_dir = (
            Path("fewshot_results")
            / "diffae_light_triple_scatter"
            / f"{ckpt_path.stem}_seed{args.seed}"
        ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def waveform_flat_to_matrix(flat_waveform: np.ndarray, n_channels: int, n_time_points: int) -> np.ndarray:
    return flat_waveform.reshape(n_time_points, n_channels).T


def summarize_metric(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
    }


def set_torch_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def reconstruct_batches(
    ctx: DiffAELightContext,
    loader: TripleScatterBatcher,
    batch_indices: Sequence[int],
    *,
    sample_seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    batch_np, meta = loader.load_batch(batch_indices)
    batch_norm = ctx.data_stats.normalize(batch_np)
    x_ref = torch.from_numpy(batch_norm.astype(np.float32)).to(ctx.device)
    set_torch_seed(sample_seed, ctx.device)
    rec = sample_diffae_light(ctx, x_ref, pbar=False)
    rec_np = ctx.data_stats.denormalize(rec.cpu().numpy())
    rec_flat = np.clip(rec_np[:, 0, :], 0.0, None)
    truth_flat = batch_np[:, :, 0]
    return rec_flat, truth_flat, meta


@torch.inference_mode()
def evaluate_reconstruction(
    ctx: DiffAELightContext,
    loader: TripleScatterBatcher,
    *,
    batch_size: int,
    base_seed: int,
    max_batches: Optional[int],
    desc: str,
) -> Dict[str, object]:
    per_event_mse: List[np.ndarray] = []
    per_event_mae: List[np.ndarray] = []
    per_event_profile_mae: List[np.ndarray] = []
    n_eval = 0

    batches = list(loader.iter_indices(batch_size, shuffle=False))
    if max_batches is not None:
        batches = batches[:max_batches]

    pbar = tqdm(batches, desc=desc, ncols=120)
    for batch_id, batch_indices in enumerate(pbar):
        rec_flat, truth_flat, _ = reconstruct_batches(
            ctx,
            loader,
            batch_indices,
            sample_seed=base_seed + batch_id,
        )

        diff = rec_flat - truth_flat
        mse = np.mean(diff ** 2, axis=1)
        mae = np.mean(np.abs(diff), axis=1)

        truth_profile = truth_flat.reshape(truth_flat.shape[0], loader.n_time_points, loader.n_channels).sum(axis=2)
        rec_profile = rec_flat.reshape(rec_flat.shape[0], loader.n_time_points, loader.n_channels).sum(axis=2)
        profile_mae = np.mean(np.abs(rec_profile - truth_profile), axis=1)

        per_event_mse.append(mse)
        per_event_mae.append(mae)
        per_event_profile_mae.append(profile_mae)
        n_eval += int(truth_flat.shape[0])
        pbar.set_postfix(mse=float(np.mean(mse)))

    mse_all = np.concatenate(per_event_mse) if per_event_mse else np.empty(0, dtype=np.float32)
    mae_all = np.concatenate(per_event_mae) if per_event_mae else np.empty(0, dtype=np.float32)
    profile_mae_all = np.concatenate(per_event_profile_mae) if per_event_profile_mae else np.empty(0, dtype=np.float32)

    return {
        "num_events": int(n_eval),
        "mse": summarize_metric(mse_all),
        "rmse_mean": float(np.sqrt(np.mean(mse_all))) if mse_all.size else float("nan"),
        "mae": summarize_metric(mae_all),
        "profile_mae": summarize_metric(profile_mae_all),
        "per_event_mse": mse_all,
        "per_event_mae": mae_all,
        "per_event_profile_mae": profile_mae_all,
    }


def fine_tune_one_pass(
    ctx: DiffAELightContext,
    loader: TripleScatterBatcher,
    *,
    batch_size: int,
    shuffle_seed: int,
    max_batches: Optional[int],
) -> List[float]:
    if ctx.optim is None:
        raise RuntimeError("Context optimizer is missing; build the context with for_training=True.")

    ctx.encoder.train()
    ctx.decoder.train()
    if ctx.ema_encoder is not None:
        ctx.ema_encoder.eval()
    if ctx.ema_decoder is not None:
        ctx.ema_decoder.eval()

    amp_enabled = ctx.cfg.training.use_amp and ctx.device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float32
    batches = list(loader.iter_indices(batch_size, shuffle=True, seed=shuffle_seed))
    if max_batches is not None:
        batches = batches[:max_batches]

    losses: List[float] = []
    pbar = tqdm(batches, desc="Fine-tune", ncols=120)
    for batch_indices in pbar:
        batch_np, _ = loader.load_batch(batch_indices)
        batch_norm = ctx.data_stats.normalize(batch_np)
        batch_size_actual = int(batch_norm.shape[0])

        x0 = torch.from_numpy(batch_norm.astype(np.float32)).to(ctx.device)
        x0_flat = x0.reshape(batch_size_actual * ctx.n_nodes, 1)
        ctx.optim.zero_grad(set_to_none=True)

        with torch.amp.autocast(ctx.device.type, dtype=amp_dtype, enabled=amp_enabled):
            z, mu, logvar = _encode_with_context(ctx, x0_flat, batch_size_actual)
            t = torch.randint(
                0,
                ctx.cfg.diffusion.timesteps,
                (batch_size_actual,),
                device=ctx.device,
                dtype=torch.long,
            )
            t_emb = sinusoidal_embedding(t, ctx.cfg.conditioning.time_dim)
            cond_full = torch.cat([z, t_emb], dim=-1)

            sqrt_ab = ctx.schedule["sqrt_alphas_cumprod"][t].view(batch_size_actual, 1, 1)
            sqrt_om = ctx.schedule["sqrt_one_minus_alphas_cumprod"][t].view(batch_size_actual, 1, 1)
            snr_t = ctx.schedule["snr"][t].view(batch_size_actual)

            noise = torch.randn_like(x0)
            x_t = sqrt_ab * x0 + sqrt_om * noise
            x_t_flat = x_t.reshape(batch_size_actual * ctx.n_nodes, 1)

            pred_flat = ctx.decoder(x_t_flat, ctx.decoder_pyramid, cond_full, batch_size=batch_size_actual)
            pred = pred_flat.reshape(batch_size_actual, ctx.n_nodes, 1)

            if ctx.cfg.diffusion.parametrization == "eps":
                target = noise
            elif ctx.cfg.diffusion.parametrization == "v":
                target = sqrt_ab * noise - sqrt_om * x0
            else:
                raise ValueError("parametrization must be 'eps' or 'v'")

            loss_per_sample = F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2))
            if ctx.cfg.diffusion.p2_gamma > 0.0:
                weight = torch.pow(ctx.cfg.diffusion.p2_k + snr_t, -ctx.cfg.diffusion.p2_gamma)
                loss_per_sample = loss_per_sample * weight
            loss = loss_per_sample.mean()

            if ctx.cfg.encoder.use_stochastic and mu is not None and logvar is not None:
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = loss + ctx.cfg.encoder.kl_weight * kl

        if not torch.isfinite(loss):
            ctx.optim.zero_grad(set_to_none=True)
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(ctx.encoder.parameters()) + list(ctx.decoder.parameters()),
            max_norm=ctx.cfg.training.grad_clip,
        )
        ctx.optim.step()

        if ctx.ema_encoder is not None:
            _ema_update(ctx.ema_encoder, ctx.encoder, ctx.cfg.training.ema_decay)
        if ctx.ema_decoder is not None:
            _ema_update(ctx.ema_decoder, ctx.decoder, ctx.cfg.training.ema_decay)

        loss_value = float(loss.item())
        losses.append(loss_value)
        pbar.set_postfix(loss=loss_value)

    return losses


def save_loss_plot(losses: Sequence[float], output_path: Path) -> None:
    if not losses:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(losses) + 1), losses, linewidth=1.5, color="#005f73")
    ax.set_xlabel("Fine-tune step")
    ax.set_ylabel("Loss")
    ax.set_title("Triple-scatter one-pass fine-tuning loss")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_metric_histogram(
    before: np.ndarray,
    after: np.ndarray,
    output_path: Path,
    *,
    title: str,
    xlabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(before, bins=40, alpha=0.55, label="Before fine-tune", color="#9b2226")
    ax.hist(after, bins=40, alpha=0.55, label="After fine-tune", color="#0a9396")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_example_reconstructions(
    loader: TripleScatterBatcher,
    truth_batch: np.ndarray,
    before_batch: np.ndarray,
    after_batch: np.ndarray,
    output_path: Path,
) -> None:
    n_examples = int(before_batch.shape[0])
    fig, axes = plt.subplots(n_examples, 4, figsize=(16, 3.4 * n_examples), squeeze=False)

    for row in range(n_examples):
        truth_mat = waveform_flat_to_matrix(truth_batch[row], loader.n_channels, loader.n_time_points)
        before_mat = waveform_flat_to_matrix(before_batch[row], loader.n_channels, loader.n_time_points)
        after_mat = waveform_flat_to_matrix(after_batch[row], loader.n_channels, loader.n_time_points)
        vmax = float(max(truth_mat.max(), before_mat.max(), after_mat.max(), 1e-6))

        for col, (mat, title) in enumerate(
            [
                (truth_mat, "Truth"),
                (before_mat, "Before"),
                (after_mat, "After"),
            ]
        ):
            ax = axes[row][col]
            im = ax.imshow(mat, aspect="auto", origin="lower", cmap="magma", vmin=0.0, vmax=vmax)
            ax.set_title(title)
            ax.set_xlabel("Time bin")
            ax.set_ylabel("Channel")
            if col == 2:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax_prof = axes[row][3]
        time_axis = np.arange(loader.n_time_points)
        truth_profile = truth_mat.sum(axis=0)
        before_profile = before_mat.sum(axis=0)
        after_profile = after_mat.sum(axis=0)
        ax_prof.plot(time_axis, truth_profile, linewidth=1.5, color="black", label="Truth")
        ax_prof.plot(time_axis, before_profile, linewidth=1.2, color="#9b2226", label="Before")
        ax_prof.plot(time_axis, after_profile, linewidth=1.2, color="#0a9396", label="After")
        ax_prof.set_title("Summed waveform")
        ax_prof.set_xlabel("Time bin")
        ax_prof.set_ylabel("Amplitude")
        ax_prof.grid(alpha=0.3)
        if row == 0:
            ax_prof.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_finetuned_checkpoint(ctx: DiffAELightContext, path: Path, source_checkpoint: Path) -> None:
    state = {
        "encoder": ctx.encoder.state_dict(),
        "decoder": ctx.decoder.state_dict(),
        "ema_encoder": ctx.ema_encoder.state_dict() if ctx.ema_encoder is not None else ctx.encoder.state_dict(),
        "ema_decoder": ctx.ema_decoder.state_dict() if ctx.ema_decoder is not None else ctx.decoder.state_dict(),
        "optim": ctx.optim.state_dict() if ctx.optim is not None else None,
        "data_stats": {"mean": ctx.data_stats.mean, "std": ctx.data_stats.std},
        "source_checkpoint": str(source_checkpoint),
        "note": "Few-shot triple-scatter fine-tune checkpoint",
    }
    torch.save(state, path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Few-shot triple-scatter fine-tuning for DiffAE Light")
    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit DiffAE Light checkpoint to load")
    parser.add_argument("--checkpoint-root", type=str, default="checkpoints", help="Root directory to search for diffae_light checkpoints")
    parser.add_argument("--latent-dim", type=int, default=None, help="Checkpoint latent dim hint or explicit override")

    parser.add_argument("--tritium-h5", type=str, default=None, help="Override SS waveform H5 path")
    parser.add_argument("--channel-positions", type=str, default=None, help="Override channel-position file path")

    parser.add_argument("--train-events", type=int, default=8000, help="Number of triple-scatter training events")
    parser.add_argument("--val-events", type=int, default=2000, help="Number of held-out triple-scatter evaluation events")
    parser.add_argument("--batch-size", type=int, default=8, help="Fine-tuning batch size")
    parser.add_argument("--eval-batch-size", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Fine-tuning learning rate")

    parser.add_argument("--delta-min", type=int, default=None, help="Minimum triple-scatter shift in bins")
    parser.add_argument("--delta-max", type=int, default=None, help="Maximum triple-scatter shift in bins")
    parser.add_argument("--ns-per-bin", type=float, default=None, help="Time-bin size in ns")

    parser.add_argument("--parametrization", choices=("eps", "v"), default=None, help="Override diffusion parametrization")
    parser.add_argument("--timesteps", type=int, default=None, help="Override diffusion timesteps")
    parser.add_argument("--radius", type=float, default=None, help="Override graph radius")
    parser.add_argument("--z-hops", type=int, default=None, help="Override graph z_hops")
    parser.add_argument("--z-sep", type=float, default=None, help="Override graph z spacing")
    parser.add_argument("--lpe-dim", type=int, default=None, help="Override graph Laplacian PE dim")
    parser.add_argument("--weighted-edges", action="store_true", help="Use weighted graph edges")

    parser.add_argument("--encoder-hidden-dim", type=int, default=None, help="Manual override for encoder hidden dim")
    parser.add_argument("--model-hidden-dim", type=int, default=None, help="Manual override for decoder hidden dim")
    parser.add_argument("--encoder-depth", type=int, default=None, help="Manual override for encoder depth")
    parser.add_argument("--model-depth", type=int, default=None, help="Manual override for decoder depth")
    parser.add_argument("--encoder-blocks-per-stage", type=int, default=None, help="Manual override for encoder blocks/stage")
    parser.add_argument("--model-blocks-per-stage", type=int, default=None, help="Manual override for decoder blocks/stage")

    parser.add_argument("--device", type=str, default=None, help="Torch device override, e.g. cpu or cuda")
    parser.add_argument("--disable-amp", action="store_true", help="Disable AMP even on CUDA")

    parser.add_argument("--seed", type=int, default=0, help="Base seed for split generation and training")
    parser.add_argument("--eval-seed", type=int, default=None, help="Base seed for deterministic held-out sampling")
    parser.add_argument("--num-example-plots", type=int, default=6, help="Number of held-out examples to plot")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Optional cap for fine-tune batches")
    parser.add_argument("--max-eval-batches", type=int, default=None, help="Optional cap for eval batches")

    parser.add_argument("--output-dir", type=str, default=None, help="Experiment artifact directory")
    parser.add_argument("--save-finetuned-checkpoint", action="store_true", help="Write fine-tuned weights into the experiment output directory")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    ckpt_path = resolve_checkpoint(args)
    cfg = infer_config_from_checkpoint(ckpt_path)
    cfg = apply_overrides(cfg, args)

    eval_seed = args.eval_seed if args.eval_seed is not None else args.seed + 10_000
    output_dir = make_output_dir(args, ckpt_path)

    ss_loader = TritiumSSDataLoader(cfg.paths.tritium_h5, cfg.paths.channel_positions)
    total_events = args.train_events + args.val_events
    all_specs = sample_triple_scatter_specs(
        total_events,
        ss_loader.n_samples,
        cfg.ms_data.delta_min,
        cfg.ms_data.delta_max,
        args.seed,
    )
    train_specs = all_specs.subset(np.arange(args.train_events, dtype=np.int64))
    val_specs = all_specs.subset(np.arange(args.train_events, total_events, dtype=np.int64))
    train_loader = TripleScatterBatcher(cfg.paths.tritium_h5, cfg.paths.channel_positions, train_specs, cfg.ms_data.ns_per_bin)
    val_loader = TripleScatterBatcher(cfg.paths.tritium_h5, cfg.paths.channel_positions, val_specs, cfg.ms_data.ns_per_bin)

    train_specs.save(output_dir / "train_specs.npz")
    val_specs.save(output_dir / "val_specs.npz")

    ctx = DiffAELightContext.build(cfg, for_training=True, verbose=True, use_ms_data=True)
    ctx.load_checkpoint(str(ckpt_path), load_optim=False)
    ctx.checkpoint_dir = str(output_dir)
    ctx.plot_dir = str(output_dir)
    if ctx.optim is None:
        raise RuntimeError("Failed to initialize optimizer for fine-tuning.")
    for group in ctx.optim.param_groups:
        group["lr"] = cfg.training.lr
        group.setdefault("initial_lr", cfg.training.lr)

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Training triple-scatter events: {len(train_specs)}")
    print(f"Held-out triple-scatter events: {len(val_specs)}")
    print(f"Artifacts will be written to: {output_dir}")

    before_metrics = evaluate_reconstruction(
        ctx,
        val_loader,
        batch_size=args.eval_batch_size,
        base_seed=eval_seed,
        max_batches=args.max_eval_batches,
        desc="Held-out reconstruction (before)",
    )

    losses = fine_tune_one_pass(
        ctx,
        train_loader,
        batch_size=args.batch_size,
        shuffle_seed=args.seed + 1,
        max_batches=args.max_train_batches,
    )

    after_metrics = evaluate_reconstruction(
        ctx,
        val_loader,
        batch_size=args.eval_batch_size,
        base_seed=eval_seed,
        max_batches=args.max_eval_batches,
        desc="Held-out reconstruction (after)",
    )

    n_examples = min(args.num_example_plots, val_loader.n_samples)
    example_indices = np.arange(n_examples, dtype=np.int64)
    truth_examples, _ = val_loader.load_batch(example_indices)

    # Reconstruct the same examples with the pre-fine-tuned weights by reloading a fresh context.
    ctx_before = DiffAELightContext.build(cfg, for_training=True, verbose=False, use_ms_data=True)
    ctx_before.load_checkpoint(str(ckpt_path), load_optim=False)
    before_examples, _, _ = reconstruct_batches(
        ctx=ctx_before,
        loader=val_loader,
        batch_indices=example_indices,
        sample_seed=eval_seed + 777,
    )
    after_examples, _, _ = reconstruct_batches(
        ctx=ctx,
        loader=val_loader,
        batch_indices=example_indices,
        sample_seed=eval_seed + 777,
    )

    save_example_reconstructions(
        val_loader,
        truth_examples[:, :, 0],
        before_examples,
        after_examples,
        output_dir / "heldout_examples_before_after.png",
    )
    save_loss_plot(losses, output_dir / "finetune_loss.png")
    save_metric_histogram(
        before=np.asarray(before_metrics["per_event_mse"]),
        after=np.asarray(after_metrics["per_event_mse"]),
        output_path=output_dir / "heldout_mse_hist.png",
        title="Held-out triple-scatter reconstruction MSE",
        xlabel="Per-event MSE",
    )

    np.savez_compressed(
        output_dir / "heldout_event_metrics.npz",
        before_mse=np.asarray(before_metrics["per_event_mse"]),
        after_mse=np.asarray(after_metrics["per_event_mse"]),
        before_mae=np.asarray(before_metrics["per_event_mae"]),
        after_mae=np.asarray(after_metrics["per_event_mae"]),
        before_profile_mae=np.asarray(before_metrics["per_event_profile_mae"]),
        after_profile_mae=np.asarray(after_metrics["per_event_profile_mae"]),
    )

    metrics_summary = {
        "checkpoint": str(ckpt_path),
        "output_dir": str(output_dir),
        "config": asdict(cfg),
        "train_events": int(args.train_events),
        "heldout_events": int(args.val_events),
        "evaluated_events_before": int(before_metrics["num_events"]),
        "evaluated_events_after": int(after_metrics["num_events"]),
        "before": {
            "mse": before_metrics["mse"],
            "rmse_mean": before_metrics["rmse_mean"],
            "mae": before_metrics["mae"],
            "profile_mae": before_metrics["profile_mae"],
        },
        "after": {
            "mse": after_metrics["mse"],
            "rmse_mean": after_metrics["rmse_mean"],
            "mae": after_metrics["mae"],
            "profile_mae": after_metrics["profile_mae"],
        },
        "delta": {
            "mse_mean": float(after_metrics["mse"]["mean"] - before_metrics["mse"]["mean"]),
            "mae_mean": float(after_metrics["mae"]["mean"] - before_metrics["mae"]["mean"]),
            "profile_mae_mean": float(after_metrics["profile_mae"]["mean"] - before_metrics["profile_mae"]["mean"]),
        },
        "fine_tune": {
            "num_steps": int(len(losses)),
            "loss_first": float(losses[0]) if losses else None,
            "loss_last": float(losses[-1]) if losses else None,
            "loss_mean": float(np.mean(losses)) if losses else None,
        },
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    with open(output_dir / "run_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    if args.save_finetuned_checkpoint:
        save_finetuned_checkpoint(ctx, output_dir / "diffae_light_triple_scatter_finetuned.pt", ckpt_path)

    print("\nHeld-out reconstruction summary")
    print(f"  Before MSE mean: {before_metrics['mse']['mean']:.6f}")
    print(f"  After  MSE mean: {after_metrics['mse']['mean']:.6f}")
    print(f"  Before MAE mean: {before_metrics['mae']['mean']:.6f}")
    print(f"  After  MAE mean: {after_metrics['mae']['mean']:.6f}")
    print(f"  Before profile-MAE mean: {before_metrics['profile_mae']['mean']:.6f}")
    print(f"  After  profile-MAE mean: {after_metrics['profile_mae']['mean']:.6f}")
    print(f"  Saved metrics and plots to: {output_dir}")


if __name__ == "__main__":
    main()
