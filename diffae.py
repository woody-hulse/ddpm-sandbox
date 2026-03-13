"""
DiffAE: Diffusion Autoencoder with Graph Encoder.

Uses a graph encoder to map events to latent representations,
which condition the diffusion process for reconstruction.
"""
import os
import sys
import glob
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union
from copy import deepcopy

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm

from data import Graph, visualize_event, visualize_event_z, SparseGraph
from lz_data_loader import TritiumSSDataLoader, OnlineMSBatcher
from config import Config, default_config, get_config, print_config

from models.graph_unet import (
    GraphDDPMUNet, GraphResBlock, TopKPool, FiLMFromCond,
    build_block_diagonal_adj, _unpool_like
)
from diffusion.schedule import build_cosine_schedule, sinusoidal_embedding
from utils.sparse_ops import to_coalesced_coo, subgraph_coo, to_binary
from utils.visualization import build_xy_adjacency_radius

from ae import (
    DiffAEDataStats, apply_lopsided_augmentation, visualize_event_3d,
    Conv1DEncoder, MLPEncoder, MLPDecoder,
)
from graphae import GraphEncoder, SimpleGraphDecoder  # type: ignore[import]


DataLoaderType = Union[TritiumSSDataLoader, OnlineMSBatcher]


@dataclass
class DiffAEContext:
    """Holds all model components for DiffAE training/inference."""
    cfg: Config
    device: torch.device
    loader: DataLoaderType
    graph: SparseGraph
    A_sparse: torch.Tensor
    pos: torch.Tensor
    n_channels: int
    n_time_points: int
    n_nodes: int
    data_stats: DiffAEDataStats
    schedule: dict
    encoder: nn.Module
    decoder: nn.Module
    latent_proj: nn.Module
    checkpoint_dir: str = ""
    plot_dir: str = ""
    ema_encoder: Optional[nn.Module] = None
    ema_decoder: Optional[nn.Module] = None
    ema_latent_proj: Optional[nn.Module] = None
    regressive_decoder: Optional[nn.Module] = None
    ema_regressive_decoder: Optional[nn.Module] = None
    optim: Optional[torch.optim.Optimizer] = None
    use_ms_data: bool = False

    @classmethod
    def build(cls, cfg: Config, for_training: bool = True, verbose: bool = True, use_ms_data: bool = True) -> 'DiffAEContext':
        device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if verbose:
            print(f"Using device: {device}")

        if use_ms_data:
            loader = OnlineMSBatcher(
                cfg.paths.tritium_h5,
                cfg.paths.channel_positions,
                delta_min=cfg.ms_data.delta_min,
                delta_max=cfg.ms_data.delta_max,
                ns_per_bin=cfg.ms_data.ns_per_bin,
                seed=cfg.ms_data.seed,
            )
            if verbose:
                print(f"Using online MS data: delta=[{cfg.ms_data.delta_min}, {cfg.ms_data.delta_max}] bins")
        else:
            loader = TritiumSSDataLoader(cfg.paths.tritium_h5, cfg.paths.channel_positions)

        graph = loader.load_adjacency_sparse(
            z_sep=cfg.graph.z_sep,
            radius=cfg.graph.radius,
            z_hops=cfg.graph.z_hops
        )
        A_sparse = graph.adjacency.to(device)
        pos = graph.positions_xyz.to(device)
        n_channels = loader.n_channels
        n_time_points = loader.n_time_points
        n_nodes = n_channels * n_time_points

        if verbose:
            print(f"Graph: {n_nodes} nodes, {A_sparse._nnz()} edges")
            print("Computing data statistics...")

        data_stats = DiffAEDataStats.from_loader(loader, n_samples=1000, batch_size=32)
        if verbose:
            print(f"Data mean: {data_stats.mean:.4f}, std: {data_stats.std:.4f}")

        schedule = build_cosine_schedule(cfg.diffusion.timesteps, device)

        encoder_type = (getattr(cfg.encoder, "encoder_type", "graph") or "graph").lower()
        if encoder_type == "cnn":
            encoder = Conv1DEncoder(
                in_dim=cfg.model.in_dim,
                latent_dim=cfg.encoder.latent_dim,
                n_nodes=n_nodes,
                dropout=cfg.encoder.dropout,
                use_stochastic=cfg.encoder.use_stochastic,
            ).to(device)
        elif encoder_type == "mlp":
            encoder = MLPEncoder(
                in_dim=cfg.model.in_dim,
                hidden_dim=cfg.encoder.hidden_dim,
                latent_dim=cfg.encoder.latent_dim,
                n_nodes=n_nodes,
                num_layers=getattr(cfg.encoder, "mlp_encoder_layers", 3),
                dropout=cfg.encoder.dropout,
                use_stochastic=cfg.encoder.use_stochastic,
            ).to(device)
        else:
            encoder = GraphEncoder(
                in_dim=cfg.model.in_dim,
                hidden_dim=cfg.encoder.hidden_dim,
                latent_dim=cfg.encoder.latent_dim,
                depth=cfg.encoder.depth,
                blocks_per_stage=cfg.encoder.blocks_per_stage,
                pool_ratio=cfg.encoder.pool_ratio,
                dropout=cfg.encoder.dropout,
                pos_dim=cfg.model.pos_dim,
                use_stochastic=cfg.encoder.use_stochastic,
            ).to(device)

        latent_proj = nn.Sequential(
            nn.Linear(cfg.encoder.latent_dim, cfg.conditioning.cond_proj_dim * 2),
            nn.SiLU(),
            nn.Linear(cfg.conditioning.cond_proj_dim * 2, cfg.conditioning.cond_proj_dim),
        ).to(device)

        for m in latent_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        decoder = GraphDDPMUNet(
            in_dim=cfg.model.in_dim,
            cond_dim=cfg.encoder.latent_dim + cfg.conditioning.time_dim,
            hidden_dim=cfg.model.hidden_dim,
            depth=cfg.model.depth,
            blocks_per_stage=cfg.model.blocks_per_stage,
            pool_ratio=cfg.model.pool_ratio,
            out_dim=cfg.model.out_dim,
            dropout=cfg.model.dropout,
            pos_dim=cfg.model.pos_dim,
            skip_scale=getattr(cfg.model, 'skip_scale', 1.0),
        ).to(device)

        regressive_decoder = None
        if cfg.encoder.use_regressive_head:
            decoder_type = (getattr(cfg.encoder, "decoder_type", "mlp") or "mlp").lower()
            if decoder_type == "mlp":
                regressive_decoder = MLPDecoder(
                    latent_dim=cfg.encoder.latent_dim,
                    hidden_dim=cfg.encoder.hidden_dim,
                    out_dim=cfg.model.out_dim,
                    n_nodes=n_nodes,
                    num_layers=getattr(cfg.encoder, "mlp_decoder_layers", 3),
                    dropout=cfg.encoder.dropout,
                ).to(device)
            else:
                regressive_decoder = SimpleGraphDecoder(
                    latent_dim=cfg.encoder.latent_dim,
                    hidden_dim=cfg.encoder.hidden_dim,
                    out_dim=cfg.model.out_dim,
                    n_nodes=n_nodes,
                    depth=cfg.encoder.depth,
                    blocks_per_stage=cfg.encoder.blocks_per_stage,
                    dropout=cfg.encoder.dropout,
                    pos_dim=cfg.model.pos_dim,
                ).to(device)

        ema_encoder = None
        ema_decoder = None
        ema_latent_proj = None
        ema_regressive_decoder = None
        optim = None
        subdir = cfg.paths.diffae_subdir.format(latent_dim=cfg.encoder.latent_dim)
        checkpoint_dir = os.path.join(cfg.paths.checkpoint_dir, subdir)
        plot_dir = os.path.join(cfg.paths.plot_dir, subdir)

        if for_training:
            ema_encoder = deepcopy(encoder).to(device)
            ema_decoder = deepcopy(decoder).to(device)
            ema_latent_proj = deepcopy(latent_proj).to(device)
            all_params = (
                list(encoder.parameters()) +
                list(decoder.parameters()) +
                list(latent_proj.parameters())
            )
            if regressive_decoder is not None:
                ema_regressive_decoder = deepcopy(regressive_decoder).to(device)
                all_params += list(regressive_decoder.parameters())
            optim = torch.optim.AdamW(
                all_params,
                lr=cfg.training.lr,
                betas=(0.9, 0.999),
                weight_decay=cfg.training.weight_decay,
            )
            os.makedirs(checkpoint_dir, exist_ok=True)

        if verbose:
            n_enc = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
            n_dec = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
            n_proj = sum(p.numel() for p in latent_proj.parameters() if p.requires_grad)
            n_reg = sum(p.numel() for p in regressive_decoder.parameters() if p.requires_grad) if regressive_decoder else 0
            print(f"Encoder parameters: {n_enc:,}")
            print(f"Decoder parameters: {n_dec:,}")
            print(f"Latent projection parameters: {n_proj:,}")
            if regressive_decoder:
                print(f"Regressive decoder parameters: {n_reg:,}")
            print(f"Total trainable parameters: {n_enc + n_dec + n_proj + n_reg:,}")

        return cls(
            cfg=cfg,
            device=device,
            loader=loader,
            graph=graph,
            A_sparse=A_sparse,
            pos=pos,
            n_channels=n_channels,
            n_time_points=n_time_points,
            n_nodes=n_nodes,
            data_stats=data_stats,
            schedule=schedule,
            encoder=encoder,
            decoder=decoder,
            latent_proj=latent_proj,
            checkpoint_dir=checkpoint_dir,
            plot_dir=plot_dir,
            ema_encoder=ema_encoder,
            ema_decoder=ema_decoder,
            ema_latent_proj=ema_latent_proj,
            regressive_decoder=regressive_decoder,
            ema_regressive_decoder=ema_regressive_decoder,
            optim=optim,
            use_ms_data=use_ms_data,
        )

    def latest_checkpoint(self) -> Optional[str]:
        files = glob.glob(os.path.join(self.checkpoint_dir, "diffae_epoch_*.pt"))
        if not files:
            return None

        def _epoch_num(path: str) -> int:
            base = os.path.basename(path)
            stem = os.path.splitext(base)[0]
            try:
                return int(stem.split("_")[-1])
            except (ValueError, IndexError):
                return -1

        return max(files, key=_epoch_num)

    def save_checkpoint(self, epoch: int) -> str:
        state = {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "latent_proj": self.latent_proj.state_dict(),
            "ema_encoder": self.ema_encoder.state_dict() if self.ema_encoder else self.encoder.state_dict(),
            "ema_decoder": self.ema_decoder.state_dict() if self.ema_decoder else self.decoder.state_dict(),
            "ema_latent_proj": self.ema_latent_proj.state_dict() if self.ema_latent_proj else self.latent_proj.state_dict(),
            "optim": self.optim.state_dict() if self.optim else None,
            "epoch": epoch,
            "data_stats": {"mean": self.data_stats.mean, "std": self.data_stats.std},
        }
        if self.regressive_decoder is not None:
            state["regressive_decoder"] = self.regressive_decoder.state_dict()
        if self.ema_regressive_decoder is not None:
            state["ema_regressive_decoder"] = self.ema_regressive_decoder.state_dict()
        path = os.path.join(self.checkpoint_dir, f"diffae_epoch_{epoch:04d}.pt")
        torch.save(state, path)
        return path

    def load_checkpoint(self, path: str, load_optim: bool = True) -> int:
        chk = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(chk["encoder"], strict=False)
        self.decoder.load_state_dict(chk["decoder"], strict=False)
        self.latent_proj.load_state_dict(chk["latent_proj"])
        if self.ema_encoder is not None and "ema_encoder" in chk:
            self.ema_encoder.load_state_dict(chk["ema_encoder"], strict=False)
        if self.ema_decoder is not None and "ema_decoder" in chk:
            self.ema_decoder.load_state_dict(chk["ema_decoder"], strict=False)
        if self.ema_latent_proj is not None and "ema_latent_proj" in chk:
            self.ema_latent_proj.load_state_dict(chk["ema_latent_proj"])
        elif self.ema_latent_proj is not None:
            self.ema_latent_proj.load_state_dict(chk["latent_proj"])
        if self.regressive_decoder is not None and "regressive_decoder" in chk:
            self.regressive_decoder.load_state_dict(chk["regressive_decoder"])
        if self.ema_regressive_decoder is not None and "ema_regressive_decoder" in chk:
            self.ema_regressive_decoder.load_state_dict(chk["ema_regressive_decoder"])
        if load_optim and self.optim is not None and chk.get("optim"):
            try:
                self.optim.load_state_dict(chk["optim"])
            except (ValueError, RuntimeError) as e:
                print(f"  Optimizer state skipped (param count changed, fine-tuning): {e}")
        if "data_stats" in chk:
            self.data_stats.mean = chk["data_stats"]["mean"]
            self.data_stats.std = chk["data_stats"]["std"]
        return int(chk.get("epoch", 0))

    def find_best_checkpoint(self) -> Optional[str]:
        """Find the best available checkpoint, preferring same latent_dim but falling back to others."""
        same_latent = self.latest_checkpoint()
        if same_latent is not None:
            return same_latent

        parent_dir = os.path.dirname(self.checkpoint_dir)
        if not os.path.isdir(parent_dir):
            return None

        all_ckpts = []
        for subdir in os.listdir(parent_dir):
            if subdir.startswith("diffae_z"):
                subdir_path = os.path.join(parent_dir, subdir)
                ckpt_files = sorted(glob.glob(os.path.join(subdir_path, "diffae_epoch_*.pt")))
                if ckpt_files:
                    all_ckpts.append(ckpt_files[-1])

        if all_ckpts:
            return max(all_ckpts, key=os.path.getmtime)
        return None

    def load_checkpoint_partial(self, path: str, verbose: bool = True) -> Tuple[int, bool]:
        """
        Load checkpoint with partial weight loading for different latent sizes.

        Loads all compatible weights from checkpoint and keeps latent-dependent
        layers freshly initialized if sizes don't match.

        Returns:
            (epoch, is_full_load): epoch from checkpoint, True if all weights loaded
        """
        chk = torch.load(path, map_location=self.device)

        encoder_latent_keys = {'to_latent.weight', 'to_latent.bias',
                               'to_mu.weight', 'to_mu.bias',
                               'to_logvar.weight', 'to_logvar.bias',
                               'latent_norm.weight', 'latent_norm.bias',
                               'pre_latent_norm.weight', 'pre_latent_norm.bias'}

        def load_partial(model: nn.Module, state_dict: dict, skip_keys: set, name: str) -> List[str]:
            """Load state dict, skipping keys with mismatched sizes."""
            model_dict = model.state_dict()
            loaded_keys = []
            skipped_keys = []

            for key, value in state_dict.items():
                if key in skip_keys:
                    skipped_keys.append(key)
                    continue
                if key in model_dict:
                    if model_dict[key].shape == value.shape:
                        model_dict[key] = value
                        loaded_keys.append(key)
                    else:
                        skipped_keys.append(key)

            model.load_state_dict(model_dict)
            if verbose and skipped_keys:
                print(f"  {name}: loaded {len(loaded_keys)}/{len(state_dict)} keys, skipped: {skipped_keys}")
            return skipped_keys

        latent_proj_keys = {'0.weight', '0.bias'}

        all_skipped = []

        all_skipped.extend(load_partial(self.encoder, chk["encoder"], encoder_latent_keys, "encoder"))
        all_skipped.extend(load_partial(self.decoder, chk["decoder"], set(), "decoder"))
        all_skipped.extend(load_partial(self.latent_proj, chk["latent_proj"], latent_proj_keys, "latent_proj"))

        if self.ema_encoder is not None and "ema_encoder" in chk:
            load_partial(self.ema_encoder, chk["ema_encoder"], encoder_latent_keys, "ema_encoder")
        if self.ema_decoder is not None and "ema_decoder" in chk:
            load_partial(self.ema_decoder, chk["ema_decoder"], set(), "ema_decoder")

        if "data_stats" in chk:
            self.data_stats.mean = chk["data_stats"]["mean"]
            self.data_stats.std = chk["data_stats"]["std"]

        is_full_load = len(all_skipped) == 0
        epoch = int(chk.get("epoch", 0))
        return epoch, is_full_load


@torch.no_grad()
def save_encoded_dataset(
    ctx: DiffAEContext,
    output_path: str,
    encoder: Optional[nn.Module] = None,
    batch_size: int = 32,
    n_samples: int = 10000,
    verbose: bool = True,
) -> str:
    """
    Encode MS events and save latent vectors with delta_mu to h5.

    This is designed for the aux task: it encodes MS events and saves the
    latents along with delta_mu targets so that aux training only needs to
    load pre-computed embeddings (no encoder calls during MLP training).

    Args:
        ctx: DiffAE context with loader, graph, etc.
        output_path: Path to save the encoded dataset h5 file.
        encoder: Encoder to use (defaults to ctx.ema_encoder or ctx.encoder).
        batch_size: Batch size for encoding.
        n_samples: Number of MS samples to encode (only for OnlineMSBatcher).
        verbose: Print progress information.

    Returns:
        Path to the saved h5 file.
    """
    if encoder is None:
        encoder = ctx.ema_encoder if ctx.ema_encoder is not None else ctx.encoder
    encoder.eval()

    latent_dim = ctx.cfg.encoder.latent_dim

    all_latents = []
    all_delta_mu = []
    all_delta_bins = []
    all_xc1 = []
    all_yc1 = []
    all_xc2 = []
    all_yc2 = []

    n_batches = (n_samples + batch_size - 1) // batch_size
    pbar = tqdm(range(n_batches), desc="Encoding MS dataset", disable=not verbose, ncols=120)

    samples_encoded = 0
    for batch_idx in pbar:
        remaining = n_samples - samples_encoded
        actual_batch_size = min(batch_size, remaining)
        if actual_batch_size <= 0:
            break

        wf_col, cond, *_ = ctx.loader.get_batch(actual_batch_size)

        if ctx.use_ms_data:
            xc1 = cond[:, 0]
            yc1 = cond[:, 1]
            xc2 = cond[:, 2]
            yc2 = cond[:, 3]
            delta_mu = cond[:, 4]
            delta_bins = cond[:, 5]
            all_xc1.append(xc1)
            all_yc1.append(yc1)
            all_xc2.append(xc2)
            all_yc2.append(yc2)
            all_delta_mu.append(delta_mu)
            all_delta_bins.append(delta_bins)

        wf_norm = ctx.data_stats.normalize(wf_col)
        x = torch.from_numpy(wf_norm).to(ctx.device)
        x_flat = x.view(actual_batch_size * ctx.n_nodes, 1)

        z, _, _ = encoder(x_flat, ctx.A_sparse, ctx.pos, batch_size=actual_batch_size)
        all_latents.append(z.cpu().numpy())
        samples_encoded += actual_batch_size

    latents = np.concatenate(all_latents, axis=0)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        f.create_dataset('latents', data=latents, dtype=np.float32)

        if all_delta_mu:
            f.create_dataset('delta_mu', data=np.concatenate(all_delta_mu), dtype=np.float32)
            f.create_dataset('delta_bins', data=np.concatenate(all_delta_bins), dtype=np.float32)
            f.create_dataset('xc1', data=np.concatenate(all_xc1), dtype=np.float32)
            f.create_dataset('yc1', data=np.concatenate(all_yc1), dtype=np.float32)
            f.create_dataset('xc2', data=np.concatenate(all_xc2), dtype=np.float32)
            f.create_dataset('yc2', data=np.concatenate(all_yc2), dtype=np.float32)

        f.attrs['latent_dim'] = latent_dim
        f.attrs['n_samples'] = samples_encoded
        f.attrs['data_mean'] = ctx.data_stats.mean
        f.attrs['data_std'] = ctx.data_stats.std
        f.attrs['is_ms_data'] = ctx.use_ms_data

    if verbose:
        print(f"Saved encoded MS dataset to {output_path}: {samples_encoded} samples, latent_dim={latent_dim}")

    return output_path


@torch.no_grad()
def sample_diffae(
    encoder: nn.Module,
    decoder: nn.Module,
    latent_proj: nn.Module,
    schedule: dict,
    A_sparse: torch.Tensor,
    pos: torch.Tensor,
    time_dim: int,
    x_ref: torch.Tensor,
    parametrization: str = 'v',
    pbar: bool = False,
) -> torch.Tensor:
    """
    Sample from DiffAE by encoding reference events then decoding via diffusion.

    Args:
        encoder: Graph encoder
        decoder: Diffusion U-Net decoder
        latent_proj: Latent to conditioning projection
        schedule: Diffusion schedule
        A_sparse: Graph adjacency
        pos: Node positions
        time_dim: Time embedding dimension
        x_ref: Reference events to encode (B, N, 1)
        parametrization: 'v' or 'eps'
        pbar: Show progress bar

    Returns:
        Reconstructed samples (B, 1, N)
    """
    B, N, C = x_ref.shape
    device = x_ref.device

    x_ref_flat = x_ref.view(B * N, C)
    z, _, _ = encoder(x_ref_flat, A_sparse, pos, batch_size=B)
    _ = latent_proj  # conditioning now uses z directly

    x = torch.randn((B, N, C), device=device)
    T = schedule['betas'].shape[0]

    for i in tqdm(reversed(range(T)), desc="Sampling", disable=not pbar, total=T, ncols=150):
        betas_t = schedule['betas'][i]
        sqrt_one_minus_ab_t = schedule['sqrt_one_minus_alphas_cumprod'][i]
        alpha_bar_t = schedule['alphas_cumprod'][i]
        alpha_bar_prev_t = schedule['alphas_cumprod_prev'][i]

        t_emb = sinusoidal_embedding(torch.tensor([i], device=device), time_dim)
        t_emb_batch = t_emb.expand(B, -1)
        cond_full = torch.cat([z, t_emb_batch], dim=-1)

        x_flat = x.view(B * N, C)
        pred_flat = decoder(x_flat, A_sparse, cond_full, pos, batch_size=B)
        pred = pred_flat.view(B, N, C)

        if parametrization == 'eps':
            eps_theta = pred
            x0_pred = (x - sqrt_one_minus_ab_t * eps_theta) / torch.clamp(torch.sqrt(alpha_bar_t), min=1e-8)
        elif parametrization == 'v':
            a = torch.clamp(torch.sqrt(alpha_bar_t), min=1e-8)
            b_coef = torch.clamp(torch.sqrt(1.0 - alpha_bar_t), min=1e-8)
            x0_pred = a * x - b_coef * pred
        else:
            raise ValueError("parametrization must be 'eps' or 'v'")

        coef1 = betas_t * torch.sqrt(torch.clamp(alpha_bar_prev_t, min=1e-12)) / torch.clamp(1.0 - alpha_bar_t, min=1e-12)
        coef2 = torch.clamp(1.0 - alpha_bar_prev_t, min=0.0) * torch.sqrt(torch.clamp(1.0 - betas_t, min=1e-12)) / torch.clamp(1.0 - alpha_bar_t, min=1e-12)
        mean = coef1 * x0_pred + coef2 * x

        if i > 0:
            posterior_var = schedule['posterior_variance'][i]
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(posterior_var) * noise
        else:
            x = mean

    return x.permute(0, 2, 1)  # (B, C, N)


@torch.no_grad()
def sample_from_latent(
    decoder: nn.Module,
    latent_proj: nn.Module,
    schedule: dict,
    A_sparse: torch.Tensor,
    pos: torch.Tensor,
    time_dim: int,
    z: torch.Tensor,
    n_nodes: int,
    parametrization: str = 'v',
    pbar: bool = False
) -> torch.Tensor:
    """
    Sample from a given latent representation (for interpolation, etc.).
    """
    B = z.shape[0]
    device = z.device
    C = 1

    _ = latent_proj  # conditioning now uses z directly

    x = torch.randn((B, n_nodes, C), device=device)
    T = schedule['betas'].shape[0]

    for i in tqdm(reversed(range(T)), desc="Sampling", disable=not pbar, total=T, ncols=150):
        betas_t = schedule['betas'][i]
        sqrt_one_minus_ab_t = schedule['sqrt_one_minus_alphas_cumprod'][i]
        alpha_bar_t = schedule['alphas_cumprod'][i]
        alpha_bar_prev_t = schedule['alphas_cumprod_prev'][i]

        t_emb = sinusoidal_embedding(torch.tensor([i], device=device), time_dim)
        t_emb_batch = t_emb.expand(B, -1)
        cond_full = torch.cat([z, t_emb_batch], dim=-1)

        x_flat = x.view(B * n_nodes, C)
        pred_flat = decoder(x_flat, A_sparse, cond_full, pos, batch_size=B)
        pred = pred_flat.view(B, n_nodes, C)

        if parametrization == 'eps':
            eps_theta = pred
            x0_pred = (x - sqrt_one_minus_ab_t * eps_theta) / torch.clamp(torch.sqrt(alpha_bar_t), min=1e-8)
        elif parametrization == 'v':
            a = torch.clamp(torch.sqrt(alpha_bar_t), min=1e-8)
            b_coef = torch.clamp(torch.sqrt(1.0 - alpha_bar_t), min=1e-8)
            x0_pred = a * x - b_coef * pred
        else:
            raise ValueError("parametrization must be 'eps' or 'v'")

        coef1 = betas_t * torch.sqrt(torch.clamp(alpha_bar_prev_t, min=1e-12)) / torch.clamp(1.0 - alpha_bar_t, min=1e-12)
        coef2 = torch.clamp(1.0 - alpha_bar_prev_t, min=0.0) * torch.sqrt(torch.clamp(1.0 - betas_t, min=1e-12)) / torch.clamp(1.0 - alpha_bar_t, min=1e-12)
        mean = coef1 * x0_pred + coef2 * x

        if i > 0:
            posterior_var = schedule['posterior_variance'][i]
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(posterior_var) * noise
        else:
            x = mean

    out = x.permute(0, 2, 1)
    return out


@torch.no_grad()
def sample_diffae_partial(
    encoder: nn.Module,
    decoder: nn.Module,
    latent_proj: nn.Module,
    schedule: dict,
    A_sparse: torch.Tensor,
    pos: torch.Tensor,
    time_dim: int,
    x_ref: torch.Tensor,
    t_start: int,
    parametrization: str = 'v',
) -> torch.Tensor:
    """Reconstruct by forward-noising x_ref to t_start, then reverse-denoising.

    Unlike ``sample_diffae`` which starts from pure noise (t=T), this starts
    from x_{t_start} = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) * noise,
    preserving progressively more of the original signal as t_start decreases.

    Returns: reconstructed samples (B, 1, N).
    """
    B, N, C = x_ref.shape
    device = x_ref.device
    T = schedule['betas'].shape[0]
    t_start = min(t_start, T - 1)

    x_ref_flat = x_ref.view(B * N, C)
    z, _, _ = encoder(x_ref_flat, A_sparse, pos, batch_size=B)
    _ = latent_proj  # conditioning now uses z directly

    sqrt_ab = schedule['sqrt_alphas_cumprod'][t_start]
    sqrt_om = schedule['sqrt_one_minus_alphas_cumprod'][t_start]
    noise = torch.randn_like(x_ref)
    x = sqrt_ab * x_ref + sqrt_om * noise

    for i in reversed(range(t_start + 1)):
        betas_t = schedule['betas'][i]
        sqrt_one_minus_ab_t = schedule['sqrt_one_minus_alphas_cumprod'][i]
        alpha_bar_t = schedule['alphas_cumprod'][i]
        alpha_bar_prev_t = schedule['alphas_cumprod_prev'][i]

        t_emb = sinusoidal_embedding(torch.tensor([i], device=device), time_dim)
        t_emb_batch = t_emb.expand(B, -1)
        cond_full = torch.cat([z, t_emb_batch], dim=-1)

        x_flat = x.view(B * N, C)
        pred_flat = decoder(x_flat, A_sparse, cond_full, pos, batch_size=B)
        pred = pred_flat.view(B, N, C)

        if parametrization == 'eps':
            eps_theta = pred
            x0_pred = (x - sqrt_one_minus_ab_t * eps_theta) / torch.clamp(torch.sqrt(alpha_bar_t), min=1e-8)
        elif parametrization == 'v':
            a = torch.clamp(torch.sqrt(alpha_bar_t), min=1e-8)
            b_coef = torch.clamp(torch.sqrt(1.0 - alpha_bar_t), min=1e-8)
            x0_pred = a * x - b_coef * pred
        else:
            raise ValueError("parametrization must be 'eps' or 'v'")

        coef1 = betas_t * torch.sqrt(torch.clamp(alpha_bar_prev_t, min=1e-12)) / torch.clamp(1.0 - alpha_bar_t, min=1e-12)
        coef2 = torch.clamp(1.0 - alpha_bar_prev_t, min=0.0) * torch.sqrt(torch.clamp(1.0 - betas_t, min=1e-12)) / torch.clamp(1.0 - alpha_bar_t, min=1e-12)
        mean = coef1 * x0_pred + coef2 * x

        if i > 0:
            posterior_var = schedule['posterior_variance'][i]
            x = mean + torch.sqrt(posterior_var) * torch.randn_like(x)
        else:
            x = mean

    return x.permute(0, 2, 1)


def train_diffae(cfg: Config = default_config):
    """Main DiffAE training function."""
    print_config(cfg, include_encoder=True)

    ctx = DiffAEContext.build(cfg, for_training=True, verbose=True)

    device_t = ctx.device
    encoder = ctx.encoder
    decoder = ctx.decoder
    latent_proj = ctx.latent_proj
    regressive_decoder = ctx.regressive_decoder
    ema_encoder = ctx.ema_encoder
    ema_decoder = ctx.ema_decoder
    ema_regressive_decoder = ctx.ema_regressive_decoder
    optim = ctx.optim
    schedule = ctx.schedule
    data_stats = ctx.data_stats
    A_sparse = ctx.A_sparse
    pos = ctx.pos
    n_nodes = ctx.n_nodes
    n_channels = ctx.n_channels
    n_time_points = ctx.n_time_points
    graph = ctx.graph
    tr = ctx.loader
    channel_positions = tr.channel_positions
    use_regressive = cfg.encoder.use_regressive_head and regressive_decoder is not None

    start_epoch = 0
    if cfg.resume:
        last = ctx.latest_checkpoint()
        if last is not None:
            try:
                start_epoch = ctx.load_checkpoint(last) + 1
                print(f"Resumed from epoch {start_epoch}")
            except Exception as e:
                print(f"Could not resume exact checkpoint: {e}")
                start_epoch = 0

        if start_epoch == 0:
            best_ckpt = ctx.find_best_checkpoint()
            if best_ckpt is not None:
                print(f"Attempting partial load from: {best_ckpt}")
                try:
                    _, is_full = ctx.load_checkpoint_partial(best_ckpt, verbose=True)
                    if is_full:
                        print("Full checkpoint loaded (different directory)")
                    else:
                        print("Partial weights loaded - latent layers freshly initialized")
                except Exception as e:
                    print(f"Could not load partial checkpoint: {e}")

    for g in optim.param_groups:
        g["lr"] = cfg.training.lr

    B = cfg.training.batch_size

    global_step = start_epoch * cfg.training.steps_per_epoch
    encoded_output_path = os.path.join(ctx.checkpoint_dir, "encoded_ms_latents.h5")

    if cfg.training.lopsided_aug:
        print(f"  Lopsided augmentation ON: frac={cfg.training.lopsided_frac}, sigma={cfg.training.lopsided_sigma}")

    for epoch in range(start_epoch, cfg.training.epochs):
        encoder.train()
        decoder.train()
        latent_proj.train()
        if use_regressive:
            regressive_decoder.train()
        epoch_loss = 0.0
        epoch_kl = 0.0
        epoch_reg_loss = 0.0
        pbar = tqdm(range(cfg.training.steps_per_epoch), desc=f"Epoch {epoch+1}/{cfg.training.epochs}", ncols=120, file=sys.stdout)

        for step in pbar:
            batch_np, _, sample_idx = tr.get_batch(B)
            if cfg.training.lopsided_aug:
                batch_np = apply_lopsided_augmentation(
                    batch_np, frac=cfg.training.lopsided_frac, sigma=cfg.training.lopsided_sigma,
                    sample_indices=sample_idx)
            batch_np = data_stats.normalize(batch_np)

            x0 = torch.from_numpy(batch_np.astype(np.float32)).to(device_t)  # (B, N, 1)
            x0_flat = x0.view(B * n_nodes, 1)

            z, mu, logvar = encoder(x0_flat, A_sparse, pos, batch_size=B)

            if step == 0 and epoch % 50 == 0:
                with torch.no_grad():
                    z_std = z.std().item()
                    z_sim = 0.0
                    if B > 1:
                        z_norm = z / (z.norm(dim=1, keepdim=True) + 1e-8)
                        z_sim = (z_norm @ z_norm.T).fill_diagonal_(0).abs().mean().item()
                    print(f"\n  [Monitor] Latent z: std={z_std:.4f}, within-batch similarity={z_sim:.4f}")
                    if z_sim > 0.95:
                        print(f"  [WARNING] Latent similarity is very high - encoder may be collapsing!")

            t = torch.randint(0, cfg.diffusion.timesteps, (B,), device=device_t, dtype=torch.long)
            t_emb = sinusoidal_embedding(t, cfg.conditioning.time_dim)
            cond_full = torch.cat([z, t_emb], dim=-1)

            sqrt_ab = schedule['sqrt_alphas_cumprod'][t].view(B, 1, 1)
            sqrt_om = schedule['sqrt_one_minus_alphas_cumprod'][t].view(B, 1, 1)
            snr_t = schedule['snr'][t].view(B)

            noise = torch.randn_like(x0)
            x_t = sqrt_ab * x0 + sqrt_om * noise
            x_t_flat = x_t.view(B * n_nodes, 1)

            pred_flat = decoder(x_t_flat, A_sparse, cond_full, pos, batch_size=B)
            pred = pred_flat.view(B, n_nodes, 1)

            if cfg.diffusion.parametrization == "eps":
                target = noise
            elif cfg.diffusion.parametrization == "v":
                target = sqrt_ab * noise - sqrt_om * x0
            else:
                raise ValueError("parametrization must be 'eps' or 'v'")

            mse_per_sample = F.mse_loss(pred, target, reduction='none').mean(dim=(1, 2))

            if cfg.diffusion.p2_gamma > 0.0:
                weight = torch.pow(cfg.diffusion.p2_k + snr_t, -cfg.diffusion.p2_gamma)
                mse_per_sample = mse_per_sample * weight

            loss = mse_per_sample.mean()

            if use_regressive:
                reg_flat = regressive_decoder(z, A_sparse, pos, batch_size=B)
                reg_pred = reg_flat.view(B, n_nodes, 1)
                reg_loss = F.mse_loss(reg_pred, x0, reduction='mean')
                loss = loss + cfg.encoder.regressive_head_weight * reg_loss
                epoch_reg_loss += reg_loss.item()

            if cfg.encoder.use_stochastic and mu is not None and logvar is not None:
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = loss + cfg.encoder.kl_weight * kl_loss
                epoch_kl += kl_loss.item()

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  WARNING: NaN/Inf loss at step {step}! Skipping...")
                optim.zero_grad(set_to_none=True)
                continue

            epoch_loss += float(loss.item())

            optim.zero_grad(set_to_none=True)
            loss.backward()

            clip_params = list(encoder.parameters()) + list(decoder.parameters()) + list(latent_proj.parameters())
            if use_regressive:
                clip_params += list(regressive_decoder.parameters())
            torch.nn.utils.clip_grad_norm_(clip_params, max_norm=cfg.training.grad_clip)
            optim.step()

            with torch.no_grad():
                for p_ema, p in zip(ema_encoder.parameters(), encoder.parameters()):
                    p_ema.data.mul_(cfg.training.ema_decay).add_(p.data, alpha=1.0 - cfg.training.ema_decay)
                for p_ema, p in zip(ema_decoder.parameters(), decoder.parameters()):
                    p_ema.data.mul_(cfg.training.ema_decay).add_(p.data, alpha=1.0 - cfg.training.ema_decay)
                for p_ema, p in zip(ctx.ema_latent_proj.parameters(), latent_proj.parameters()):
                    p_ema.data.mul_(cfg.training.ema_decay).add_(p.data, alpha=1.0 - cfg.training.ema_decay)
                if use_regressive:
                    for p_ema, p in zip(ema_regressive_decoder.parameters(), regressive_decoder.parameters()):
                        p_ema.data.mul_(cfg.training.ema_decay).add_(p.data, alpha=1.0 - cfg.training.ema_decay)
            global_step += 1

            postfix = {"loss": epoch_loss / (step + 1)}
            if cfg.encoder.use_stochastic:
                postfix["kl"] = epoch_kl / (step + 1)
            if use_regressive:
                postfix["reg"] = epoch_reg_loss / (step + 1)
            pbar.set_postfix(**postfix)

        if (epoch + 1) % cfg.training.checkpoint_every == 0:
            ctx.save_checkpoint(epoch)

        if cfg.training.encode_dataset_every > 0 and (epoch + 1) % cfg.training.encode_dataset_every == 0:
            ema_encoder.eval()
            save_encoded_dataset(ctx, encoded_output_path, encoder=ema_encoder, batch_size=B * 4, n_samples=cfg.training.encode_n_samples)
            encoder.train()

        if cfg.visualize and (epoch % cfg.training.visualize_every == 0 or epoch == cfg.training.epochs - 1):
            ema_encoder.eval()
            ema_decoder.eval()
            ctx.ema_latent_proj.eval()
            with torch.no_grad():
                b_vis = min(cfg.training.batch_size, 4)
                batch_np, _, sample_idx = tr.get_batch(b_vis)
                if cfg.training.lopsided_aug:
                    batch_np = apply_lopsided_augmentation(
                        batch_np, frac=cfg.training.lopsided_frac, sigma=cfg.training.lopsided_sigma,
                        sample_indices=sample_idx)
                batch_np_norm = data_stats.normalize(batch_np)
                x_ref = torch.from_numpy(batch_np_norm.astype(np.float32)).to(device_t)

                samples = sample_diffae(
                    encoder=ema_encoder,
                    decoder=ema_decoder,
                    latent_proj=ctx.ema_latent_proj,
                    schedule=schedule,
                    A_sparse=A_sparse,
                    pos=pos,
                    time_dim=cfg.conditioning.time_dim,
                    x_ref=x_ref,
                    parametrization=cfg.diffusion.parametrization,
                )
                samples_denorm = data_stats.denormalize(samples.cpu().numpy())
                samples_denorm = np.clip(samples_denorm, 0, None)

                true_data = batch_np[:, :, 0]
                gen_data = samples_denorm[:, 0, :]
                print(f"\n  [Vis] True data - mean: {true_data.mean():.4f}, std: {true_data.std():.4f}")
                print(f"  [Vis] Gen data  - mean: {gen_data.mean():.4f}, std: {gen_data.std():.4f}")

            plots_dir = f"{ctx.plot_dir}/epoch_{epoch}"
            os.makedirs(plots_dir, exist_ok=True)

            adj2d = build_xy_adjacency_radius(channel_positions, radius=cfg.graph.radius)
            Gxy = Graph(adjacency=adj2d, positions_xy=channel_positions, positions_z=np.zeros(n_channels, dtype=np.float32))
            Gz = Graph(adjacency=np.eye(n_channels, dtype=np.float32), positions_xy=channel_positions, positions_z=np.arange(n_time_points, dtype=np.float32))

            for idx in range(samples.shape[0]):
                rec_int = samples_denorm[idx, 0]
                true_int = batch_np[idx, :, 0]

                rec_xy = rec_int.reshape(n_channels, n_time_points, order='F').sum(axis=1)
                true_xy = true_int.reshape(n_channels, n_time_points, order='F').sum(axis=1)
                rec_z = rec_int.reshape(n_channels, n_time_points, order='F')
                true_z = true_int.reshape(n_channels, n_time_points, order='F')

                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                visualize_event(Gxy, true_xy, None, ax=axes[0])
                axes[0].set_title("Ground truth")
                visualize_event(Gxy, rec_xy, None, ax=axes[1])
                axes[1].set_title("DiffAE reconstruction")
                plt.tight_layout()
                fig.savefig(f"{plots_dir}/event_{idx}_xy.png")
                plt.close(fig)

                fig, axes = plt.subplots(1, 2, figsize=(10, 3))
                visualize_event_z(Gz, true_z, None, ax=axes[0])
                axes[0].set_title("Ground truth")
                visualize_event_z(Gz, rec_z, None, ax=axes[1])
                axes[1].set_title("DiffAE reconstruction")
                plt.tight_layout()
                fig.savefig(f"{plots_dir}/event_{idx}_z.png")
                plt.close(fig)

                # Temporal cross-sections for strongest channels.
                top_k = min(4, n_channels)
                top_channels = np.argsort(true_xy)[-top_k:][::-1]
                fig, axes = plt.subplots(top_k, 1, figsize=(10, 2.2 * top_k), sharex=True)
                if top_k == 1:
                    axes = [axes]
                t_axis = np.arange(n_time_points)
                for ax, ch in zip(axes, top_channels):
                    ax.plot(t_axis, true_z[ch], color="black", linewidth=1.2, label="truth")
                    ax.plot(t_axis, rec_z[ch], color="#d62728", linewidth=1.0, alpha=0.9, label="recon")
                    ax.set_ylabel(f"ch {ch}")
                    ax.grid(alpha=0.25, linewidth=0.4)
                axes[0].legend(loc="upper right", fontsize=8)
                axes[-1].set_xlabel("time bin")
                fig.suptitle("Cross-section comparison (top channels)")
                plt.tight_layout()
                fig.savefig(f"{plots_dir}/event_{idx}_cross_sections.png")
                plt.close(fig)


def interpolate_latents(cfg: Config = default_config, n_steps: int = 5):
    """Generate interpolations between two events in latent space."""
    ctx = DiffAEContext.build(cfg, for_training=False, verbose=True)

    latest_ckpt = ctx.latest_checkpoint()
    if latest_ckpt is None:
        raise FileNotFoundError(f"No checkpoints found in {ctx.checkpoint_dir}")
    print(f"Loading checkpoint: {latest_ckpt}")
    ctx.load_checkpoint(latest_ckpt, load_optim=False)

    ctx.encoder.eval()
    ctx.decoder.eval()

    with torch.no_grad():
        batch_np, *_ = ctx.loader.get_batch(2)
        batch_np_norm = ctx.data_stats.normalize(batch_np)
        x_ref = torch.from_numpy(batch_np_norm.astype(np.float32)).to(ctx.device)

        x_ref_flat = x_ref.view(2 * ctx.n_nodes, 1)
        z, _, _ = ctx.encoder(x_ref_flat, ctx.A_sparse, ctx.pos, batch_size=2)
        z1, z2 = z[0], z[1]

        alphas = torch.linspace(0, 1, n_steps, device=ctx.device)
        z_interp = torch.stack([(1 - a) * z1 + a * z2 for a in alphas])

        samples = sample_from_latent(
            decoder=ctx.decoder,
            latent_proj=ctx.latent_proj,
            schedule=ctx.schedule,
            A_sparse=ctx.A_sparse,
            pos=ctx.pos,
            time_dim=cfg.conditioning.time_dim,
            z=z_interp,
            n_nodes=ctx.n_nodes,
            parametrization=cfg.diffusion.parametrization,
            pbar=True,
        )

        samples_denorm = ctx.data_stats.denormalize(samples.cpu().numpy())
        samples_denorm = np.clip(samples_denorm, 0, None)

    plots_dir = f"{ctx.plot_dir}/interpolation"
    os.makedirs(plots_dir, exist_ok=True)

    channel_positions = ctx.loader.channel_positions
    adj2d = build_xy_adjacency_radius(channel_positions, radius=cfg.graph.radius)

    fig, axes = plt.subplots(1, n_steps + 2, figsize=(3 * (n_steps + 2), 3))

    true1 = batch_np[0, :, 0].reshape(ctx.n_channels, ctx.n_time_points, order='F').sum(axis=1)
    true2 = batch_np[1, :, 0].reshape(ctx.n_channels, ctx.n_time_points, order='F').sum(axis=1)

    Gxy = Graph(adjacency=adj2d, positions_xy=channel_positions, positions_z=np.zeros(ctx.n_channels, dtype=np.float32))
    visualize_event(Gxy, true1, None, ax=axes[0])
    axes[0].set_title("Event A")

    for i in range(n_steps):
        interp_xy = samples_denorm[i, 0].reshape(ctx.n_channels, ctx.n_time_points, order='F').sum(axis=1)
        visualize_event(Gxy, interp_xy, None, ax=axes[i + 1])
        axes[i + 1].set_title(f"α={alphas[i].item():.2f}")

    visualize_event(Gxy, true2, None, ax=axes[-1])
    axes[-1].set_title("Event B")

    plt.tight_layout()
    fig.savefig(f"{plots_dir}/interpolation.png", dpi=150)
    plt.close(fig)
    print(f"Saved interpolation to {plots_dir}/interpolation.png")

    return samples_denorm


if __name__ == "__main__":
    cfg = get_config(epochs=20_000)
    train_diffae(cfg)
