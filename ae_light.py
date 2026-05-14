"""
AE Light: autoencoder built from the light graph encoder and latent decoder.
"""
import glob
import os
import sys
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import h5py
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm

matplotlib.use("Agg")

from ae import (
    Conv1DEncoder,
    DiffAEDataStats,
    MLPDecoder,
    MLPEncoder,
    apply_lopsided_augmentation,
    corrupt_ae_input,
)
from config import Config, default_config, get_config, print_config
from data import Graph, SparseGraph, visualize_event, visualize_event_z
from diffae_light import (
    GraphPyramid,
    LightGraphEncoder,
    LightLatentDecoder,
    _ema_update,
    _load_compatible_state_dict,
    build_graph_pyramid,
)
from data_loader import OnlineMSBatcher, TritiumSSDataLoader
from utils.run_paths import (
    ensure_parent_dir,
    epoch_plot_dir,
    latest_checkpoint,
    latest_checkpoint_across_runs,
    resolve_model_run_dirs,
)
from utils.visualization import build_xy_adjacency_radius


DataLoaderType = Union[TritiumSSDataLoader, OnlineMSBatcher]


@dataclass
class AELightContext:
    cfg: Config
    device: torch.device
    loader: DataLoaderType
    graph: SparseGraph
    n_channels: int
    n_time_points: int
    n_nodes: int
    data_stats: DiffAEDataStats
    encoder: nn.Module
    decoder: nn.Module
    encoder_pyramid: Optional[GraphPyramid]
    decoder_pyramid: GraphPyramid
    checkpoint_dir: str = ""
    plot_dir: str = ""
    ema_encoder: Optional[nn.Module] = None
    ema_decoder: Optional[nn.Module] = None
    optim: Optional[torch.optim.Optimizer] = None
    use_ms_data: bool = False

    @property
    def A_sparse(self) -> torch.Tensor:
        return self.decoder_pyramid.levels[0].adj

    @property
    def pos(self) -> torch.Tensor:
        return self.decoder_pyramid.levels[0].pos

    @classmethod
    def build(
        cls,
        cfg: Config,
        for_training: bool = True,
        verbose: bool = True,
        use_ms_data: bool = True,
    ) -> "AELightContext":
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
            z_hops=cfg.graph.z_hops,
            weighted=cfg.graph.weighted_edges,
            lpe_dim=cfg.graph.lpe_dim,
        )

        n_channels = loader.n_channels
        n_time_points = loader.n_time_points
        n_nodes = n_channels * n_time_points

        if verbose:
            print(f"Graph: {n_nodes} nodes, {graph.adjacency._nnz()} edges")
            print("Computing data statistics...")

        data_stats = DiffAEDataStats.from_loader(loader, n_samples=1000, batch_size=32)
        if verbose:
            print(f"Data mean: {data_stats.mean:.4f}, std: {data_stats.std:.4f}")

        decoder_pyramid = build_graph_pyramid(
            graph=graph,
            n_channels=n_channels,
            n_time_points=n_time_points,
            depth=cfg.model.depth,
            pool_ratio=cfg.model.pool_ratio,
            pos_dim=cfg.model.pos_dim,
            weighted_edges=cfg.graph.weighted_edges,
            device=device,
        )
        if verbose:
            print(f"Decoder pyramid nodes: {[level.n_nodes for level in decoder_pyramid.levels]}")

        encoder_type = cfg.encoder.encoder_type.lower()
        encoder_pyramid: Optional[GraphPyramid] = None
        if encoder_type == "graph":
            same_pyramid = (
                cfg.encoder.depth == cfg.model.depth
                and abs(cfg.encoder.pool_ratio - cfg.model.pool_ratio) < 1e-8
            )
            if same_pyramid:
                encoder_pyramid = decoder_pyramid
            else:
                encoder_pyramid = build_graph_pyramid(
                    graph=graph,
                    n_channels=n_channels,
                    n_time_points=n_time_points,
                    depth=cfg.encoder.depth,
                    pool_ratio=cfg.encoder.pool_ratio,
                    pos_dim=cfg.model.pos_dim,
                    weighted_edges=cfg.graph.weighted_edges,
                    device=device,
                )
            lpe0 = encoder_pyramid.levels[0].lpe
            actual_lpe_dim = 0 if lpe0 is None else int(lpe0.size(1))
            encoder = LightGraphEncoder(
                in_dim=cfg.model.in_dim,
                hidden_dim=cfg.encoder.hidden_dim,
                latent_dim=cfg.encoder.latent_dim,
                latent_head_dim=cfg.encoder.latent_head_dim,
                anchor_dim=cfg.encoder.latent_anchor_dim,
                anchor_count=cfg.encoder.latent_anchor_count,
                anchor_value_dim=cfg.encoder.latent_anchor_value_dim,
                num_scales=len(encoder_pyramid.levels),
                blocks_per_stage=cfg.encoder.blocks_per_stage,
                dropout=cfg.encoder.dropout,
                pos_dim=cfg.model.pos_dim,
                lpe_dim=actual_lpe_dim,
                use_stochastic=cfg.encoder.use_stochastic,
            ).to(device)
        elif encoder_type == "cnn":
            encoder = Conv1DEncoder(
                in_dim=cfg.model.in_dim,
                latent_dim=cfg.encoder.latent_dim,
                n_nodes=n_nodes,
                dropout=cfg.encoder.dropout,
                use_stochastic=cfg.encoder.use_stochastic,
                channels=tuple(cfg.encoder.conv_channels),
                kernel_size=cfg.encoder.conv_kernel_size,
                pool_size=cfg.encoder.conv_pool_size,
            ).to(device)
        elif encoder_type == "mlp":
            encoder = MLPEncoder(
                in_dim=cfg.model.in_dim,
                hidden_dim=cfg.encoder.mlp_hidden_dim,
                latent_dim=cfg.encoder.latent_dim,
                n_nodes=n_nodes,
                num_layers=cfg.encoder.mlp_encoder_layers,
                dropout=cfg.encoder.dropout,
                use_stochastic=cfg.encoder.use_stochastic,
            ).to(device)
        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")

        decoder_type = cfg.encoder.decoder_type.lower()
        if decoder_type == "mlp":
            decoder = MLPDecoder(
                latent_dim=cfg.encoder.latent_dim,
                hidden_dim=cfg.encoder.mlp_decoder_hidden_dim,
                out_dim=cfg.model.out_dim,
                n_nodes=n_nodes,
                num_layers=cfg.encoder.mlp_decoder_layers,
                dropout=cfg.encoder.dropout,
            ).to(device)
        else:
            decoder = LightLatentDecoder(
                out_dim=cfg.model.out_dim,
                hidden_dim=cfg.encoder.regressive_hidden_dim,
                cond_dim=cfg.encoder.latent_dim,
                num_scales=len(decoder_pyramid.levels),
                blocks_per_stage=cfg.encoder.blocks_per_stage,
                dropout=cfg.encoder.dropout,
                pos_dim=cfg.model.pos_dim,
                anchor_count=cfg.encoder.latent_anchor_count,
                anchor_value_dim=max(cfg.encoder.latent_anchor_value_dim, cfg.encoder.regressive_hidden_dim // 4),
            ).to(device)

        ema_encoder = None
        ema_decoder = None
        optim = None
        checkpoint_dir, plot_dir = resolve_model_run_dirs(
            cfg,
            "ae_light_subdir",
            create=for_training,
        )

        if for_training:
            ema_encoder = deepcopy(encoder).to(device)
            ema_decoder = deepcopy(decoder).to(device)
            optim = torch.optim.AdamW(
                list(encoder.parameters()) + list(decoder.parameters()),
                lr=cfg.training.lr,
                betas=(0.9, 0.999),
                weight_decay=cfg.training.weight_decay,
            )
        if verbose:
            n_enc = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
            n_dec = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
            print(f"Encoder parameters: {n_enc:,}")
            print(f"Decoder parameters: {n_dec:,}")
            print(f"Total trainable parameters: {n_enc + n_dec:,}")

        return cls(
            cfg=cfg,
            device=device,
            loader=loader,
            graph=graph,
            n_channels=n_channels,
            n_time_points=n_time_points,
            n_nodes=n_nodes,
            data_stats=data_stats,
            encoder=encoder,
            decoder=decoder,
            encoder_pyramid=encoder_pyramid,
            decoder_pyramid=decoder_pyramid,
            checkpoint_dir=checkpoint_dir,
            plot_dir=plot_dir,
            ema_encoder=ema_encoder,
            ema_decoder=ema_decoder,
            optim=optim,
            use_ms_data=use_ms_data,
        )

    def latest_checkpoint(self) -> Optional[str]:
        return latest_checkpoint(self.checkpoint_dir, "ae_light_epoch_*.pt")

    def save_checkpoint(self, epoch: int) -> str:
        state = {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "ema_encoder": self.ema_encoder.state_dict() if self.ema_encoder else self.encoder.state_dict(),
            "ema_decoder": self.ema_decoder.state_dict() if self.ema_decoder else self.decoder.state_dict(),
            "optim": self.optim.state_dict() if self.optim else None,
            "epoch": epoch,
            "data_stats": {"mean": self.data_stats.mean, "std": self.data_stats.std},
        }
        path = os.path.join(self.checkpoint_dir, f"ae_light_epoch_{epoch:04d}.pt")
        torch.save(state, path)
        return path

    def load_checkpoint(self, path: str, load_optim: bool = True) -> int:
        chk = torch.load(path, map_location=self.device)
        _load_compatible_state_dict(self.encoder, chk["encoder"], "encoder")
        _load_compatible_state_dict(self.decoder, chk["decoder"], "decoder")
        if self.ema_encoder is not None and "ema_encoder" in chk:
            _load_compatible_state_dict(self.ema_encoder, chk["ema_encoder"], "ema_encoder")
        if self.ema_decoder is not None and "ema_decoder" in chk:
            _load_compatible_state_dict(self.ema_decoder, chk["ema_decoder"], "ema_decoder")
        if load_optim and self.optim is not None and chk.get("optim"):
            try:
                self.optim.load_state_dict(chk["optim"])
            except (ValueError, RuntimeError) as exc:
                print(f"  Optimizer state skipped: {exc}")
        if "data_stats" in chk:
            self.data_stats.mean = chk["data_stats"]["mean"]
            self.data_stats.std = chk["data_stats"]["std"]
        return int(chk.get("epoch", 0))

    def find_best_checkpoint(self) -> Optional[str]:
        same_latent = self.latest_checkpoint()
        if same_latent is not None:
            return same_latent

        return latest_checkpoint_across_runs(self.checkpoint_dir, "ae_light_z", "ae_light_epoch_*.pt")


def _encode_with_context(
    ctx: AELightContext,
    x_flat: torch.Tensor,
    batch_size: int,
    encoder: Optional[nn.Module] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    enc = ctx.encoder if encoder is None else encoder
    if isinstance(enc, LightGraphEncoder):
        if ctx.encoder_pyramid is None:
            raise RuntimeError("Light graph encoder requested but encoder_pyramid is missing.")
        return enc(x_flat, ctx.encoder_pyramid, batch_size=batch_size)
    return enc(x_flat, ctx.A_sparse, ctx.pos, batch_size=batch_size)


def _decode_with_context(
    ctx: AELightContext,
    z: torch.Tensor,
    batch_size: int,
    decoder: Optional[nn.Module] = None,
) -> torch.Tensor:
    dec = ctx.decoder if decoder is None else decoder
    if isinstance(dec, LightLatentDecoder):
        return dec(z, ctx.decoder_pyramid, batch_size=batch_size)
    return dec(z, ctx.A_sparse, ctx.pos, batch_size=batch_size)


@torch.no_grad()
def save_encoded_dataset(
    ctx: AELightContext,
    output_path: str,
    encoder: Optional[nn.Module] = None,
    batch_size: int = 32,
    n_samples: int = 10000,
    verbose: bool = True,
) -> str:
    enc = ctx.ema_encoder if encoder is None and ctx.ema_encoder is not None else (ctx.encoder if encoder is None else encoder)
    enc.eval()

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
    for _ in pbar:
        actual_batch_size = min(batch_size, n_samples - samples_encoded)
        if actual_batch_size <= 0:
            break

        wf_col, cond, *_ = ctx.loader.get_batch(actual_batch_size)
        if ctx.use_ms_data:
            all_xc1.append(cond[:, 0])
            all_yc1.append(cond[:, 1])
            all_xc2.append(cond[:, 2])
            all_yc2.append(cond[:, 3])
            all_delta_mu.append(cond[:, 4])
            all_delta_bins.append(cond[:, 5])

        wf_norm = ctx.data_stats.normalize(wf_col)
        x_flat = torch.from_numpy(wf_norm.astype(np.float32)).to(ctx.device).reshape(actual_batch_size * ctx.n_nodes, 1)
        z, _, _ = _encode_with_context(ctx, x_flat, actual_batch_size, encoder=enc)
        all_latents.append(z.cpu().numpy())
        samples_encoded += actual_batch_size

    ensure_parent_dir(output_path)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("latents", data=np.concatenate(all_latents, axis=0), dtype=np.float32)
        if all_delta_mu:
            f.create_dataset("delta_mu", data=np.concatenate(all_delta_mu), dtype=np.float32)
            f.create_dataset("delta_bins", data=np.concatenate(all_delta_bins), dtype=np.float32)
            f.create_dataset("xc1", data=np.concatenate(all_xc1), dtype=np.float32)
            f.create_dataset("yc1", data=np.concatenate(all_yc1), dtype=np.float32)
            f.create_dataset("xc2", data=np.concatenate(all_xc2), dtype=np.float32)
            f.create_dataset("yc2", data=np.concatenate(all_yc2), dtype=np.float32)
        f.attrs["latent_dim"] = ctx.cfg.encoder.latent_dim
        f.attrs["n_samples"] = samples_encoded
        f.attrs["data_mean"] = ctx.data_stats.mean
        f.attrs["data_std"] = ctx.data_stats.std
        f.attrs["is_ms_data"] = ctx.use_ms_data

    if verbose:
        print(f"Saved encoded MS dataset to {output_path}: {samples_encoded} samples, latent_dim={ctx.cfg.encoder.latent_dim}")
    return output_path


@torch.no_grad()
def reconstruct_ae_light(
    ctx: AELightContext,
    x_ref: torch.Tensor,
    encoder: Optional[nn.Module] = None,
    decoder: Optional[nn.Module] = None,
) -> torch.Tensor:
    batch_size, n_nodes, channels = x_ref.shape
    x_flat = x_ref.reshape(batch_size * n_nodes, channels)
    z, _, _ = _encode_with_context(ctx, x_flat, batch_size, encoder=encoder)
    rec_flat = _decode_with_context(ctx, z, batch_size, decoder=decoder)
    return rec_flat.reshape(batch_size, n_nodes, channels).permute(0, 2, 1)


@torch.no_grad()
def sample_from_latent(
    ctx: AELightContext,
    z: torch.Tensor,
    decoder: Optional[nn.Module] = None,
) -> torch.Tensor:
    batch_size = z.shape[0]
    rec_flat = _decode_with_context(ctx, z, batch_size, decoder=decoder)
    return rec_flat.reshape(batch_size, ctx.n_nodes, 1).permute(0, 2, 1)


def train_ae_light(cfg: Optional[Config] = None) -> None:
    cfg = get_config() if cfg is None else cfg
    print("=" * 50)
    print("AE Light Training")
    print("=" * 50)
    print_config(cfg, include_encoder=True)

    ctx = AELightContext.build(cfg, for_training=True, verbose=True)
    encoder = ctx.encoder
    decoder = ctx.decoder
    ema_encoder = ctx.ema_encoder
    ema_decoder = ctx.ema_decoder
    optim = ctx.optim
    assert optim is not None

    if cfg.resume:
        last = ctx.latest_checkpoint()
        if last is not None:
            start_epoch = ctx.load_checkpoint(last) + 1
            print(f"Resumed from epoch {start_epoch}")
        else:
            start_epoch = 0
    else:
        start_epoch = 0

    for group in optim.param_groups:
        group["lr"] = cfg.training.lr

    batch_size = cfg.training.batch_size
    steps_per_epoch = cfg.training.resolved_steps_per_epoch(ctx.loader.n_samples)
    encoded_output_path = os.path.join(ctx.checkpoint_dir, "ae_light_encoded_ms_latents.h5")
    amp_enabled = cfg.training.use_amp and ctx.device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float32

    if cfg.training.lopsided_aug:
        print(f"  Lopsided augmentation ON: frac={cfg.training.lopsided_frac}, sigma={cfg.training.lopsided_sigma}")
    print(
        f"  AE denoising: {cfg.training.ae_denoising} "
        f"(noise_std={cfg.training.ae_input_noise_std}, mask_prob={cfg.training.ae_mask_prob}), "
        f"latent_l1={cfg.training.ae_latent_l1_weight}"
    )

    for epoch in range(start_epoch, cfg.training.epochs):
        encoder.train()
        decoder.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_l1 = 0.0
        epoch_kl = 0.0
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{cfg.training.epochs}", ncols=120, file=sys.stdout)

        for step in pbar:
            batch_np, _, sample_idx = ctx.loader.get_batch(batch_size)
            if cfg.training.lopsided_aug:
                batch_np = apply_lopsided_augmentation(
                    batch_np,
                    frac=cfg.training.lopsided_frac,
                    sigma=cfg.training.lopsided_sigma,
                    sample_indices=sample_idx,
                )
            batch_np = ctx.data_stats.normalize(batch_np)

            x0 = torch.from_numpy(batch_np.astype(np.float32)).to(ctx.device)
            x_in = x0
            if cfg.training.ae_denoising:
                x_in = corrupt_ae_input(
                    x0,
                    noise_std=cfg.training.ae_input_noise_std,
                    mask_prob=cfg.training.ae_mask_prob,
                )
            x_in_flat = x_in.reshape(batch_size * ctx.n_nodes, 1)

            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast(ctx.device.type, dtype=amp_dtype, enabled=amp_enabled):
                z, mu, logvar = _encode_with_context(ctx, x_in_flat, batch_size)
                rec_flat = _decode_with_context(ctx, z, batch_size)
                rec = rec_flat.reshape(batch_size, ctx.n_nodes, 1)
                recon_loss = F.mse_loss(rec, x0, reduction="mean")
                l1_loss = z.abs().mean() if cfg.training.ae_latent_l1_weight > 0 else torch.zeros((), device=z.device)
                loss = recon_loss + cfg.training.ae_latent_l1_weight * l1_loss
                if cfg.encoder.use_stochastic and mu is not None and logvar is not None:
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = loss + cfg.encoder.kl_weight * kl_loss
                else:
                    kl_loss = torch.zeros((), device=z.device)

            if not torch.isfinite(loss):
                print(f"  WARNING: non-finite loss at step {step}, skipping")
                optim.zero_grad(set_to_none=True)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()),
                max_norm=cfg.training.grad_clip,
            )
            optim.step()

            if ema_encoder is not None:
                _ema_update(ema_encoder, encoder, cfg.training.ema_decay)
            if ema_decoder is not None:
                _ema_update(ema_decoder, decoder, cfg.training.ema_decay)

            epoch_loss += float(loss.item())
            epoch_recon += float(recon_loss.item())
            epoch_l1 += float(l1_loss.item())
            epoch_kl += float(kl_loss.item())
            postfix = {
                "loss": epoch_loss / (step + 1),
                "recon": epoch_recon / (step + 1),
                "l1": epoch_l1 / (step + 1),
            }
            if cfg.encoder.use_stochastic:
                postfix["kl"] = epoch_kl / (step + 1)
            pbar.set_postfix(**postfix)

        if (epoch + 1) % cfg.training.checkpoint_every == 0:
            ctx.save_checkpoint(epoch)

        if cfg.training.encode_dataset_every > 0 and (epoch + 1) % cfg.training.encode_dataset_every == 0 and ema_encoder is not None:
            ema_encoder.eval()
            save_encoded_dataset(
                ctx,
                encoded_output_path,
                encoder=ema_encoder,
                batch_size=batch_size,
                n_samples=cfg.training.encode_n_samples,
            )
            encoder.train()

        if cfg.visualize and (epoch % cfg.training.visualize_every == 0 or epoch == cfg.training.epochs - 1):
            enc_vis = ema_encoder if ema_encoder is not None else encoder
            dec_vis = ema_decoder if ema_decoder is not None else decoder
            enc_vis.eval()
            dec_vis.eval()
            with torch.no_grad():
                b_vis = min(batch_size, 4)
                batch_np, _, sample_idx = ctx.loader.get_batch(b_vis)
                if cfg.training.lopsided_aug:
                    batch_np = apply_lopsided_augmentation(
                        batch_np,
                        frac=cfg.training.lopsided_frac,
                        sigma=cfg.training.lopsided_sigma,
                        sample_indices=sample_idx,
                    )
                batch_np_norm = ctx.data_stats.normalize(batch_np)
                x_ref = torch.from_numpy(batch_np_norm.astype(np.float32)).to(ctx.device)
                samples = reconstruct_ae_light(ctx, x_ref, encoder=enc_vis, decoder=dec_vis)
                samples_denorm = np.clip(ctx.data_stats.denormalize(samples.cpu().numpy()), 0, None)

            plots_dir = epoch_plot_dir(ctx.plot_dir, epoch)
            channel_positions = ctx.loader.channel_positions
            adj2d = build_xy_adjacency_radius(channel_positions, radius=cfg.graph.radius)
            graph_xy = Graph(adjacency=adj2d, positions_xy=channel_positions, positions_z=np.zeros(ctx.n_channels, dtype=np.float32))
            graph_z = Graph(
                adjacency=None,
                positions_xy=channel_positions,
                positions_z=np.concatenate([range(ctx.n_time_points) for _ in range(ctx.n_channels)]),
            )

            for idx in range(samples.shape[0]):
                rec_int = samples_denorm[idx, 0]
                true_int = batch_np[idx, :, 0]
                rec_xy = rec_int.reshape(ctx.n_channels, ctx.n_time_points, order="F").sum(axis=1)
                true_xy = true_int.reshape(ctx.n_channels, ctx.n_time_points, order="F").sum(axis=1)
                rec_z = rec_int.reshape(ctx.n_channels, ctx.n_time_points, order="F")
                true_z = true_int.reshape(ctx.n_channels, ctx.n_time_points, order="F")

                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                visualize_event(graph_xy, true_xy, None, ax=axes[0])
                axes[0].set_title("Ground truth")
                visualize_event(graph_xy, rec_xy, None, ax=axes[1])
                axes[1].set_title("AE Light reconstruction")
                plt.tight_layout()
                fig.savefig(os.path.join(plots_dir, f"event_{idx}_xy.png"))
                plt.close(fig)

                fig, axes = plt.subplots(1, 2, figsize=(10, 3))
                visualize_event_z(graph_z, true_z, None, ax=axes[0])
                axes[0].set_title("Ground truth")
                visualize_event_z(graph_z, rec_z, None, ax=axes[1])
                axes[1].set_title("AE Light reconstruction")
                plt.tight_layout()
                fig.savefig(os.path.join(plots_dir, f"event_{idx}_z.png"))
                plt.close(fig)


if __name__ == "__main__":
    train_ae_light(get_config())
