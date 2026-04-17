"""
DiffAE Light: a static sparse-graph diffusion autoencoder optimized for large
layered graphs.

Key design choices:
- No dynamic pooling or per-step subgraph rebuilding.
- No block-diagonal batched adjacency. Sparse matmuls run on a single graph
  with the batch packed into the feature dimension.
- No edge-wise message tensors in the hotpath.
- No gradient checkpointing.
- Static temporal coarsening for multiscale context, built once at startup.
"""
import glob
import math
import os
import sys
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import h5py
import matplotlib
import numpy as np
import scipy.sparse as sp
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
)
from config import Config, default_config, get_config, print_config
from data import Graph, SparseGraph, visualize_event, visualize_event_z
from diffusion.schedule import build_cosine_schedule, sinusoidal_embedding
from lz_data_loader import OnlineMSBatcher, TritiumSSDataLoader
from plot_style import COLORS, apply_style
from utils.sparse_ops import to_coalesced_coo
from utils.visualization import build_xy_adjacency_radius


if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

DataLoaderType = Union[TritiumSSDataLoader, OnlineMSBatcher]


@dataclass
class PyramidLevel:
    adj: torch.Tensor
    row_sum: torch.Tensor
    pos: torch.Tensor
    lpe: Optional[torch.Tensor]
    fine_to_coarse: Optional[torch.Tensor]
    coarse_counts: Optional[torch.Tensor]
    n_nodes: int
    n_time_points: int


@dataclass
class GraphPyramid:
    levels: List[PyramidLevel]
    n_channels: int


@dataclass
class DiffAELightContext:
    cfg: Config
    device: torch.device
    loader: DataLoaderType
    graph: SparseGraph
    n_channels: int
    n_time_points: int
    n_nodes: int
    data_stats: DiffAEDataStats
    schedule: dict
    encoder: nn.Module
    decoder: nn.Module
    encoder_pyramid: Optional[GraphPyramid]
    decoder_pyramid: GraphPyramid
    checkpoint_dir: str = ""
    plot_dir: str = ""
    ema_encoder: Optional[nn.Module] = None
    ema_decoder: Optional[nn.Module] = None
    regressive_decoder: Optional[nn.Module] = None
    ema_regressive_decoder: Optional[nn.Module] = None
    optim: Optional[torch.optim.Optimizer] = None
    use_ms_data: bool = False

    @property
    def A_sparse(self) -> torch.Tensor:
        return self.decoder_pyramid.levels[0].adj

    @property
    def pos(self) -> torch.Tensor:
        return self.decoder_pyramid.levels[0].pos

    @property
    def lpe(self) -> Optional[torch.Tensor]:
        if self.encoder_pyramid is None:
            return None
        return self.encoder_pyramid.levels[0].lpe

    @classmethod
    def build(
        cls,
        cfg: Config,
        for_training: bool = True,
        verbose: bool = True,
        use_ms_data: bool = True,
    ) -> "DiffAELightContext":
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

        schedule = build_cosine_schedule(cfg.diffusion.timesteps, device)

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
            scales = [lvl.n_nodes for lvl in decoder_pyramid.levels]
            print(f"Decoder pyramid nodes: {scales}")

        encoder_type = (getattr(cfg.encoder, "encoder_type", "graph") or "graph").lower()
        encoder_pyramid: Optional[GraphPyramid] = None
        actual_lpe_dim = 0
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
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")

        decoder = LightGraphDiffusionDecoder(
            in_dim=cfg.model.in_dim,
            out_dim=cfg.model.out_dim,
            hidden_dim=cfg.model.hidden_dim,
            cond_dim=cfg.encoder.latent_dim + cfg.conditioning.time_dim,
            num_scales=len(decoder_pyramid.levels),
            blocks_per_stage=cfg.model.blocks_per_stage,
            dropout=cfg.model.dropout,
            pos_dim=cfg.model.pos_dim,
            pos_dropout=cfg.model.pos_dropout,
            skip_scale=getattr(cfg.model, "skip_scale", 1.0),
        ).to(device)

        regressive_decoder: Optional[nn.Module] = None
        if cfg.encoder.use_regressive_head:
            decoder_type = (getattr(cfg.encoder, "decoder_type", "graph") or "graph").lower()
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
                regressive_decoder = LightLatentDecoder(
                    out_dim=cfg.model.out_dim,
                    hidden_dim=cfg.encoder.hidden_dim,
                    cond_dim=cfg.encoder.latent_dim,
                    num_scales=len(decoder_pyramid.levels),
                    blocks_per_stage=cfg.encoder.blocks_per_stage,
                    dropout=cfg.encoder.dropout,
                    pos_dim=cfg.model.pos_dim,
                ).to(device)

        ema_encoder = None
        ema_decoder = None
        ema_regressive_decoder = None
        optim = None

        checkpoint_dir = os.path.join(cfg.paths.checkpoint_dir, f"diffae_light_z{cfg.encoder.latent_dim}")
        plot_dir = os.path.join(cfg.paths.plot_dir, f"diffae_light_z{cfg.encoder.latent_dim}")

        if for_training:
            ema_encoder = deepcopy(encoder).to(device)
            ema_decoder = deepcopy(decoder).to(device)
            all_params = list(encoder.parameters()) + list(decoder.parameters())
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
            os.makedirs(plot_dir, exist_ok=True)

        if verbose:
            n_enc = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
            n_dec = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
            n_reg = sum(p.numel() for p in regressive_decoder.parameters() if p.requires_grad) if regressive_decoder else 0
            print(f"Encoder parameters: {n_enc:,}")
            print(f"Decoder parameters: {n_dec:,}")
            if regressive_decoder is not None:
                print(f"Regressive decoder parameters: {n_reg:,}")
            print(f"Total trainable parameters: {n_enc + n_dec + n_reg:,}")

        return cls(
            cfg=cfg,
            device=device,
            loader=loader,
            graph=graph,
            n_channels=n_channels,
            n_time_points=n_time_points,
            n_nodes=n_nodes,
            data_stats=data_stats,
            schedule=schedule,
            encoder=encoder,
            decoder=decoder,
            encoder_pyramid=encoder_pyramid,
            decoder_pyramid=decoder_pyramid,
            checkpoint_dir=checkpoint_dir,
            plot_dir=plot_dir,
            ema_encoder=ema_encoder,
            ema_decoder=ema_decoder,
            regressive_decoder=regressive_decoder,
            ema_regressive_decoder=ema_regressive_decoder,
            optim=optim,
            use_ms_data=use_ms_data,
        )

    def latest_checkpoint(self) -> Optional[str]:
        files = glob.glob(os.path.join(self.checkpoint_dir, "diffae_light_epoch_*.pt"))
        if not files:
            return None
        return max(files, key=lambda path: int(os.path.splitext(os.path.basename(path))[0].split("_")[-1]))

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
        if self.regressive_decoder is not None:
            state["regressive_decoder"] = self.regressive_decoder.state_dict()
        if self.ema_regressive_decoder is not None:
            state["ema_regressive_decoder"] = self.ema_regressive_decoder.state_dict()
        path = os.path.join(self.checkpoint_dir, f"diffae_light_epoch_{epoch:04d}.pt")
        torch.save(state, path)
        return path

    def load_checkpoint(self, path: str, load_optim: bool = True) -> int:
        chk = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(chk["encoder"], strict=False)
        self.decoder.load_state_dict(chk["decoder"], strict=False)
        if self.ema_encoder is not None and "ema_encoder" in chk:
            self.ema_encoder.load_state_dict(chk["ema_encoder"], strict=False)
        if self.ema_decoder is not None and "ema_decoder" in chk:
            self.ema_decoder.load_state_dict(chk["ema_decoder"], strict=False)
        if self.regressive_decoder is not None and "regressive_decoder" in chk:
            self.regressive_decoder.load_state_dict(chk["regressive_decoder"], strict=False)
        if self.ema_regressive_decoder is not None and "ema_regressive_decoder" in chk:
            self.ema_regressive_decoder.load_state_dict(chk["ema_regressive_decoder"], strict=False)
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
        return self.latest_checkpoint()


def _to_scipy_csr(adj: torch.Tensor) -> sp.csr_matrix:
    adj = to_coalesced_coo(adj.detach().cpu())
    idx = adj.indices().numpy()
    vals = adj.values().numpy().astype(np.float32, copy=False)
    return sp.coo_matrix((vals, (idx[0], idx[1])), shape=adj.shape, dtype=np.float32).tocsr()


def _from_scipy_sparse(adj: sp.spmatrix, device: torch.device) -> torch.Tensor:
    coo = adj.tocoo()
    idx = np.vstack([coo.row.astype(np.int64, copy=False), coo.col.astype(np.int64, copy=False)])
    vals = coo.data.astype(np.float32, copy=False)
    out = torch.sparse_coo_tensor(torch.from_numpy(idx), torch.from_numpy(vals), size=coo.shape).coalesce()
    out = out.to(device)
    try:
        return out.to_sparse_csr()
    except RuntimeError:
        return out


def _normalize_positions(pos: np.ndarray, pos_dim: int) -> np.ndarray:
    cur = pos[:, :pos_dim].astype(np.float32, copy=False)
    mean = cur.mean(axis=0, keepdims=True)
    std = cur.std(axis=0, keepdims=True)
    return (cur - mean) / (std + 1e-6)


def _normalize_scipy_adj(adj: sp.csr_matrix) -> sp.csr_matrix:
    adj = adj.tolil(copy=True)
    adj.setdiag(1.0)
    adj = adj.tocsr()
    adj.sum_duplicates()
    deg = np.asarray(adj.sum(axis=1)).ravel().astype(np.float32, copy=False)
    inv_sqrt = 1.0 / np.sqrt(np.clip(deg, 1e-12, None))
    d = sp.diags(inv_sqrt, dtype=np.float32, format="csr")
    return (d @ adj @ d).tocsr()


def _pool_group_size(pool_ratio: float) -> int:
    if pool_ratio >= 1.0:
        return 1
    return max(1, int(round(1.0 / max(pool_ratio, 1e-6))))


def _build_temporal_mapping(n_channels: int, n_time_points: int, pool_ratio: float) -> Optional[Tuple[np.ndarray, np.ndarray, int, sp.csr_matrix]]:
    group_size = _pool_group_size(pool_ratio)
    if group_size <= 1 or n_time_points <= 1:
        return None
    n_fine = n_channels * n_time_points
    coarse_time = int(math.ceil(n_time_points / group_size))
    n_coarse = n_channels * coarse_time
    fine_idx = np.arange(n_fine, dtype=np.int64)
    fine_t = fine_idx // n_channels
    fine_c = fine_idx % n_channels
    coarse_t = fine_t // group_size
    fine_to_coarse = coarse_t * n_channels + fine_c
    coarse_counts = np.bincount(fine_to_coarse, minlength=n_coarse).astype(np.float32, copy=False)
    assign = sp.csr_matrix(
        (np.ones(n_fine, dtype=np.float32), (fine_idx, fine_to_coarse)),
        shape=(n_fine, n_coarse),
        dtype=np.float32,
    )
    return fine_to_coarse, coarse_counts, coarse_time, assign


def _coarsen_dense_feature(feat: Optional[np.ndarray], fine_to_coarse: np.ndarray, coarse_counts: np.ndarray) -> Optional[np.ndarray]:
    if feat is None:
        return None
    out = np.zeros((coarse_counts.shape[0], feat.shape[1]), dtype=np.float32)
    np.add.at(out, fine_to_coarse, feat.astype(np.float32, copy=False))
    out /= coarse_counts[:, None]
    return out


def build_graph_pyramid(
    graph: SparseGraph,
    n_channels: int,
    n_time_points: int,
    depth: int,
    pool_ratio: float,
    pos_dim: int,
    weighted_edges: bool,
    device: torch.device,
) -> GraphPyramid:
    cur_adj = _to_scipy_csr(graph.adjacency)
    cur_pos = graph.positions_xyz.cpu().numpy().astype(np.float32, copy=False)
    cur_lpe = None if graph.lpe is None else graph.lpe.cpu().numpy().astype(np.float32, copy=False)
    cur_time = n_time_points
    levels: List[PyramidLevel] = []

    for level_idx in range(depth + 1):
        mapping = None
        if level_idx < depth:
            mapping = _build_temporal_mapping(n_channels, cur_time, pool_ratio)

        norm_adj = _normalize_scipy_adj(cur_adj)
        row_sum = np.asarray(norm_adj.sum(axis=1)).ravel().astype(np.float32, copy=False)
        fine_to_coarse_t = None
        coarse_counts_t = None
        next_adj = None
        next_pos = None
        next_lpe = None
        next_time = None

        if mapping is not None:
            fine_to_coarse, coarse_counts, next_time, assign = mapping
            next_adj = (assign.T @ cur_adj @ assign).tocsr()
            next_adj.sum_duplicates()
            if not weighted_edges:
                next_adj.data = np.ones_like(next_adj.data, dtype=np.float32)
            next_pos = _coarsen_dense_feature(cur_pos, fine_to_coarse, coarse_counts)
            next_lpe = _coarsen_dense_feature(cur_lpe, fine_to_coarse, coarse_counts)
            fine_to_coarse_t = torch.from_numpy(fine_to_coarse.astype(np.int64, copy=False)).to(device)
            coarse_counts_t = torch.from_numpy(coarse_counts.astype(np.float32, copy=False)).to(device)

        levels.append(
            PyramidLevel(
                adj=_from_scipy_sparse(norm_adj, device),
                row_sum=torch.from_numpy(row_sum).to(device),
                pos=torch.from_numpy(_normalize_positions(cur_pos, pos_dim)).to(device),
                lpe=None if cur_lpe is None else torch.from_numpy(cur_lpe).to(device),
                fine_to_coarse=fine_to_coarse_t,
                coarse_counts=coarse_counts_t,
                n_nodes=cur_adj.shape[0],
                n_time_points=cur_time,
            )
        )

        if mapping is None or next_adj is None or next_pos is None or next_time is None:
            break

        cur_adj = next_adj
        cur_pos = next_pos
        cur_lpe = next_lpe
        cur_time = next_time

    return GraphPyramid(levels=levels, n_channels=n_channels)


def _repeat_graph_feature(x: torch.Tensor, batch_size: int) -> torch.Tensor:
    return x.unsqueeze(0).expand(batch_size, -1, -1).reshape(batch_size * x.size(0), x.size(1))


def _expand_condition(cond: torch.Tensor, n_nodes: int) -> torch.Tensor:
    return cond.unsqueeze(1).expand(-1, n_nodes, -1).reshape(cond.size(0) * n_nodes, cond.size(1))


def _apply_affine(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, batch_size: int, n_nodes: int) -> torch.Tensor:
    scale_e = _expand_condition(scale, n_nodes)
    shift_e = _expand_condition(shift, n_nodes)
    return x * (1.0 + scale_e) + shift_e


def sparse_batch_mm(adj: torch.Tensor, x: torch.Tensor, batch_size: int) -> torch.Tensor:
    n_in = adj.size(1)
    n_out = adj.size(0)
    h_dim = x.size(1)
    x_mat = x.reshape(batch_size, n_in, h_dim).permute(1, 0, 2).reshape(n_in, batch_size * h_dim)
    with torch.amp.autocast(x.device.type, enabled=False):
        adj_f = adj.float() if adj.dtype != torch.float32 else adj
        y_mat = torch.sparse.mm(adj_f, x_mat.float())
    y_mat = y_mat.to(x.dtype)
    return y_mat.reshape(n_out, batch_size, h_dim).permute(1, 0, 2).reshape(batch_size * n_out, h_dim)


def _expand_row_sum(row_sum: torch.Tensor, batch_size: int, dtype: torch.dtype) -> torch.Tensor:
    return row_sum.to(dtype).unsqueeze(0).expand(batch_size, -1).reshape(batch_size * row_sum.numel(), 1)


def sparse_batch_mean_std(adj: torch.Tensor, row_sum: torch.Tensor, x: torch.Tensor, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    denom = torch.clamp(_expand_row_sum(row_sum, batch_size, x.dtype), min=1e-6)
    mean_num = sparse_batch_mm(adj, x, batch_size)
    second_num = sparse_batch_mm(adj, x * x, batch_size)
    mean = mean_num / denom
    second = second_num / denom
    std = torch.sqrt(torch.clamp(second - mean * mean, min=0.0) + 1e-6)
    return mean, std


def temporal_downsample(x: torch.Tensor, fine_to_coarse: torch.Tensor, coarse_counts: torch.Tensor, batch_size: int) -> torch.Tensor:
    n_fine = fine_to_coarse.numel()
    n_coarse = coarse_counts.numel()
    h_dim = x.size(1)
    x_mat = x.reshape(batch_size, n_fine, h_dim).permute(1, 0, 2).reshape(n_fine, batch_size * h_dim)
    out = x_mat.new_zeros((n_coarse, batch_size * h_dim))
    out.index_add_(0, fine_to_coarse, x_mat)
    out = out / coarse_counts[:, None].to(out.dtype)
    return out.reshape(n_coarse, batch_size, h_dim).permute(1, 0, 2).reshape(batch_size * n_coarse, h_dim)


def temporal_upsample(x: torch.Tensor, fine_to_coarse: torch.Tensor, batch_size: int) -> torch.Tensor:
    n_fine = fine_to_coarse.numel()
    n_coarse = int(fine_to_coarse.max().item()) + 1
    h_dim = x.size(1)
    x_mat = x.reshape(batch_size, n_coarse, h_dim).permute(1, 0, 2).reshape(n_coarse, batch_size * h_dim)
    out = x_mat.index_select(0, fine_to_coarse)
    return out.reshape(n_fine, batch_size, h_dim).permute(1, 0, 2).reshape(batch_size * n_fine, h_dim)


class LightEncoderBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.msg_proj = nn.Linear(hidden_dim * 3, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, row_sum: torch.Tensor, batch_size: int) -> torch.Tensor:
        h = self.norm1(x)
        mean, std = sparse_batch_mean_std(adj, row_sum, h, batch_size)
        x = x + self.dropout(self.msg_proj(torch.cat([h, mean, std], dim=-1)))
        return x + self.dropout(self.ff(self.norm2(x)))


class LightConditionedBlock(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.msg_proj = nn.Linear(hidden_dim * 3, hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.cond_proj = nn.Linear(cond_dim, hidden_dim * 6)
        self.node_proj = nn.Linear(hidden_dim, hidden_dim * 4)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        row_sum: torch.Tensor,
        cond: torch.Tensor,
        node_cond: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        n_nodes = adj.size(0)
        cond_params = self.cond_proj(cond).reshape(batch_size, 6, self.hidden_dim)
        shift1, scale1, gate1, shift2, scale2, gate2 = cond_params.unbind(dim=1)
        node_shift1, node_scale1, node_shift2, node_scale2 = self.node_proj(node_cond).chunk(4, dim=-1)

        h = _apply_affine(self.norm1(x), scale1, shift1, batch_size, n_nodes)
        h = h * (1.0 + node_scale1) + node_shift1
        mean, std = sparse_batch_mean_std(adj, row_sum, h, batch_size)
        gate1_e = torch.sigmoid(_expand_condition(gate1, n_nodes))
        x = x + gate1_e * self.dropout(self.msg_proj(torch.cat([h, mean, std], dim=-1)))

        h2 = _apply_affine(self.norm2(x), scale2, shift2, batch_size, n_nodes)
        h2 = h2 * (1.0 + node_scale2) + node_shift2
        gate2_e = torch.sigmoid(_expand_condition(gate2, n_nodes))
        return x + gate2_e * self.dropout(self.ff(h2))


class LightGraphEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_scales: int,
        blocks_per_stage: int,
        dropout: float,
        pos_dim: int,
        lpe_dim: int,
        use_stochastic: bool,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.use_stochastic = use_stochastic

        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.pos_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(pos_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_scales)
        ])
        self.lpe_proj = nn.Linear(lpe_dim, hidden_dim) if lpe_dim > 0 else None
        self.stages = nn.ModuleList([
            nn.ModuleList([LightEncoderBlock(hidden_dim, dropout=dropout) for _ in range(blocks_per_stage)])
            for _ in range(num_scales)
        ])

        readout_dim = num_scales * hidden_dim * 3
        if use_stochastic:
            self.to_mu = nn.Sequential(
                nn.Linear(readout_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.to_logvar = nn.Sequential(
                nn.Linear(readout_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, latent_dim),
            )
        else:
            self.to_latent = nn.Sequential(
                nn.Linear(readout_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, latent_dim),
            )

    def forward(
        self,
        x: torch.Tensor,
        pyramid: GraphPyramid,
        batch_size: int = 1,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        h = self.in_proj(x)
        readouts: List[torch.Tensor] = []

        for scale_idx, level in enumerate(pyramid.levels):
            pos_emb = _repeat_graph_feature(self.pos_mlps[scale_idx](level.pos.to(h.dtype)), batch_size)
            h = h + pos_emb
            if scale_idx == 0 and level.lpe is not None and self.lpe_proj is not None:
                h = h + _repeat_graph_feature(self.lpe_proj(level.lpe.to(h.dtype)), batch_size)

            for block in self.stages[scale_idx]:
                h = block(h, level.adj, level.row_sum, batch_size)

            h_view = h.reshape(batch_size, level.n_nodes, self.hidden_dim)
            readouts.append(
                torch.cat(
                    [
                        h_view.mean(dim=1),
                        h_view.std(dim=1, correction=0) + 1e-6,
                        h_view.amax(dim=1),
                    ],
                    dim=-1,
                )
            )

            if level.fine_to_coarse is not None and level.coarse_counts is not None:
                h = temporal_downsample(h, level.fine_to_coarse, level.coarse_counts, batch_size)

        h_readout = torch.cat(readouts, dim=-1)
        if self.use_stochastic:
            mu = self.to_mu(h_readout)
            logvar = torch.clamp(self.to_logvar(h_readout), min=-10.0, max=2.0)
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
            return z, mu, logvar
        return self.to_latent(h_readout), None, None


class LightGraphDiffusionDecoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        cond_dim: int,
        num_scales: int,
        blocks_per_stage: int,
        dropout: float,
        pos_dim: int,
        pos_dropout: float,
        skip_scale: float,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.skip_scale = skip_scale
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.input_adapters = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_scales)])
        self.pos_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(pos_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_scales)
        ])
        self.pos_drop = nn.Dropout(pos_dropout)
        self.node_cond_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cond_dim + pos_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_scales)
        ])
        self.down_scale_cond = nn.ModuleList([nn.Linear(cond_dim, hidden_dim) for _ in range(num_scales)])
        self.up_scale_cond = nn.ModuleList([nn.Linear(cond_dim, hidden_dim) for _ in range(num_scales)])
        self.start_proj = nn.Linear(cond_dim, hidden_dim)
        self.down_stages = nn.ModuleList([
            nn.ModuleList([LightConditionedBlock(hidden_dim, cond_dim, dropout=dropout) for _ in range(blocks_per_stage)])
            for _ in range(num_scales)
        ])
        self.bottleneck = nn.ModuleList([LightConditionedBlock(hidden_dim, cond_dim, dropout=dropout) for _ in range(blocks_per_stage)])
        self.up_stages = nn.ModuleList([
            nn.ModuleList([LightConditionedBlock(hidden_dim, cond_dim, dropout=dropout) for _ in range(blocks_per_stage)])
            for _ in range(num_scales)
        ])
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def _build_input_pyramid(self, x: torch.Tensor, pyramid: GraphPyramid, batch_size: int) -> List[torch.Tensor]:
        h = self.in_proj(x)
        levels: List[torch.Tensor] = []
        for scale_idx, level in enumerate(pyramid.levels):
            pos_emb = _repeat_graph_feature(self.pos_mlps[scale_idx](level.pos.to(h.dtype)), batch_size)
            levels.append(self.input_adapters[scale_idx](h) + self.pos_drop(pos_emb))
            if level.fine_to_coarse is not None and level.coarse_counts is not None:
                h = temporal_downsample(h, level.fine_to_coarse, level.coarse_counts, batch_size)
        return levels

    def _build_node_cond_pyramid(
        self,
        cond: torch.Tensor,
        pyramid: GraphPyramid,
        batch_size: int,
        dtype: torch.dtype,
    ) -> List[torch.Tensor]:
        node_levels: List[torch.Tensor] = []
        for scale_idx, level in enumerate(pyramid.levels):
            pos_rep = _repeat_graph_feature(level.pos.to(dtype), batch_size)
            cond_rep = _expand_condition(cond, level.n_nodes)
            node_levels.append(self.node_cond_mlps[scale_idx](torch.cat([cond_rep, pos_rep], dim=-1)))
        return node_levels

    def forward(self, x: torch.Tensor, pyramid: GraphPyramid, cond: torch.Tensor, batch_size: int = 1) -> torch.Tensor:
        x_pyr = self._build_input_pyramid(x, pyramid, batch_size)
        node_cond_pyr = self._build_node_cond_pyramid(cond, pyramid, batch_size, x.dtype)
        last_idx = len(pyramid.levels) - 1
        h = x_pyr[0]
        skips: List[torch.Tensor] = []

        for scale_idx, level in enumerate(pyramid.levels):
            if scale_idx > 0:
                h = h + x_pyr[scale_idx]
            h = h + _expand_condition(self.down_scale_cond[scale_idx](cond), level.n_nodes)
            node_cond = node_cond_pyr[scale_idx]
            for block in self.down_stages[scale_idx]:
                h = block(h, level.adj, level.row_sum, cond, node_cond, batch_size)
            skips.append(h)
            if scale_idx < last_idx:
                assert level.fine_to_coarse is not None and level.coarse_counts is not None
                h = temporal_downsample(h, level.fine_to_coarse, level.coarse_counts, batch_size)

        h = h + _expand_condition(self.start_proj(cond), pyramid.levels[last_idx].n_nodes)
        coarsest_level = pyramid.levels[last_idx]
        coarsest_node_cond = node_cond_pyr[last_idx]
        for block in self.bottleneck:
            h = block(h, coarsest_level.adj, coarsest_level.row_sum, cond, coarsest_node_cond, batch_size)

        for scale_idx in reversed(range(len(pyramid.levels))):
            level = pyramid.levels[scale_idx]
            h = h + _expand_condition(self.up_scale_cond[scale_idx](cond), level.n_nodes)
            node_cond = node_cond_pyr[scale_idx]
            for block in self.up_stages[scale_idx]:
                h = block(h, level.adj, level.row_sum, cond, node_cond, batch_size)
            if scale_idx > 0:
                fine_level = pyramid.levels[scale_idx - 1]
                assert fine_level.fine_to_coarse is not None
                h = temporal_upsample(h, fine_level.fine_to_coarse, batch_size)
                h = h + self.skip_scale * skips[scale_idx - 1]

        return self.out_proj(F.silu(self.out_norm(h)))


class LightLatentDecoder(nn.Module):
    def __init__(
        self,
        out_dim: int,
        hidden_dim: int,
        cond_dim: int,
        num_scales: int,
        blocks_per_stage: int,
        dropout: float,
        pos_dim: int,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pos_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(pos_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_scales)
        ])
        self.node_cond_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cond_dim + pos_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_scales)
        ])
        self.scale_cond = nn.ModuleList([nn.Linear(cond_dim, hidden_dim) for _ in range(num_scales)])
        self.start_proj = nn.Linear(cond_dim, hidden_dim)
        self.stages = nn.ModuleList([
            nn.ModuleList([LightConditionedBlock(hidden_dim, cond_dim, dropout=dropout) for _ in range(blocks_per_stage)])
            for _ in range(num_scales)
        ])
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def _build_node_cond_pyramid(
        self,
        cond: torch.Tensor,
        pyramid: GraphPyramid,
        batch_size: int,
        dtype: torch.dtype,
    ) -> List[torch.Tensor]:
        node_levels: List[torch.Tensor] = []
        for scale_idx, level in enumerate(pyramid.levels):
            pos_rep = _repeat_graph_feature(level.pos.to(dtype), batch_size)
            cond_rep = _expand_condition(cond, level.n_nodes)
            node_levels.append(self.node_cond_mlps[scale_idx](torch.cat([cond_rep, pos_rep], dim=-1)))
        return node_levels

    def forward(self, z: torch.Tensor, pyramid: GraphPyramid, batch_size: int = 1) -> torch.Tensor:
        last_idx = len(pyramid.levels) - 1
        last_level = pyramid.levels[last_idx]
        node_cond_pyr = self._build_node_cond_pyramid(z, pyramid, batch_size, z.dtype)
        h = _expand_condition(self.start_proj(z), last_level.n_nodes)
        h = h + _repeat_graph_feature(self.pos_mlps[last_idx](last_level.pos.to(h.dtype)), batch_size)

        for scale_idx in reversed(range(len(pyramid.levels))):
            level = pyramid.levels[scale_idx]
            h = h + _expand_condition(self.scale_cond[scale_idx](z), level.n_nodes)
            node_cond = node_cond_pyr[scale_idx]
            for block in self.stages[scale_idx]:
                h = block(h, level.adj, level.row_sum, z, node_cond, batch_size)
            if scale_idx > 0:
                fine_level = pyramid.levels[scale_idx - 1]
                assert fine_level.fine_to_coarse is not None
                h = temporal_upsample(h, fine_level.fine_to_coarse, batch_size)
                h = h + _repeat_graph_feature(self.pos_mlps[scale_idx - 1](fine_level.pos.to(h.dtype)), batch_size)

        return self.out_proj(F.silu(self.out_norm(h)))


def _encode_with_context(
    ctx: DiffAELightContext,
    x_flat: torch.Tensor,
    batch_size: int,
    encoder: Optional[nn.Module] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    enc = ctx.encoder if encoder is None else encoder
    if isinstance(enc, LightGraphEncoder):
        if ctx.encoder_pyramid is None:
            raise RuntimeError("Graph encoder requested but encoder_pyramid is missing.")
        return enc(x_flat, ctx.encoder_pyramid, batch_size=batch_size)
    return enc(x_flat, ctx.A_sparse, ctx.pos, batch_size=batch_size)  # type: ignore[misc]


def _ema_update(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        ema_params = [p for p in ema_model.parameters()]
        model_params = [p for p in model.parameters()]
        torch._foreach_mul_(ema_params, decay)
        torch._foreach_add_(ema_params, model_params, alpha=1.0 - decay)


@torch.no_grad()
def save_encoded_dataset(
    ctx: DiffAELightContext,
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
        remaining = n_samples - samples_encoded
        actual_batch_size = min(batch_size, remaining)
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
        x = torch.from_numpy(wf_norm.astype(np.float32)).to(ctx.device)
        x_flat = x.reshape(actual_batch_size * ctx.n_nodes, 1)
        z, _, _ = _encode_with_context(ctx, x_flat, actual_batch_size, encoder=enc)
        all_latents.append(z.cpu().numpy())
        samples_encoded += actual_batch_size

    latents = np.concatenate(all_latents, axis=0)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("latents", data=latents, dtype=np.float32)
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
        print(f"Saved encoded MS dataset to {output_path}: {samples_encoded} samples")
    return output_path


@torch.no_grad()
def sample_diffae_light(
    ctx: DiffAELightContext,
    x_ref: torch.Tensor,
    encoder: Optional[nn.Module] = None,
    decoder: Optional[nn.Module] = None,
    parametrization: Optional[str] = None,
    pbar: bool = False,
) -> torch.Tensor:
    enc = ctx.ema_encoder if encoder is None and ctx.ema_encoder is not None else (ctx.encoder if encoder is None else encoder)
    dec = ctx.ema_decoder if decoder is None and ctx.ema_decoder is not None else (ctx.decoder if decoder is None else decoder)
    parametrization = ctx.cfg.diffusion.parametrization if parametrization is None else parametrization

    bsz, n_nodes, channels = x_ref.shape
    x_ref_flat = x_ref.reshape(bsz * n_nodes, channels)
    z, _, _ = _encode_with_context(ctx, x_ref_flat, bsz, encoder=enc)

    x = torch.randn((bsz, n_nodes, channels), device=x_ref.device)
    t_total = ctx.schedule["betas"].shape[0]

    for step in tqdm(reversed(range(t_total)), desc="Sampling", disable=not pbar, total=t_total, ncols=150):
        betas_t = ctx.schedule["betas"][step]
        sqrt_one_minus_ab_t = ctx.schedule["sqrt_one_minus_alphas_cumprod"][step]
        alpha_bar_t = ctx.schedule["alphas_cumprod"][step]
        alpha_bar_prev_t = ctx.schedule["alphas_cumprod_prev"][step]

        t_emb = sinusoidal_embedding(torch.tensor([step], device=x_ref.device), ctx.cfg.conditioning.time_dim)
        cond_full = torch.cat([z, t_emb.expand(bsz, -1)], dim=-1)

        x_flat = x.reshape(bsz * n_nodes, channels)
        pred_flat = dec(x_flat, ctx.decoder_pyramid, cond_full, batch_size=bsz)  # type: ignore[misc]
        pred = pred_flat.reshape(bsz, n_nodes, channels)

        if parametrization == "eps":
            x0_pred = (x - sqrt_one_minus_ab_t * pred) / torch.clamp(torch.sqrt(alpha_bar_t), min=1e-8)
        elif parametrization == "v":
            a = torch.clamp(torch.sqrt(alpha_bar_t), min=1e-8)
            b_coef = torch.clamp(torch.sqrt(1.0 - alpha_bar_t), min=1e-8)
            x0_pred = a * x - b_coef * pred
        else:
            raise ValueError("parametrization must be 'eps' or 'v'")

        coef1 = betas_t * torch.sqrt(torch.clamp(alpha_bar_prev_t, min=1e-12)) / torch.clamp(1.0 - alpha_bar_t, min=1e-12)
        coef2 = torch.clamp(1.0 - alpha_bar_prev_t, min=0.0) * torch.sqrt(torch.clamp(1.0 - betas_t, min=1e-12)) / torch.clamp(1.0 - alpha_bar_t, min=1e-12)
        mean = coef1 * x0_pred + coef2 * x

        if step > 0:
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(ctx.schedule["posterior_variance"][step]) * noise
        else:
            x = mean

    return x.permute(0, 2, 1)


def train_diffae_light(cfg: Config = default_config) -> None:
    print_config(cfg, include_encoder=True)
    ctx = DiffAELightContext.build(cfg, for_training=True, verbose=True)

    device_t = ctx.device
    encoder = ctx.encoder
    decoder = ctx.decoder
    regressive_decoder = ctx.regressive_decoder
    ema_encoder = ctx.ema_encoder
    ema_decoder = ctx.ema_decoder
    ema_regressive_decoder = ctx.ema_regressive_decoder
    optim = ctx.optim
    assert optim is not None

    schedule = ctx.schedule
    data_stats = ctx.data_stats
    n_nodes = ctx.n_nodes
    n_channels = ctx.n_channels
    n_time_points = ctx.n_time_points
    tr = ctx.loader
    channel_positions = tr.channel_positions
    use_regressive = cfg.encoder.use_regressive_head and regressive_decoder is not None

    start_epoch = 0
    if cfg.resume:
        last = ctx.latest_checkpoint()
        if last is not None:
            start_epoch = ctx.load_checkpoint(last) + 1
            print(f"Resumed from epoch {start_epoch}")

    for group in optim.param_groups:
        group["lr"] = cfg.training.lr
        group.setdefault("initial_lr", cfg.training.lr)

    total_epochs = cfg.training.epochs
    warmup = cfg.training.warmup_epochs

    def _lr_lambda(epoch: int) -> float:
        if warmup > 0 and epoch < warmup:
            return float(epoch + 1) / float(warmup)
        if cfg.training.lr_schedule == "cosine":
            progress = (epoch - warmup) / max(1, total_epochs - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=_lr_lambda, last_epoch=start_epoch - 1)

    batch_size = cfg.training.batch_size
    steps_per_epoch = tr.n_samples // batch_size
    print(f"  Steps per epoch: {tr.n_samples} samples // batch {batch_size} = {steps_per_epoch}")

    encoded_output_path = os.path.join(ctx.checkpoint_dir, cfg.paths.diffae_latents_file)
    amp_enabled = cfg.training.use_amp and device_t.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float32

    for epoch in range(start_epoch, cfg.training.epochs):
        encoder.train()
        decoder.train()
        if use_regressive:
            regressive_decoder.train()

        epoch_loss = 0.0
        epoch_kl = 0.0
        epoch_reg_loss = 0.0
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{cfg.training.epochs}", ncols=120, file=sys.stdout)

        for step in pbar:
            batch_np, _, sample_idx = tr.get_batch(batch_size)
            if cfg.training.lopsided_aug:
                batch_np = apply_lopsided_augmentation(
                    batch_np,
                    frac=cfg.training.lopsided_frac,
                    sigma=cfg.training.lopsided_sigma,
                    sample_indices=sample_idx,
                )
            batch_np = data_stats.normalize(batch_np)

            x0 = torch.from_numpy(batch_np.astype(np.float32)).to(device_t)
            x0_flat = x0.reshape(batch_size * n_nodes, 1)
            optim.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_t.type, dtype=amp_dtype, enabled=amp_enabled):
                z, mu, logvar = _encode_with_context(ctx, x0_flat, batch_size)

                if step == 0 and epoch % 50 == 0:
                    with torch.no_grad():
                        z_std = z.float().std().item()
                        z_sim = 0.0
                        if batch_size > 1:
                            z_norm = z.float() / (z.float().norm(dim=1, keepdim=True) + 1e-8)
                            z_sim = (z_norm @ z_norm.T).fill_diagonal_(0).abs().mean().item()
                        print(f"\n  [Monitor] Latent z: std={z_std:.4f}, within-batch similarity={z_sim:.4f}")

                t = torch.randint(0, cfg.diffusion.timesteps, (batch_size,), device=device_t, dtype=torch.long)
                t_emb = sinusoidal_embedding(t, cfg.conditioning.time_dim)
                cond_full = torch.cat([z, t_emb], dim=-1)

                sqrt_ab = schedule["sqrt_alphas_cumprod"][t].view(batch_size, 1, 1)
                sqrt_om = schedule["sqrt_one_minus_alphas_cumprod"][t].view(batch_size, 1, 1)
                snr_t = schedule["snr"][t].view(batch_size)

                noise = torch.randn_like(x0)
                x_t = sqrt_ab * x0 + sqrt_om * noise
                x_t_flat = x_t.reshape(batch_size * n_nodes, 1)

                pred_flat = decoder(x_t_flat, ctx.decoder_pyramid, cond_full, batch_size=batch_size)  # type: ignore[misc]
                pred = pred_flat.reshape(batch_size, n_nodes, 1)

                if cfg.diffusion.parametrization == "eps":
                    target = noise
                elif cfg.diffusion.parametrization == "v":
                    target = sqrt_ab * noise - sqrt_om * x0
                else:
                    raise ValueError("parametrization must be 'eps' or 'v'")

                mse_per_sample = F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2))
                if cfg.diffusion.p2_gamma > 0.0:
                    weight = torch.pow(cfg.diffusion.p2_k + snr_t, -cfg.diffusion.p2_gamma)
                    mse_per_sample = mse_per_sample * weight
                loss = mse_per_sample.mean()

                if cfg.encoder.use_stochastic and mu is not None and logvar is not None:
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = loss + cfg.encoder.kl_weight * kl_loss
                    epoch_kl += float(kl_loss.item())

            if not torch.isfinite(loss):
                print(f"  WARNING: non-finite loss at step {step}, skipping")
                optim.zero_grad(set_to_none=True)
                continue

            loss.backward()

            reg_term_value = 0.0
            if use_regressive:
                with torch.amp.autocast(device_t.type, dtype=amp_dtype, enabled=amp_enabled):
                    z_reg, _, _ = _encode_with_context(ctx, x0_flat, batch_size)
                    if isinstance(regressive_decoder, LightLatentDecoder):
                        reg_flat = regressive_decoder(z_reg, ctx.decoder_pyramid, batch_size=batch_size)
                    else:
                        reg_flat = regressive_decoder(z_reg, ctx.A_sparse, ctx.pos, batch_size=batch_size)  # type: ignore[misc]
                    reg_pred = reg_flat.reshape(batch_size, n_nodes, 1)
                    reg_loss = F.mse_loss(reg_pred, x0, reduction="mean")
                    reg_term = cfg.encoder.regressive_head_weight * reg_loss

                if torch.isfinite(reg_term):
                    reg_term.backward()
                    reg_term_value = float(reg_term.item())
                    epoch_reg_loss += float(reg_loss.item())

            epoch_loss += float(loss.item()) + reg_term_value
            clip_params = list(encoder.parameters()) + list(decoder.parameters())
            if use_regressive:
                clip_params += list(regressive_decoder.parameters())
            torch.nn.utils.clip_grad_norm_(clip_params, max_norm=cfg.training.grad_clip)
            optim.step()

            if ema_encoder is not None:
                _ema_update(ema_encoder, encoder, cfg.training.ema_decay)
            if ema_decoder is not None:
                _ema_update(ema_decoder, decoder, cfg.training.ema_decay)
            if use_regressive and ema_regressive_decoder is not None:
                _ema_update(ema_regressive_decoder, regressive_decoder, cfg.training.ema_decay)

            postfix = {"loss": epoch_loss / (step + 1)}
            if cfg.encoder.use_stochastic:
                postfix["kl"] = epoch_kl / (step + 1)
            if use_regressive:
                postfix["reg"] = epoch_reg_loss / (step + 1)
            pbar.set_postfix(**postfix)

        scheduler.step()

        if (epoch + 1) % cfg.training.checkpoint_every == 0:
            ctx.save_checkpoint(epoch)

        if cfg.training.encode_dataset_every > 0 and (epoch + 1) % cfg.training.encode_dataset_every == 0:
            if ema_encoder is not None:
                ema_encoder.eval()
                save_encoded_dataset(
                    ctx,
                    encoded_output_path,
                    encoder=ema_encoder,
                    batch_size=batch_size,
                    n_samples=cfg.training.encode_n_samples,
                )
                encoder.train()
            if device_t.type == "cuda":
                torch.cuda.empty_cache()

        if cfg.visualize and (epoch % cfg.training.visualize_every == 0 or epoch == cfg.training.epochs - 1):
            enc_vis = ema_encoder if ema_encoder is not None else encoder
            dec_vis = ema_decoder if ema_decoder is not None else decoder
            enc_vis.eval()
            dec_vis.eval()

            with torch.no_grad():
                b_vis = min(batch_size, 4)
                batch_np_raw, _, sample_idx = tr.get_batch(b_vis)
                if cfg.training.lopsided_aug:
                    batch_np_raw = apply_lopsided_augmentation(
                        batch_np_raw,
                        frac=cfg.training.lopsided_frac,
                        sigma=cfg.training.lopsided_sigma,
                        sample_indices=sample_idx,
                    )
                batch_np_norm = data_stats.normalize(batch_np_raw)
                x_ref = torch.from_numpy(batch_np_norm.astype(np.float32)).to(device_t)
                samples = sample_diffae_light(ctx, x_ref, encoder=enc_vis, decoder=dec_vis, pbar=False)
                samples_denorm = data_stats.denormalize(samples.cpu().numpy())
                samples_denorm = np.clip(samples_denorm, 0, None)

            plots_dir = os.path.join(ctx.plot_dir, f"epoch_{epoch}")
            os.makedirs(plots_dir, exist_ok=True)

            adj2d = build_xy_adjacency_radius(channel_positions, radius=cfg.graph.radius)
            gxy = Graph(adjacency=adj2d, positions_xy=channel_positions, positions_z=np.zeros(n_channels, dtype=np.float32))
            gz = Graph(adjacency=np.eye(n_channels, dtype=np.float32), positions_xy=channel_positions, positions_z=np.arange(n_time_points, dtype=np.float32))

            apply_style()
            for idx in range(samples.shape[0]):
                rec_int = samples_denorm[idx, 0]
                true_int = batch_np_raw[idx, :, 0]

                rec_xy = rec_int.reshape(n_channels, n_time_points, order="F").sum(axis=1)
                true_xy = true_int.reshape(n_channels, n_time_points, order="F").sum(axis=1)
                rec_z = rec_int.reshape(n_channels, n_time_points, order="F")
                true_z = true_int.reshape(n_channels, n_time_points, order="F")

                fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
                visualize_event(gxy, true_xy, None, ax=axes[0])
                axes[0].set_title("Ground truth")
                visualize_event(gxy, rec_xy, None, ax=axes[1])
                axes[1].set_title("DiffAE Light reconstruction")
                fig.tight_layout()
                fig.savefig(os.path.join(plots_dir, f"event_{idx}_xy.png"))
                plt.close(fig)

                fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
                visualize_event_z(gz, true_z, None, ax=axes[0])
                axes[0].set_title("Ground truth")
                visualize_event_z(gz, rec_z, None, ax=axes[1])
                axes[1].set_title("DiffAE Light reconstruction")
                fig.tight_layout()
                fig.savefig(os.path.join(plots_dir, f"event_{idx}_z.png"))
                plt.close(fig)

                top_k = min(4, n_channels)
                top_channels = np.argsort(true_xy)[-top_k:][::-1]
                fig, axes = plt.subplots(top_k, 1, figsize=(9, 2.0 * top_k), sharex=True)
                if top_k == 1:
                    axes = [axes]
                t_axis = np.arange(n_time_points)
                for ax, ch in zip(axes, top_channels):
                    ax.plot(t_axis, true_z[ch], color=COLORS["truth"], linewidth=1.2, label="Truth")
                    ax.plot(t_axis, rec_z[ch], color=COLORS["diffae"], linewidth=1.0, alpha=0.9, label="DiffAE Light")
                    ax.set_ylabel(f"ch {ch}")
                axes[0].legend(loc="upper right", handlelength=1.2)
                axes[-1].set_xlabel("Time bin")
                fig.suptitle("Channel cross-sections (top 4 by charge)")
                fig.tight_layout()
                fig.savefig(os.path.join(plots_dir, f"event_{idx}_cross_sections.png"))
                plt.close(fig)

            encoder.train()
            decoder.train()
            if device_t.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    cfg = get_config(epochs=20_000)
    train_diffae_light(cfg)
