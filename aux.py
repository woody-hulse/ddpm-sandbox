"""
Auxiliary task training for evaluating latent representations.

Trains MLPs on encoded event representations from:
1. DiffAE encoder
2. Regular (deterministic) autoencoder
3. True MS event properties (baseline)

Compares their performance on predicting delta_mu (time separation).
"""
import os
import argparse
import time
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from config import Config, default_config, MSDataConfig, AuxTaskConfig
from lz_data_loader import TritiumSSDataLoader, create_3d_adjacency_matrix_sparse_
from data import SparseGraph
from diffae import GraphEncoder, DiffAEContext, DiffAEDataStats, sample_diffae
from diffusion.schedule import build_cosine_schedule
from ae import GraphVAEEncoder
from models.graph_unet import TopKPool, build_block_diagonal_adj, GraphDDPMUNet
from utils.sparse_ops import to_binary, subgraph_coo


class GraphAE(nn.Module):
    """
    Graph-based autoencoder (deterministic) using the same encoder structure as DiffAE.
    Uses GraphEncoder for encoding and MLP for decoding.
    """
    def __init__(
        self,
        in_dim: int,
        n_nodes: int,
        hidden_dim: int = 32,
        latent_dim: int = 64,
        depth: int = 4,
        blocks_per_stage: int = 2,
        pool_ratio: float = 0.5,
        dropout: float = 0.0,
        pos_dim: int = 3,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.n_nodes = n_nodes
        self.latent_dim = latent_dim
        self.register_buffer('data_mean', torch.tensor(0.0))
        self.register_buffer('data_std', torch.tensor(1.0))

        self.encoder = GraphEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            depth=depth,
            blocks_per_stage=blocks_per_stage,
            pool_ratio=pool_ratio,
            dropout=dropout,
            pos_dim=pos_dim,
            use_stochastic=False,
        )
        
        decoder_hidden = hidden_dim * 4
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, decoder_hidden),
            nn.LayerNorm(decoder_hidden),
            nn.SiLU(),
            nn.Linear(decoder_hidden, decoder_hidden),
            nn.LayerNorm(decoder_hidden),
            nn.SiLU(),
            nn.Linear(decoder_hidden, decoder_hidden),
            nn.LayerNorm(decoder_hidden),
            nn.SiLU(),
            nn.Linear(decoder_hidden, n_nodes * in_dim),
        )

    def set_data_stats(self, mean: float, std: float):
        self.data_mean.fill_(mean)
        self.data_std.fill_(std)
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.data_mean) / self.data_std
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.data_std + self.data_mean

    def encode(
        self, 
        x: torch.Tensor, 
        adj: torch.Tensor, 
        pos: torch.Tensor, 
        batch_size: int,
        normalize: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if normalize:
            x = self.normalize(x)
        z, mu, logvar = self.encoder(x, adj, pos, batch_size=batch_size)
        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        out = self.decoder(z)
        return out.view(z.shape[0], self.n_nodes, self.in_dim)

    def forward(
        self, 
        x: torch.Tensor, 
        adj: torch.Tensor, 
        pos: torch.Tensor, 
        batch_size: int,
        normalize: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if normalize:
            x_norm = self.normalize(x)
        else:
            x_norm = x
        z, mu, logvar = self.encoder(x_norm, adj, pos, batch_size=batch_size)
        x_recon = self.decode(z)
        return x_recon, z, mu, logvar


def load_graph_ae(
    ckpt_path: str,
    device: torch.device,
    cfg: Optional[Config] = None,
) -> Tuple[GraphAE, int]:
    """Load Graph AE from checkpoint. Returns (model, latent_dim)."""
    if cfg is None:
        cfg = default_config
    ckpt = torch.load(ckpt_path, map_location=device)
    latent_dim = int(ckpt["latent_dim"])
    n_nodes = int(ckpt["n_nodes"])
    in_dim = int(ckpt["in_dim"])
    hidden_dim = int(ckpt["hidden_dim"])
    depth = int(ckpt["depth"])
    blocks_per_stage = int(ckpt["blocks_per_stage"])
    pool_ratio = float(ckpt["pool_ratio"])
    dropout = float(ckpt["dropout"])
    pos_dim = int(ckpt["pos_dim"])
    model = GraphAE(
        in_dim=in_dim,
        n_nodes=n_nodes,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        depth=depth,
        blocks_per_stage=blocks_per_stage,
        pool_ratio=pool_ratio,
        dropout=dropout,
        pos_dim=pos_dim,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.set_data_stats(float(ckpt["data_mean"]), float(ckpt["data_std"]))
    return model, latent_dim


class MLP(nn.Module):
    """Simple MLP for auxiliary prediction tasks."""
    def __init__(
        self,
        in_dim: int,
        hidden_dims: list,
        out_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def shift_waveform(waveform: np.ndarray, shift_bins: int) -> np.ndarray:
    """Shift waveform along time axis with zero-padding.
    
    Args:
        waveform: Array of shape (C, T) or (T,)
        shift_bins: Number of bins to shift (positive = shift right/later)
    
    Returns:
        Shifted waveform with same shape, zero-padded
    """
    if waveform.ndim == 1:
        T = waveform.shape[0]
        shifted = np.zeros_like(waveform)
        if shift_bins == 0:
            return waveform.copy()
        elif shift_bins > 0:
            if shift_bins < T:
                shifted[shift_bins:] = waveform[:T - shift_bins]
        else:
            abs_shift = abs(shift_bins)
            if abs_shift < T:
                shifted[:T - abs_shift] = waveform[abs_shift:]
        return shifted
    else:
        C, T = waveform.shape
        shifted = np.zeros_like(waveform)
        if shift_bins == 0:
            return waveform.copy()
        elif shift_bins > 0:
            if shift_bins < T:
                shifted[:, shift_bins:] = waveform[:, :T - shift_bins]
        else:
            abs_shift = abs(shift_bins)
            if abs_shift < T:
                shifted[:, :T - abs_shift] = waveform[:, abs_shift:]
        return shifted


class OnlineMSDataset(Dataset):
    """Dataset that generates multi-scatter events online by co-adding single-scatter events.
    
    Instead of loading pre-generated MS data from a file, this dataset loads SS (single-scatter)
    data and generates MS events on-the-fly by randomly combining pairs of SS events with
    time shifts.
    """
    def __init__(
        self,
        ss_h5_path: str,
        n_events: int = 10000,
        ms_config: Optional[MSDataConfig] = None,
        transform=None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            ss_h5_path: Path to single-scatter H5 file with 'waveforms', 'xc', 'yc', 'dt' datasets
            n_events: Number of MS events to generate per epoch
            ms_config: MSDataConfig with delta_min, delta_max, ns_per_bin settings
            transform: Optional transform to apply to waveforms
            seed: Random seed for reproducibility (overrides ms_config.seed if provided)
        """
        if ms_config is None:
            ms_config = MSDataConfig()
        
        self.ms_config = ms_config
        self.n_events = n_events
        self.transform = transform
        self.seed = seed if seed is not None else ms_config.seed
        
        with h5py.File(ss_h5_path, 'r') as f:
            self.ss_waveforms = f['waveforms'][:].astype(np.float32)
            self.ss_xc = f['xc'][:].astype(np.float32)
            self.ss_yc = f['yc'][:].astype(np.float32)
            self.ss_dt = f['dt'][:].astype(np.float32)
        
        self.n_ss = len(self.ss_waveforms)
        self.n_channels = self.ss_waveforms.shape[1]
        self.n_time = self.ss_waveforms.shape[2]
        self.input_dim = self.n_channels * self.n_time
        
        self._rng = np.random.default_rng(self.seed)
        self._regenerate_indices()

    def _regenerate_indices(self):
        """Pre-generate random indices and shifts for all events in this epoch."""
        self._idx1 = self._rng.integers(0, self.n_ss, size=self.n_events)
        self._idx2 = self._rng.integers(0, self.n_ss, size=self.n_events)
        mask = self._idx1 == self._idx2
        while mask.any():
            self._idx2[mask] = self._rng.integers(0, self.n_ss, size=mask.sum())
            mask = self._idx1 == self._idx2
        
        self._delta_bins = self._rng.integers(
            self.ms_config.delta_min,
            self.ms_config.delta_max + 1,
            size=self.n_events
        )

    def reseed(self, seed: Optional[int] = None):
        """Reseed the RNG and regenerate indices for a new epoch."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._regenerate_indices()

    def __len__(self):
        return self.n_events

    def __getitem__(self, idx):
        idx1 = self._idx1[idx]
        idx2 = self._idx2[idx]
        delta_bins = int(self._delta_bins[idx])
        
        wf1 = self.ss_waveforms[idx1]
        wf2 = self.ss_waveforms[idx2]
        wf2_shifted = shift_waveform(wf2, delta_bins)
        ms_wf = wf1 + wf2_shifted
        
        wf_flat = ms_wf.T.reshape(-1, 1)
        
        if self.transform:
            wf_flat = self.transform(wf_flat)
        
        delta_mu = delta_bins * self.ms_config.ns_per_bin
        targets = {
            'delta_mu': np.float32(delta_mu),
            'delta_bins': np.float32(delta_bins),
            'dx': self.ss_xc[idx2] - self.ss_xc[idx1],
            'dy': self.ss_yc[idx2] - self.ss_yc[idx1],
        }
        
        return torch.from_numpy(wf_flat), targets


class MSDataset(OnlineMSDataset):
    """Alias for OnlineMSDataset for backward compatibility.
    
    Previously this class loaded pre-generated MS data from an H5 file.
    Now it generates MS data online from single-scatter data.
    """
    def __init__(
        self,
        ss_h5_path: str,
        n_events: int = 10000,
        ms_config: Optional[MSDataConfig] = None,
        transform=None,
        seed: Optional[int] = 42,
    ):
        super().__init__(
            ss_h5_path=ss_h5_path,
            n_events=n_events,
            ms_config=ms_config,
            transform=transform,
            seed=seed,
        )


class EncodedMSDataset(Dataset):
    """Dataset that loads pre-encoded MS latents from an h5 file.
    
    This is the fast path for aux task training: latents are pre-computed
    by DiffAE or GraphAE and saved to disk, so no encoder calls are needed
    during MLP training.
    """
    def __init__(self, h5_path: str):
        """
        Args:
            h5_path: Path to encoded latents h5 file (from save_encoded_dataset)
        """
        with h5py.File(h5_path, 'r') as f:
            self.latents = f['latents'][:].astype(np.float32)
            self.delta_mu = f['delta_mu'][:].astype(np.float32) if 'delta_mu' in f else None
            self.delta_bins = f['delta_bins'][:].astype(np.float32) if 'delta_bins' in f else None
            self.xc1 = f['xc1'][:].astype(np.float32) if 'xc1' in f else None
            self.yc1 = f['yc1'][:].astype(np.float32) if 'yc1' in f else None
            self.xc2 = f['xc2'][:].astype(np.float32) if 'xc2' in f else None
            self.yc2 = f['yc2'][:].astype(np.float32) if 'yc2' in f else None
            self.latent_dim = int(f.attrs.get('latent_dim', self.latents.shape[1]))
            self.is_ms_data = bool(f.attrs.get('is_ms_data', self.delta_mu is not None))
        
        if self.delta_mu is None:
            raise ValueError(f"Encoded dataset {h5_path} does not contain delta_mu - was it encoded from MS data?")
    
    def __len__(self):
        return len(self.latents)
    
    def __getitem__(self, idx):
        z = torch.from_numpy(self.latents[idx])
        targets = {
            'delta_mu': self.delta_mu[idx],
            'delta_bins': self.delta_bins[idx] if self.delta_bins is not None else 0.0,
            'dx': (self.xc2[idx] - self.xc1[idx]) if self.xc1 is not None else 0.0,
            'dy': (self.yc2[idx] - self.yc1[idx]) if self.yc1 is not None else 0.0,
        }
        return z, targets


def train_aux_mlp_on_latents(
    mlp: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-3,
    target_key: str = 'delta_mu',
) -> Tuple[list, list, List[float]]:
    """Train auxiliary MLP directly on pre-encoded latents (fast path).
    
    This is the efficient version - latents are already computed, so we just
    train a simple MLP without any encoder calls.
    """
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    train_losses = []
    val_losses = []
    val_times = []
    t0 = time.perf_counter()
    
    for epoch in range(epochs):
        mlp.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for z, targets in train_loader:
            z = z.to(device)
            target = targets[target_key].to(device).float()
            
            pred = mlp(z).squeeze(-1)
            loss = F.mse_loss(pred, target)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        train_losses.append(epoch_loss / n_batches)
        
        mlp.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for z, targets in val_loader:
                z = z.to(device)
                target = targets[target_key].to(device).float()
                pred = mlp(z).squeeze(-1)
                val_loss += F.mse_loss(pred, target).item()
                n_val += 1
        
        val_losses.append(val_loss / n_val)
        val_times.append(time.perf_counter() - t0)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train_loss={train_losses[-1]:.4f}, val_loss={val_losses[-1]:.4f}")
    
    return train_losses, val_losses, val_times


def evaluate_aux_mlp_on_latents(
    mlp: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    target_key: str = 'delta_mu',
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate auxiliary MLP on pre-encoded latents."""
    mlp.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for z, targets in test_loader:
            z = z.to(device)
            target = targets[target_key].numpy()
            pred = mlp(z).squeeze(-1).cpu().numpy()
            all_preds.append(pred)
            all_targets.append(target)
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    mae = np.mean(np.abs(all_preds - all_targets))
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    
    return mae, rmse, all_preds, all_targets


def build_graph_for_shape(
    cfg: Config,
    n_channels: int,
    n_time: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build adjacency and positions to match (n_channels, n_time) for use with MS or SS data."""
    if n_channels == 1:
        xy = np.array([[0.0, 0.0]], dtype=np.float32)
    else:
        path = cfg.paths.channel_positions
        if path.endswith(('.h5', '.hdf5')):
            with h5py.File(path, 'r') as f:
                if 'TA_PMTs_xy' in f:
                    xy = np.asarray(f['TA_PMTs_xy'][:], dtype=np.float32) / 10.0
                elif 'positions' in f:
                    xy = np.asarray(f['positions'][:], dtype=np.float32)
                elif 'xy' in f:
                    xy = np.asarray(f['xy'][:], dtype=np.float32)
                else:
                    raise ValueError(f"No positions in {path}")
        elif path.endswith('.npy'):
            xy = np.load(path).astype(np.float32)
        else:
            raise ValueError(f"Unsupported positions path: {path}")
        if xy.shape[0] < n_channels:
            raise ValueError(f"Positions file has {xy.shape[0]} channels, need {n_channels}")
        xy = xy[:n_channels]
    
    graph = create_3d_adjacency_matrix_sparse_(
        xy,
        num_layers=n_time,
        r_within=cfg.graph.radius,
        positions_xy_profile=xy,
        z_hops=cfg.graph.z_hops,
        self_loops=True,
        z_spacing=cfg.graph.z_sep,
    )
    return graph.adjacency.to(device), graph.positions_xyz.to(device)


@dataclass
class AuxTrainingResult:
    model_name: str
    train_losses: list
    val_losses: list
    test_mae: float
    test_rmse: float
    predictions: np.ndarray
    targets: np.ndarray
    val_times: Optional[List[float]] = None


def train_aux_model(
    encoder: nn.Module,
    mlp: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    A_sparse: torch.Tensor,
    pos: torch.Tensor,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-3,
    target_key: str = 'delta_mu',
    encoder_type: str = 'diffae',
) -> Tuple[list, list, List[float]]:
    """Train auxiliary MLP on encoded representations."""
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    encoder.eval()
    train_losses = []
    val_losses = []
    val_times = []
    t0 = time.perf_counter()
    
    for epoch in range(epochs):
        mlp.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for wf, targets in train_loader:
            wf = wf.to(device)
            target = targets[target_key].to(device).float()
            B = wf.shape[0]
            
            with torch.no_grad():
                wf_flat = wf.view(B * wf.shape[1], 1)
                if encoder_type == 'diffae':
                    z, _, _ = encoder(wf_flat, A_sparse, pos, batch_size=B)
                elif encoder_type == 'ae':
                    z, _, _, _ = encoder(wf_flat, A_sparse, pos, batch_size=B)
                else:  # regular_ae
                    z = encoder.encode(wf_flat, A_sparse, pos, batch_size=B)
            
            pred = mlp(z).squeeze(-1)
            loss = F.mse_loss(pred, target)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        train_losses.append(epoch_loss / n_batches)
        
        mlp.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for wf, targets in val_loader:
                wf = wf.to(device)
                target = targets[target_key].to(device).float()
                B = wf.shape[0]
                
                wf_flat = wf.view(B * wf.shape[1], 1)
                if encoder_type == 'diffae':
                    z, _, _ = encoder(wf_flat, A_sparse, pos, batch_size=B)
                elif encoder_type == 'ae':
                    z, _, _, _ = encoder(wf_flat, A_sparse, pos, batch_size=B)
                else:  # regular_ae
                    z = encoder.encode(wf_flat, A_sparse, pos, batch_size=B)
                
                pred = mlp(z).squeeze(-1)
                val_loss += F.mse_loss(pred, target).item()
                n_val += 1
        
        val_losses.append(val_loss / n_val)
        val_times.append(time.perf_counter() - t0)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train_loss={train_losses[-1]:.4f}, val_loss={val_losses[-1]:.4f}")
    
    return train_losses, val_losses, val_times


def evaluate_aux_model(
    encoder: nn.Module,
    mlp: nn.Module,
    test_loader: DataLoader,
    A_sparse: torch.Tensor,
    pos: torch.Tensor,
    device: torch.device,
    target_key: str = 'delta_mu',
    encoder_type: str = 'diffae',
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate auxiliary MLP on test set."""
    encoder.eval()
    mlp.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for wf, targets in test_loader:
            wf = wf.to(device)
            target = targets[target_key].numpy()
            B = wf.shape[0]
            
            wf_flat = wf.view(B * wf.shape[1], 1)
            if encoder_type == 'diffae':
                z, _, _ = encoder(wf_flat, A_sparse, pos, batch_size=B)
            elif encoder_type == 'ae':
                z, _, _, _ = encoder(wf_flat, A_sparse, pos, batch_size=B)
            else:  # regular_ae
                z = encoder.encode(wf_flat, A_sparse, pos, batch_size=B)
            
            pred = mlp(z).squeeze(-1).cpu().numpy()
            all_preds.append(pred)
            all_targets.append(target)
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    mae = np.mean(np.abs(all_preds - all_targets))
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    
    return mae, rmse, all_preds, all_targets


def train_baseline_mlp(
    mlp: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-3,
    target_key: str = 'delta_mu',
) -> Tuple[list, list, List[float]]:
    """Train MLP directly on flattened waveforms (baseline)."""
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    train_losses = []
    val_losses = []
    val_times = []
    t0 = time.perf_counter()
    
    for epoch in range(epochs):
        mlp.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for wf, targets in train_loader:
            wf = wf.to(device).view(wf.shape[0], -1)
            target = targets[target_key].to(device).float()
            
            pred = mlp(wf).squeeze(-1)
            loss = F.mse_loss(pred, target)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        train_losses.append(epoch_loss / n_batches)
        
        mlp.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for wf, targets in val_loader:
                wf = wf.to(device).view(wf.shape[0], -1)
                target = targets[target_key].to(device).float()
                pred = mlp(wf).squeeze(-1)
                val_loss += F.mse_loss(pred, target).item()
                n_val += 1
        
        val_losses.append(val_loss / n_val)
        val_times.append(time.perf_counter() - t0)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train_loss={train_losses[-1]:.4f}, val_loss={val_losses[-1]:.4f}")
    
    return train_losses, val_losses, val_times


def evaluate_baseline_mlp(
    mlp: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    target_key: str = 'delta_mu',
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate baseline MLP on test set."""
    mlp.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for wf, targets in test_loader:
            wf = wf.to(device).view(wf.shape[0], -1)
            target = targets[target_key].numpy()
            pred = mlp(wf).squeeze(-1).cpu().numpy()
            all_preds.append(pred)
            all_targets.append(target)
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    mae = np.mean(np.abs(all_preds - all_targets))
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    
    return mae, rmse, all_preds, all_targets


def train_aux_model_graph_vae(
    vae: nn.Module,
    mlp: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    A_sparse: torch.Tensor,
    pos: torch.Tensor,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-3,
    target_key: str = 'delta_mu',
) -> Tuple[list, list, List[float]]:
    """Train auxiliary MLP on graph VAE encoded representations."""
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    vae.eval()
    train_losses = []
    val_losses = []
    val_times = []
    t0 = time.perf_counter()
    
    for epoch in range(epochs):
        mlp.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for wf, targets in train_loader:
            wf = wf.to(device)
            target = targets[target_key].to(device).float()
            B = wf.shape[0]
            wf_flat = wf.view(B * wf.shape[1], 1)
            
            with torch.no_grad():
                z, _, _ = vae.encode(wf_flat, A_sparse, pos, batch_size=B)
            
            pred = mlp(z).squeeze(-1)
            loss = F.mse_loss(pred, target)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        train_losses.append(epoch_loss / n_batches)
        
        mlp.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for wf, targets in val_loader:
                wf = wf.to(device)
                target = targets[target_key].to(device).float()
                B = wf.shape[0]
                wf_flat = wf.view(B * wf.shape[1], 1)
                z, _, _ = vae.encode(wf_flat, A_sparse, pos, batch_size=B)
                pred = mlp(z).squeeze(-1)
                val_loss += F.mse_loss(pred, target).item()
                n_val += 1
        
        val_losses.append(val_loss / n_val)
        val_times.append(time.perf_counter() - t0)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train_loss={train_losses[-1]:.4f}, val_loss={val_losses[-1]:.4f}")
    
    return train_losses, val_losses, val_times


def evaluate_aux_model_graph_vae(
    vae: nn.Module,
    mlp: nn.Module,
    test_loader: DataLoader,
    A_sparse: torch.Tensor,
    pos: torch.Tensor,
    device: torch.device,
    target_key: str = 'delta_mu',
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate auxiliary MLP on graph VAE encoded representations."""
    vae.eval()
    mlp.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for wf, targets in test_loader:
            wf = wf.to(device)
            target = targets[target_key].numpy()
            B = wf.shape[0]
            wf_flat = wf.view(B * wf.shape[1], 1)
            z, _, _ = vae.encode(wf_flat, A_sparse, pos, batch_size=B)
            pred = mlp(z).squeeze(-1).cpu().numpy()
            all_preds.append(pred)
            all_targets.append(target)
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    mae = np.mean(np.abs(all_preds - all_targets))
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    
    return mae, rmse, all_preds, all_targets


def plot_results(results: Dict[str, AuxTrainingResult], output_dir: str):
    """Plot comparison of different models (separate figures, coordinated colors)."""
    os.makedirs(output_dir, exist_ok=True)
    names = list(results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(names), 10)))[:len(names)]
    color_map = {n: colors[i] for i, n in enumerate(names)}

    fig1, ax1 = plt.subplots(figsize=(6, 4))
    for name, res in results.items():
        ax1.plot(res.val_losses, color=color_map[name], label=name)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation MSE Loss')
    ax1.set_title('Validation Loss')
    ax1.legend()
    ax1.set_yscale('log')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/aux_validation_loss_epoch.png', dpi=150)
    plt.close()

    has_times = any(getattr(r, 'val_times', None) is not None and len(getattr(r, 'val_times', [])) == len(r.val_losses) for r in results.values())
    if has_times:
        fig1b, ax1b = plt.subplots(figsize=(6, 4))
        for name, res in results.items():
            times = getattr(res, 'val_times', None)
            if times is not None and len(times) == len(res.val_losses):
                ax1b.plot(times, res.val_losses, color=color_map[name], label=name)
        ax1b.set_xlabel('Time (s)')
        ax1b.set_ylabel('Validation MSE Loss')
        ax1b.set_title('Validation Loss vs Time')
        ax1b.legend()
        ax1b.set_yscale('log')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/aux_validation_loss_time.png', dpi=150)
        plt.close()

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    x = np.arange(len(names))
    width = 0.35
    for i, n in enumerate(names):
        c = color_map[n]
        ax2.bar(x[i] - width/2, results[n].test_mae, width, color=c, alpha=0.9)
        ax2.bar(x[i] + width/2, results[n].test_rmse, width, color=c, alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15)
    ax2.set_ylabel('Error (ns)')
    ax2.set_title('Test Set Performance')
    ax2.legend(
        [Patch(facecolor='gray', alpha=0.9), Patch(facecolor='gray', alpha=0.5)],
        ['MAE', 'RMSE'],
        framealpha=0.9,
    )
    plt.tight_layout()
    plt.savefig(f'{output_dir}/aux_test_performance.png', dpi=150)
    plt.close()

    fig3, ax3 = plt.subplots(figsize=(6, 5))
    for name, res in results.items():
        ax3.scatter(res.targets, res.predictions, alpha=0.3, s=5, color=color_map[name], label=name)
    all_vals = np.concatenate([r.targets for r in results.values()] +
                               [r.predictions for r in results.values()])
    lims = [all_vals.min() - 10, all_vals.max() + 10]
    ax3.plot(lims, lims, 'k--', alpha=0.5)
    ax3.set_xlim(lims)
    ax3.set_ylim(lims)
    ax3.set_xlabel('True Δμ (ns)')
    ax3.set_ylabel('Predicted Δμ (ns)')
    ax3.set_title('Predictions vs Ground Truth')
    ax3.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/aux_predictions_vs_truth.png', dpi=150)
    plt.close()

    print(f"\nResults saved to {output_dir}/ (aux_validation_loss_epoch.png, aux_validation_loss_time.png, aux_test_performance.png, aux_predictions_vs_truth.png)")
    print("\n" + "="*50)
    print("Summary")
    print("="*50)
    for name, res in results.items():
        print(f"{name:20s}: MAE={res.test_mae:.2f} ns, RMSE={res.test_rmse:.2f} ns")


def _get_diffae_recon_batch(
    encoder: nn.Module,
    decoder: nn.Module,
    latent_proj: nn.Module,
    schedule: dict,
    data_stats: DiffAEDataStats,
    A_sparse: torch.Tensor,
    pos: torch.Tensor,
    time_dim: int,
    wf: torch.Tensor,
    device: torch.device,
    n_channels: int,
    n_time: int,
    parametrization: str = 'v',
) -> np.ndarray:
    """Run DiffAE encode + decode for one batch; return (B, C, T) numpy."""
    B = wf.shape[0]
    wf_np = wf.cpu().numpy()
    wf_norm = data_stats.normalize(wf_np)
    wf_flat = wf_norm.reshape(B, -1, order='F')
    x_ref = torch.from_numpy(wf_flat).float().to(device).view(B, -1, 1)
    samples = sample_diffae(
        encoder=encoder,
        decoder=decoder,
        latent_proj=latent_proj,
        schedule=schedule,
        A_sparse=A_sparse,
        pos=pos,
        time_dim=time_dim,
        x_ref=x_ref,
        parametrization=parametrization,
        pbar=False,
    )
    out = samples.cpu().numpy()
    out = data_stats.denormalize(out)
    out = np.clip(out, 0, None)
    out = np.stack([out[b, 0, :].reshape(n_channels, n_time, order='F') for b in range(B)])
    return out


def plot_reconstruction_overlays(
    data_loader: DataLoader,
    device: torch.device,
    n_channels: int,
    n_time: int,
    output_dir: str,
    num_examples: int = 6,
    vae: Optional[nn.Module] = None,
    diffae_encoder: Optional[nn.Module] = None,
    diffae_decoder: Optional[nn.Module] = None,
    diffae_latent_proj: Optional[nn.Module] = None,
    diffae_schedule: Optional[dict] = None,
    diffae_data_stats: Optional[DiffAEDataStats] = None,
    A_sparse: Optional[torch.Tensor] = None,
    pos: Optional[torch.Tensor] = None,
    diffae_time_dim: int = 64,
    cfg: Optional[Config] = None,
):
    """Plot original summed waveform, VAE reconstruction, and DiffAE reconstruction (1x3 per example)."""
    os.makedirs(output_dir, exist_ok=True)
    has_vae = vae is not None and A_sparse is not None and pos is not None
    has_diffae = all([
        diffae_encoder, diffae_decoder, diffae_latent_proj,
        diffae_schedule, diffae_data_stats, A_sparse is not None, pos is not None,
    ])
    if not has_vae and not has_diffae:
        return
    if cfg is None:
        cfg = default_config
    parametrization = getattr(cfg.diffusion, 'parametrization', 'v')
    n_nodes = n_channels * n_time

    collected: List[Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]] = []
    with torch.no_grad():
        for wf, _ in data_loader:
            wf = wf.to(device)
            orig_np = wf.cpu().numpy()
            vae_recon_np = None
            diffae_recon_np = None
            if has_vae:
                vae.eval()
                B = wf.shape[0]
                wf_flat = wf.view(B * wf.shape[1], 1)
                recon, _, _, _ = vae(wf_flat, A_sparse, pos, batch_size=B)
                recon_denorm = vae.denormalize(recon)
                vae_recon_np = recon_denorm.view(wf.shape).cpu().numpy()
            if has_diffae:
                diffae_encoder.eval()
                diffae_decoder.eval()
                diffae_recon_np = _get_diffae_recon_batch(
                    diffae_encoder, diffae_decoder, diffae_latent_proj,
                    diffae_schedule, diffae_data_stats, A_sparse, pos,
                    diffae_time_dim, wf, device, n_channels, n_time, parametrization,
                )
            for b in range(wf.shape[0]):
                collected.append((
                    orig_np[b],
                    vae_recon_np[b] if vae_recon_np is not None else None,
                    diffae_recon_np[b] if diffae_recon_np is not None else None,
                ))
                if len(collected) >= num_examples:
                    break
            if len(collected) >= num_examples:
                break
    if not collected:
        return

    def flat_to_2d(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr).squeeze()
        if arr.ndim == 1:
            return arr.reshape(n_time, n_channels).T
        return arr

    time_ns = np.arange(n_time) * 10
    for idx, (orig, vae_rec, diffae_rec) in enumerate(collected):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))
        orig_2d = flat_to_2d(orig)
        summed_orig = orig_2d.sum(axis=0)

        axes[0].plot(time_ns, summed_orig, color='C0', alpha=0.8)
        axes[0].set_xlabel('Time (ns)')
        axes[0].set_ylabel('Summed amplitude')
        axes[0].set_title('Original')

        if vae_rec is not None:
            vae_2d = flat_to_2d(vae_rec)
            summed_vae = vae_2d.sum(axis=0)
            axes[1].plot(time_ns, summed_vae, color='C1', alpha=0.8)
        else:
            axes[1].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_xlabel('Time (ns)')
        axes[1].set_title('AE reconstruction')

        if diffae_rec is not None:
            diffae_2d = flat_to_2d(diffae_rec)
            summed_diffae = diffae_2d.sum(axis=0)
            axes[2].plot(time_ns, summed_diffae, color='C2', alpha=0.8)
        else:
            axes[2].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_xlabel('Time (ns)')
        axes[2].set_title('DiffAE reconstruction')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/reconstruction_example_{idx + 1}.png', dpi=150)
        plt.close()
    print(f"  Reconstruction examples saved to {output_dir}/reconstruction_example_1.png ... reconstruction_example_{len(collected)}.png")


@dataclass
class LatentSizeResult:
    """Results for a single latent size and model type."""
    latent_dim: int
    model_type: str  # 'baseline', 'ae', 'diffae'
    mae_values: list  # MAE from each trial
    rmse_values: list  # RMSE from each trial
    
    @property
    def mae_mean(self) -> float:
        return np.mean(self.mae_values)
    
    @property
    def mae_std(self) -> float:
        return np.std(self.mae_values)
    
    @property
    def rmse_mean(self) -> float:
        return np.mean(self.rmse_values)
    
    @property
    def rmse_std(self) -> float:
        return np.std(self.rmse_values)


def run_single_aux_trial(
    encoder: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    A_sparse: torch.Tensor,
    pos: torch.Tensor,
    device: torch.device,
    latent_dim: int,
    aux_epochs: int,
    encoder_type: str,
    target_key: str = 'delta_mu',
) -> Tuple[float, float]:
    """Run a single auxiliary training trial and return MAE, RMSE."""
    mlp = MLP(
        in_dim=latent_dim,
        hidden_dims=[128, 64],
        out_dim=1,
        dropout=0.1,
    ).to(device)
    
    train_aux_model(
        encoder, mlp, train_loader, val_loader,
        A_sparse, pos, device,
        epochs=aux_epochs, target_key=target_key,
        encoder_type=encoder_type
    )
    
    mae, rmse, _, _ = evaluate_aux_model(
        encoder, mlp, test_loader,
        A_sparse, pos, device,
        target_key=target_key, encoder_type=encoder_type
    )
    
    return mae, rmse


def run_baseline_trial(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    n_nodes: int,
    device: torch.device,
    aux_epochs: int,
    target_key: str = 'delta_mu',
) -> Tuple[float, float]:
    """Run a single baseline MLP trial and return MAE, RMSE."""
    mlp = MLP(
        in_dim=n_nodes,
        hidden_dims=[256, 128, 64],
        out_dim=1,
        dropout=0.1,
    ).to(device)
    
    train_baseline_mlp(
        mlp, train_loader, val_loader, device,
        epochs=aux_epochs, target_key=target_key
    )
    
    mae, rmse, _, _ = evaluate_baseline_mlp(
        mlp, test_loader, device, target_key=target_key
    )
    
    return mae, rmse


def load_ae_encoder(
    cfg: Config,
    latent_dim: int,
    device: torch.device,
) -> Optional[nn.Module]:
    """Load AE encoder for a specific latent dimension."""
    ae_subdir = cfg.paths.ae_subdir.format(latent_dim=latent_dim)
    ae_ckpt_dir = os.path.join(cfg.paths.checkpoint_dir, ae_subdir)
    
    if not os.path.isdir(ae_ckpt_dir):
        return None
    
    ckpt_files = sorted([f for f in os.listdir(ae_ckpt_dir) if f.startswith('vae_epoch_')])
    if not ckpt_files:
        return None
    
    ckpt_path = os.path.join(ae_ckpt_dir, ckpt_files[-1])
    
    encoder = GraphVAEEncoder(
        in_dim=cfg.model.in_dim,
        hidden_dim=cfg.encoder.hidden_dim,
        latent_dim=latent_dim,
        depth=cfg.encoder.depth,
        blocks_per_stage=cfg.encoder.blocks_per_stage,
        pool_ratio=cfg.encoder.pool_ratio,
        dropout=cfg.encoder.dropout,
        pos_dim=cfg.model.pos_dim,
        use_stochastic=True,
    ).to(device)
    
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'ema_encoder' in ckpt:
        encoder.load_state_dict(ckpt['ema_encoder'])
    else:
        encoder.load_state_dict(ckpt['encoder'])
    
    return encoder


def load_diffae_encoder(
    cfg: Config,
    latent_dim: int,
    device: torch.device,
) -> Optional[nn.Module]:
    """Load DiffAE encoder for a specific latent dimension."""
    diffae_subdir = cfg.paths.diffae_subdir.format(latent_dim=latent_dim)
    diffae_ckpt_dir = os.path.join(cfg.paths.checkpoint_dir, diffae_subdir)
    
    if not os.path.isdir(diffae_ckpt_dir):
        return None
    
    ckpt_files = sorted([f for f in os.listdir(diffae_ckpt_dir) if f.startswith('diffae_epoch_')])
    if not ckpt_files:
        return None
    
    ckpt_path = os.path.join(diffae_ckpt_dir, ckpt_files[-1])
    
    encoder = GraphEncoder(
        in_dim=cfg.model.in_dim,
        hidden_dim=cfg.encoder.hidden_dim,
        latent_dim=latent_dim,
        depth=cfg.encoder.depth,
        blocks_per_stage=cfg.encoder.blocks_per_stage,
        pool_ratio=cfg.encoder.pool_ratio,
        dropout=cfg.encoder.dropout,
        pos_dim=cfg.model.pos_dim,
        use_stochastic=cfg.encoder.use_stochastic,
    ).to(device)
    
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'ema_encoder' in ckpt:
        encoder.load_state_dict(ckpt['ema_encoder'])
    else:
        encoder.load_state_dict(ckpt['encoder'])
    
    return encoder


def run_latent_size_comparison(
    latent_dims: List[int],
    cfg: Config,
    ss_data_path: str,
    output_dir: str,
    n_ms_events: int = 10000,
    n_trials: int = 3,
    aux_epochs: int = 50,
    batch_size: int = 32,
    target_key: str = 'delta_mu',
    metric: str = 'mae',
    ms_seed: int = 42,
):
    """
    Train auxiliary tasks at varying encoded sizes and plot comparison.
    
    Args:
        latent_dims: List of latent dimensions to test (e.g., [8, 16, 32, 64, 128, 256])
        cfg: Configuration object
        ss_data_path: Path to single-scatter data for online MS generation
        output_dir: Directory to save plots
        n_ms_events: Number of MS events to generate per epoch
        n_trials: Number of trials per latent size for error bars
        aux_epochs: Epochs for auxiliary MLP training
        batch_size: Batch size for training
        target_key: Target to predict ('delta_mu', 'delta_bins', etc.)
        metric: Which metric to plot ('mae' or 'rmse')
        ms_seed: Random seed for MS data generation
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"\nGenerating online MS data from SS data: {ss_data_path}")
    print(f"  Events: {n_ms_events}, delta range: [{cfg.ms_data.delta_min}, {cfg.ms_data.delta_max}] bins")
    dataset = OnlineMSDataset(
        ss_h5_path=ss_data_path,
        n_events=n_ms_events,
        ms_config=cfg.ms_data,
        seed=ms_seed,
    )
    n_total = len(dataset)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val
    
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"  Train: {n_train}, Val: {n_val}, Test: {n_test}")
    print(f"  MS waveform shape: (C={dataset.n_channels}, T={dataset.n_time}) -> input_dim={dataset.input_dim}")
    
    A_sparse, pos = build_graph_for_shape(cfg, dataset.n_channels, dataset.n_time, device)
    n_nodes = dataset.input_dim
    
    results: Dict[str, List[LatentSizeResult]] = {
        'baseline': [],
        'ae': [],
        'diffae': [],
    }
    
    print("\n" + "="*60)
    print("Training baseline MLP (raw waveforms) - does not vary with latent size")
    print("="*60)
    
    baseline_maes = []
    baseline_rmses = []
    for trial in range(n_trials):
        print(f"  Trial {trial+1}/{n_trials}...")
        mae, rmse = run_baseline_trial(
            train_loader, val_loader, test_loader,
            dataset.input_dim, device, aux_epochs, target_key
        )
        baseline_maes.append(mae)
        baseline_rmses.append(rmse)
        print(f"    MAE: {mae:.2f} ns, RMSE: {rmse:.2f} ns")
    
    baseline_result = LatentSizeResult(
        latent_dim=0,
        model_type='baseline',
        mae_values=baseline_maes,
        rmse_values=baseline_rmses,
    )
    results['baseline'].append(baseline_result)
    print(f"  Baseline mean MAE: {baseline_result.mae_mean:.2f} ± {baseline_result.mae_std:.2f} ns")
    
    for latent_dim in latent_dims:
        print(f"\n" + "="*60)
        print(f"Latent dimension: {latent_dim} (log2 = {np.log2(latent_dim):.1f})")
        print("="*60)
        
        ae_encoder = load_ae_encoder(cfg, latent_dim, device)
        if ae_encoder is not None:
            print(f"\n  Loading AE encoder from checkpoint...")
            ae_encoder.eval()
            ae_maes = []
            ae_rmses = []
            for trial in range(n_trials):
                print(f"    AE Trial {trial+1}/{n_trials}...")
                mae, rmse = run_single_aux_trial(
                    ae_encoder, train_loader, val_loader, test_loader,
                    A_sparse, pos, device, latent_dim, aux_epochs,
                    encoder_type='ae', target_key=target_key
                )
                ae_maes.append(mae)
                ae_rmses.append(rmse)
            
            ae_result = LatentSizeResult(
                latent_dim=latent_dim,
                model_type='ae',
                mae_values=ae_maes,
                rmse_values=ae_rmses,
            )
            results['ae'].append(ae_result)
            print(f"    AE mean MAE: {ae_result.mae_mean:.2f} ± {ae_result.mae_std:.2f} ns")
        else:
            print(f"  No AE checkpoint found for latent_dim={latent_dim}")
        
        diffae_encoder = load_diffae_encoder(cfg, latent_dim, device)
        if diffae_encoder is not None:
            print(f"\n  Loading DiffAE encoder from checkpoint...")
            diffae_encoder.eval()
            diffae_maes = []
            diffae_rmses = []
            for trial in range(n_trials):
                print(f"    DiffAE Trial {trial+1}/{n_trials}...")
                mae, rmse = run_single_aux_trial(
                    diffae_encoder, train_loader, val_loader, test_loader,
                    A_sparse, pos, device, latent_dim, aux_epochs,
                    encoder_type='diffae', target_key=target_key
                )
                diffae_maes.append(mae)
                diffae_rmses.append(rmse)
            
            diffae_result = LatentSizeResult(
                latent_dim=latent_dim,
                model_type='diffae',
                mae_values=diffae_maes,
                rmse_values=diffae_rmses,
            )
            results['diffae'].append(diffae_result)
            print(f"    DiffAE mean MAE: {diffae_result.mae_mean:.2f} ± {diffae_result.mae_std:.2f} ns")
        else:
            print(f"  No DiffAE checkpoint found for latent_dim={latent_dim}")
    
    plot_latent_size_comparison(results, output_dir, metric=metric, target_key=target_key)
    
    return results


def plot_latent_size_comparison(
    results: Dict[str, List[LatentSizeResult]],
    output_dir: str,
    metric: str = 'mae',
    target_key: str = 'delta_mu',
):
    """
    Plot latent size vs performance comparison.
    
    X-axis: log2(latent_dim)
    Y-axis: MAE or RMSE (linear scale)
    Baseline: horizontal dashed line
    AE/DiffAE: points with error bars
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if results['baseline']:
        baseline = results['baseline'][0]
        if metric == 'mae':
            mean_val = baseline.mae_mean
            std_val = baseline.mae_std
        else:
            mean_val = baseline.rmse_mean
            std_val = baseline.rmse_std
        
        ax.axhline(y=mean_val, color='gray', linestyle='--', linewidth=2, label='Baseline (raw)')
        ax.axhspan(mean_val - std_val, mean_val + std_val, alpha=0.2, color='gray')
    
    colors = {'ae': 'blue', 'diffae': 'red'}
    markers = {'ae': 'o', 'diffae': 's'}
    labels = {'ae': 'AE', 'diffae': 'DiffAE'}
    
    for model_type in ['ae', 'diffae']:
        model_results = results[model_type]
        if not model_results:
            continue
        
        latent_dims = [r.latent_dim for r in model_results]
        log2_dims = [np.log2(d) for d in latent_dims]
        
        if metric == 'mae':
            means = [r.mae_mean for r in model_results]
            stds = [r.mae_std for r in model_results]
        else:
            means = [r.rmse_mean for r in model_results]
            stds = [r.rmse_std for r in model_results]
        
        ax.errorbar(
            log2_dims, means, yerr=stds,
            fmt=markers[model_type] + '-',
            color=colors[model_type],
            capsize=5,
            capthick=2,
            linewidth=2,
            markersize=8,
            label=labels[model_type],
        )
    
    if results['ae'] or results['diffae']:
        all_dims = []
        for model_type in ['ae', 'diffae']:
            all_dims.extend([r.latent_dim for r in results[model_type]])
        if all_dims:
            unique_dims = sorted(set(all_dims))
            log2_ticks = [np.log2(d) for d in unique_dims]
            ax.set_xticks(log2_ticks)
            ax.set_xticklabels([str(d) for d in unique_dims])
    
    ax.set_xlabel('Latent Dimension', fontsize=12)
    metric_label = 'MAE' if metric == 'mae' else 'RMSE'
    ax.set_ylabel(f'{metric_label} (ns)', fontsize=12)
    ax.set_title(f'Auxiliary Task Performance vs Latent Size\n(Predicting {target_key})', fontsize=14)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'latent_size_comparison_{metric}.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    print(f"\nPlot saved to {plot_path}")
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    if results['baseline']:
        baseline = results['baseline'][0]
        print(f"Baseline (raw): MAE={baseline.mae_mean:.2f}±{baseline.mae_std:.2f}, "
              f"RMSE={baseline.rmse_mean:.2f}±{baseline.rmse_std:.2f}")
    
    for model_type in ['ae', 'diffae']:
        if results[model_type]:
            print(f"\n{labels.get(model_type, model_type)}:")
            for r in results[model_type]:
                print(f"  z={r.latent_dim:4d}: MAE={r.mae_mean:.2f}±{r.mae_std:.2f}, "
                      f"RMSE={r.rmse_mean:.2f}±{r.rmse_std:.2f}")


def find_encoded_latents(cfg: Config, model_type: str, latent_dim: int) -> Optional[str]:
    """Find pre-encoded latent file for a given model type and latent dim."""
    if model_type == 'diffae':
        subdir = cfg.paths.diffae_subdir.format(latent_dim=latent_dim)
        filename = "encoded_ms_latents.h5"
    elif model_type == 'ae':
        subdir = cfg.paths.ae_subdir.format(latent_dim=latent_dim)
        filename = "vae_encoded_ms_latents.h5"
    else:
        return None
    
    path = os.path.join(cfg.paths.checkpoint_dir, subdir, filename)
    if os.path.exists(path):
        return path
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Train auxiliary predictors on pre-encoded latent representations',
        epilog="""
This script trains simple MLPs to predict delta_mu from pre-encoded latents.
The latents must be pre-computed by running diffae.py or ae.py with MS data.

Examples:
  # Train on DiffAE latents (auto-find in checkpoint dir)
  python aux.py --latent-dim 64

  # Specify path to encoded latents directly  
  python aux.py --diffae-latents checkpoints/diffae_z64/encoded_ms_latents.h5

  # Compare DiffAE and AE latents
  python aux.py --diffae-latents path/to/diffae.h5 --ae-latents path/to/ae.h5
        """
    )
    aux_cfg = default_config.aux_task
    parser.add_argument('--diffae-latents', type=str, default=None,
                        help='Path to pre-encoded DiffAE latents h5 file')
    parser.add_argument('--ae-latents', type=str, default=None,
                        help='Path to pre-encoded AE latents h5 file')
    parser.add_argument('--latent-dim', type=int, default=None,
                        help='Latent dimension to auto-find encoded files (default: from config)')
    parser.add_argument('--aux-epochs', type=int, default=aux_cfg.epochs,
                        help=f'Epochs for auxiliary MLP training (default: {aux_cfg.epochs})')
    parser.add_argument('--batch-size', type=int, default=aux_cfg.batch_size,
                        help=f'Batch size (default: {aux_cfg.batch_size})')
    parser.add_argument('--lr', type=float, default=aux_cfg.lr,
                        help=f'Learning rate (default: {aux_cfg.lr})')
    parser.add_argument('--output-dir', type=str, default=aux_cfg.output_dir,
                        help=f'Output directory (default: {aux_cfg.output_dir})')
    parser.add_argument('--n-trials', type=int, default=1,
                        help='Number of trials for error bars')
    parser.add_argument('--metric', type=str, default='mae', choices=['mae', 'rmse'],
                        help='Metric to report')
    
    args = parser.parse_args()
    
    cfg = default_config
    latent_dim = args.latent_dim if args.latent_dim is not None else cfg.encoder.latent_dim
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    diffae_path = args.diffae_latents
    ae_path = args.ae_latents
    
    if diffae_path is None:
        diffae_path = find_encoded_latents(cfg, 'diffae', latent_dim)
    if ae_path is None:
        ae_path = find_encoded_latents(cfg, 'ae', latent_dim)
    
    if diffae_path is None and ae_path is None:
        print("ERROR: No encoded latent files found!")
        print(f"  Looked for DiffAE: {cfg.paths.checkpoint_dir}/{cfg.paths.diffae_subdir.format(latent_dim=latent_dim)}/encoded_ms_latents.h5")
        print(f"  Looked for AE: {cfg.paths.checkpoint_dir}/{cfg.paths.ae_subdir.format(latent_dim=latent_dim)}/vae_encoded_ms_latents.h5")
        print("\nTo generate encoded latents, run:")
        print("  python diffae.py   # for DiffAE")
        print("  python ae.py       # for Graph AE")
        print("\nMake sure to train with MS data (default) and save encoded latents.")
        return
    
    results = {}
    os.makedirs(args.output_dir, exist_ok=True)
    
    for model_name, latents_path in [('DiffAE', diffae_path), ('AE', ae_path)]:
        if latents_path is None:
            print(f"\n{model_name}: No encoded latents found, skipping.")
            continue
        
        print(f"\n" + "="*60)
        print(f"Loading {model_name} encoded latents from: {latents_path}")
        print("="*60)
        
        dataset = EncodedMSDataset(latents_path)
        print(f"  Samples: {len(dataset)}, latent_dim: {dataset.latent_dim}")
        
        n_total = len(dataset)
        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)
        n_test = n_total - n_train - n_val
        
        train_set, val_set, test_set = random_split(
            dataset, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
        
        print(f"  Train: {n_train}, Val: {n_val}, Test: {n_test}")
        
        all_maes = []
        all_rmses = []
        
        for trial in range(args.n_trials):
            if args.n_trials > 1:
                print(f"\n  Trial {trial+1}/{args.n_trials}")
            
            mlp = MLP(
                in_dim=dataset.latent_dim,
                hidden_dims=[128, 64],
                out_dim=1,
                dropout=0.1,
            ).to(device)
            
            train_losses, val_losses, val_times = train_aux_mlp_on_latents(
                mlp, train_loader, val_loader, device,
                epochs=args.aux_epochs, lr=args.lr, target_key='delta_mu'
            )
            
            mae, rmse, preds, targets = evaluate_aux_mlp_on_latents(
                mlp, test_loader, device, target_key='delta_mu'
            )
            
            all_maes.append(mae)
            all_rmses.append(rmse)
            print(f"    MAE: {mae:.2f} ns, RMSE: {rmse:.2f} ns")
        
        mean_mae = np.mean(all_maes)
        std_mae = np.std(all_maes) if len(all_maes) > 1 else 0.0
        mean_rmse = np.mean(all_rmses)
        std_rmse = np.std(all_rmses) if len(all_rmses) > 1 else 0.0
        
        results[model_name] = AuxTrainingResult(
            model_name=model_name,
            train_losses=train_losses,
            val_losses=val_losses,
            test_mae=mean_mae,
            test_rmse=mean_rmse,
            predictions=preds,
            targets=targets,
            val_times=val_times,
        )
        
        print(f"\n  {model_name} Final: MAE={mean_mae:.2f}±{std_mae:.2f} ns, RMSE={mean_rmse:.2f}±{std_rmse:.2f} ns")
    
    if results:
        plot_results(results, args.output_dir)
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    for name, res in results.items():
        print(f"  {name}: MAE={res.test_mae:.2f} ns, RMSE={res.test_rmse:.2f} ns")


if __name__ == '__main__':
    main()
