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

from config import Config, default_config
from lz_data_loader import TritiumSSDataLoader
from diffae import GraphEncoder, DiffAEContext, DiffAEDataStats
from ae import GraphVAEEncoder
from models.graph_unet import TopKPool, build_block_diagonal_adj
from utils.sparse_ops import to_binary, subgraph_coo


class GraphAEEncoderBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.eps = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        neighbor_sum = torch.sparse.mm(adj, h)
        h = (1 + self.eps) * h + neighbor_sum
        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)
        h = self.dropout(h)
        return x + h


class GraphAEDecoderBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)
        h = self.dropout(h)
        return x + h


class GraphAutoencoder(nn.Module):
    """
    Regular graph autoencoder (non-diffusion) for comparison.
    Encoder uses graph convolutions + pooling, decoder uses MLPs.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        latent_dim: int,
        n_nodes: int,
        depth: int = 3,
        blocks_per_stage: int = 2,
        pool_ratio: float = 0.5,
        dropout: float = 0.0,
        pos_dim: int = 3,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_nodes = n_nodes
        self.depth = depth
        self.pool_ratio = pool_ratio

        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.pos_mlp = nn.Sequential(
            nn.Linear(pos_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        for _ in range(depth):
            stage_blocks = nn.ModuleList([
                GraphAEEncoderBlock(hidden_dim, dropout)
                for _ in range(blocks_per_stage)
            ])
            self.enc_blocks.append(stage_blocks)
            self.pools.append(TopKPool(hidden_dim, ratio=pool_ratio))

        self.final_enc_blocks = nn.ModuleList([
            GraphAEEncoderBlock(hidden_dim, dropout)
            for _ in range(blocks_per_stage)
        ])

        self.to_latent = nn.Linear(hidden_dim, latent_dim)

        self.from_latent = nn.Linear(latent_dim, hidden_dim * 16)
        
        self.dec_blocks = nn.ModuleList([
            GraphAEDecoderBlock(hidden_dim, dropout)
            for _ in range(depth * blocks_per_stage)
        ])
        
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, in_dim),
        )

        self._cached_block_adj = {}

    def _get_block_adj(self, adj: torch.Tensor, batch_size: int) -> torch.Tensor:
        key = (adj.device, adj.size(), adj._nnz(), batch_size)
        if key in self._cached_block_adj:
            return self._cached_block_adj[key]
        adj_binary = to_binary(adj)
        block_adj = build_block_diagonal_adj(adj_binary, batch_size)
        self._cached_block_adj[key] = block_adj
        return block_adj

    def encode(
        self, 
        x: torch.Tensor, 
        adj: torch.Tensor, 
        pos: torch.Tensor,
        batch_size: int = 1
    ) -> torch.Tensor:
        """Encode to latent representation."""
        N_single = adj.size(0)
        adj0 = self._get_block_adj(adj, batch_size).to(device=x.device, dtype=x.dtype)

        h = self.in_proj(x)
        pos_tiled = pos.repeat(batch_size, 1)
        pos_emb = self.pos_mlp(pos_tiled.to(x.dtype))
        h = h + pos_emb

        nodes_per_graph = N_single
        h_cur, adj_cur = h, adj0

        for d in range(self.depth):
            for blk in self.enc_blocks[d]:
                h_cur = blk(h_cur, adj_cur)
            h_pool, keep_idx, k_per_graph = self.pools[d](h_cur, nodes_per_graph)
            K = h_pool.size(0)
            adj_next = to_binary(subgraph_coo(adj_cur, keep_idx, K))
            h_cur, adj_cur, nodes_per_graph = h_pool, adj_next, k_per_graph

        for blk in self.final_enc_blocks:
            h_cur = blk(h_cur, adj_cur)

        h_graph = h_cur.view(batch_size, nodes_per_graph, self.hidden_dim)
        h_agg = h_graph.mean(dim=1)
        z = self.to_latent(h_agg)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent to reconstruction."""
        B = z.shape[0]
        h = self.from_latent(z)
        h = h.view(B, 16, self.hidden_dim)
        h = F.interpolate(h.transpose(1, 2), size=self.n_nodes, mode='linear', align_corners=False)
        h = h.transpose(1, 2)
        h = h.reshape(B * self.n_nodes, self.hidden_dim)
        
        for blk in self.dec_blocks:
            h = blk(h)
        
        out = self.out_proj(h)
        return out.view(B, self.n_nodes, self.in_dim)

    def forward(
        self, 
        x: torch.Tensor, 
        adj: torch.Tensor, 
        pos: torch.Tensor,
        batch_size: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass returning reconstruction and latent."""
        z = self.encode(x, adj, pos, batch_size)
        x_recon = self.decode(z)
        return x_recon, z


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


class MSDataset(Dataset):
    """Dataset for multi-scatter events."""
    def __init__(self, h5_path: str, transform=None):
        with h5py.File(h5_path, 'r') as f:
            self.waveforms = f['waveforms'][:]
            self.delta_mu = f['delta_mu'][:].astype(np.float32)
            self.delta_bins = f['delta_bins'][:].astype(np.float32)
            self.xc1 = f['xc1'][:].astype(np.float32)
            self.yc1 = f['yc1'][:].astype(np.float32)
            self.xc2 = f['xc2'][:].astype(np.float32)
            self.yc2 = f['yc2'][:].astype(np.float32)
        
        self.n_channels = self.waveforms.shape[1]
        self.n_time = self.waveforms.shape[2]
        self.transform = transform

    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, idx):
        wf = self.waveforms[idx].astype(np.float32)
        wf_flat = wf.T.reshape(-1, 1)
        
        if self.transform:
            wf_flat = self.transform(wf_flat)
        
        targets = {
            'delta_mu': self.delta_mu[idx],
            'delta_bins': self.delta_bins[idx],
            'dx': self.xc2[idx] - self.xc1[idx],
            'dy': self.yc2[idx] - self.yc1[idx],
        }
        
        return torch.from_numpy(wf_flat), targets


@dataclass
class AuxTrainingResult:
    model_name: str
    train_losses: list
    val_losses: list
    test_mae: float
    test_rmse: float
    predictions: np.ndarray
    targets: np.ndarray


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
) -> Tuple[list, list]:
    """Train auxiliary MLP on encoded representations."""
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    encoder.eval()
    train_losses = []
    val_losses = []
    
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
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train_loss={train_losses[-1]:.4f}, val_loss={val_losses[-1]:.4f}")
    
    return train_losses, val_losses


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
) -> Tuple[list, list]:
    """Train MLP directly on flattened waveforms (baseline)."""
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    train_losses = []
    val_losses = []
    
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
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train_loss={train_losses[-1]:.4f}, val_loss={val_losses[-1]:.4f}")
    
    return train_losses, val_losses


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


def plot_results(results: Dict[str, AuxTrainingResult], output_dir: str):
    """Plot comparison of different models."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for name, res in results.items():
        axes[0].plot(res.train_losses, label=f'{name} (train)')
        axes[0].plot(res.val_losses, '--', label=f'{name} (val)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Training Curves')
    axes[0].legend()
    axes[0].set_yscale('log')
    
    names = list(results.keys())
    maes = [results[n].test_mae for n in names]
    rmses = [results[n].test_rmse for n in names]
    
    x = np.arange(len(names))
    width = 0.35
    axes[1].bar(x - width/2, maes, width, label='MAE')
    axes[1].bar(x + width/2, rmses, width, label='RMSE')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=15)
    axes[1].set_ylabel('Error (ns)')
    axes[1].set_title('Test Set Performance')
    axes[1].legend()
    
    for name, res in results.items():
        axes[2].scatter(res.targets, res.predictions, alpha=0.3, s=5, label=name)
    
    all_vals = np.concatenate([r.targets for r in results.values()] + 
                               [r.predictions for r in results.values()])
    lims = [all_vals.min() - 10, all_vals.max() + 10]
    axes[2].plot(lims, lims, 'k--', alpha=0.5)
    axes[2].set_xlim(lims)
    axes[2].set_ylim(lims)
    axes[2].set_xlabel('True Δμ (ns)')
    axes[2].set_ylabel('Predicted Δμ (ns)')
    axes[2].set_title('Predictions vs Ground Truth')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/aux_comparison.png', dpi=150)
    plt.close()
    
    print(f"\nResults saved to {output_dir}/aux_comparison.png")
    print("\n" + "="*50)
    print("Summary")
    print("="*50)
    for name, res in results.items():
        print(f"{name:20s}: MAE={res.test_mae:.2f} ns, RMSE={res.test_rmse:.2f} ns")


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
    ms_data_path: str,
    output_dir: str,
    n_trials: int = 3,
    aux_epochs: int = 50,
    batch_size: int = 32,
    target_key: str = 'delta_mu',
    metric: str = 'mae',  # 'mae' or 'rmse'
):
    """
    Train auxiliary tasks at varying encoded sizes and plot comparison.
    
    Args:
        latent_dims: List of latent dimensions to test (e.g., [8, 16, 32, 64, 128, 256])
        cfg: Configuration object
        ms_data_path: Path to multi-scatter data
        output_dir: Directory to save plots
        n_trials: Number of trials per latent size for error bars
        aux_epochs: Epochs for auxiliary MLP training
        batch_size: Batch size for training
        target_key: Target to predict ('delta_mu', 'delta_bins', etc.)
        metric: Which metric to plot ('mae' or 'rmse')
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"\nLoading MS data from {ms_data_path}...")
    dataset = MSDataset(ms_data_path)
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
    
    ss_loader = TritiumSSDataLoader(cfg.paths.tritium_h5, cfg.paths.channel_positions)
    graph = ss_loader.load_adjacency_sparse(
        z_sep=cfg.graph.z_sep,
        radius=cfg.graph.radius,
        z_hops=cfg.graph.z_hops
    )
    A_sparse = graph.adjacency.to(device)
    pos = graph.positions_xyz.to(device)
    n_nodes = ss_loader.n_channels * ss_loader.n_time_points
    
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
            n_nodes, device, aux_epochs, target_key
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


def main():
    parser = argparse.ArgumentParser(description='Train auxiliary predictors on latent representations')
    parser.add_argument('--ms-data', type=str, default='data/tritium_ms_42.h5',
                        help='Path to MS data')
    parser.add_argument('--diffae-ckpt', type=str, default=None,
                        help='Path to DiffAE checkpoint (optional)')
    parser.add_argument('--train-ae', action='store_true',
                        help='Train regular AE from scratch')
    parser.add_argument('--ae-epochs', type=int, default=100,
                        help='Epochs for AE training')
    parser.add_argument('--aux-epochs', type=int, default=50,
                        help='Epochs for auxiliary MLP training')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--output-dir', type=str, default='aux_results',
                        help='Output directory')
    
    parser.add_argument('--compare-latent-sizes', action='store_true',
                        help='Run latent size comparison across multiple pre-trained models')
    parser.add_argument('--latent-dims', type=str, default='8,16,32,64,128,256',
                        help='Comma-separated list of latent dimensions to compare')
    parser.add_argument('--n-trials', type=int, default=3,
                        help='Number of trials per latent size for error bars')
    parser.add_argument('--metric', type=str, default='mae', choices=['mae', 'rmse'],
                        help='Metric to plot (mae or rmse)')
    
    args = parser.parse_args()
    
    if args.compare_latent_sizes:
        latent_dims = [int(x.strip()) for x in args.latent_dims.split(',')]
        print(f"Running latent size comparison for dims: {latent_dims}")
        run_latent_size_comparison(
            latent_dims=latent_dims,
            cfg=default_config,
            ms_data_path=args.ms_data,
            output_dir=args.output_dir,
            n_trials=args.n_trials,
            aux_epochs=args.aux_epochs,
            batch_size=args.batch_size,
            metric=args.metric,
        )
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    cfg = default_config
    
    print(f"\nLoading MS data from {args.ms_data}...")
    dataset = MSDataset(args.ms_data)
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
    
    ss_loader = TritiumSSDataLoader(cfg.paths.tritium_h5, cfg.paths.channel_positions)
    graph = ss_loader.load_adjacency_sparse(
        z_sep=cfg.graph.z_sep,
        radius=cfg.graph.radius,
        z_hops=cfg.graph.z_hops
    )
    A_sparse = graph.adjacency.to(device)
    pos = graph.positions_xyz.to(device)
    n_nodes = ss_loader.n_channels * ss_loader.n_time_points
    
    results = {}
    
    print("\n" + "="*50)
    print("Training baseline MLP (raw waveforms)")
    print("="*50)
    
    baseline_mlp = MLP(
        in_dim=n_nodes,
        hidden_dims=[256, 128, 64],
        out_dim=1,
        dropout=0.1,
    ).to(device)
    
    train_losses, val_losses = train_baseline_mlp(
        baseline_mlp, train_loader, val_loader, device,
        epochs=args.aux_epochs, target_key='delta_mu'
    )
    
    mae, rmse, preds, targets = evaluate_baseline_mlp(
        baseline_mlp, test_loader, device, target_key='delta_mu'
    )
    
    results['Baseline'] = AuxTrainingResult(
        model_name='Baseline',
        train_losses=train_losses,
        val_losses=val_losses,
        test_mae=mae,
        test_rmse=rmse,
        predictions=preds,
        targets=targets,
    )
    print(f"  Test MAE: {mae:.2f} ns, RMSE: {rmse:.2f} ns")
    
    if args.train_ae:
        print("\n" + "="*50)
        print("Training Regular Graph Autoencoder")
        print("="*50)
        
        regular_ae = GraphAutoencoder(
            in_dim=1,
            hidden_dim=cfg.encoder.hidden_dim,
            latent_dim=cfg.encoder.latent_dim,
            n_nodes=n_nodes,
            depth=cfg.encoder.depth,
            blocks_per_stage=cfg.encoder.blocks_per_stage,
            pool_ratio=cfg.encoder.pool_ratio,
            dropout=cfg.encoder.dropout,
            pos_dim=cfg.model.pos_dim,
        ).to(device)
        
        ae_optimizer = torch.optim.AdamW(regular_ae.parameters(), lr=1e-3, weight_decay=1e-4)
        
        for epoch in range(args.ae_epochs):
            regular_ae.train()
            epoch_loss = 0.0
            n_batches = 0
            
            for wf, _ in train_loader:
                wf = wf.to(device)
                B = wf.shape[0]
                wf_flat = wf.view(B * wf.shape[1], 1)
                
                recon, z = regular_ae(wf_flat, A_sparse, pos, batch_size=B)
                loss = F.mse_loss(recon, wf)
                
                ae_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(regular_ae.parameters(), 1.0)
                ae_optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            if (epoch + 1) % 20 == 0:
                print(f"  AE Epoch {epoch+1}/{args.ae_epochs}: recon_loss={epoch_loss/n_batches:.6f}")
        
        print("\nTraining auxiliary MLP on Regular AE representations...")
        ae_aux_mlp = MLP(
            in_dim=cfg.encoder.latent_dim,
            hidden_dims=[128, 64],
            out_dim=1,
            dropout=0.1,
        ).to(device)
        
        train_losses, val_losses = train_aux_model(
            regular_ae, ae_aux_mlp, train_loader, val_loader,
            A_sparse, pos, device,
            epochs=args.aux_epochs, target_key='delta_mu',
            encoder_type='regular_ae'
        )
        
        mae, rmse, preds, targets = evaluate_aux_model(
            regular_ae, ae_aux_mlp, test_loader,
            A_sparse, pos, device,
            target_key='delta_mu', encoder_type='regular_ae'
        )
        
        results['Regular AE'] = AuxTrainingResult(
            model_name='Regular AE',
            train_losses=train_losses,
            val_losses=val_losses,
            test_mae=mae,
            test_rmse=rmse,
            predictions=preds,
            targets=targets,
        )
        print(f"  Test MAE: {mae:.2f} ns, RMSE: {rmse:.2f} ns")
    
    diffae_ckpt = args.diffae_ckpt
    if diffae_ckpt is None:
        diffae_subdir = cfg.paths.diffae_subdir.format(latent_dim=cfg.encoder.latent_dim)
        diffae_ckpt_dir = os.path.join(cfg.paths.checkpoint_dir, diffae_subdir)
        if os.path.isdir(diffae_ckpt_dir):
            ckpt_files = sorted([f for f in os.listdir(diffae_ckpt_dir) if f.startswith('diffae_epoch_')])
            if ckpt_files:
                diffae_ckpt = os.path.join(diffae_ckpt_dir, ckpt_files[-1])
    
    if diffae_ckpt and os.path.exists(diffae_ckpt):
        print("\n" + "="*50)
        print(f"Loading DiffAE from {diffae_ckpt}")
        print("="*50)
        
        diffae_encoder = GraphEncoder(
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
        
        ckpt = torch.load(diffae_ckpt, map_location=device)
        if 'ema_encoder' in ckpt:
            diffae_encoder.load_state_dict(ckpt['ema_encoder'])
        else:
            diffae_encoder.load_state_dict(ckpt['encoder'])
        
        print("\nTraining auxiliary MLP on DiffAE representations...")
        diffae_aux_mlp = MLP(
            in_dim=cfg.encoder.latent_dim,
            hidden_dims=[128, 64],
            out_dim=1,
            dropout=0.1,
        ).to(device)
        
        train_losses, val_losses = train_aux_model(
            diffae_encoder, diffae_aux_mlp, train_loader, val_loader,
            A_sparse, pos, device,
            epochs=args.aux_epochs, target_key='delta_mu',
            encoder_type='diffae'
        )
        
        mae, rmse, preds, targets = evaluate_aux_model(
            diffae_encoder, diffae_aux_mlp, test_loader,
            A_sparse, pos, device,
            target_key='delta_mu', encoder_type='diffae'
        )
        
        results['DiffAE'] = AuxTrainingResult(
            model_name='DiffAE',
            train_losses=train_losses,
            val_losses=val_losses,
            test_mae=mae,
            test_rmse=rmse,
            predictions=preds,
            targets=targets,
        )
        print(f"  Test MAE: {mae:.2f} ns, RMSE: {rmse:.2f} ns")
    else:
        print("\nNo DiffAE checkpoint found. Run diffae.py first to train the model.")
    
    plot_results(results, args.output_dir)


if __name__ == '__main__':
    main()
