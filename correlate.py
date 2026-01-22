import os
import argparse
import glob
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm

from config import Config, default_config
from lz_data_loader import TritiumSSDataLoader
from models.graph_unet import GraphDDPMUNet
from diffusion.schedule import build_cosine_schedule, sinusoidal_embedding
from ddpm_sparse import sample_core, standardize_batch

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_checkpoint(cfg: Config, device: torch.device) -> Tuple[nn.Module, nn.Module, dict]:
    """Load trained DDPM model from checkpoint."""
    cond_proj = nn.Sequential(
        nn.Linear(cfg.conditioning.cond_in_dim, 64),
        nn.SiLU(),
        nn.Linear(64, cfg.conditioning.cond_proj_dim)
    ).to(device)

    core = GraphDDPMUNet(
        in_dim=cfg.model.in_dim,
        cond_dim=cfg.conditioning.cond_proj_dim + cfg.conditioning.time_dim,
        hidden_dim=cfg.model.hidden_dim,
        depth=cfg.model.depth,
        blocks_per_stage=cfg.model.blocks_per_stage,
        pool_ratio=cfg.model.pool_ratio,
        out_dim=cfg.model.out_dim,
        dropout=cfg.model.dropout,
        pos_dim=cfg.model.pos_dim,
    ).to(device)

    checkpoint_files = sorted(glob.glob(os.path.join(cfg.paths.checkpoint_dir, "ckpt_epoch_*.pt")))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {cfg.paths.checkpoint_dir}")

    latest_ckpt = checkpoint_files[-1]
    print(f"Loading checkpoint: {latest_ckpt}")
    
    chk = torch.load(latest_ckpt, map_location=device)
    core.load_state_dict(chk["core"])
    cond_proj.load_state_dict(chk["cond_proj"])
    
    schedule = build_cosine_schedule(cfg.diffusion.timesteps, device)
    
    return core, cond_proj, schedule


def compute_z_profile_metrics(z_profile: np.ndarray, debug: bool = False) -> Optional[Dict]:
    """
    Compute metrics for a z-profile (summed over channels).
    
    Parameters:
    -----------
    z_profile : ndarray
        1D array of intensity summed over all channels for each time point
    debug : bool
        Print debug information
        
    Returns:
    --------
    dict or None
        Dictionary of profile metrics
    """
    if np.sum(z_profile) == 0 or np.max(z_profile) == 0:
        if debug:
            print("Warning: Empty or zero profile")
        return None

    z_profile = np.abs(z_profile)
    time_axis = np.arange(len(z_profile))

    if debug:
        print(f"Profile stats: min={np.min(z_profile):.3f}, max={np.max(z_profile):.3f}, "
              f"mean={np.mean(z_profile):.3f}, sum={np.sum(z_profile):.3f}")

    peak_idx = np.argmax(z_profile)
    peak_amplitude = z_profile[peak_idx]
    peak_time = time_axis[peak_idx]
    total_integral = np.trapz(z_profile, time_axis)

    threshold_10 = 0.1 * peak_amplitude
    threshold_90 = 0.9 * peak_amplitude

    rise_10_idx = np.where(z_profile[:peak_idx] >= threshold_10)[0]
    rise_10_time = time_axis[rise_10_idx[0]] if len(rise_10_idx) > 0 else time_axis[0]

    rise_90_idx = np.where(z_profile[:peak_idx] >= threshold_90)[0]
    rise_90_time = time_axis[rise_90_idx[0]] if len(rise_90_idx) > 0 else peak_time

    rise_time = rise_90_time - rise_10_time

    fall_90_idx = np.where(z_profile[peak_idx:] <= threshold_90)[0]
    fall_90_time = time_axis[peak_idx + fall_90_idx[0]] if len(fall_90_idx) > 0 else peak_time

    fall_10_idx = np.where(z_profile[peak_idx:] <= threshold_10)[0]
    fall_10_time = time_axis[peak_idx + fall_10_idx[0]] if len(fall_10_idx) > 0 else time_axis[-1]

    fall_time = fall_10_time - fall_90_time

    half_max = 0.5 * peak_amplitude
    left_idx = np.where(z_profile[:peak_idx] >= half_max)[0]
    left_time = time_axis[left_idx[0]] if len(left_idx) > 0 else time_axis[0]

    right_idx = np.where(z_profile[peak_idx:] <= half_max)[0]
    right_time = time_axis[peak_idx + right_idx[0]] if len(right_idx) > 0 else time_axis[-1]

    fwhm = max(0, right_time - left_time)

    std_dev = np.sqrt(np.average((time_axis - peak_time)**2, weights=z_profile + 1e-8))
    
    skewness = stats.skew(z_profile)
    kurtosis = stats.kurtosis(z_profile)

    return {
        'peak_amplitude': peak_amplitude,
        'peak_time': peak_time,
        'total_integral': total_integral,
        'rise_time': rise_time,
        'fall_time': fall_time,
        'fwhm': fwhm,
        'std_dev': std_dev,
        'skewness': skewness,
        'kurtosis': kurtosis,
    }


def compute_xy_metrics(xy_profile: np.ndarray, channel_positions: np.ndarray) -> Optional[Dict]:
    """
    Compute metrics for xy distribution (summed over time).
    
    Parameters:
    -----------
    xy_profile : ndarray
        1D array of intensity for each channel (summed over time)
    channel_positions : ndarray
        (n_channels, 2) array of channel x,y positions
        
    Returns:
    --------
    dict or None
        Dictionary of spatial metrics
    """
    if np.sum(xy_profile) == 0:
        return None

    xy_profile = np.abs(xy_profile)
    total = np.sum(xy_profile)
    
    weights = xy_profile / (total + 1e-8)
    centroid_x = np.sum(weights * channel_positions[:, 0])
    centroid_y = np.sum(weights * channel_positions[:, 1])
    
    dx = channel_positions[:, 0] - centroid_x
    dy = channel_positions[:, 1] - centroid_y
    spread_x = np.sqrt(np.sum(weights * dx**2))
    spread_y = np.sqrt(np.sum(weights * dy**2))
    
    return {
        'total_intensity': total,
        'centroid_x': centroid_x,
        'centroid_y': centroid_y,
        'spread_x': spread_x,
        'spread_y': spread_y,
    }


def extract_profiles(data: np.ndarray, n_channels: int, n_time: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract z-profile (sum over channels) and xy-profile (sum over time).
    
    Parameters:
    -----------
    data : ndarray
        Flattened data of shape (n_channels * n_time,) or (n_channels * n_time, 1)
    n_channels : int
        Number of spatial channels
    n_time : int
        Number of time points
        
    Returns:
    --------
    z_profile : ndarray
        Shape (n_time,) - intensity summed over all channels
    xy_profile : ndarray
        Shape (n_channels,) - intensity summed over all time points
    """
    data = data.flatten()
    data_2d = data.reshape(n_channels, n_time, order='F')
    
    z_profile = data_2d.sum(axis=0)
    xy_profile = data_2d.sum(axis=1)
    
    return z_profile, xy_profile


@torch.no_grad()
def generate_samples(
    core: nn.Module,
    cond_proj: nn.Module,
    schedule: dict,
    data_loader: TritiumSSDataLoader,
    graph,
    cfg: Config,
    n_samples: int,
    device: torch.device,
    batch_size: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate DDPM samples conditioned on real data conditions.
    
    Returns:
    --------
    real_data : ndarray of shape (n_samples, n_nodes, 1)
    generated_data : ndarray of shape (n_samples, n_nodes, 1)
    conditions : ndarray of shape (n_samples, cond_dim)
    """
    core.eval()
    cond_proj.eval()
    
    A_sparse = graph.adjacency.to(device)
    pos = graph.positions_xyz.to(device)
    n_nodes = data_loader.n_channels * data_loader.n_time_points
    
    real_list = []
    gen_list = []
    cond_list = []
    
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    for _ in tqdm(range(n_batches), desc="Generating samples"):
        current_batch = min(batch_size, n_samples - len(real_list))
        if current_batch <= 0:
            break
            
        batch_np, batch_cond = data_loader.get_batch(current_batch)
        batch_cond_t = torch.from_numpy(batch_cond.astype(np.float32)).to(device)
        
        batch_np_std = standardize_batch(batch_np)
        
        cond_base = cond_proj(batch_cond_t)
        
        samples = sample_core(
            core=core,
            schedule=schedule,
            A_sparse=A_sparse,
            cond_proj=cond_base,
            time_dim=cfg.conditioning.time_dim,
            shape=(current_batch, 1, n_nodes),
            parametrization=cfg.diffusion.parametrization,
            pos=pos,
            pbar=False,
        ).cpu().numpy()
        
        for i in range(current_batch):
            real_list.append(batch_np_std[i])
            gen_list.append(samples[i, 0, :, np.newaxis])
            cond_list.append(batch_cond[i])
    
    return np.array(real_list), np.array(gen_list), np.array(cond_list)


def collect_metrics(
    data_list: List[np.ndarray],
    n_channels: int,
    n_time: int,
    channel_positions: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Collect z-profile and xy-profile metrics for a list of samples."""
    z_metrics_list = []
    xy_metrics_list = []
    
    for data in data_list:
        z_profile, xy_profile = extract_profiles(data, n_channels, n_time)
        
        z_met = compute_z_profile_metrics(z_profile)
        xy_met = compute_xy_metrics(xy_profile, channel_positions)
        
        if z_met is not None:
            z_metrics_list.append(z_met)
        if xy_met is not None:
            xy_metrics_list.append(xy_met)
    
    z_metrics = {}
    if z_metrics_list:
        for key in z_metrics_list[0].keys():
            z_metrics[key] = np.array([m[key] for m in z_metrics_list])
    
    xy_metrics = {}
    if xy_metrics_list:
        for key in xy_metrics_list[0].keys():
            xy_metrics[key] = np.array([m[key] for m in xy_metrics_list])
    
    return z_metrics, xy_metrics


def plot_correlation_matrix(
    metrics_a: Dict[str, np.ndarray],
    metrics_b: Dict[str, np.ndarray],
    save_path: str,
    title: str = "Correlation Analysis",
    label_a: str = "Dataset A",
    label_b: str = "Dataset B",
):
    """Create correlation plots for all metrics."""
    metrics_names = list(metrics_a.keys())
    n_metrics = len(metrics_names)
    
    if n_metrics == 0:
        print(f"Warning: No metrics to plot for {title}")
        return []
    
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = np.atleast_2d(axes).flatten()
    
    correlations = []
    
    for i, metric in enumerate(metrics_names):
        ax = axes[i]
        
        vals_a = metrics_a[metric]
        vals_b = metrics_b[metric]
        
        n = min(len(vals_a), len(vals_b))
        vals_a = vals_a[:n]
        vals_b = vals_b[:n]
        
        p99_a = np.percentile(np.abs(vals_a), 99)
        p99_b = np.percentile(np.abs(vals_b), 99)
        max_val = max(p99_a, p99_b)
        
        valid_mask = np.isfinite(vals_a) & np.isfinite(vals_b)
        vals_a_clean = vals_a[valid_mask]
        vals_b_clean = vals_b[valid_mask]
        
        if len(vals_a_clean) < 10:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        corr = np.corrcoef(vals_a_clean, vals_b_clean)[0, 1]
        if np.isfinite(corr):
            correlations.append(corr)
        
        hb = ax.hexbin(vals_a_clean, vals_b_clean, gridsize=40, cmap='viridis', mincnt=1, alpha=0.8)
        
        min_val = min(np.min(vals_a_clean), np.min(vals_b_clean))
        max_val = max(np.max(vals_a_clean), np.max(vals_b_clean))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8, label='y=x')
        
        try:
            slope, intercept, r_val, p_val, std_err = stats.linregress(vals_a_clean, vals_b_clean)
            x_fit = np.linspace(min_val, max_val, 100)
            y_fit = slope * x_fit + intercept
            ax.plot(x_fit, y_fit, 'orange', linewidth=2, alpha=0.8, label=f'Fit (R²={r_val**2:.3f})')
        except:
            pass
        
        rmse = np.sqrt(np.mean((vals_a_clean - vals_b_clean)**2))
        
        display_name = metric.replace('_', ' ').title()
        ax.set_xlabel(f'{label_a} {display_name}', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'{label_b} {display_name}', fontsize=11, fontweight='bold')
        ax.set_title(f'{display_name}\nρ = {corr:.4f}, RMSE = {rmse:.3e}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    for i in range(n_metrics, len(axes)):
        fig.delaxes(axes[i])
    
    if correlations:
        avg_corr = np.mean(correlations)
        fig.suptitle(f'{title}\nAverage Correlation: {avg_corr:.4f}', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return correlations


def plot_single_distribution(
    metrics: Dict[str, np.ndarray],
    save_path: str,
    title: str = "Distribution",
):
    """Plot distributions for all metrics from a single dataset."""
    metrics_names = list(metrics.keys())
    n_metrics = len(metrics_names)
    
    if n_metrics == 0:
        return
    
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.atleast_2d(axes).flatten()
    
    for i, metric in enumerate(metrics_names):
        ax = axes[i]
        vals = metrics[metric]
        
        p1 = np.percentile(vals, 1)
        p99 = np.percentile(vals, 99)
        vals_clean = vals[(vals >= p1) & (vals <= p99)]
        
        bins = 50
        ax.hist(vals_clean, bins=bins, alpha=0.7, density=True, color='steelblue', 
                edgecolor='black', linewidth=0.5)
        
        mean_val = np.mean(vals_clean)
        std_val = np.std(vals_clean)
        
        ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, linewidth=2, label=f'Mean: {mean_val:.3f}')
        
        display_name = metric.replace('_', ' ').title()
        ax.set_xlabel(display_name, fontsize=11, fontweight='bold')
        ax.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax.set_title(f'{display_name}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        stats_text = f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}\nN: {len(vals_clean)}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    for i in range(n_metrics, len(axes)):
        fig.delaxes(axes[i])
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_distribution_comparison(
    metrics_a: Dict[str, np.ndarray],
    metrics_b: Dict[str, np.ndarray],
    save_path: str,
    label_a: str = "Dataset A",
    label_b: str = "Dataset B",
):
    """Plot distribution comparisons for all metrics."""
    metrics_names = list(metrics_a.keys())
    n_metrics = len(metrics_names)
    
    if n_metrics == 0:
        return
    
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.atleast_2d(axes).flatten()
    
    for i, metric in enumerate(metrics_names):
        ax = axes[i]
        
        vals_a = metrics_a[metric]
        vals_b = metrics_b[metric]
        
        p99_a = np.percentile(np.abs(vals_a), 99)
        p99_b = np.percentile(np.abs(vals_b), 99)
        max_val = max(p99_a, p99_b) * 1.1
        min_val = min(np.min(vals_a), np.min(vals_b))
        
        vals_a_clean = vals_a[(vals_a <= max_val) & (vals_a >= min_val)]
        vals_b_clean = vals_b[(vals_b <= max_val) & (vals_b >= min_val)]
        
        bins = np.linspace(min_val, max_val, 50)
        ax.hist(vals_a_clean, bins=bins, alpha=0.6, label=label_a, density=True, color='blue', edgecolor='black', linewidth=0.5)
        ax.hist(vals_b_clean, bins=bins, alpha=0.6, label=label_b, density=True, color='red', edgecolor='black', linewidth=0.5)
        
        mean_a, std_a = np.mean(vals_a_clean), np.std(vals_a_clean)
        mean_b, std_b = np.mean(vals_b_clean), np.std(vals_b_clean)
        
        ks_stat, ks_p = stats.ks_2samp(vals_a_clean, vals_b_clean)
        
        ax.axvline(mean_a, color='blue', linestyle='--', alpha=0.8, linewidth=2)
        ax.axvline(mean_b, color='red', linestyle='--', alpha=0.8, linewidth=2)
        
        display_name = metric.replace('_', ' ').title()
        ax.set_xlabel(display_name, fontsize=11, fontweight='bold')
        ax.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax.set_title(f'{display_name}\nKS p-value: {ks_p:.4f}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        stats_text = f'{label_a}: {mean_a:.3f} ± {std_a:.3f}\n{label_b}: {mean_b:.3f} ± {std_b:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    for i in range(n_metrics, len(axes)):
        fig.delaxes(axes[i])
    
    fig.suptitle('Distribution Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_sample_profiles(
    data_a: List[np.ndarray],
    data_b: List[np.ndarray],
    n_channels: int,
    n_time: int,
    save_path: str,
    label_a: str = "Dataset A",
    label_b: str = "Dataset B",
    n_samples: int = 5,
):
    """Plot sample z-profiles side by side."""
    fig, axes = plt.subplots(n_samples, 2, figsize=(12, 3 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    time_axis = np.arange(n_time)
    
    for i in range(min(n_samples, len(data_a), len(data_b))):
        z_a, _ = extract_profiles(data_a[i], n_channels, n_time)
        z_b, _ = extract_profiles(data_b[i], n_channels, n_time)
        
        axes[i, 0].plot(time_axis, z_a, 'b-', linewidth=2, label=label_a)
        axes[i, 0].set_title(f'{label_a} Sample {i+1}', fontweight='bold')
        axes[i, 0].set_xlabel('Time bin')
        axes[i, 0].set_ylabel('Intensity (summed over channels)')
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].legend()
        
        axes[i, 1].plot(time_axis, z_b, 'r-', linewidth=2, label=label_b)
        axes[i, 1].set_title(f'{label_b} Sample {i+1}', fontweight='bold')
        axes[i, 1].set_xlabel('Time bin')
        axes[i, 1].set_ylabel('Intensity (summed over channels)')
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].legend()
        
        y_min = min(np.min(z_a), np.min(z_b))
        y_max = max(np.max(z_a), np.max(z_b))
        margin = (y_max - y_min) * 0.1
        axes[i, 0].set_ylim(y_min - margin, y_max + margin)
        axes[i, 1].set_ylim(y_min - margin, y_max + margin)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


def generate_report(
    metrics_a: Dict[str, np.ndarray],
    metrics_b: Dict[str, np.ndarray],
    save_path: str,
    title: str,
    label_a: str,
    label_b: str,
):
    """Generate text report of correlation analysis."""
    with open(save_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"{title}\n")
        f.write("=" * 80 + "\n\n")
        
        correlations = []
        
        for metric in metrics_a.keys():
            vals_a = metrics_a[metric]
            vals_b = metrics_b[metric]
            
            n = min(len(vals_a), len(vals_b))
            vals_a = vals_a[:n]
            vals_b = vals_b[:n]
            
            valid_mask = np.isfinite(vals_a) & np.isfinite(vals_b)
            vals_a_clean = vals_a[valid_mask]
            vals_b_clean = vals_b[valid_mask]
            
            if len(vals_a_clean) < 10:
                f.write(f"\n{metric}:\n")
                f.write(f"  ERROR: Insufficient valid data\n")
                continue
            
            corr = np.corrcoef(vals_a_clean, vals_b_clean)[0, 1]
            if np.isfinite(corr):
                correlations.append(corr)
            
            mean_a, std_a = np.mean(vals_a_clean), np.std(vals_a_clean)
            mean_b, std_b = np.mean(vals_b_clean), np.std(vals_b_clean)
            rmse = np.sqrt(np.mean((vals_a_clean - vals_b_clean)**2))
            ks_stat, ks_p = stats.ks_2samp(vals_a_clean, vals_b_clean)
            
            f.write(f"\n{metric}:\n")
            f.write(f"  Valid samples: {len(vals_a_clean)}\n")
            f.write(f"  Correlation: {corr:.4f}\n")
            f.write(f"  {label_a}: {mean_a:.4f} ± {std_a:.4f}\n")
            f.write(f"  {label_b}: {mean_b:.4f} ± {std_b:.4f}\n")
            f.write(f"  RMSE: {rmse:.4e}\n")
            f.write(f"  KS p-value: {ks_p:.4f}\n")
        
        if correlations:
            f.write(f"\n{'='*40}\n")
            f.write(f"SUMMARY:\n")
            f.write(f"  Average correlation: {np.mean(correlations):.4f}\n")
            f.write(f"  Min correlation: {np.min(correlations):.4f}\n")
            f.write(f"  Max correlation: {np.max(correlations):.4f}\n")


def find_nearest_neighbor(
    conditions: np.ndarray,
    exclude_self: bool = True,
) -> np.ndarray:
    """
    For each event, find the nearest neighbor by (xc, yc, dt, intensity).
    
    Parameters:
    -----------
    conditions : ndarray
        Shape (n, cond_dim) where columns are [xc, yc, dt, intensity, ...]
    exclude_self : bool
        If True, don't match an event to itself
        
    Returns:
    --------
    indices : array of nearest neighbor indices for each event
    """
    xc = conditions[:, 0]
    yc = conditions[:, 1]
    dt = conditions[:, 2]
    intensity = conditions[:, 3]
    
    xc_norm = (xc - xc.mean()) / (xc.std() + 1e-8)
    yc_norm = (yc - yc.mean()) / (yc.std() + 1e-8)
    dt_norm = (dt - dt.mean()) / (dt.std() + 1e-8)
    int_norm = (intensity - intensity.mean()) / (intensity.std() + 1e-8)
    
    features = np.stack([xc_norm, yc_norm, dt_norm, int_norm], axis=1)
    
    indices = []
    for i in range(len(features)):
        dists = np.sum((features - features[i]) ** 2, axis=1)
        if exclude_self:
            dists[i] = float('inf')
        best_j = np.argmin(dists)
        indices.append(best_j)
    
    return np.array(indices)


def run_real_vs_real_analysis(
    data_loader: TritiumSSDataLoader,
    n_samples: int,
    output_dir: str,
):
    """Compare pairs of real events matched by conditioning parameters (xc, yc, dt, intensity)."""
    print("\n" + "=" * 60)
    print("REAL vs REAL ANALYSIS (Matched by xc, yc, dt, intensity)")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading {n_samples} real samples...")
    
    all_data = []
    all_cond = []
    batch_size = 64
    
    for _ in tqdm(range((n_samples + batch_size - 1) // batch_size), desc="Loading data"):
        remaining = n_samples - len(all_data)
        if remaining <= 0:
            break
        current_batch = min(batch_size, remaining)
        batch_data, batch_cond = data_loader.get_batch(current_batch)
        for i in range(current_batch):
            all_data.append(batch_data[i])
            all_cond.append(batch_cond[i])
    
    all_data = np.array(all_data)
    all_cond = np.array(all_cond)
    
    print(f"Finding nearest neighbor for each sample by (xc, yc, dt, intensity)...")
    neighbor_indices = find_nearest_neighbor(all_cond, exclude_self=True)
    
    cond_a = all_cond
    cond_b = all_cond[neighbor_indices]
    
    param_diff = np.abs(cond_a[:, :4] - cond_b[:, :4])
    print(f"  Avg |xc_a - xc_b|: {np.mean(param_diff[:, 0]):.3f}")
    print(f"  Avg |yc_a - yc_b|: {np.mean(param_diff[:, 1]):.3f}")
    print(f"  Avg |dt_a - dt_b|: {np.mean(param_diff[:, 2]):.3f}")
    print(f"  Avg intensity ratio: {np.mean(cond_a[:, 3] / (cond_b[:, 3] + 1e-8)):.3f}")
    
    n_channels = data_loader.n_channels
    n_time = data_loader.n_time_points
    channel_positions = data_loader.channel_positions
    
    batch_a = standardize_batch(all_data)
    batch_b = standardize_batch(all_data[neighbor_indices])
    
    print("Computing extracted metrics...")
    z_metrics_a, xy_metrics_a = collect_metrics(batch_a, n_channels, n_time, channel_positions)
    z_metrics_b, xy_metrics_b = collect_metrics(batch_b, n_channels, n_time, channel_positions)
    
    cond_metrics_a = {
        'xc (true)': cond_a[:, 0],
        'yc (true)': cond_a[:, 1],
        'dt (true)': cond_a[:, 2],
        'intensity (true)': cond_a[:, 3],
    }
    cond_metrics_b = {
        'xc (true)': cond_b[:, 0],
        'yc (true)': cond_b[:, 1],
        'dt (true)': cond_b[:, 2],
        'intensity (true)': cond_b[:, 3],
    }
    
    print("Generating plots...")
    
    cond_corr = plot_correlation_matrix(
        cond_metrics_a, cond_metrics_b,
        os.path.join(output_dir, 'conditioning_params_correlation.png'),
        title="Real vs Real: Conditioning Parameters (xc, yc, dt, intensity)",
        label_a="Event A", label_b="Event B (nearest neighbor)"
    )
    
    z_corr = plot_correlation_matrix(
        z_metrics_a, z_metrics_b,
        os.path.join(output_dir, 'z_profile_correlation.png'),
        title="Real vs Real: Z-Profile (Time) Extracted Metrics",
        label_a="Event A", label_b="Event B (nearest neighbor)"
    )
    
    xy_corr = plot_correlation_matrix(
        xy_metrics_a, xy_metrics_b,
        os.path.join(output_dir, 'xy_profile_correlation.png'),
        title="Real vs Real: XY-Profile (Spatial) Extracted Metrics",
        label_a="Event A", label_b="Event B (nearest neighbor)"
    )
    
    plot_single_distribution(
        cond_metrics_a,
        os.path.join(output_dir, 'conditioning_params_distributions.png'),
        title="Real Data: Conditioning Parameters Distribution"
    )
    
    plot_single_distribution(
        z_metrics_a,
        os.path.join(output_dir, 'z_profile_distributions.png'),
        title="Real Data: Z-Profile Metrics Distribution"
    )
    
    plot_single_distribution(
        xy_metrics_a,
        os.path.join(output_dir, 'xy_profile_distributions.png'),
        title="Real Data: XY-Profile Metrics Distribution"
    )
    
    plot_sample_profiles(
        batch_a, batch_b, n_channels, n_time,
        os.path.join(output_dir, 'sample_z_profiles.png'),
        label_a="Event A", label_b="Event B (nearest)", n_samples=5
    )
    
    generate_report(
        cond_metrics_a, cond_metrics_b,
        os.path.join(output_dir, 'conditioning_params_report.txt'),
        "Real vs Real: Conditioning Parameters", "Event A", "Event B"
    )
    generate_report(
        z_metrics_a, z_metrics_b,
        os.path.join(output_dir, 'z_profile_report.txt'),
        "Real vs Real: Z-Profile Analysis", "Event A", "Event B"
    )
    generate_report(
        xy_metrics_a, xy_metrics_b,
        os.path.join(output_dir, 'xy_profile_report.txt'),
        "Real vs Real: XY-Profile Analysis", "Event A", "Event B"
    )
    
    print(f"\nResults saved to: {output_dir}")
    if cond_corr:
        print(f"  Conditioning params avg correlation: {np.mean(cond_corr):.4f}")
    if z_corr:
        print(f"  Z-profile avg correlation: {np.mean(z_corr):.4f}")
    if xy_corr:
        print(f"  XY-profile avg correlation: {np.mean(xy_corr):.4f}")


def run_real_vs_generated_analysis(
    cfg: Config,
    data_loader: TritiumSSDataLoader,
    n_samples: int,
    output_dir: str,
    device: torch.device,
):
    """Compare real data vs DDPM-generated data conditioned on the same parameters."""
    print("\n" + "=" * 60)
    print("REAL vs GENERATED ANALYSIS")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading trained model...")
    core, cond_proj, schedule = load_checkpoint(cfg, device)
    
    print("Loading graph structure...")
    graph = data_loader.load_adjacency_sparse(
        z_sep=cfg.graph.z_sep,
        radius=cfg.graph.radius,
        z_hops=cfg.graph.z_hops
    )
    
    print(f"Generating {n_samples} samples conditioned on real event parameters...")
    real_data, gen_data, conditions = generate_samples(
        core, cond_proj, schedule, data_loader, graph, cfg,
        n_samples, device, batch_size=8
    )
    
    n_channels = data_loader.n_channels
    n_time = data_loader.n_time_points
    channel_positions = data_loader.channel_positions
    
    print("Computing metrics...")
    z_metrics_real, xy_metrics_real = collect_metrics(real_data, n_channels, n_time, channel_positions)
    z_metrics_gen, xy_metrics_gen = collect_metrics(gen_data, n_channels, n_time, channel_positions)
    
    true_params = {
        'xc (true)': conditions[:, 0],
        'yc (true)': conditions[:, 1],
        'dt (true)': conditions[:, 2],
        'intensity (true)': conditions[:, 3],
    }
    
    gen_extracted = {
        'centroid_x (gen)': xy_metrics_gen.get('centroid_x', np.zeros(len(conditions))),
        'centroid_y (gen)': xy_metrics_gen.get('centroid_y', np.zeros(len(conditions))),
        'peak_time (gen)': z_metrics_gen.get('peak_time', np.zeros(len(conditions))),
        'total_intensity (gen)': xy_metrics_gen.get('total_intensity', np.zeros(len(conditions))),
    }
    
    real_extracted = {
        'centroid_x (real)': xy_metrics_real.get('centroid_x', np.zeros(len(conditions))),
        'centroid_y (real)': xy_metrics_real.get('centroid_y', np.zeros(len(conditions))),
        'peak_time (real)': z_metrics_real.get('peak_time', np.zeros(len(conditions))),
        'total_intensity (real)': xy_metrics_real.get('total_intensity', np.zeros(len(conditions))),
    }
    
    print("Generating plots...")
    
    true_vs_gen = {
        'xc': (true_params['xc (true)'], gen_extracted['centroid_x (gen)']),
        'yc': (true_params['yc (true)'], gen_extracted['centroid_y (gen)']),
        'dt': (true_params['dt (true)'], gen_extracted['peak_time (gen)']),
        'intensity': (true_params['intensity (true)'], gen_extracted['total_intensity (gen)']),
    }
    true_metrics = {k: v[0] for k, v in true_vs_gen.items()}
    gen_metrics = {k: v[1] for k, v in true_vs_gen.items()}
    
    param_corr = plot_correlation_matrix(
        true_metrics, gen_metrics,
        os.path.join(output_dir, 'true_vs_generated_params.png'),
        title="True Conditioning vs Generated Extracted Parameters",
        label_a="True (conditioning)", label_b="Generated (extracted)"
    )
    
    true_vs_real = {
        'xc': (true_params['xc (true)'], real_extracted['centroid_x (real)']),
        'yc': (true_params['yc (true)'], real_extracted['centroid_y (real)']),
        'dt': (true_params['dt (true)'], real_extracted['peak_time (real)']),
        'intensity': (true_params['intensity (true)'], real_extracted['total_intensity (real)']),
    }
    true_metrics_r = {k: v[0] for k, v in true_vs_real.items()}
    real_metrics_r = {k: v[1] for k, v in true_vs_real.items()}
    
    real_param_corr = plot_correlation_matrix(
        true_metrics_r, real_metrics_r,
        os.path.join(output_dir, 'true_vs_real_params.png'),
        title="True Conditioning vs Real Extracted Parameters (baseline)",
        label_a="True (conditioning)", label_b="Real (extracted)"
    )
    
    z_corr = plot_correlation_matrix(
        z_metrics_real, z_metrics_gen,
        os.path.join(output_dir, 'z_profile_correlation.png'),
        title="Real vs Generated: Z-Profile (Time) Metrics",
        label_a="Real", label_b="Generated"
    )
    
    plot_distribution_comparison(
        z_metrics_real, z_metrics_gen,
        os.path.join(output_dir, 'z_profile_distributions.png'),
        label_a="Real", label_b="Generated"
    )
    
    xy_corr = plot_correlation_matrix(
        xy_metrics_real, xy_metrics_gen,
        os.path.join(output_dir, 'xy_profile_correlation.png'),
        title="Real vs Generated: XY-Profile (Spatial) Metrics",
        label_a="Real", label_b="Generated"
    )
    
    plot_distribution_comparison(
        xy_metrics_real, xy_metrics_gen,
        os.path.join(output_dir, 'xy_profile_distributions.png'),
        label_a="Real", label_b="Generated"
    )
    
    plot_sample_profiles(
        real_data, gen_data, n_channels, n_time,
        os.path.join(output_dir, 'sample_z_profiles.png'),
        label_a="Real", label_b="Generated", n_samples=5
    )
    
    generate_report(
        true_metrics, gen_metrics,
        os.path.join(output_dir, 'true_vs_generated_params_report.txt'),
        "True Conditioning vs Generated Extracted", "True", "Generated"
    )
    generate_report(
        z_metrics_real, z_metrics_gen,
        os.path.join(output_dir, 'z_profile_report.txt'),
        "Real vs Generated: Z-Profile Analysis", "Real", "Generated"
    )
    generate_report(
        xy_metrics_real, xy_metrics_gen,
        os.path.join(output_dir, 'xy_profile_report.txt'),
        "Real vs Generated: XY-Profile Analysis", "Real", "Generated"
    )
    
    print(f"\nResults saved to: {output_dir}")
    if param_corr:
        print(f"  True vs Generated params avg correlation: {np.mean(param_corr):.4f}")
    if real_param_corr:
        print(f"  True vs Real params avg correlation: {np.mean(real_param_corr):.4f} (baseline)")
    if z_corr:
        print(f"  Z-profile avg correlation: {np.mean(z_corr):.4f}")
    if xy_corr:
        print(f"  XY-profile avg correlation: {np.mean(xy_corr):.4f}")


def main():
    parser = argparse.ArgumentParser(description="DDPM Correlation Analysis")
    parser.add_argument('--n_samples', type=int, default=4096, help='Number of samples for analysis')
    parser.add_argument('--output_dir', type=str, default='correlation_analysis', help='Output directory')
    parser.add_argument('--mode', type=str, choices=['all', 'real_vs_real', 'real_vs_generated'], 
                        default='all', help='Analysis mode')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    args = parser.parse_args()
    
    cfg = default_config
    
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading data...")
    data_loader = TritiumSSDataLoader(cfg.paths.tritium_h5, cfg.paths.channel_positions)
    print(f"  Channels: {data_loader.n_channels}")
    print(f"  Time points: {data_loader.n_time_points}")
    print(f"  Total samples: {data_loader.n_samples}")
    
    if args.mode in ['all', 'real_vs_real']:
        run_real_vs_real_analysis(
            data_loader,
            args.n_samples,
            os.path.join(args.output_dir, 'real_vs_real'),
        )
    
    if args.mode in ['all', 'real_vs_generated']:
        run_real_vs_generated_analysis(
            cfg,
            data_loader,
            args.n_samples,
            os.path.join(args.output_dir, 'real_vs_generated'),
            device,
        )
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
