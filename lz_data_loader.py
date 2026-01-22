import h5py
import numpy as np
from typing import Tuple, List
from tqdm import tqdm
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree
import torch

from data import SparseGraph, Graph

def create_adjacency_matrix(channel_positions: np.ndarray, 
                          connection_type: str = 'knn',
                          k: int = 6,
                          radius: float = None,
                          distance_metric: str = 'euclidean') -> np.ndarray:
    n_channels = len(channel_positions)
    
    if distance_metric == 'euclidean':
        distances = squareform(pdist(channel_positions, metric='euclidean'))
    elif distance_metric == 'manhattan':
        distances = squareform(pdist(channel_positions, metric='manhattan'))
    else:
        raise ValueError(f"Unsupported distance metric: {distance_metric}")
    
    adjacency = np.zeros((n_channels, n_channels))
    
    if connection_type == 'knn':
        for i in range(n_channels):
            neighbor_indices = np.argsort(distances[i])[1:k+1]  # Skip self (index 0)
            for j in neighbor_indices:
                adjacency[i, j] = 1.0
                adjacency[j, i] = 1.0
        
        np.fill_diagonal(adjacency, 1)
                
    elif connection_type == 'radius':
        if radius is None:
            # Auto-compute radius as 1.5x median nearest neighbor distance
            nearest_distances = np.array([np.sort(distances[i])[1] for i in range(n_channels)])
            radius = 1.5 * np.median(nearest_distances)
            print(f"  Auto-computed radius: {radius:.3f}")
        
        adjacency = (distances <= radius).astype(float)
        np.fill_diagonal(adjacency, 1)
        
    elif connection_type == 'full':
        # Fully connected with inverse distance weighting
        adjacency = 1.0 / (distances + 1e-8)
        np.fill_diagonal(adjacency, 1)
        
    else:
        raise ValueError(f"Unsupported connection type: {connection_type}")
    
    return adjacency


def create_3d_adjacency_matrix(
    channel_positions: np.ndarray,
    num_layers: int,
    *,
    radius: float = 1.0,
    z_hops: int = 2,
    distance_metric: str = "euclidean",
    k: int = 5,
    include_self: bool = True,
) -> np.ndarray:
    n_channels = int(channel_positions.shape[0])
    A = create_adjacency_matrix(
        channel_positions,
        connection_type="knn",
        k=k,
        radius=radius,
        distance_metric=distance_metric,
    ).astype(np.uint8)

    A = ((A + A.T) > 0).astype(np.uint8)
    if include_self:
        np.fill_diagonal(A, 1)

    N = num_layers * n_channels
    adj = np.zeros((N, N), dtype=np.uint8)

    for i in range(num_layers):
        r = slice(i * n_channels, (i + 1) * n_channels)
        adj[r, r] = A

    for hop in range(1, z_hops + 1):
        for i in range(num_layers - hop):
            r1 = slice(i * n_channels, (i + 1) * n_channels)
            r2 = slice((i + hop) * n_channels, (i + hop + 1) * n_channels)
            adj[r1, r2] = A
            adj[r2, r1] = A

    return adj


def create_3d_adjacency_matrix_sparse(channel_positions: np.ndarray,
                                      num_layers: int,
                                      z_sep : float = 8.0,
                                      radius : float = 25.0) -> torch.Tensor:

    nodes_xy = channel_positions

    # Define positions for nodes_xyz
    nodes_xyz = np.zeros((nodes_xy.shape[0] * num_layers, 3))
    nodes_xyz[:, :2] = np.tile(nodes_xy, (num_layers, 1))
    nodes_xyz[:, 2] = np.repeat(np.arange(num_layers) * z_sep, nodes_xy.shape[0])

    # Define COO sparse adjacency based on radius (layer-major ordering)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    N = int(nodes_xyz.shape[0])
    max_layer_hops = int(np.ceil(radius / max(z_sep, 1e-8))) * num_layers
    for i in tqdm(range(N)):
        j_min = max(0, int(i - max_layer_hops))
        j_max = min(N - 1, int(i + max_layer_hops))
        xi = nodes_xyz[i]
        for j in range(j_min, j_max + 1):
            if np.linalg.norm(xi - nodes_xyz[j]) <= radius:
                rows.append(i)
                cols.append(j)
                vals.append(1.0)
    if len(rows) == 0:
        A = torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
            (N, N)
        ).coalesce()
    else:
        idx = np.vstack([np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)])
        val = np.asarray(vals, dtype=np.float32)
        A = torch.sparse_coo_tensor(torch.from_numpy(idx), torch.from_numpy(val), (N, N)).coalesce()

    return A


def create_3d_adjacency_matrix_sparse(channel_positions: np.ndarray,
                                      num_layers: int,
                                      z_sep: float = 8.0,
                                      radius: float = 25.0,
                                      weighted: bool = False) -> torch.Tensor:
    # Build 3D coordinates
    M = channel_positions.shape[0]
    L = num_layers
    N = M * L

    nodes_xyz = np.empty((N, 3), dtype=np.float32)
    nodes_xyz[:, :2] = np.tile(channel_positions.astype(np.float32), (L, 1))
    nodes_xyz[:, 2]  = np.repeat(np.arange(L, dtype=np.float32) * z_sep, M)

    # KD-tree radius graph
    tree = cKDTree(nodes_xyz)
    A = tree.sparse_distance_matrix(tree, max_distance=radius, output_type='coo_matrix')

    # If you want the graph to be undirected/symmetric, uncomment these two lines:
    A = (A + A.T)
    A.sum_duplicates()

    # Make sure we're in COO *now* (after any ops above)
    A = A.tocoo()

    # Values: either 1.0 (unweighted) or distances
    if weighted:
        vals = A.data.astype(np.float32, copy=False)
    else:
        vals = np.ones_like(A.data, dtype=np.float32)

    idx = np.vstack([A.row.astype(np.int64, copy=False),
                     A.col.astype(np.int64, copy=False)])

    A_t = torch.sparse_coo_tensor(torch.from_numpy(idx),
                                  torch.from_numpy(vals),
                                  (N, N)).coalesce()
    return A_t

def _layer_radial_adjacency(xy: np.ndarray, r_within: float) -> np.ndarray:
    L = xy.shape[0]
    diff = xy[:, None, :] - xy[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    A = (dist2 <= (r_within ** 2)).astype(np.float32)
    np.fill_diagonal(A, 0.0)
    A = np.maximum(A, A.T)
    return A

def create_3d_adjacency_matrix_sparse_(
    channel_positions: np.ndarray,
    num_layers: int,
    r_within: float,
    positions_xy_profile: np.ndarray,
    z_hops: int = 2,
    self_loops : bool = True,
    z_spacing: float = 10.0
) -> SparseGraph:
    num_layer_nodes = len(channel_positions)
    L, T, N = num_layer_nodes, num_layers, num_layer_nodes * num_layers
    xy_profile = np.asarray(positions_xy_profile, dtype=np.float32)
    z_vals = (np.arange(T, dtype=np.float32) * float(z_spacing))
    positions_z  = np.repeat(z_vals, L).astype(np.float32)

    A_layer = _layer_radial_adjacency(xy_profile, r_within)
    cross = (A_layer + np.eye(L, dtype=np.float32))
    cross[cross > 0] = 1.0

    # Build sparse indices without materializing (N,N)
    layer_r, layer_c = np.nonzero(A_layer)
    layer_r = layer_r.astype(np.int64, copy=False)
    layer_c = layer_c.astype(np.int64, copy=False)
    # cross = A_layer OR I
    cross_r = np.concatenate([layer_r, np.arange(L, dtype=np.int64)])
    cross_c = np.concatenate([layer_c, np.arange(L, dtype=np.int64)])

    rows_list: List[np.ndarray] = []
    cols_list: List[np.ndarray] = []

    # diagonal blocks (per-layer A_layer)
    for t in range(T):
        offset = t * L
        rows_list.append(layer_r + offset)
        cols_list.append(layer_c + offset)

    # off-diagonal blocks within z_hops using cross pattern, add symmetric counterpart via transpose
    for hop in range(1, z_hops + 1):
        for t in range(0, T - hop):
            i0 = t * L
            j0 = (t + hop) * L
            # block (t, t+hop): use cross
            rows_list.append(cross_r + i0)
            cols_list.append(cross_c + j0)
            # symmetric block (t+hop, t): use cross.T -> swap r/c
            rows_list.append(cross_c + j0)
            cols_list.append(cross_r + i0)

    # optional self-loops on all nodes
    if self_loops:
        diag = np.arange(N, dtype=np.int64)
        rows_list.append(diag)
        cols_list.append(diag)

    if len(rows_list) == 0:
        idx_t = torch.empty((2, 0), dtype=torch.long)
        val_t = torch.empty((0,), dtype=torch.float32)
    else:
        rows_all = np.concatenate(rows_list)
        cols_all = np.concatenate(cols_list)
        idx = np.vstack([rows_all, cols_all])
        val = np.ones(idx.shape[1], dtype=np.float32)
        idx_t = torch.from_numpy(idx)
        val_t = torch.from_numpy(val)
    A_sparse = torch.sparse_coo_tensor(idx_t, val_t, (N, N)).coalesce()

    # Positions in (N,3): tile xy per layer, pair with z
    xy_tiled = np.tile(xy_profile.astype(np.float32, copy=False), (T, 1))
    pos_xyz = np.concatenate([xy_tiled, positions_z.reshape(-1, 1)], axis=1).astype(np.float32, copy=False)
    pos_xyz_t = torch.from_numpy(pos_xyz)

    return SparseGraph(adjacency=A_sparse, positions_xyz=pos_xyz_t)


class TritiumDataLoader:
    
    def __init__(self, h5_file_path: str, channel_positions_path: str):
        self.h5_file_path = h5_file_path
        self.channel_positions_path = channel_positions_path
        
        with h5py.File(h5_file_path, 'r') as f:
            self.n_samples = f['waveforms'].shape[0]
            self.n_channels = f['waveforms'].shape[1] 
            self.n_time_points = f['waveforms'].shape[2]
        self.channel_positions = self._load_channel_positions()
        
    def _load_channel_positions(self) -> np.ndarray:
        if self.channel_positions_path.endswith('.h5') or self.channel_positions_path.endswith('.hdf5'):
            with h5py.File(self.channel_positions_path, 'r') as f:
                if 'TA_PMTs_xy' in f:
                    positions = f['TA_PMTs_xy'][:] / 10.0
                else:
                    available_keys = list(f.keys())
                    if 'positions' in f:
                        positions = f['positions'][:]
                    elif 'xy' in f:
                        positions = f['xy'][:]
                    else:
                        raise ValueError(f"Could not find channel positions in HDF5 file. Available keys: {available_keys}")
        elif self.channel_positions_path.endswith('.npy'):
            positions = np.load(self.channel_positions_path)
        elif self.channel_positions_path.endswith('.npz'):
            data = np.load(self.channel_positions_path)
            if 'positions' in data:
                positions = data['positions']
            else:
                positions = data[list(data.keys())[0]]
        elif self.channel_positions_path.endswith(('.txt', '.csv')):
            positions = np.loadtxt(self.channel_positions_path, delimiter=',')
        else:
            raise ValueError(f"Unsupported file format for channel positions: {self.channel_positions_path}")
        
        if positions.shape != (self.n_channels, 2):
            raise ValueError(f"Channel positions shape {positions.shape} does not match expected ({self.n_channels}, 2)")
        
        
        return positions
    
    def load_batch(self, batch_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        unique_indices, inverse_indices = np.unique(batch_indices, return_inverse=True)
        
        with h5py.File(self.h5_file_path, 'r') as f:
            waveforms_unique = f['waveforms'][unique_indices]
            xc_unique = f['xc'][unique_indices]
            yc_unique = f['yc'][unique_indices]
            dt_unique = f['dt'][unique_indices]
        
        waveforms = waveforms_unique[inverse_indices]
        xc = xc_unique[inverse_indices]
        yc = yc_unique[inverse_indices]
        dt = dt_unique[inverse_indices]
        
        true_positions = np.column_stack([xc, yc, dt])
        
        return waveforms, true_positions, xc, yc, dt
    
    def create_batches(self, batch_size: int, shuffle: bool = True) -> List[np.ndarray]:
        indices = np.arange(self.n_samples)
        if shuffle:
            np.random.shuffle(indices)
        
        batches = []
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            batches.append(batch_indices)
            
        return batches
    
    def get_sample_data(self, n_samples: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = np.random.choice(self.n_samples, size=min(n_samples, self.n_samples), replace=False)
        return self.load_batch(indices)
    
    def analyze_data_statistics(self) -> dict:
        
        sample_size = min(1000, self.n_samples)
        sample_indices = np.random.choice(self.n_samples, size=sample_size, replace=False)
        
        waveforms, true_positions, xc, yc, dt = self.load_batch(sample_indices)
        
        intensities = np.sum(waveforms, axis=(1, 2))
        
        stats = {
            'n_samples': self.n_samples,
            'n_channels': self.n_channels, 
            'n_time_points': self.n_time_points,
            'waveform_shape': waveforms.shape,
            'intensity_range': (np.min(intensities), np.max(intensities)),
            'intensity_mean': np.mean(intensities),
            'intensity_std': np.std(intensities),
            'true_x_range': (np.min(xc), np.max(xc)),
            'true_y_range': (np.min(yc), np.max(yc)),
            'true_t_range': (np.min(dt), np.max(dt)),
            'true_x_mean': np.mean(xc),
            'true_y_mean': np.mean(yc),
            'true_t_mean': np.mean(dt),
            'true_x_std': np.std(xc),
            'true_y_std': np.std(yc),
            'true_t_std': np.std(dt),
        }
        
        
        return stats

    def positions_xyz_layer_major(self, *, device: Optional[str] = None, normalize_z: bool = True) -> torch.Tensor:
        """Return positions as (T*C, 3) in layer-major order to match 3D adjacency.
        Order: all channels at z=0, then all channels at z=1, etc.
        If normalize_z, z is mapped to [-1, 1]."""
        dev = torch.device(device) if device is not None else None
        xy = torch.from_numpy(self.channel_positions.astype(np.float32))
        xy = xy.repeat(self.n_time_points, 1)
        z = torch.arange(self.n_time_points, dtype=torch.float32)
        z = z.repeat_interleave(self.n_channels).unsqueeze(1)
        if normalize_z:
            z = (z / max(self.n_time_points - 1, 1)) * 2.0 - 1.0
        pos = torch.cat([xy, z], dim=1).float()
        if dev is not None:
            pos = pos.to(dev)
        return pos

    def adjacency_sparse_layer_major(self, *, z_sep: float = 8.0, radius: float = 25.0) -> torch.Tensor:
        """Build sparse COO adjacency (T*C, T*C) matching layer-major node order."""
        A = create_3d_adjacency_matrix_sparse(self.channel_positions, num_layers=self.n_time_points, z_sep=z_sep, radius=radius)
        return A.coalesce()


def create_tritium_batch_generator(h5_file_path: str, channel_positions_path: str,
                                 batch_size: int = 16) -> callable:
    loader = TritiumDataLoader(h5_file_path, channel_positions_path)
    
    def batch_generator():
        while True:
            batch_indices = np.random.choice(loader.n_samples, size=batch_size, replace=True)
            waveforms, centers, xc, yc = loader.load_batch(batch_indices)
            
            intensities = np.sum(waveforms, axis=(1, 2))
            params = np.column_stack([centers, intensities])
            
            params_expanded = np.zeros((batch_size, 1, 4))
            params_expanded[:, 0, :] = params
            
            yield waveforms, params_expanded
    
    return batch_generator


def visualize_tritium_data(h5_file_path: str, channel_positions_path: str,
                          save_dir: str = 'tritium_analysis', 
                          n_samples: int = 6) -> None:
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    loader = TritiumDataLoader(h5_file_path, channel_positions_path)
    waveforms, true_positions, xc, yc, dt = loader.get_sample_data(n_samples)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(min(n_samples, len(axes))):
        ax = axes[i]
        
        im = ax.imshow(waveforms[i], aspect='auto', origin='lower', cmap='viridis')
        
        ax.axvline(dt[i], color='red', linestyle='--', linewidth=2, 
                  label=f'True time: t={dt[i]:.1f}')
        
        ax.set_title(f'Sample {i+1}\nTrue: ({xc[i]:.3f}, {yc[i]:.3f}, {dt[i]:.1f})')
        ax.set_xlabel('Time')
        ax.set_ylabel('Channel')
        ax.legend()
        plt.colorbar(im, ax=ax)
    
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/tritium_data_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].hist(xc, bins=30, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('True X Position')
    axes[0].set_ylabel('Count')
    axes[0].set_title('X Position Distribution')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(yc, bins=30, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('True Y Position') 
    axes[1].set_ylabel('Count')
    axes[1].set_title('Y Position Distribution')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].hist(dt, bins=30, alpha=0.7, edgecolor='black')
    axes[2].set_xlabel('True Time')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Time Distribution')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/position_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    scatter = ax.scatter(xc, yc, c=dt, s=50, alpha=0.7, cmap='viridis')
    ax.set_xlabel('True X Position')
    ax.set_ylabel('True Y Position')
    ax.set_title('2D Position Distribution (colored by time)')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='True Time')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/position_2d_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    stats = loader.analyze_data_statistics()
    
    
    return stats


if __name__ == "__main__":
    import os
    
    h5_file = "tritium_ss.h5"
    
    if not os.path.exists(h5_file):
        exit(1)
    channel_positions_file = "test_channel_positions.h5"
    
    if not os.path.exists(channel_positions_file):
        n_channels = 253
        grid_size = int(np.ceil(np.sqrt(n_channels)))
        positions = []
        for i in range(n_channels):
            x = i % grid_size
            y = i // grid_size
            positions.append([x, y])
        positions = np.array(positions, dtype=np.float32)
        
        with h5py.File(channel_positions_file, 'w') as f:
            f.create_dataset('TA_PMTs_xy', data=positions)
    loader = TritiumDataLoader(h5_file, channel_positions_file)
    
    batch_gen = create_tritium_batch_generator(h5_file, channel_positions_file, batch_size=4)
    waveforms, params = next(batch_gen())
    
    stats = visualize_tritium_data(h5_file, channel_positions_file, n_samples=6)


def create_conditional_data(centers: np.ndarray, intensities: np.ndarray, max_events: int = 4) -> np.ndarray:
    N = len(centers)
    conditions = np.zeros((N, max_events, 4))
    
    conditions[:, 0, :3] = centers  # x, y, t
    conditions[:, 0, 3] = intensities  # intensity
    
    return conditions

def generate_tritium_batch(batch_size: int, tritium_loader: TritiumDataLoader) -> Tuple[np.ndarray, np.ndarray]:
    batch_indices = np.random.choice(tritium_loader.n_samples, size=batch_size, replace=True)
    waveforms, true_positions, xc, yc, dt = tritium_loader.load_batch(batch_indices)
    
    intensities = np.sum(waveforms, axis=(1, 2))

    centers = true_positions
    conditions = create_conditional_data(centers, intensities, max_events=4)
    conditions = conditions.reshape(batch_size, -1) # Flatten to (batch_size, 16)

    # deliver waveforms flattened in layer-major order
    waveforms = waveforms.reshape(batch_size, -1, 1, order='F')

    return waveforms, conditions


class TritiumSSDataLoader:
    def __init__(self, h5_file_path: str, channel_positions_path: str):
        self.h5_file_path = h5_file_path
        self.channel_positions_path = channel_positions_path
        with h5py.File(h5_file_path, 'r') as f:
            self.n_samples = f['waveforms'].shape[0]
            self.n_channels = f['waveforms'].shape[1]
            self.n_time_points = f['waveforms'].shape[2]
        self.channel_positions = self._load_channel_positions()

    def _load_channel_positions(self) -> np.ndarray:
        path = self.channel_positions_path
        if path.endswith(('.h5', '.hdf5')):
            with h5py.File(path, 'r') as f:
                if 'TA_PMTs_xy' in f:
                    positions = f['TA_PMTs_xy'][:] / 10.0
                elif 'positions' in f:
                    positions = f['positions'][:]
                elif 'xy' in f:
                    positions = f['xy'][:]
                else:
                    raise ValueError('positions dataset not found in HDF5')
        elif path.endswith('.npy'):
            positions = np.load(path)
        elif path.endswith('.npz'):
            data = np.load(path)
            positions = data['positions'] if 'positions' in data else data[list(data.keys())[0]]
        elif path.endswith(('.txt', '.csv')):
            positions = np.loadtxt(path, delimiter=',')
        else:
            raise ValueError(f'Unsupported file format: {path}')
        if positions.shape != (self.n_channels, 2):
            raise ValueError(f'Channel positions shape {positions.shape} does not match ({self.n_channels}, 2)')
        return positions.astype(np.float32)

    def load_positions(self, *, layer_major: bool = False, normalize_z: bool = True) -> np.ndarray:
        if not layer_major:
            return self.channel_positions.copy()
        C, T = self.n_channels, self.n_time_points
        xy = np.repeat(self.channel_positions.astype(np.float32), T, axis=0)
        z = np.repeat(np.arange(T, dtype=np.float32), C)[:, None]
        if normalize_z:
            z = (z / max(T - 1, 1)) * 2.0 - 1.0
        return np.concatenate([xy, z], axis=1)

    def load_adjacency_sparse(self, z_sep: float = 10.0, radius: float = 20.0, weighted: bool = False, z_hops: int = 4) -> torch.Tensor:
        return create_3d_adjacency_matrix_sparse_(
            self.channel_positions,
            num_layers=self.n_time_points,
            r_within=radius,
            positions_xy_profile=self.channel_positions,
            z_hops=z_hops,
            self_loops=True,
            z_spacing=z_sep,
        )
        # return create_3d_adjacency_matrix_sparse(self.channel_positions, num_layers=self.n_time_points, z_sep=z_sep, radius=radius, weighted=weighted).coalesce()

    def get_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        idx = np.random.randint(0, self.n_samples, size=batch_size, dtype=np.int64)
        # h5py requires strictly increasing indices; load unique sorted, then reconstruct
        uniq_idx, inverse = np.unique(idx, return_inverse=True)
        with h5py.File(self.h5_file_path, 'r') as f:
            wf_u = f['waveforms'][uniq_idx]  # (U,C,T)
            xc_u = f['xc'][uniq_idx].astype(np.float32)
            yc_u = f['yc'][uniq_idx].astype(np.float32)
            dt_u = f['dt'][uniq_idx].astype(np.float32)
        # Reconstruct batch with duplicates/original order
        wf = wf_u[inverse]
        xc = xc_u[inverse]
        yc = yc_u[inverse]
        dt = dt_u[inverse]
        intensities = np.sum(wf, axis=(1, 2)).astype(np.float32)
        # Column-order (layer-major) flatten: (C,T) -> (T*C,1)
        wf_col = np.transpose(wf, (0, 2, 1)).reshape(batch_size, -1, 1).astype(np.float32)
        cond = np.stack([xc, yc, dt, intensities, np.zeros_like(intensities)], axis=1)
        return wf_col, cond

    def batcher(self, batch_size: int):
        while True:
            yield self.get_batch(batch_size)