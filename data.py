import math
from dataclasses import dataclass
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


@dataclass
class Graph:
    adjacency: np.ndarray  # (N, N)
    positions_xy: np.ndarray  # (N, 2)
    positions_z: np.ndarray  # (K,)

@dataclass
class SparseGraph:
    adjacency: torch.Tensor  # (N, N)
    positions_xyz: torch.Tensor  # (N, 3)


def graph_to_sparse_graph(graph: Graph) -> SparseGraph:
    positions_xy = graph.positions_xy # (N, 2)
    positions_z = graph.positions_z # (K,)
    
    # Create positions_xyz of size N * K. Each z is an xy graph
    positions_xy = np.asarray(positions_xy, dtype=np.float32)
    positions_z = np.expand_dims(np.asarray(positions_z, dtype=np.float32), axis=1)
    n = int(positions_xy.shape[0])
    k = int(positions_z.shape[0]) // n
    xy_tiled = np.tile(positions_xy, (k, 1))  # (K*N, 2)
    positions_xyz = np.concatenate([xy_tiled, positions_z], axis=1).astype(np.float32)  # (K*N, 3)
    A = np.asarray(graph.adjacency, dtype=np.float32)

    # convert A to sparse COO tensor
    idx = np.stack([np.nonzero(A)], axis=0)[0]
    val = A[idx[0], idx[1]].astype(np.float32)
    adj_sparse = torch.sparse_coo_tensor(idx, val, size=(n * k, n * k)).coalesce()
    return SparseGraph(adjacency=adj_sparse, positions_xyz=torch.from_numpy(positions_xyz))


@dataclass
class EventParams:
    center_xy: np.ndarray  # (2,)
    height_z: float  # interpreted as center along z
    sigma_xy: float
    sigma_z: float


def create_graph(num_nodes: int, radius: float = 1.0, k_neighbors: int = 1, seed: Optional[int] = None) -> Graph:
    """Create a filled circular triangular lattice and connect nearby neighbors.

    Nodes lie on a hexagonal (triangular) grid clipped to a disk of given radius.
    Adjacency connects nodes whose pairwise distance <= a * (k_neighbors + 0.1),
    where a is the lattice spacing inferred from num_nodes.
    """
    if num_nodes < 3:
        raise ValueError("num_nodes must be >= 3")
    if k_neighbors < 1:
        raise ValueError("k_neighbors must be >= 1")

    # Infer lattice spacing a from desired count within disk area.
    # Triangular lattice density = 2 / (sqrt(3) * a^2).
    a = radius * math.sqrt((2.0 * math.pi) / (math.sqrt(3.0) * float(num_nodes)))
    a = max(a, 1e-6)
    dy = a * math.sqrt(3.0) / 2.0

    ys = np.arange(-radius - dy, radius + dy, dy)
    points: List[Tuple[float, float]] = []
    for row_index, y in enumerate(ys):
        shift = (a / 2.0) if (row_index % 2 == 1) else 0.0
        xs = np.arange(-radius - a, radius + a, a)
        xs = xs + shift
        for x in xs:
            if x * x + y * y <= radius * radius + (a * 0.5) ** 2 and len(points) < num_nodes:
                points.append((float(x), float(y)))

    positions_xy = np.array(points, dtype=np.float32)
    if positions_xy.shape[0] < 3:
        raise RuntimeError("Lattice generation produced too few nodes; adjust parameters.")

    # Build adjacency by distance threshold.
    k = int(k_neighbors)
    thresh = a * (k + 0.1)
    diff = positions_xy[:, None, :] - positions_xy[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    adjacency = (dist <= thresh).astype(np.float32)
    adjacency = np.maximum(adjacency, adjacency.T)

    return Graph(adjacency=adjacency, positions_xy=positions_xy, positions_z=np.zeros(positions_xy.shape[0]))


def create_full_graph(num_layer_nodes: int, num_layers: int, radius: float = 1.0, k_neighbors: int = 3) -> Graph:
    """Create a full graph with num_layer_nodes nodes per layer and num_layers layers."""

    layer_graphs = [
        create_graph(num_layer_nodes, radius=radius, k_neighbors=k_neighbors)
        for i in range(num_layers)
    ]

    adjacency = np.zeros((num_layer_nodes * num_layers, num_layer_nodes * num_layers))
    for i in range(num_layers):
        for j in range(num_layers):
            adjacency[i * num_layer_nodes:(i + 1) * num_layer_nodes, j * num_layer_nodes:(j + 1) * num_layer_nodes] = layer_graphs[i].adjacency
    

    # Connect each layer to neighboring layers
    for i in range(num_layers - 1):
        adjacency[i * num_layer_nodes:(i + 1) * num_layer_nodes, (i + 1) * num_layer_nodes:(i + 2) * num_layer_nodes] = 1
        adjacency[(i + 1) * num_layer_nodes:(i + 2) * num_layer_nodes, i * num_layer_nodes:(i + 1) * num_layer_nodes] = 1

    np.fill_diagonal(adjacency, 1)
    positions_xy = layer_graphs[0].positions_xy
    positions_z = np.concatenate([np.ones(layer_graphs[0].positions_xy.shape[0]) * i for i in range(num_layers)], axis=0)

    return Graph(adjacency=adjacency, positions_xy=positions_xy, positions_z=positions_z)


def _triangular_lattice_xy(num_nodes: int, radius: float) -> np.ndarray:
    a = radius * math.sqrt((2.0 * math.pi) / (math.sqrt(3.0) * float(num_nodes)))
    a = max(a, 1e-6)
    dy = a * math.sqrt(3.0) / 2.0
    ys = np.arange(-radius - dy, radius + dy, dy, dtype=np.float32)
    pts: List[Tuple[float, float]] = []
    for r, y in enumerate(ys):
        shift = (a / 2.0) if (r % 2 == 1) else 0.0
        xs = np.arange(-radius - a, radius + a, a, dtype=np.float32) + shift
        for x in xs:
            if x * x + y * y <= radius * radius + (a * 0.5) ** 2 and len(pts) < num_nodes:
                pts.append((float(x), float(y)))
    xy = np.array(pts, dtype=np.float32)
    if xy.shape[0] < num_nodes:
        need = num_nodes - xy.shape[0]
        pad = (np.random.rand(need, 2).astype(np.float32) - 0.5) * (0.05 * a)
        xy = np.vstack([xy, pad])
    return xy[:num_nodes]


def _layer_radial_adjacency(xy: np.ndarray, r_within: float) -> np.ndarray:
    L = xy.shape[0]
    diff = xy[:, None, :] - xy[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    A = (dist2 <= (r_within ** 2)).astype(np.float32)
    np.fill_diagonal(A, 0.0)
    A = np.maximum(A, A.T)
    return A

def create_3d_radial_graph(
    num_layer_nodes: int,
    num_layers: int,
    r_within: float,
    *,
    positions_xy_profile: Optional[np.ndarray] = None,
    z_profile: Optional[np.ndarray] = None,
    radius: float = 1.0,
    z_spacing: float = 1.0,
    z_hops: int = 2,
    self_loops: bool = True
) -> Graph:
    L, T, N = num_layer_nodes, num_layers, num_layer_nodes * num_layers
    xy_profile = _triangular_lattice_xy(L, radius) if positions_xy_profile is None else np.asarray(positions_xy_profile, dtype=np.float32)
    z_vals = (np.arange(T, dtype=np.float32) * float(z_spacing)) if z_profile is None else np.asarray(z_profile, dtype=np.float32)
    positions_z  = np.repeat(z_vals, L).astype(np.float32)

    A_layer = _layer_radial_adjacency(xy_profile, r_within)
    cross = (A_layer + np.eye(L, dtype=np.float32))
    cross[cross > 0] = 1.0

    A = np.zeros((N, N), dtype=np.float32)

    # diagonal blocks
    for layer in range(T):
        i0, i1 = layer * L, (layer + 1) * L
        A[i0:i1, i0:i1] = A_layer

    # off-diagonal: propagate identity + radial pattern across ±1..±z_hops
    for hop in range(1, z_hops + 1):
        for layer in range(T):
            up = layer + hop
            down = layer - hop
            i0, i1 = layer * L, (layer + 1) * L
            if up < T:
                j0, j1 = up * L, (up + 1) * L
                A[i0:i1, j0:j1] = np.maximum(A[i0:i1, j0:j1], cross)
                A[j0:j1, i0:i1] = np.maximum(A[j0:j1, i0:i1], cross.T)
            if down >= 0:
                j0, j1 = down * L, (down + 1) * L
                A[i0:i1, j0:j1] = np.maximum(A[i0:i1, j0:j1], cross)
                A[j0:j1, i0:i1] = np.maximum(A[j0:j1, i0:i1], cross.T)

    if self_loops:
        np.fill_diagonal(A, 1.0)

    return Graph(adjacency=A, positions_xy=xy_profile, positions_z=positions_z)

def simulate_event(
    graph: Graph,
    z_limits: Tuple[float, float] = (0.0, 1.0),
    sigma_xy: float = 0.3,
    sigma_z: float = 0.2,
    num_z: int = 16,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, EventParams]:
    """Simulate a separable 3D Gaussian evaluated along node-local z vectors.

    Returns intensities of shape (N, M) where M=num_z. intensities[i, m] =
    exp(-0.5 * ||(x_i,y_i) - center_xy||^2 / sigma_xy^2) * exp(-0.5 * (z_m - height_z)^2 / sigma_z^2).
    """
    rng = np.random.default_rng(seed)
    xy = graph.positions_xy
    # Bound event center within convex hull (disk): sample uniformly in disk
    r = math.sqrt(rng.random()) * np.max(np.linalg.norm(xy, axis=1))
    theta = 2.0 * math.pi * rng.random()
    center_xy = np.array([r * math.cos(theta), r * math.sin(theta)], dtype=np.float32)

    height_z = float(rng.uniform(z_limits[0], z_limits[1]))
    params = EventParams(center_xy=center_xy, height_z=height_z, sigma_xy=float(sigma_xy), sigma_z=float(sigma_z))
    # z samples shared across nodes
    if num_z < 1:
        raise ValueError("num_z must be >= 1")
    z_samples = np.linspace(z_limits[0], z_limits[1], int(num_z), dtype=np.float32)

    # Separable Gaussian
    g_xy = np.exp(-0.5 * (np.sum((xy - params.center_xy[None, :]) ** 2, axis=1) / (params.sigma_xy ** 2)))  # (N,)
    g_z = np.exp(-0.5 * ((z_samples - params.height_z) ** 2) / (params.sigma_z ** 2))  # (M,)
    intensities = g_xy[:, None] @ g_z[None, :]

    # Add random noise to the intensities
    intensities = intensities + intensities * np.random.normal(1, 1, intensities.shape)
    
    intensities = intensities.astype(np.float32)

    return intensities, params

def simulate_event_3d(
    graph: Graph,
    z_limits: Tuple[float, float] = (0.0, 1.0),
    sigma_xy: float = 0.3,
    sigma_z: float = 0.2,
    num_z: int = 16,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, EventParams]:
    intensities, params = simulate_event(graph, z_limits, sigma_xy, sigma_z, num_z, seed)
    return intensities.flatten('F')[:, None], params


def normalize_adjacency(adjacency: np.ndarray) -> torch.Tensor:
    a = adjacency.astype(np.float32)
    n = a.shape[0]
    a = a + np.eye(n, dtype=np.float32)
    d = np.sum(a, axis=1) 
    d_sqrt_inv = 1.0 / np.sqrt(np.maximum(d, 1e-8)).astype(np.float32)
    a_hat = (d_sqrt_inv[:, None] * a) * d_sqrt_inv[None, :]
    return torch.from_numpy(a_hat)

def visualize_event(graph: Graph, intensities: np.ndarray, params: Optional[EventParams] = None, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Scatter plot of node intensities; if (N,M), sums over z for color."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    xy = graph.positions_xy
    node_color = intensities
    if node_color.ndim == 2:
        node_color = node_color.sum(axis=1)
    # draw edges
    rows, cols = np.where(graph.adjacency[:len(xy), :len(xy)] > 0.0)
    for i, j in zip(rows, cols):
        if i < j:
            ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]], color="lightgray", linewidth=0.5, zorder=1)

    sc = ax.scatter(xy[:, 0], xy[:, 1], c=node_color, cmap="viridis", s=50, zorder=2)
    if params is not None:
        ax.scatter([params.center_xy[0]], [params.center_xy[1]], c="red", s=60, marker="x", zorder=3)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("intensity")
    if params is not None:
        ax.set_title(f"Event center=({params.center_xy[0]:.2f},{params.center_xy[1]:.2f}), height_z={params.height_z:.2f}, "+
                    f"sigma_xy={params.sigma_xy:.2f}, sigma_z={params.sigma_z:.2f}")
    return ax


def visualize_event_z(graph: Graph, intensities: np.ndarray, params: Optional[EventParams] = None, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Scatter plot of node intensities; if (N,M), sums over z for color."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    xy = graph.positions_xy
    node_color = intensities
    
    if node_color.ndim == 2:
        node_color=node_color.sum(axis=0)
    
    if params is not None:
        ax.set_title(f"Event center=({params.center_xy[0]:.2f},{params.center_xy[1]:.2f}), height_z={params.height_z:.2f}, "+
                    f"sigma_xy={params.sigma_xy:.2f}, sigma_z={params.sigma_z:.2f}")
    ax.set_xlabel("z")
    ax.set_ylabel("intensity")
    ax.plot(np.linspace(0, 1, node_color.shape[0]), node_color)
    if params is not None:
        ax.axvline(params.height_z, color="red", linestyle="--")
        
    return ax


def visualize_graph_3d(
    adjacency,
    positions: np.ndarray,
    *,
    edge_alpha: float = 0.15,
    edge_color: str = "gray",
    node_color: Optional[np.ndarray] = None,
    node_size: int = 6,
    max_edges: int = 200000,
    ax: Optional[plt.Axes] = None,
):
    """Plot a 3D graph given a sparse adjacency and vertex positions (N,3).

    adjacency: torch.sparse_coo_tensor | scipy.sparse.coo_matrix | np.ndarray
    positions: np.ndarray of shape (N,3)
    """
    import numpy as _np
    import torch as _torch
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    pos = _np.asarray(positions, dtype=_np.float32)
    assert pos.ndim == 2 and pos.shape[1] == 3, f"positions must be (N,3), got {pos.shape}"
    N = pos.shape[0]

    if hasattr(adjacency, "is_sparse") and isinstance(adjacency, _torch.Tensor) and adjacency.is_sparse:
        A = adjacency.coalesce()
        rows = A.indices()[0].detach().cpu().numpy()
        cols = A.indices()[1].detach().cpu().numpy()
    elif hasattr(adjacency, "tocoo"):
        A = adjacency.tocoo()
        rows = _np.asarray(A.row, dtype=_np.int64)
        cols = _np.asarray(A.col, dtype=_np.int64)
    else:
        A = _np.asarray(adjacency)
        rows, cols = _np.nonzero(A)

    # Optionally subsample edges for speed
    E = rows.shape[0]
    if E > max_edges:
        sel = _np.random.default_rng(0).choice(E, size=max_edges, replace=False)
        rows = rows[sel]
        cols = cols[sel]

    if ax is None:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")

    # Draw edges
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    for i, j in zip(rows, cols):
        if i == j:
            continue
        ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], color=edge_color, alpha=edge_alpha, linewidth=0.5, zorder=1)

    # Draw nodes
    if node_color is None:
        # ax.scatter(x, y, z, s=node_size, c="#1f77b4", depthshade=True, zorder=2)
        pass
    else:
        c = _np.asarray(node_color)
        if c.ndim == 2:
            c = c.sum(axis=1)
        # ax.scatter(x, y, z, s=node_size, c=c, cmap="viridis", depthshade=True, zorder=2)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("3D Graph")
    return ax


def create_batch(
    graph: Graph,
    batch_size: int,
    z_limits: Tuple[float, float] = (0.0, 1.0),
    sigma_xy: float = 0.3,
    sigma_z: float = 0.2,
    num_z: int = 64,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, List[EventParams]]:
    """Generate a batch of events over the same graph.

    Returns:
        intensities: (B, N, M) with M=num_z
        params_list: list length B of EventParams
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    rng = np.random.default_rng(seed)
    # Derive seeds for reproducibility but varied events
    seeds = rng.integers(0, np.iinfo(np.int32).max, size=batch_size, endpoint=True)
    intensities_list: List[np.ndarray] = []
    params_list: List[EventParams] = []
    for s in seeds:
        intensities, params = simulate_event(
            graph,
            z_limits=z_limits,
            sigma_xy=sigma_xy,
            sigma_z=sigma_z,
            num_z=num_z,
            seed=int(s),
        )
        intensities_list.append(intensities)
        params_list.append(params)
    batch = np.stack(intensities_list, axis=0)
    return batch, params_list


def create_batch_3d(
    graph: Graph,
    batch_size: int,
    z_limits: Tuple[float, float] = (0.0, 1.0),
    sigma_xy: float = 0.3,
    sigma_z: float = 0.2,
    num_z: int = 64,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, List[EventParams]]:
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, np.iinfo(np.int32).max, size=batch_size, endpoint=True)
    intensities_list: List[np.ndarray] = []
    params_list: List[EventParams] = []
    for s in seeds:
        intensities, params = simulate_event_3d(graph, z_limits, sigma_xy, sigma_z, num_z, int(s))
        intensities_list.append(intensities)
        params_list.append(params)
    batch = np.stack(intensities_list, axis=0)
    return batch, params_list


# -----------------------------
# Perlin noise utilities (3D)
# -----------------------------
def _fade(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6 - 15) + 10)


def _lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    return a + t * (b - a)


def _grad(hashv: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    # 12 gradient directions
    h = hashv & 15
    u = np.where(h < 8, x, y)
    v = np.where(h < 4, y, np.where((h == 12) | (h == 14), x, z))
    return ((np.where((h & 1) == 0, u, -u)) + (np.where((h & 2) == 0, v, -v)))


def perlin_noise_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray, frequency: float = 1.0, seed: Optional[int] = None) -> np.ndarray:
    """3D Perlin noise in [-1, 1] evaluated elementwise on arrays x,y,z (broadcastable)."""
    if seed is None:
        seed = 0
    rng = np.random.default_rng(seed)
    p = np.arange(256, dtype=np.int32)
    rng.shuffle(p)
    p = np.concatenate([p, p])

    # scale inputs by frequency
    x = np.asarray(x, dtype=np.float32) * float(frequency)
    y = np.asarray(y, dtype=np.float32) * float(frequency)
    z = np.asarray(z, dtype=np.float32) * float(frequency)

    xi = np.floor(x).astype(np.int32) & 255
    yi = np.floor(y).astype(np.int32) & 255
    zi = np.floor(z).astype(np.int32) & 255

    xf = x - np.floor(x)
    yf = y - np.floor(y)
    zf = z - np.floor(z)

    u = _fade(xf)
    v = _fade(yf)
    w = _fade(zf)

    # hash coordinates of the cube corners
    aaa = p[p[p[    xi ]+    yi ]+    zi ]
    aba = p[p[p[    xi ]+yi + 1]+    zi ]
    aab = p[p[p[    xi ]+    yi ]+zi + 1]
    abb = p[p[p[    xi ]+yi + 1]+zi + 1]
    baa = p[p[p[xi + 1]+    yi ]+    zi ]
    bba = p[p[p[xi + 1]+yi + 1]+    zi ]
    bab = p[p[p[xi + 1]+    yi ]+zi + 1]
    bbb = p[p[p[xi + 1]+yi + 1]+zi + 1]

    # gradients at corners
    x1 = _lerp(_grad(aaa, xf    , yf    , zf    ), _grad(baa, xf-1  , yf    , zf    ), u)
    x2 = _lerp(_grad(aba, xf    , yf-1  , zf    ), _grad(bba, xf-1  , yf-1  , zf    ), u)
    y1 = _lerp(x1, x2, v)

    x1 = _lerp(_grad(aab, xf    , yf    , zf-1  ), _grad(bab, xf-1  , yf    , zf-1  ), u)
    x2 = _lerp(_grad(abb, xf    , yf-1  , zf-1  ), _grad(bbb, xf-1  , yf-1  , zf-1  ), u)
    y2 = _lerp(x1, x2, v)

    out = _lerp(y1, y2, w)
    # Normalize approx to [-1,1]
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def create_batch_3d_perlin(
    graph: Graph,
    batch_size: int,
    z_limits: Tuple[float, float] = (0.0, 1.0),
    sigma_xy: float = 0.3,
    sigma_z: float = 0.2,
    num_z: int = 64,
    seed: Optional[int] = None,
    perlin_frequency: float = 2.0,
    perlin_weight: float = 1.0,
) -> Tuple[np.ndarray, List[EventParams]]:
    """Like create_batch_3d, but multiplies the 3D Gaussian with a 3D Perlin field.

    Returns:
        batch: (B, N_total, 1) where N_total = N_nodes * num_z (same as create_batch_3d)
        params_list: list of EventParams
    """
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, np.iinfo(np.int32).max, size=batch_size, endpoint=True)

    N_nodes = graph.positions_xy.shape[0]
    # Coordinates
    xy = np.asarray(graph.positions_xy, dtype=np.float32)  # (N,2)
    z_samples = np.linspace(z_limits[0], z_limits[1], int(num_z), dtype=np.float32)  # (M,)

    intensities_list: List[np.ndarray] = []
    params_list: List[EventParams] = []
    for s in seeds:
        base, params = simulate_event(graph, z_limits, sigma_xy, sigma_z, num_z, int(s))  # (N,M)

        # Build broadcastable grids for Perlin: (N,M)
        x_grid = np.repeat(xy[:, 0:1], num_z, axis=1)
        y_grid = np.repeat(xy[:, 1:2], num_z, axis=1)
        z_grid = np.repeat(z_samples[None, :], N_nodes, axis=0)

        noise = perlin_noise_3d(x_grid, y_grid, z_grid, frequency=perlin_frequency, seed=int(s))  # (N,M) in [-1,1]
        # Normalize within-sample to enhance contrast, then map to [0,1]
        n_min, n_max = float(noise.min()), float(noise.max())
        if n_max - n_min < 1e-6:
            noise01 = np.zeros_like(noise, dtype=np.float32) + 0.5
        else:
            noise01 = (noise - n_min) / (n_max - n_min)

        # Multiplicative modulation around 1.0 with visible amplitude
        mod = (1.0 - 0.5 * perlin_weight) + perlin_weight * noise01  # in [1-0.5*w, 1+0.5*w]
        combined = base * mod
        # Ensure non-negative
        combined = np.clip(combined.astype(np.float32), 0.0, None)

        # Column-major flatten to match layer-major convention (z changes slowest)
        combined_vec = combined.reshape(-1, 1, order='F')
        intensities_list.append(combined_vec)
        params_list.append(params)

    batch = np.stack(intensities_list, axis=0)  # (B, N*M, 1)
    return batch, params_list


if __name__ == "__main__":

    '''
    tr = TritiumDataLoader("data/tritium_ss.h5", "data/pmt_xy.h5")
    channel_positions = tr.channel_positions
    n_nodes = tr.n_channels * tr.n_time_points

    pos_xy = torch.from_numpy(channel_positions.astype(np.float32))  # (C,2)
    pos_xy = pos_xy.repeat(tr.n_time_points, 1)  # (T*C,2)
    pos_z = torch.arange(tr.n_time_points, dtype=torch.float32).repeat_interleave(tr.n_channels).unsqueeze(1)  # (T*C,1)
    pos_z = (pos_z / max(tr.n_time_points - 1, 1)) * 2.0 - 1.0
    pos = torch.cat([pos_xy, pos_z], dim=1).float()  # (T*C,3)
    '''

    
    # A_sparse = create_3d_adjacency_matrix_sparse(channel_positions, num_layers=tr.n_time_points, z_sep=10.0, radius=17.0)
    # A_sparse = A_sparse.coalesce()

    # visualize_graph_3d(A_sparse, pos)
    # plt.show()

    """
    n_channels = 32
    n_time_points = 16
    n_nodes = n_channels * n_time_points
    graph = create_3d_radial_graph(num_layer_nodes=n_channels, num_layers=n_time_points, r_within=0.5)
    sparse_graph = graph_to_sparse_graph(graph)
    A_sparse = sparse_graph.adjacency
    pos = sparse_graph.positions_xyz

    visualize_graph_3d(A_sparse, pos)
    plt.show()
    """

