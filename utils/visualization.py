import numpy as np


def build_xy_adjacency_radius(positions_xy: np.ndarray, radius: float) -> np.ndarray:
    diff = positions_xy[:, None, :] - positions_xy[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    A = (dist2 <= (radius * radius)).astype(np.float32)
    np.fill_diagonal(A, 1.0)
    A = np.maximum(A, A.T)
    return A
