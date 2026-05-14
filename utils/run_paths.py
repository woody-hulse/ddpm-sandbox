import glob
import os
from typing import Optional, Tuple

from config import Config


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def ensure_parent_dir(path: str) -> str:
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    return path


def resolve_model_run_dirs(
    cfg: Config,
    subdir_attr: str,
    *,
    create: bool = False,
) -> Tuple[str, str]:
    checkpoint_dir, plot_dir = cfg.paths.run_dirs(subdir_attr, cfg.encoder.latent_dim)
    if create:
        ensure_dir(checkpoint_dir)
        ensure_dir(plot_dir)
    return checkpoint_dir, plot_dir


def epoch_plot_dir(plot_dir: str, epoch: int, *, create: bool = True) -> str:
    path = os.path.join(plot_dir, f"epoch_{epoch}")
    if create:
        ensure_dir(path)
    return path


def latest_checkpoint(checkpoint_dir: str, pattern: str) -> Optional[str]:
    files = glob.glob(os.path.join(checkpoint_dir, pattern))
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


def latest_checkpoint_across_runs(
    checkpoint_dir: str,
    sibling_prefix: str,
    pattern: str,
) -> Optional[str]:
    parent_dir = os.path.dirname(checkpoint_dir)
    if not os.path.isdir(parent_dir):
        return None

    candidates = []
    for subdir in os.listdir(parent_dir):
        if not subdir.startswith(sibling_prefix):
            continue
        match = latest_checkpoint(os.path.join(parent_dir, subdir), pattern)
        if match is not None:
            candidates.append(match)

    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)
