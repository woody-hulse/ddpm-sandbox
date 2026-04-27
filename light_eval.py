#!/usr/bin/env python3
"""
Run the inference-only evaluation suite for AE Light and DiffAE Light.

This script adapts the repo's reusable evaluation workflows to the light model
variants and writes every artifact under `light_eval/` by default.

Included experiments:
- Reconstruction metrics on SS events (adapted from eval_recon.py)
- RQ comparison, residual plots, and histogram overlays on MS events
  (adapted from compare_rqs.py and plot_rq_distributions.py)
- Latent manifold plots colored by |delta_mu|, lopsided labels, and waveform
  roughness (adapted from plot_umap.py)
- DiffAE latent-prior sampling from random latent draws
  (adapted from sample_diffae_light_latents.py)
- SS-vs-MS linear probe + manifold scatter (adapted from diagnose/probe_ss_ms.py)
- Anomaly-separability plots and Mahalanobis ranking heatmap
  (adapted from anomaly_probe.py)

Legacy scripts that are intentionally not run are recorded in the summary JSON
when they rely on non-light-only internals with no light-model equivalent.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py

os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
DIAGNOSE_DIR = os.path.join(ROOT, "diagnose")
FULL_PMT_XY_PATH = os.path.join(ROOT, "data", "pmt_xy.h5")
if DIAGNOSE_DIR not in sys.path:
    sys.path.insert(0, DIAGNOSE_DIR)

from ae_light import AELightContext, reconstruct_ae_light, save_encoded_dataset as save_ae_light_dataset
from compare_rqs import collect_rqs, compute_rqs, plot_rq_comparison, wf_to_z_profile
from config import Config, default_config
from diffae_light import (
    DiffAELightContext,
    LightGraphEncoder,
    sample_diffae_light,
    sample_from_latent_diffae_light,
    save_encoded_dataset as save_diffae_light_dataset,
)
from eval_recon import (
    distribution_metrics,
    evaluate_all,
    multi_sample_metrics,
    physics_marginals,
    plot_results,
    print_table,
    to_2d,
)
from lz_data_loader import TritiumSSDataLoader
from plot_rq_distributions import plot_distributions
from plot_real_event_3d import plot_waveform_3d_scatter
from plot_style import COLORS, MODEL_COLORS, apply_style


apply_style()


@dataclass
class LoadedLightModel:
    key: str
    display_name: str
    color: str
    latent_dim: int
    checkpoint_dir: str
    checkpoint_path: str
    epoch: int
    ctx: Any


@dataclass
class SharedSampleStore:
    path: str
    n_samples: int
    n_nodes: int
    n_channels: int
    n_time_points: int
    diffae_samples: int


def method_label(method: str) -> str:
    return {"pca": "PCA", "umap": "UMAP", "tsne": "t-SNE"}[method]


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def write_json(path: str, payload: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, sort_keys=True)


def metric_summary(metrics: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for model_name, model_metrics in metrics.items():
        out[model_name] = {}
        for metric_name, arr in model_metrics.items():
            arr_f = np.asarray(arr, dtype=np.float64)
            out[model_name][metric_name] = {
                "mean": float(np.nanmean(arr_f)),
                "std": float(np.nanstd(arr_f)),
            }
    return out


def shared_sample_store_metadata(
    models: Sequence[LoadedLightModel],
    n_samples: int,
    n_nodes: int,
    n_channels: int,
    n_time_points: int,
    diffae_samples: int,
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "n_samples": int(n_samples),
        "n_nodes": int(n_nodes),
        "n_channels": int(n_channels),
        "n_time_points": int(n_time_points),
        "diffae_samples": int(diffae_samples),
    }
    for model in models:
        prefix = model.key
        meta[f"{prefix}_latent_dim"] = int(model.latent_dim)
        meta[f"{prefix}_epoch"] = int(model.epoch)
        meta[f"{prefix}_checkpoint_path"] = os.path.abspath(model.checkpoint_path)
    return meta


def _build_shared_store(path: str, attrs: Dict[str, Any]) -> SharedSampleStore:
    return SharedSampleStore(
        path=path,
        n_samples=int(attrs["n_samples"]),
        n_nodes=int(attrs["n_nodes"]),
        n_channels=int(attrs["n_channels"]),
        n_time_points=int(attrs["n_time_points"]),
        diffae_samples=int(attrs["diffae_samples"]),
    )


def validate_shared_sample_store(
    path: str,
    models: Sequence[LoadedLightModel],
    n_samples: int,
    diffae_samples: int,
) -> SharedSampleStore:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Shared sample cache not found at {path}. Run light_eval.py with --regenerate to create it."
        )

    ref_ctx = models[0].ctx
    expected = shared_sample_store_metadata(
        models=models,
        n_samples=n_samples,
        n_nodes=ref_ctx.n_nodes,
        n_channels=ref_ctx.n_channels,
        n_time_points=ref_ctx.n_time_points,
        diffae_samples=max(1, int(diffae_samples)),
    )
    optional_model_attr_suffixes = ("_latent_dim", "_epoch", "_checkpoint_path")
    missing_optional_attrs: List[str] = []

    with h5py.File(path, "r") as f:
        attrs = {key: f.attrs[key] for key in f.attrs.keys()}
        for key, expected_value in expected.items():
            actual = attrs.get(key)
            if isinstance(actual, bytes):
                actual = actual.decode("utf-8")
            if isinstance(expected_value, np.generic):
                expected_value = expected_value.item()
            if actual is None and key.endswith(optional_model_attr_suffixes):
                missing_optional_attrs.append(key)
                continue
            if actual != expected_value:
                raise RuntimeError(
                    f"Shared sample cache mismatch for '{key}': expected {expected_value!r}, got {actual!r}. "
                    "Rerun with --regenerate to rebuild the cache."
                )

        required_shapes = {
            "raw": (n_samples, ref_ctx.n_nodes, 1),
            "delta_mu": (n_samples,),
            "diffae_light_samples": (max(1, int(diffae_samples)), n_samples, ref_ctx.n_nodes),
        }
        for model in models:
            required_shapes[model.key] = (n_samples, ref_ctx.n_nodes)

        for dataset_name, expected_shape in required_shapes.items():
            if dataset_name not in f:
                raise RuntimeError(
                    f"Shared sample cache is missing dataset '{dataset_name}'. "
                    "Rerun with --regenerate to rebuild the cache."
                )
            actual_shape = tuple(int(v) for v in f[dataset_name].shape)
            if actual_shape != expected_shape:
                raise RuntimeError(
                    f"Shared sample cache dataset '{dataset_name}' has shape {actual_shape}, "
                    f"expected {expected_shape}. Rerun with --regenerate to rebuild the cache."
                )

    if missing_optional_attrs:
        print(
            "\nShared sample cache is missing backward-compatible metadata attributes: "
            + ", ".join(sorted(missing_optional_attrs))
            + ". Reusing cache based on structural validation only."
        )
    print(f"\nUsing existing shared MS sample cache: {path}")
    return _build_shared_store(path, expected)


def create_shared_sample_store(
    models: Sequence[LoadedLightModel],
    output_root: str,
    n_samples: int,
    batch_size: int,
    diffae_samples: int,
    regenerate: bool,
) -> SharedSampleStore:
    diffae_samples = max(1, int(diffae_samples))
    path = os.path.join(output_root, "shared_ms_samples.h5")
    if not regenerate:
        return validate_shared_sample_store(
            path=path,
            models=models,
            n_samples=n_samples,
            diffae_samples=diffae_samples,
        )

    ref_ctx = models[0].ctx
    n_nodes = ref_ctx.n_nodes
    n_channels = ref_ctx.n_channels
    n_time_points = ref_ctx.n_time_points
    diffae_model = next((model for model in models if model.key == "diffae_light"), None)
    if diffae_model is None:
        raise RuntimeError("DiffAE model is required for the shared sample store.")

    print(f"\nPrecomputing shared MS sample cache: {n_samples} events")
    with h5py.File(path, "w") as f:
        for key, value in shared_sample_store_metadata(
            models=models,
            n_samples=n_samples,
            n_nodes=n_nodes,
            n_channels=n_channels,
            n_time_points=n_time_points,
            diffae_samples=diffae_samples,
        ).items():
            f.attrs[key] = value

        raw_ds = f.create_dataset("raw", shape=(n_samples, n_nodes, 1), dtype=np.float16)
        delta_mu_ds = f.create_dataset("delta_mu", shape=(n_samples,), dtype=np.float32)
        for model in models:
            f.create_dataset(model.key, shape=(n_samples, n_nodes), dtype=np.float16)
        f.create_dataset("diffae_light_samples", shape=(diffae_samples, n_samples, n_nodes), dtype=np.float16)

        written = 0
        while written < n_samples:
            bs = min(batch_size, n_samples - written)
            wf_col, cond, *_ = ref_ctx.loader.get_batch(bs)
            raw_ds[written:written + bs] = wf_col.astype(np.float16)
            delta_mu_ds[written:written + bs] = cond[:, 4].astype(np.float32)

            for model in models:
                if model.key == "diffae_light":
                    diff_samples = []
                    for _ in range(diffae_samples):
                        diff_samples.append(reconstruct_raw_flat_batch(model, wf_col))
                    diff_stack = np.stack(diff_samples, axis=0).astype(np.float16)
                    f["diffae_light_samples"][:, written:written + bs, :] = diff_stack
                    f[model.key][written:written + bs] = diff_stack[0]
                else:
                    rec = reconstruct_raw_flat_batch(model, wf_col).astype(np.float16)
                    f[model.key][written:written + bs] = rec

            written += bs
            print(f"  cached {written}/{n_samples}")

    return SharedSampleStore(
        path=path,
        n_samples=n_samples,
        n_nodes=n_nodes,
        n_channels=n_channels,
        n_time_points=n_time_points,
        diffae_samples=diffae_samples,
    )


def load_shared_raw(store: SharedSampleStore, n_samples: Optional[int] = None) -> np.ndarray:
    n = store.n_samples if n_samples is None else min(n_samples, store.n_samples)
    with h5py.File(store.path, "r") as f:
        return np.asarray(f["raw"][:n], dtype=np.float32)


def load_shared_delta_mu(store: SharedSampleStore, n_samples: Optional[int] = None) -> np.ndarray:
    n = store.n_samples if n_samples is None else min(n_samples, store.n_samples)
    with h5py.File(store.path, "r") as f:
        return np.asarray(f["delta_mu"][:n], dtype=np.float32)


def load_shared_reconstruction(store: SharedSampleStore, model_key: str, n_samples: Optional[int] = None) -> np.ndarray:
    n = store.n_samples if n_samples is None else min(n_samples, store.n_samples)
    with h5py.File(store.path, "r") as f:
        return np.asarray(f[model_key][:n], dtype=np.float32)


def load_shared_diffae_samples(store: SharedSampleStore, n_samples: Optional[int] = None) -> np.ndarray:
    n = store.n_samples if n_samples is None else min(n_samples, store.n_samples)
    with h5py.File(store.path, "r") as f:
        return np.asarray(f["diffae_light_samples"][:, :n], dtype=np.float32)


def load_pmt_positions(path: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        if "TA_PMTs_xy" in f:
            xy = np.asarray(f["TA_PMTs_xy"][:], dtype=np.float32)
            if np.max(np.abs(xy)) > 30.0:
                xy = xy / 10.0
            return xy
        if "xy" in f:
            return np.asarray(f["xy"][:], dtype=np.float32)
        raise ValueError(f"No recognized xy dataset in {path}; available keys: {list(f.keys())}")


def map_subset_pmts_to_full(subset_xy: np.ndarray, full_xy: np.ndarray, tol: float = 1e-5) -> np.ndarray:
    mapping = np.empty(subset_xy.shape[0], dtype=np.int32)
    used: set[int] = set()
    for i, xy in enumerate(subset_xy):
        d = np.linalg.norm(full_xy - xy[None, :], axis=1)
        j = int(np.argmin(d))
        if float(d[j]) > tol:
            raise ValueError(f"Could not match subset PMT {i} to full detector geometry within tolerance {tol}.")
        if j in used:
            raise ValueError(f"Subset PMT {i} maps to duplicate full-detector index {j}.")
        mapping[i] = j
        used.add(j)
    return mapping


def expand_charge_to_full_detector(charge_subset: np.ndarray, subset_to_full: np.ndarray, n_full_pmts: int) -> np.ndarray:
    charge_full = np.zeros(n_full_pmts, dtype=np.float32)
    charge_full[subset_to_full] = np.asarray(charge_subset, dtype=np.float32)
    return charge_full


def expand_waveform_to_full_detector(
    waveform_subset_ct: np.ndarray,
    subset_to_full: np.ndarray,
    n_full_pmts: int,
) -> np.ndarray:
    waveform_full = np.zeros((n_full_pmts, waveform_subset_ct.shape[1]), dtype=np.float32)
    waveform_full[subset_to_full] = np.asarray(waveform_subset_ct, dtype=np.float32)
    return waveform_full


def select_nonzero_example_indices(raw_flat: np.ndarray, n_examples: int, seed: int) -> np.ndarray:
    n_total = raw_flat.shape[0]
    if n_total == 0:
        return np.empty((0,), dtype=np.int32)
    nnz = np.count_nonzero(raw_flat, axis=1)
    candidates = np.flatnonzero(nnz > 0)
    if candidates.size == 0:
        raise ValueError("No nonzero events found in the shared cache.")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(candidates, size=min(n_examples, candidates.size), replace=False)
    return np.sort(chosen.astype(np.int32))


def resolve_encoder(ctx: Any) -> nn.Module:
    return ctx.encoder


def get_ss_loader(loader: Any) -> Any:
    return loader.ss_loader if hasattr(loader, "ss_loader") else loader


@torch.no_grad()
def encode_raw_flat_batch(model: LoadedLightModel, batch_np: np.ndarray, batch_size: int) -> np.ndarray:
    ctx = model.ctx
    encoder = resolve_encoder(ctx)
    encoder.eval()
    all_z = []
    for start in range(0, len(batch_np), batch_size):
        end = min(start + batch_size, len(batch_np))
        chunk = batch_np[start:end]
        bs = end - start
        x = torch.from_numpy(ctx.data_stats.normalize(chunk).astype(np.float32)).to(ctx.device)
        x_flat = x.reshape(bs * ctx.n_nodes, 1)
        if isinstance(encoder, LightGraphEncoder):
            if ctx.encoder_pyramid is None:
                raise RuntimeError(f"{model.display_name}: encoder_pyramid is missing for graph encoder.")
            z, _, _ = encoder(x_flat, ctx.encoder_pyramid, batch_size=bs)
        else:
            z, _, _ = encoder(x_flat, ctx.A_sparse, ctx.pos, batch_size=bs)
        all_z.append(z.detach().cpu().numpy())
    return np.concatenate(all_z, axis=0)


@torch.no_grad()
def encode_ct_batch(model: LoadedLightModel, wf_ct: np.ndarray, batch_size: int) -> np.ndarray:
    bsz = wf_ct.shape[0]
    wf_col = np.transpose(wf_ct, (0, 2, 1)).reshape(bsz, -1, 1).astype(np.float32)
    return encode_raw_flat_batch(model, wf_col, batch_size=batch_size)


@torch.no_grad()
def reconstruct_raw_flat_batch(model: LoadedLightModel, batch_np: np.ndarray) -> np.ndarray:
    ctx = model.ctx
    x = torch.from_numpy(ctx.data_stats.normalize(batch_np).astype(np.float32)).to(ctx.device)
    if model.key == "ae_light":
        rec = reconstruct_ae_light(ctx, x)
    else:
        rec = sample_diffae_light(ctx, x, pbar=False)
    rec_np = ctx.data_stats.denormalize(rec.cpu().numpy())
    return np.clip(rec_np[:, 0, :], 0.0, None)


def collect_events(loader: Any, n_events: int, batch_size: int) -> np.ndarray:
    chunks = []
    collected = 0
    while collected < n_events:
        bs = min(batch_size, n_events - collected)
        wf, *_ = loader.get_batch(bs)
        chunks.append(wf.astype(np.float32))
        collected += bs
    return np.concatenate(chunks, axis=0)


def waveform_roughness(batch_np: np.ndarray, n_channels: int, n_time: int) -> np.ndarray:
    rough = np.empty(batch_np.shape[0], dtype=np.float32)
    for i in range(batch_np.shape[0]):
        z = wf_to_z_profile(batch_np[i, :, 0], n_channels, n_time)
        rough[i] = np.abs(np.diff(z)).mean()
    return rough


def make_lopsided_batch(batch_np: np.ndarray, frac: float, sigma: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    out = batch_np.copy()
    sides = np.zeros(len(out), dtype=np.int32)
    half = out.shape[1] // 2
    n_aug = max(1, int(round(len(out) * frac)))
    aug_idx = rng.choice(len(out), size=n_aug, replace=False)
    aug_sides = rng.integers(1, 3, size=n_aug)
    for idx, side in zip(aug_idx, aug_sides):
        if side == 1:
            out[idx, :half, 0] = gaussian_filter1d(out[idx, :half, 0], sigma=sigma)
        else:
            out[idx, half:, 0] = gaussian_filter1d(out[idx, half:, 0], sigma=sigma)
        sides[idx] = side
    return out, sides


def load_latent_subset(path: str, n_samples: int, seed: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    with h5py.File(path, "r") as f:
        total = int(f["latents"].shape[0])
        n = min(n_samples, total)
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(total, size=n, replace=False))
        latents = np.asarray(f["latents"][idx], dtype=np.float32)
        delta_mu = np.asarray(f["delta_mu"][idx], dtype=np.float32) if "delta_mu" in f else None
    return latents, delta_mu


def ensure_latent_export(
    model: LoadedLightModel,
    out_dir: str,
    batch_size: int,
    n_samples: int,
) -> str:
    ensure_dir(out_dir)
    filename = f"{model.key}_z{model.latent_dim}_encoded_ms_latents.h5"
    path = os.path.join(out_dir, filename)
    if os.path.exists(path):
        return path
    if model.key == "ae_light":
        save_ae_light_dataset(model.ctx, path, encoder=model.ctx.encoder, batch_size=batch_size, n_samples=n_samples, verbose=True)
    else:
        save_diffae_light_dataset(model.ctx, path, encoder=model.ctx.encoder, batch_size=batch_size, n_samples=n_samples, verbose=True)
    return path


def load_latents_from_h5(path: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        if "latents" not in f:
            raise KeyError(f"{path} does not contain a `latents` dataset.")
        latents = np.asarray(f["latents"][:], dtype=np.float32)
    if latents.ndim != 2:
        raise ValueError(f"Expected latent array of shape (N, D), got {latents.shape}.")
    return latents


def fit_diffae_latent_prior(
    model: LoadedLightModel,
    shared_store: SharedSampleStore,
    prior: str,
    prior_samples: int,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    ctx = model.ctx
    latent_dim = ctx.cfg.encoder.latent_dim
    if prior == "standard":
        mean = np.zeros(latent_dim, dtype=np.float32)
        std = np.ones(latent_dim, dtype=np.float32)
        return mean, std, np.empty((0, latent_dim), dtype=np.float32), "standard_normal"

    if prior_samples <= 0:
        raise ValueError("prior_samples must be positive for the empirical latent prior.")

    checkpoint_latents = os.path.join(model.checkpoint_dir, ctx.cfg.paths.diffae_latents_file)
    if os.path.exists(checkpoint_latents):
        latents = load_latents_from_h5(checkpoint_latents)
        if latents.shape[1] != latent_dim:
            raise ValueError(
                f"Latent file dimension mismatch for {checkpoint_latents}: "
                f"expected {latent_dim}, got {latents.shape[1]}."
            )
        if latents.shape[0] > prior_samples:
            latents = latents[:prior_samples]
        source = "checkpoint_latents"
    else:
        raw = load_shared_raw(shared_store, n_samples=prior_samples)
        latents = encode_raw_flat_batch(model, raw, batch_size=batch_size)
        source = "shared_ms_cache"

    mean = latents.mean(axis=0).astype(np.float32)
    std = np.clip(latents.std(axis=0).astype(np.float32), 1e-6, None)
    return mean, std, latents.astype(np.float32), source


def sample_latent_vectors(
    n_samples: int,
    mean: np.ndarray,
    std: np.ndarray,
    temperature: float,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n_samples, mean.shape[0])).astype(np.float32)
    z = mean[None, :] + temperature * std[None, :] * z
    return torch.from_numpy(z).to(device)


def reshape_flat_waveforms(flat_waveforms: np.ndarray, n_channels: int, n_time_points: int) -> np.ndarray:
    return flat_waveforms.reshape(flat_waveforms.shape[0], n_channels, n_time_points, order="F")


def save_latent_sampling_visualization(
    waveforms: np.ndarray,
    channel_positions: np.ndarray,
    output_path: str,
    title: str,
) -> None:
    ensure_dir(os.path.dirname(output_path) or ".")
    n_samples = waveforms.shape[0]
    fig, axes = plt.subplots(n_samples, 2, figsize=(11, 3.2 * n_samples), squeeze=False)

    for idx in range(n_samples):
        charge = waveforms[idx].sum(axis=1)
        trace = waveforms[idx].sum(axis=0)

        ax_xy = axes[idx, 0]
        sc = ax_xy.scatter(
            channel_positions[:, 0],
            channel_positions[:, 1],
            c=charge,
            cmap="viridis",
            s=72,
            edgecolors="k",
            linewidths=0.25,
        )
        ax_xy.set_aspect("equal")
        ax_xy.set_title(f"Sample {idx + 1} charge map", fontweight="bold")
        ax_xy.set_xlabel("x (cm)")
        ax_xy.set_ylabel("y (cm)")
        fig.colorbar(sc, ax=ax_xy, fraction=0.046, pad=0.04, label="Integrated charge")

        ax_t = axes[idx, 1]
        ax_t.plot(np.arange(trace.shape[0]), trace, color=COLORS["diffae"], linewidth=1.4)
        ax_t.fill_between(np.arange(trace.shape[0]), trace, color=COLORS["diffae"], alpha=0.12)
        ax_t.set_title(f"Sample {idx + 1} summed waveform", fontweight="bold")
        ax_t.set_xlabel("Time bin")
        ax_t.set_ylabel("Amplitude")

    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_example_reconstructions_generic(
    raw: np.ndarray,
    recs: Dict[str, np.ndarray],
    n_channels: int,
    n_time: int,
    output_dir: str,
    n_examples: int,
    seed: int,
) -> None:
    if not recs:
        return
    ensure_dir(output_dir)
    rng = np.random.default_rng(seed)
    n_total = raw.shape[0]
    indices = np.sort(rng.choice(n_total, size=min(n_examples, n_total), replace=False))
    model_names = list(recs.keys())
    model_colors = {name: MODEL_COLORS[i % len(MODEL_COLORS)] for i, name in enumerate(model_names)}
    n_cols = 1 + len(model_names)
    time_axis = np.arange(n_time)

    fig, axes = plt.subplots(len(indices), n_cols, figsize=(3.8 * n_cols, 2.2 * len(indices)), squeeze=False)
    for row, idx in enumerate(indices):
        z_raw = wf_to_z_profile(raw[idx], n_channels, n_time)
        y_max = max(z_raw.max() * 1.15, 1.0)

        ax = axes[row, 0]
        ax.plot(time_axis, z_raw, color=COLORS["truth"], linewidth=1.3)
        ax.fill_between(time_axis, z_raw, alpha=0.10, color=COLORS["truth"])
        if row == 0:
            ax.set_title("Raw", fontweight="bold")
        ax.set_ylabel(f"#{idx}", fontweight="bold", rotation=0, labelpad=25)
        if row == len(indices) - 1:
            ax.set_xlabel("Time bin")
        ax.set_ylim(0, y_max)
        ax.set_yticks([])

        for col_idx, name in enumerate(model_names, start=1):
            z_rec = wf_to_z_profile(recs[name][idx], n_channels, n_time)
            ax = axes[row, col_idx]
            ax.plot(time_axis, z_raw, color=COLORS["truth"], linewidth=0.7, alpha=0.28, label="Raw")
            ax.plot(time_axis, z_rec, color=model_colors[name], linewidth=1.3, label=name)
            ax.fill_between(time_axis, z_rec, alpha=0.12, color=model_colors[name])
            if row == 0:
                ax.set_title(name, fontweight="bold")
                ax.legend(fontsize=8, loc="upper right", handlelength=1.2)
            if row == len(indices) - 1:
                ax.set_xlabel("Time bin")
            ax.set_ylim(0, y_max)
            ax.set_yticks([])

    fig.suptitle("Z-Profile Reconstructions", y=0.99, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(os.path.join(output_dir, "example_z_profiles.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig2, axes2 = plt.subplots(len(indices), n_cols, figsize=(3.8 * n_cols, 2.0 * len(indices)), squeeze=False)
    for row, idx in enumerate(indices):
        wf_2d_raw = raw[idx].reshape(n_channels, n_time, order="F")
        vmax = wf_2d_raw.max()
        ax = axes2[row, 0]
        ax.imshow(wf_2d_raw, aspect="auto", origin="lower", cmap="inferno", vmin=0, vmax=vmax, interpolation="nearest")
        if row == 0:
            ax.set_title("Raw", fontweight="bold")
        ax.set_ylabel(f"#{idx}", fontweight="bold", rotation=0, labelpad=25)
        ax.set_yticks([])
        if row == len(indices) - 1:
            ax.set_xlabel("Time bin")

        for col_idx, name in enumerate(model_names, start=1):
            wf_2d_rec = recs[name][idx].reshape(n_channels, n_time, order="F")
            ax = axes2[row, col_idx]
            ax.imshow(wf_2d_rec, aspect="auto", origin="lower", cmap="inferno", vmin=0, vmax=vmax, interpolation="nearest")
            if row == 0:
                ax.set_title(name, fontweight="bold")
            ax.set_yticks([])
            if row == len(indices) - 1:
                ax.set_xlabel("Time bin")

    fig2.suptitle("Waveform Reconstructions (channel by time)", y=0.99, fontweight="bold")
    fig2.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig2.savefig(os.path.join(output_dir, "example_heatmaps.png"), dpi=300, bbox_inches="tight")
    plt.close(fig2)


def plot_full_pmt_reconstruction_triptych(
    raw_flat: np.ndarray,
    diffae_flat: np.ndarray,
    ae_flat: np.ndarray,
    subset_xy: np.ndarray,
    full_xy: np.ndarray,
    n_channels: int,
    n_time: int,
    output_dir: str,
    indices: Sequence[int],
) -> None:
    ensure_dir(output_dir)
    subset_to_full = map_subset_pmts_to_full(subset_xy, full_xy)
    plot_radius = float(np.max(np.linalg.norm(full_xy, axis=1)) * 1.05)

    for idx in indices:
        raw_ct = raw_flat[idx].reshape(n_channels, n_time, order="F")
        diffae_ct = diffae_flat[idx].reshape(n_channels, n_time, order="F")
        ae_ct = ae_flat[idx].reshape(n_channels, n_time, order="F")

        charges = {
            "Original": expand_charge_to_full_detector(raw_ct.sum(axis=1), subset_to_full, full_xy.shape[0]),
            "DiffAE": expand_charge_to_full_detector(diffae_ct.sum(axis=1), subset_to_full, full_xy.shape[0]),
            "AE": expand_charge_to_full_detector(ae_ct.sum(axis=1), subset_to_full, full_xy.shape[0]),
        }
        vmax = max(float(charge.max()) for charge in charges.values())
        vmax = max(vmax, 1e-8)

        fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.1), squeeze=False)
        axes_row = axes[0]
        scatter = None
        for ax, (title, charge) in zip(axes_row, charges.items()):
            scatter = ax.scatter(
                full_xy[:, 0],
                full_xy[:, 1],
                c=charge,
                cmap="viridis",
                vmin=0.0,
                vmax=vmax,
                s=54,
                edgecolors="k",
                linewidths=0.22,
            )
            ax.set_title(title, fontweight="bold")
            ax.set_aspect("equal")
            ax.set_xlim(-plot_radius, plot_radius)
            ax.set_ylim(-plot_radius, plot_radius)
            ax.set_xlabel("x (cm)")
            ax.set_ylabel("y (cm)")
            ax.grid(False)

        fig.subplots_adjust(left=0.06, right=0.90, bottom=0.12, top=0.86, wspace=0.22)
        cax = fig.add_axes([0.92, 0.20, 0.015, 0.60])
        cbar = fig.colorbar(scatter, cax=cax)
        cbar.ax.set_ylabel("Integrated charge")
        fig.suptitle(f"Reconstruction Comparison (event {idx})", fontweight="bold")
        fig.savefig(os.path.join(output_dir, f"event_{idx:04d}_xy_full.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_full_pmt_reconstruction_3d_triptych(
    raw_flat: np.ndarray,
    diffae_flat: np.ndarray,
    ae_flat: np.ndarray,
    subset_xy: np.ndarray,
    full_xy: np.ndarray,
    n_channels: int,
    n_time: int,
    output_dir: str,
    indices: Sequence[int],
    ns_per_bin: float,
    threshold_quantile: float = 0.95,
    min_amplitude: Optional[float] = None,
) -> None:
    ensure_dir(output_dir)
    subset_to_full = map_subset_pmts_to_full(subset_xy, full_xy)
    cmap = plt.get_cmap("viridis")

    for idx in indices:
        raw_ct = raw_flat[idx].reshape(n_channels, n_time, order="F")
        diffae_ct = diffae_flat[idx].reshape(n_channels, n_time, order="F")
        ae_ct = ae_flat[idx].reshape(n_channels, n_time, order="F")
        waveforms = {
            "Original": expand_waveform_to_full_detector(raw_ct, subset_to_full, full_xy.shape[0]),
            "DiffAE": expand_waveform_to_full_detector(diffae_ct, subset_to_full, full_xy.shape[0]),
            "AE": expand_waveform_to_full_detector(ae_ct, subset_to_full, full_xy.shape[0]),
        }
        vmax = max(float(max(np.max(wf), 0.0)) for wf in waveforms.values())
        norm = Normalize(vmin=0.0, vmax=max(vmax, 1e-8))

        fig = plt.figure(figsize=(14.2, 4.8))
        axes = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]
        scatter = None
        for ax, (title, waveform) in zip(axes, waveforms.items()):
            panel = plot_waveform_3d_scatter(
                ax,
                waveform,
                full_xy,
                ns_per_bin=ns_per_bin,
                threshold_quantile=threshold_quantile,
                min_amplitude=min_amplitude,
                norm=norm,
                cmap=cmap,
                title=title,
            )
            scatter = panel["scatter"]

        fig.subplots_adjust(left=0.03, right=0.90, bottom=0.05, top=0.86, wspace=0.06)
        cax = fig.add_axes([0.92, 0.20, 0.015, 0.56])
        cbar = fig.colorbar(scatter, cax=cax)
        cbar.ax.set_ylabel("Amplitude (AU)")
        fig.suptitle(f"3D Reconstruction Comparison (event {idx})", fontweight="bold")
        png_path = os.path.join(output_dir, f"event_{idx:04d}_3d_full.png")
        pdf_path = os.path.join(output_dir, f"event_{idx:04d}_3d_full.pdf")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)


def node_vector_to_grid(values: np.ndarray, n_channels: int, n_time: int) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(n_channels, n_time, order="F")


def plot_node_metric_triptych(
    panels: Sequence[Tuple[str, np.ndarray]],
    n_channels: int,
    n_time: int,
    output_path: str,
    suptitle: str,
    colorbar_label: str,
    cmap: str = "viridis",
    center_zero: bool = False,
) -> None:
    ensure_dir(os.path.dirname(output_path) or ".")
    arrays = [node_vector_to_grid(values, n_channels, n_time) for _, values in panels]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.0), squeeze=False)
    axes_row = axes[0]

    vmin = min(float(arr.min()) for arr in arrays)
    vmax = max(float(arr.max()) for arr in arrays)
    if center_zero:
        lim = max(abs(vmin), abs(vmax))
        vmin, vmax = -lim, lim

    images = []
    for ax, (title, values), arr in zip(axes_row, panels, arrays):
        im = ax.imshow(arr, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        images.append(im)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Time bin")
        ax.set_ylabel("Channel")
        ax.grid(False)

    fig.subplots_adjust(left=0.06, right=0.90, bottom=0.10, top=0.86, wspace=0.26)
    cax = fig.add_axes([0.92, 0.22, 0.015, 0.56])
    cbar = fig.colorbar(images[-1], cax=cax)
    cbar.ax.set_ylabel(colorbar_label)
    fig.suptitle(suptitle, fontweight="bold")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_node_metric_profiles(
    metric_by_model: Dict[str, np.ndarray],
    n_channels: int,
    n_time: int,
    output_path: str,
    ylabel: str,
    title: str,
) -> None:
    ensure_dir(os.path.dirname(output_path) or ".")
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    time_axis = np.arange(n_time)
    for idx, (name, values) in enumerate(metric_by_model.items()):
        arr = node_vector_to_grid(values, n_channels, n_time).mean(axis=0)
        color = MODEL_COLORS[idx % len(MODEL_COLORS)]
        ax.plot(time_axis, arr, color=color, linewidth=1.7, label=name)
    ax.set_xlabel("Time bin")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_continuous_embedding_panels(
    panels: Sequence[Dict[str, Any]],
    output_path: str,
    title: str,
    value_label: str,
    cmap: str,
    point_size: float,
    knn_k: int,
) -> Dict[str, Dict[str, float]]:
    from plot_umap import knn_label_smoothness

    ensure_dir(os.path.dirname(output_path) or ".")
    all_color_values = np.concatenate([panel["color_values"] for panel in panels])
    vmax = float(np.percentile(all_color_values, 98)) if len(all_color_values) > 0 else 1.0
    norm = Normalize(vmin=0.0, vmax=max(vmax, 1e-8))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 5.2), squeeze=False)
    axes_row = axes[0]
    sc = None
    stats: Dict[str, Dict[str, float]] = {}

    for ax, panel in zip(axes_row, panels):
        smoothness = knn_label_smoothness(panel["latents"], panel["smooth_values"], k=knn_k)
        sc = ax.scatter(
            panel["embedding"][:, 0],
            panel["embedding"][:, 1],
            c=panel["color_values"],
            cmap=cmap,
            norm=norm,
            s=point_size,
            alpha=0.65,
            edgecolors="none",
            rasterized=True,
        )
        ax.set_title(f"{panel['name']}\nSmoothness = {smoothness:.4f}", fontweight="bold", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_box_aspect(1)
        ax.grid(False)
        stats[panel["name"]] = {"knn_smoothness": float(smoothness)}

    assert sc is not None
    cbar = fig.colorbar(sc, ax=axes_row.tolist(), shrink=0.82, pad=0.04)
    cbar.set_label(value_label)
    fig.suptitle(title, fontweight="bold")
    fig.subplots_adjust(left=0.05, right=0.93, bottom=0.08, top=0.88, wspace=0.20)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return stats


def plot_lopsided_embedding_panels(
    panels: Sequence[Dict[str, Any]],
    output_path: str,
    title: str,
    point_size: float,
) -> Dict[str, Dict[str, int]]:
    ensure_dir(os.path.dirname(output_path) or ".")
    colors = {0: COLORS["lop_none"], 1: COLORS["lop_left"], 2: COLORS["lop_right"]}
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 5.2), squeeze=False)
    axes_row = axes[0]
    counts: Dict[str, Dict[str, int]] = {}

    for ax, panel in zip(axes_row, panels):
        sides = panel["sides"]
        for side_val, label in [(0, "None"), (1, "Left"), (2, "Right")]:
            mask = sides == side_val
            if not np.any(mask):
                continue
            ax.scatter(
                panel["embedding"][mask, 0],
                panel["embedding"][mask, 1],
                c=colors[side_val],
                s=point_size,
                alpha=0.65,
                edgecolors="none",
                rasterized=True,
                label=label,
                zorder=2 if side_val == 0 else 3,
            )
        n_left = int(np.sum(sides == 1))
        n_right = int(np.sum(sides == 2))
        n_none = int(np.sum(sides == 0))
        ax.set_title(
            f"{panel['name']}\nLeft {n_left} | Right {n_right} | None {n_none}",
            fontweight="bold",
            fontsize=11,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_box_aspect(1)
        ax.grid(False)
        counts[panel["name"]] = {"left": n_left, "right": n_right, "none": n_none}

    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[0], markersize=6, label="None"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[1], markersize=6, label="Left"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[2], markersize=6, label="Right"),
    ]
    axes_row[-1].legend(handles=legend, loc="upper right", fontsize=8, frameon=True, edgecolor="0.8")
    fig.suptitle(title, fontweight="bold")
    fig.subplots_adjust(left=0.05, right=0.97, bottom=0.08, top=0.88, wspace=0.20)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return counts


def load_ae_light(latent_dim: int) -> LoadedLightModel:
    cfg = copy.deepcopy(default_config)
    cfg.encoder.latent_dim = latent_dim
    ctx = AELightContext.build(cfg, for_training=True, verbose=False, use_ms_data=True)
    ckpt = ctx.latest_checkpoint()
    if ckpt is None:
        raise FileNotFoundError(f"No AE Light checkpoint found in {ctx.checkpoint_dir}")
    epoch = ctx.load_checkpoint(ckpt, load_optim=False)
    if ctx.ema_encoder is not None:
        ctx.encoder = ctx.ema_encoder
    if ctx.ema_decoder is not None:
        ctx.decoder = ctx.ema_decoder
    ctx.encoder.eval()
    ctx.decoder.eval()
    return LoadedLightModel(
        key="ae_light",
        display_name="AE",
        color=COLORS["ae"],
        latent_dim=latent_dim,
        checkpoint_dir=ctx.checkpoint_dir,
        checkpoint_path=ckpt,
        epoch=epoch,
        ctx=ctx,
    )


def load_diffae_light(latent_dim: int) -> LoadedLightModel:
    cfg = copy.deepcopy(default_config)
    cfg.encoder.latent_dim = latent_dim
    ctx = DiffAELightContext.build(cfg, for_training=True, verbose=False, use_ms_data=True)
    ckpt = ctx.latest_checkpoint()
    if ckpt is None:
        raise FileNotFoundError(f"No DiffAE Light checkpoint found in {ctx.checkpoint_dir}")
    epoch = ctx.load_checkpoint(ckpt, load_optim=False)
    if ctx.ema_encoder is not None:
        ctx.encoder = ctx.ema_encoder
    if ctx.ema_decoder is not None:
        ctx.decoder = ctx.ema_decoder
    ctx.encoder.eval()
    ctx.decoder.eval()
    return LoadedLightModel(
        key="diffae_light",
        display_name="DiffAE",
        color=COLORS["diffae"],
        latent_dim=latent_dim,
        checkpoint_dir=ctx.checkpoint_dir,
        checkpoint_path=ckpt,
        epoch=epoch,
        ctx=ctx,
    )


def run_reconstruction_eval(
    models: Sequence[LoadedLightModel],
    shared_store: SharedSampleStore,
    out_dir: str,
    n_events: int,
) -> Dict[str, Any]:
    ensure_dir(out_dir)
    wf_all = load_shared_raw(shared_store, n_samples=n_events)
    n_channels = shared_store.n_channels
    n_time = shared_store.n_time_points
    ns_per_bin = default_config.ms_data.ns_per_bin
    channel_positions = models[0].ctx.loader.channel_positions
    true_flat = wf_all[:, :, 0]
    true_2d = to_2d(true_flat, n_channels, n_time)

    per_sample: Dict[str, Dict[str, np.ndarray]] = {}
    dist_metrics: Dict[str, Dict[str, float]] = {}
    rec_2d_dict: Dict[str, np.ndarray] = {}

    for model in models:
        rec_flat = load_shared_reconstruction(shared_store, model.key, n_samples=n_events)
        rec_2d = to_2d(rec_flat, n_channels, n_time)
        rec_2d_dict[model.display_name] = rec_2d
        per_sample[model.display_name] = evaluate_all(rec_2d, true_2d, channel_positions, ns_per_bin)
        dist_metrics[model.display_name] = distribution_metrics(
            physics_marginals(rec_2d, channel_positions, ns_per_bin),
            physics_marginals(true_2d, channel_positions, ns_per_bin),
        )

    stoch_metrics = None
    stoch_summary = None
    if shared_store.diffae_samples > 1:
        diff_samples = load_shared_diffae_samples(shared_store, n_samples=n_events)
        samples_arr = np.stack([to_2d(diff_samples[k], n_channels, n_time) for k in range(diff_samples.shape[0])], axis=0)
        stoch_metrics = multi_sample_metrics(samples_arr, true_2d, ns_per_bin)
        stoch_summary = {
            "multi_sample_std": {
                "mean": float(np.mean(stoch_metrics["multi_sample_std"])),
                "std": float(np.std(stoch_metrics["multi_sample_std"])),
            },
            "energy_dispersion_ratio": {
                "mean": float(np.mean(stoch_metrics["energy_dispersion_ratio"])),
                "std": float(np.std(stoch_metrics["energy_dispersion_ratio"])),
            },
            "rank_histogram_counts": stoch_metrics["rank_histogram_counts"].tolist(),
        }

    print_table(per_sample, dist_metrics)
    plot_results(
        per_sample,
        dist_metrics,
        true_2d,
        rec_2d_dict,
        stoch_metrics,
        channel_positions,
        ns_per_bin,
        out_dir,
    )
    summary = {
        "n_events": n_events,
        "dataset": "shared_ms_cache",
        "per_sample_metrics": metric_summary(per_sample),
        "distribution_metrics": dist_metrics,
    }
    if stoch_summary is not None:
        summary["diffae_light_stochasticity"] = stoch_summary
    write_json(os.path.join(out_dir, "metrics_summary.json"), summary)
    return summary


def run_rq_analysis(
    models: Sequence[LoadedLightModel],
    shared_store: SharedSampleStore,
    out_dir: str,
    n_samples: int,
    n_examples: int,
    seed: int,
) -> Dict[str, Any]:
    ensure_dir(out_dir)
    raw = load_shared_raw(shared_store, n_samples=n_samples)[:, :, 0]
    recon_arrays = {model.display_name: load_shared_reconstruction(shared_store, model.key, n_samples=n_samples) for model in models}
    rq_true = collect_rqs(raw, shared_store.n_channels, shared_store.n_time_points)
    rq_models = {name: collect_rqs(arr, shared_store.n_channels, shared_store.n_time_points) for name, arr in recon_arrays.items()}

    plot_rq_comparison(rq_true, rq_models, out_dir)
    for model in models:
        plot_distributions(
            rq_true,
            rq_models[model.display_name],
            label_true="Raw",
            label_gen=model.display_name,
            color_true=COLORS["truth"],
            color_gen=model.color,
            output_path=os.path.join(out_dir, f"{model.key}_rq_distributions.png"),
            title=f"Raw vs {model.display_name} RQ distributions",
        )

    plot_example_reconstructions_generic(
        raw=raw,
        recs=recon_arrays,
        n_channels=shared_store.n_channels,
        n_time=shared_store.n_time_points,
        output_dir=out_dir,
        n_examples=n_examples,
        seed=seed,
    )

    summary: Dict[str, Any] = {"n_samples": n_samples, "dataset": "shared_ms_cache", "models": {}}
    for model_name, rq_pred in rq_models.items():
        model_summary: Dict[str, Dict[str, float]] = {}
        for rq_name, true_vals in rq_true.items():
            pred_vals = rq_pred[rq_name]
            mask = np.isfinite(true_vals) & np.isfinite(pred_vals)
            if np.sum(mask) < 2:
                corr = float("nan")
                mae = float("nan")
            else:
                corr = float(pearsonr(true_vals[mask], pred_vals[mask]).statistic)
                mae = float(np.mean(np.abs(pred_vals[mask] - true_vals[mask])))
            model_summary[rq_name] = {"pearson_r": corr, "mae": mae}
        summary["models"][model_name] = model_summary

    write_json(os.path.join(out_dir, "rq_summary.json"), summary)
    return summary


def run_latent_delta_mu(
    models: Sequence[LoadedLightModel],
    shared_store: SharedSampleStore,
    out_dir: str,
    n_samples: int,
    batch_size: int,
    method: str,
    point_size: float,
    knn_k: int,
    seed: int,
) -> Dict[str, Any]:
    from plot_umap import REDUCERS

    ensure_dir(out_dir)
    panels = []
    reducer = REDUCERS[method]
    raw = load_shared_raw(shared_store, n_samples=n_samples)
    delta_mu = load_shared_delta_mu(shared_store, n_samples=n_samples)
    for model in models:
        latents = encode_raw_flat_batch(model, raw, batch_size=batch_size)
        emb = reducer(latents, seed=seed)
        panels.append(
            {
                "name": f"{model.display_name} (z={model.latent_dim})",
                "latents": latents,
                "embedding": emb,
                "smooth_values": delta_mu,
                "color_values": np.abs(delta_mu),
            }
        )
    stats = plot_continuous_embedding_panels(
        panels=panels,
        output_path=os.path.join(out_dir, f"latent_delta_mu_{method}.png"),
        title=f"{method_label(method)} of latent space colored by |delta_mu|",
        value_label=r"|delta_mu| (ns)",
        cmap="viridis",
        point_size=point_size,
        knn_k=knn_k,
    )
    summary = {"n_samples": n_samples, "dataset": "shared_ms_cache", "method": method, "models": stats}
    write_json(os.path.join(out_dir, "summary.json"), summary)
    return summary


def run_full_pmt_reconstruction_examples(
    models: Sequence[LoadedLightModel],
    shared_store: SharedSampleStore,
    out_dir: str,
    n_examples: int,
    seed: int,
) -> Dict[str, Any]:
    ensure_dir(out_dir)
    model_map = {model.display_name: model for model in models}
    if "AE" not in model_map or "DiffAE" not in model_map:
        raise RuntimeError("Full-PMT reconstruction comparison requires both AE and DiffAE models.")

    raw = load_shared_raw(shared_store)[:, :, 0]
    ae = load_shared_reconstruction(shared_store, "ae_light")
    diffae = load_shared_reconstruction(shared_store, "diffae_light")
    subset_xy = np.asarray(models[0].ctx.loader.channel_positions, dtype=np.float32)
    full_xy = load_pmt_positions(FULL_PMT_XY_PATH)
    example_indices_arr = select_nonzero_example_indices(raw, n_examples=n_examples, seed=seed)
    example_indices = [int(i) for i in example_indices_arr]

    plot_full_pmt_reconstruction_triptych(
        raw_flat=raw,
        diffae_flat=diffae,
        ae_flat=ae,
        subset_xy=subset_xy,
        full_xy=full_xy,
        n_channels=shared_store.n_channels,
        n_time=shared_store.n_time_points,
        output_dir=out_dir,
        indices=example_indices_arr,
    )
    plot_full_pmt_reconstruction_3d_triptych(
        raw_flat=raw,
        diffae_flat=diffae,
        ae_flat=ae,
        subset_xy=subset_xy,
        full_xy=full_xy,
        n_channels=shared_store.n_channels,
        n_time=shared_store.n_time_points,
        output_dir=out_dir,
        indices=example_indices_arr,
        ns_per_bin=default_config.ms_data.ns_per_bin,
    )

    summary = {
        "n_examples": int(len(example_indices)),
        "full_pmt_xy_path": os.path.abspath(FULL_PMT_XY_PATH),
        "example_indices": example_indices,
        "output_patterns": {
            "xy_full_png": os.path.abspath(os.path.join(out_dir, "event_####_xy_full.png")),
            "three_d_full_png": os.path.abspath(os.path.join(out_dir, "event_####_3d_full.png")),
            "three_d_full_pdf": os.path.abspath(os.path.join(out_dir, "event_####_3d_full.pdf")),
        },
    }
    write_json(os.path.join(out_dir, "summary.json"), summary)
    return summary


def run_latent_roughness(
    models: Sequence[LoadedLightModel],
    shared_store: SharedSampleStore,
    out_dir: str,
    n_samples: int,
    batch_size: int,
    method: str,
    point_size: float,
    knn_k: int,
    seed: int,
) -> Dict[str, Any]:
    from plot_umap import REDUCERS

    ensure_dir(out_dir)
    reducer = REDUCERS[method]
    panels = []
    raw = load_shared_raw(shared_store, n_samples=n_samples)
    rough = waveform_roughness(raw, shared_store.n_channels, shared_store.n_time_points)
    for model in models:
        latents = encode_raw_flat_batch(model, raw, batch_size=batch_size)
        emb = reducer(latents, seed=seed)
        panels.append(
            {
                "name": f"{model.display_name} (z={model.latent_dim})",
                "latents": latents,
                "embedding": emb,
                "smooth_values": rough,
                "color_values": rough,
            }
        )
    stats = plot_continuous_embedding_panels(
        panels=panels,
        output_path=os.path.join(out_dir, f"latent_roughness_{method}.png"),
        title=f"{method_label(method)} of latent space colored by waveform roughness",
        value_label=r"Waveform roughness (mean |dz|)",
        cmap="magma",
        point_size=point_size,
        knn_k=knn_k,
    )
    summary = {"n_samples": n_samples, "dataset": "shared_ms_cache", "method": method, "models": stats}
    write_json(os.path.join(out_dir, "summary.json"), summary)
    return summary


def run_latent_lopsided(
    models: Sequence[LoadedLightModel],
    shared_store: SharedSampleStore,
    out_dir: str,
    n_samples: int,
    batch_size: int,
    method: str,
    point_size: float,
    seed: int,
    frac: float,
    sigma: float,
) -> Dict[str, Any]:
    from plot_umap import REDUCERS

    ensure_dir(out_dir)
    reducer = REDUCERS[method]
    rng = np.random.default_rng(seed)
    panels = []
    raw = load_shared_raw(shared_store, n_samples=n_samples)
    lop_batch, sides = make_lopsided_batch(raw, frac=frac, sigma=sigma, rng=rng)
    for model in models:
        latents = encode_raw_flat_batch(model, lop_batch, batch_size=batch_size)
        emb = reducer(latents, seed=seed)
        panels.append(
            {
                "name": f"{model.display_name} (z={model.latent_dim})",
                "embedding": emb,
                "sides": sides,
            }
        )
    counts = plot_lopsided_embedding_panels(
        panels=panels,
        output_path=os.path.join(out_dir, f"latent_lopsided_{method}.png"),
        title=f"{method_label(method)} of latent space under lopsided augmentation",
        point_size=point_size,
    )
    summary = {
        "n_samples": n_samples,
        "dataset": "shared_ms_cache",
        "method": method,
        "lopsided_frac": frac,
        "lopsided_sigma": sigma,
        "models": counts,
    }
    write_json(os.path.join(out_dir, "summary.json"), summary)
    return summary


def run_latent_sampling(
    models: Sequence[LoadedLightModel],
    shared_store: SharedSampleStore,
    out_dir: str,
    priors: Sequence[str],
    n_samples: int,
    prior_samples: int,
    batch_size: int,
    latent_temperature: float,
    seed: int,
    pbar: bool = False,
) -> Dict[str, Any]:
    ensure_dir(out_dir)
    diffae_model = next((model for model in models if model.key == "diffae_light"), None)
    if diffae_model is None:
        raise RuntimeError("DiffAE is required for latent sampling.")

    priors = list(dict.fromkeys(priors))
    ctx = diffae_model.ctx
    channel_positions = ctx.loader.channel_positions
    summary: Dict[str, Any] = {
        "model": diffae_model.display_name,
        "latent_dim": diffae_model.latent_dim,
        "n_generated": n_samples,
        "temperature": latent_temperature,
        "priors": {},
    }

    for idx, prior in enumerate(priors):
        prior_dir = ensure_dir(os.path.join(out_dir, f"{prior}_prior"))
        prior_mean, prior_std, fitted_latents, prior_source = fit_diffae_latent_prior(
            diffae_model,
            shared_store,
            prior=prior,
            prior_samples=prior_samples,
            batch_size=batch_size,
        )
        sample_seed = seed + idx
        sampled_latents = sample_latent_vectors(
            n_samples=n_samples,
            mean=prior_mean,
            std=prior_std,
            temperature=latent_temperature,
            seed=sample_seed,
            device=ctx.device,
        )

        torch.manual_seed(sample_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sample_seed)

        samples = sample_from_latent_diffae_light(
            ctx,
            sampled_latents,
            decoder=ctx.decoder,
            pbar=pbar,
        )
        samples_denorm = np.clip(ctx.data_stats.denormalize(samples.cpu().numpy()), 0.0, None)
        flat_waveforms = samples_denorm[:, 0, :]
        waveforms = reshape_flat_waveforms(flat_waveforms, ctx.n_channels, ctx.n_time_points)

        figure_path = os.path.join(prior_dir, "diffae_latent_samples.png")
        array_path = os.path.join(prior_dir, "diffae_latent_samples.npz")
        title = f"DiffAE latent samples ({prior} prior, z={diffae_model.latent_dim})"
        save_latent_sampling_visualization(waveforms, channel_positions, figure_path, title)
        np.savez_compressed(
            array_path,
            waveforms=waveforms.astype(np.float32),
            waveforms_flat=flat_waveforms.astype(np.float32),
            latents=sampled_latents.cpu().numpy().astype(np.float32),
            latent_prior_mean=prior_mean.astype(np.float32),
            latent_prior_std=prior_std.astype(np.float32),
            fitted_latents=fitted_latents.astype(np.float32),
            seed=np.int64(sample_seed),
            latent_temperature=np.float32(latent_temperature),
        )

        generated_total_charge = waveforms.sum(axis=(1, 2))
        generated_peak = waveforms.sum(axis=1).max(axis=1)
        summary["priors"][prior] = {
            "prior_source": prior_source,
            "prior_fit_samples": int(fitted_latents.shape[0]),
            "figure_path": os.path.abspath(figure_path),
            "array_path": os.path.abspath(array_path),
            "latent_mean_l2": float(np.linalg.norm(prior_mean)),
            "latent_std_mean": float(np.mean(prior_std)),
            "generated_total_charge_mean": float(np.mean(generated_total_charge)),
            "generated_total_charge_std": float(np.std(generated_total_charge)),
            "generated_peak_mean": float(np.mean(generated_peak)),
            "generated_peak_std": float(np.std(generated_peak)),
        }

    write_json(os.path.join(out_dir, "summary.json"), summary)
    return summary


def run_node_distribution_stochasticity(
    models: Sequence[LoadedLightModel],
    shared_store: SharedSampleStore,
    out_dir: str,
    node_chunk_size: int = 128,
) -> Dict[str, Any]:
    ensure_dir(out_dir)
    n_nodes = shared_store.n_nodes
    n_channels = shared_store.n_channels
    n_time = shared_store.n_time_points

    truth_mean = np.zeros(n_nodes, dtype=np.float32)
    truth_std = np.zeros(n_nodes, dtype=np.float32)
    model_stats: Dict[str, Dict[str, np.ndarray]] = {
        model.display_name: {
            "mean": np.zeros(n_nodes, dtype=np.float32),
            "std": np.zeros(n_nodes, dtype=np.float32),
            "mean_bias": np.zeros(n_nodes, dtype=np.float32),
            "residual_std": np.zeros(n_nodes, dtype=np.float32),
        }
        for model in models
    }
    conditional_std: Dict[str, np.ndarray] = {
        model.display_name: np.zeros(n_nodes, dtype=np.float32) for model in models
    }

    with h5py.File(shared_store.path, "r") as f:
        raw_ds = f["raw"]
        diff_samples_ds = f["diffae_light_samples"]
        for start in range(0, n_nodes, node_chunk_size):
            end = min(start + node_chunk_size, n_nodes)
            raw_block = np.asarray(raw_ds[:, start:end, 0], dtype=np.float32)
            truth_mean[start:end] = raw_block.mean(axis=0)
            truth_std[start:end] = raw_block.std(axis=0)

            for model in models:
                pred_block = np.asarray(f[model.key][:, start:end], dtype=np.float32)
                model_stats[model.display_name]["mean"][start:end] = pred_block.mean(axis=0)
                model_stats[model.display_name]["std"][start:end] = pred_block.std(axis=0)
                model_stats[model.display_name]["mean_bias"][start:end] = pred_block.mean(axis=0) - truth_mean[start:end]
                model_stats[model.display_name]["residual_std"][start:end] = (pred_block - raw_block).std(axis=0)

            diff_block = np.asarray(diff_samples_ds[:, :, start:end], dtype=np.float32)
            conditional_std["DiffAE"][start:end] = diff_block.std(axis=0).mean(axis=0)
            conditional_std["AE"][start:end] = 0.0

    abs_bias_delta = np.abs(model_stats["DiffAE"]["mean_bias"]) - np.abs(model_stats["AE"]["mean_bias"])
    residual_std_delta = model_stats["DiffAE"]["residual_std"] - model_stats["AE"]["residual_std"]
    conditional_std_delta = conditional_std["DiffAE"] - conditional_std["AE"]

    plot_node_metric_triptych(
        panels=[
            ("Truth mean", truth_mean),
            ("AE mean", model_stats["AE"]["mean"]),
            ("DiffAE mean", model_stats["DiffAE"]["mean"]),
        ],
        n_channels=n_channels,
        n_time=n_time,
        output_path=os.path.join(out_dir, "node_mean_maps.png"),
        suptitle="Per-node output mean across cached samples",
        colorbar_label="Mean amplitude",
        cmap="inferno",
    )

    plot_node_metric_triptych(
        panels=[
            ("Truth std", truth_std),
            ("AE std", model_stats["AE"]["std"]),
            ("DiffAE std", model_stats["DiffAE"]["std"]),
        ],
        n_channels=n_channels,
        n_time=n_time,
        output_path=os.path.join(out_dir, "node_std_maps.png"),
        suptitle="Per-node output standard deviation across cached samples",
        colorbar_label="Std. deviation",
        cmap="magma",
    )

    plot_node_metric_triptych(
        panels=[
            ("AE mean bias", model_stats["AE"]["mean_bias"]),
            ("DiffAE mean bias", model_stats["DiffAE"]["mean_bias"]),
            ("|DiffAE|-|AE| bias", abs_bias_delta),
        ],
        n_channels=n_channels,
        n_time=n_time,
        output_path=os.path.join(out_dir, "node_mean_bias_maps.png"),
        suptitle="Per-node distribution mean bias relative to truth",
        colorbar_label="Bias",
        cmap="coolwarm",
        center_zero=True,
    )

    plot_node_metric_triptych(
        panels=[
            ("AE residual std", model_stats["AE"]["residual_std"]),
            ("DiffAE residual std", model_stats["DiffAE"]["residual_std"]),
            ("DiffAE-AE residual std", residual_std_delta),
        ],
        n_channels=n_channels,
        n_time=n_time,
        output_path=os.path.join(out_dir, "node_residual_std_maps.png"),
        suptitle="Per-node reconstruction residual dispersion",
        colorbar_label="Residual std. deviation",
        cmap="coolwarm",
        center_zero=True,
    )

    plot_node_metric_triptych(
        panels=[
            ("AE conditional std", conditional_std["AE"]),
            ("DiffAE conditional std", conditional_std["DiffAE"]),
            ("DiffAE-AE conditional std", conditional_std_delta),
        ],
        n_channels=n_channels,
        n_time=n_time,
        output_path=os.path.join(out_dir, "node_stochasticity_maps.png"),
        suptitle="Per-node conditional sample dispersion",
        colorbar_label="Conditional std. deviation",
        cmap="coolwarm",
        center_zero=True,
    )

    plot_node_metric_profiles(
        metric_by_model={
            "Truth": truth_std,
            "AE": model_stats["AE"]["std"],
            "DiffAE": model_stats["DiffAE"]["std"],
        },
        n_channels=n_channels,
        n_time=n_time,
        output_path=os.path.join(out_dir, "node_std_temporal_profiles.png"),
        ylabel="Mean std across channels",
        title="Temporal profile of nodewise marginal dispersion",
    )

    plot_node_metric_profiles(
        metric_by_model={
            "AE": model_stats["AE"]["residual_std"],
            "DiffAE": model_stats["DiffAE"]["residual_std"],
        },
        n_channels=n_channels,
        n_time=n_time,
        output_path=os.path.join(out_dir, "node_residual_std_temporal_profiles.png"),
        ylabel="Mean residual std across channels",
        title="Temporal profile of nodewise residual dispersion",
    )

    plot_node_metric_profiles(
        metric_by_model={
            "AE": conditional_std["AE"],
            "DiffAE": conditional_std["DiffAE"],
        },
        n_channels=n_channels,
        n_time=n_time,
        output_path=os.path.join(out_dir, "node_stochasticity_temporal_profiles.png"),
        ylabel="Mean conditional std across channels",
        title="Temporal profile of nodewise conditional dispersion",
    )

    summary = {
        "n_samples": shared_store.n_samples,
        "dataset": "shared_ms_cache",
        "node_metrics": {
            "truth_std_mean": float(np.mean(truth_std)),
            "ae_residual_std_mean": float(np.mean(model_stats["AE"]["residual_std"])),
            "diffae_residual_std_mean": float(np.mean(model_stats["DiffAE"]["residual_std"])),
            "ae_conditional_std_mean": float(np.mean(conditional_std["AE"])),
            "diffae_conditional_std_mean": float(np.mean(conditional_std["DiffAE"])),
            "mean_abs_bias_advantage": float(np.mean(abs_bias_delta)),
        },
    }
    write_json(os.path.join(out_dir, "summary.json"), summary)
    return summary


def run_ss_ms_probe(
    models: Sequence[LoadedLightModel],
    out_dir: str,
    n_samples: int,
    probe_epochs: int,
    method: str,
    point_size: float,
    seed: int,
    ss_shift: int,
    delta_max: int,
) -> Dict[str, Any]:
    from probe_ss_ms import generate_ms_coadded, generate_ss_coadded, plot_probe_bar, plot_scatter, reduce, train_probe

    ensure_dir(out_dir)
    ref_ctx = models[0].ctx
    ss_loader = get_ss_loader(ref_ctx.loader)
    rng = np.random.default_rng(seed)

    ss_events = generate_ss_coadded(ss_loader, n_samples, ss_shift, rng)
    ms_events = generate_ms_coadded(ss_loader, n_samples, delta_max, rng)
    all_events = np.concatenate([ss_events, ms_events], axis=0)
    labels = np.array([0] * n_samples + [1] * n_samples, dtype=np.int64)

    perm = np.random.default_rng(seed + 17).permutation(2 * n_samples)
    all_events = all_events[perm]
    labels = labels[perm]
    split = (2 * n_samples) * 3 // 4
    y_tr = torch.tensor(labels[:split], dtype=torch.long)
    y_te = torch.tensor(labels[split:], dtype=torch.long)

    scatter_panels = []
    probe_results = []
    summary = {"n_samples_per_class": n_samples, "probe_epochs": probe_epochs, "method": method, "models": {}}
    for model in models:
        z_all = encode_raw_flat_batch(model, all_events, batch_size=128)
        z_tr = torch.from_numpy(z_all[:split].astype(np.float32))
        z_te = torch.from_numpy(z_all[split:].astype(np.float32))
        tr_acc, te_acc = train_probe(z_tr, y_tr, z_te, y_te, epochs=probe_epochs)
        emb = reduce(z_all, method, seed=seed)
        scatter_panels.append((f"{model.display_name} (z={model.latent_dim})", emb, labels, te_acc))
        probe_results.append((f"{model.display_name}", tr_acc, te_acc))
        summary["models"][model.display_name] = {"train_acc": float(tr_acc), "test_acc": float(te_acc)}

    plot_scatter(
        scatter_panels,
        ss_shift,
        delta_max,
        method_label(method),
        os.path.join(out_dir, f"ss_ms_scatter_{method}.png"),
        point_size,
    )
    plot_probe_bar(probe_results, os.path.join(out_dir, "ss_ms_probe_bar.png"))
    write_json(os.path.join(out_dir, "summary.json"), summary)
    return summary


def run_anomaly_probe(
    models: Sequence[LoadedLightModel],
    out_dir: str,
    n_events: int,
    batch_size: int,
    perplexity: int,
    seed: int,
) -> Dict[str, Any]:
    from anomaly_probe import (
        ANOMALY_TYPES,
        anomaly_heatmap,
        anomaly_scores,
        fit_reduce,
        make_anomaly,
        plot_anomaly_examples,
        print_score_table,
        scatter_plot,
        to_flat,
    )

    ensure_dir(out_dir)
    rng = np.random.default_rng(seed)
    with h5py.File(default_config.paths.tritium_h5, "r") as f:
        total = int(f["waveforms"].shape[0])
        idx = np.sort(rng.choice(total, size=n_events, replace=False))
        wf_real = np.asarray(f["waveforms"][idx], dtype=np.float32)
        xc_real = np.asarray(f["xc"][idx], dtype=np.float32)
        yc_real = np.asarray(f["yc"][idx], dtype=np.float32)

    ref_ctx = models[0].ctx
    channel_pos = ref_ctx.loader.channel_positions
    n_events_loaded, n_channels, n_time = wf_real.shape

    wf_flat_all = to_flat(wf_real)
    integrals = wf_flat_all.sum(axis=1)
    med_int = float(np.median(integrals))
    base_idx = int(np.argmin(np.abs(integrals - med_int)))
    wf_base = wf_real[base_idx]
    xc_base = float(xc_real[base_idx])
    yc_base = float(yc_real[base_idx])
    z_base = wf_to_z_profile(wf_flat_all[base_idx], n_channels, n_time)
    rqs_base = compute_rqs(z_base)
    if rqs_base is None:
        rqs_base = {
            "peak_amplitude": 0.0,
            "peak_time": 0.0,
            "total_integral": 0.0,
            "rise_time": 0.0,
            "fall_time": 0.0,
            "fwhm": 0.0,
            "width_10_90": 0.0,
            "std_dev": 0.0,
        }

    proto_types = [entry[0] for entry in ANOMALY_TYPES]
    wf_protos = np.stack(
        [make_anomaly(wf_base, channel_pos, atype, xc=xc_base, yc=yc_base, rqs=rqs_base) for atype in proto_types],
        axis=0,
    )
    proto_wfs = {atype: wf_protos[i] for i, atype in enumerate(proto_types)}
    plot_anomaly_examples(wf_base, proto_wfs, channel_pos, out_dir)

    rq_real_dict = collect_rqs(wf_flat_all, n_channels, n_time)
    rq_proto_dict = collect_rqs(to_flat(wf_protos), n_channels, n_time)
    rq_names = list(rq_real_dict.keys())
    rq_real = np.column_stack([rq_real_dict[name] for name in rq_names]).astype(np.float32)
    rq_proto = np.column_stack([rq_proto_dict[name] for name in rq_names]).astype(np.float32)

    for col in range(rq_real.shape[1]):
        bad_real = ~np.isfinite(rq_real[:, col])
        if bad_real.any():
            rq_real[bad_real, col] = float(np.nanmedian(rq_real[:, col]))
        bad_proto = ~np.isfinite(rq_proto[:, col])
        if bad_proto.any():
            rq_proto[bad_proto, col] = float(np.nanmedian(rq_real[:, col]))

    rq_real_2d, rq_proto_2d = fit_reduce(rq_real, rq_proto, perplexity=perplexity, seed=seed)
    scatter_plot(
        Z_real_2d=rq_real_2d,
        Z_protos_2d=rq_proto_2d,
        title="RQ metrics",
        subtitle=f"t-SNE of pulse-shape features ({n_events_loaded} real events)",
        path=os.path.join(out_dir, "scatter_rq.png"),
    )

    scores_map: Dict[str, Optional[np.ndarray]] = {}
    d_scores_map: Dict[str, Optional[np.ndarray]] = {}
    score_summary: Dict[str, Any] = {"RQ metrics": {}}

    rq_std = rq_real.std(axis=0, keepdims=True) + 1e-8
    d_rq, pct_rq = anomaly_scores((rq_real - rq_real.mean(axis=0)) / rq_std, (rq_proto - rq_real.mean(axis=0)) / rq_std)
    scores_map["RQ metrics"] = pct_rq
    d_scores_map["RQ metrics"] = d_rq
    score_summary["RQ metrics"] = {label: float(score) for label, score in zip([t[1] for t in ANOMALY_TYPES], pct_rq)}

    for model in models:
        z_real = encode_ct_batch(model, wf_real, batch_size=batch_size)
        z_proto = encode_ct_batch(model, wf_protos, batch_size=batch_size)
        z_real_2d, z_proto_2d = fit_reduce(z_real, z_proto, perplexity=perplexity, seed=seed)
        scatter_plot(
            Z_real_2d=z_real_2d,
            Z_protos_2d=z_proto_2d,
            title=f"{model.display_name} (z={model.latent_dim})",
            subtitle=f"t-SNE of encoder latents ({n_events_loaded} real events)",
            path=os.path.join(out_dir, f"scatter_{model.key}.png"),
        )
        d_model, pct_model = anomaly_scores(z_real, z_proto)
        scores_map[model.display_name] = pct_model
        d_scores_map[model.display_name] = d_model
        score_summary[model.display_name] = {label: float(score) for label, score in zip([t[1] for t in ANOMALY_TYPES], pct_model)}

    print_score_table(scores_map, d_scores_map)
    anomaly_heatmap(scores_map, [t[1] for t in ANOMALY_TYPES], os.path.join(out_dir, "anomaly_scores.png"))
    summary = {
        "n_events": n_events_loaded,
        "perplexity": perplexity,
        "scores": score_summary,
        "example_plots": {
            "spatial": os.path.abspath(os.path.join(out_dir, "anomaly_examples_spatial.png")),
            "temporal": os.path.abspath(os.path.join(out_dir, "anomaly_examples_temporal.png")),
        },
    }
    write_json(os.path.join(out_dir, "summary.json"), summary)
    return summary


def skipped_legacy_experiments() -> Dict[str, str]:
    return {
        "diagnose/diagnose_lopsidedness.py": "Requires a light-model equivalent of sample_diffae_partial, which does not exist in diffae_light.py.",
        "diagnose/probe_lopsidedness.py": "Contains legacy DiffAE-specific conditioning and decoder-internal analyses with no matching light-model hooks.",
        "diagnose/test_diffae_conditioning.py": "Coupled to legacy DiffAE internals and direct encoder/decoder call conventions that differ from diffae_light.",
        "diagnose/diagnose_dead_dims.py": "Targets the legacy FiLM-based decoder path and decoder Jacobians that are not present in diffae_light.",
        "view_events.py": "Covered by the reconstruction example panels emitted by the RQ and reconstruction evaluations.",
    }


def print_failure_banner(name: str, out_dir: str, error: Exception, elapsed_sec: float) -> None:
    banner = "!" * 72
    print(
        f"\n{banner}\n"
        f"EXPERIMENT FAILED: {name}\n"
        f"Elapsed: {elapsed_sec:.2f}s\n"
        f"Output dir: {os.path.abspath(out_dir)}\n"
        f"Error: {error}\n"
        "Continuing with remaining experiments.\n"
        f"{banner}",
        flush=True,
    )


def run_experiment(
    summary: Dict[str, Any],
    summary_path: str,
    name: str,
    out_dir: str,
    fn,
) -> None:
    print(f"\n{'=' * 72}\nRunning: {name}\n{'=' * 72}")
    start = time.time()
    try:
        result = fn()
        summary["experiments"][name] = {
            "status": "ok",
            "elapsed_sec": float(time.time() - start),
            "output_dir": os.path.abspath(out_dir),
            "summary": result,
        }
    except Exception as exc:
        elapsed_sec = float(time.time() - start)
        print_failure_banner(name, out_dir, exc, elapsed_sec)
        traceback.print_exc(file=sys.stdout)
        summary["experiments"][name] = {
            "status": "error",
            "elapsed_sec": elapsed_sec,
            "output_dir": os.path.abspath(out_dir),
            "error": str(exc),
        }
    write_json(summary_path, summary)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference-only evaluations for AE Light and DiffAE Light.")
    parser.add_argument("--output-dir", type=str, default="light_eval")
    parser.add_argument("--regenerate", action="store_true", help="Regenerate the shared MS sample cache before running experiments.")
    parser.add_argument("--ae-latent-dim", type=int, default=default_config.encoder.latent_dim)
    parser.add_argument("--diffae-latent-dim", type=int, default=default_config.encoder.latent_dim)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shared-samples", type=int, default=8192, help="Number of cached MS events to generate once up front.")
    parser.add_argument("--recon-events", type=int, default=2048)
    parser.add_argument("--recon-compare-examples", type=int, default=6, help="Full-PMT reconstruction comparison figures to save.")
    parser.add_argument("--diffae-samples", type=int, default=4, help="Independent DiffAE Light samples per event for stochasticity metrics.")
    parser.add_argument("--rq-samples", type=int, default=500)
    parser.add_argument("--rq-examples", type=int, default=6)
    parser.add_argument("--latent-samples", type=int, default=5000)
    parser.add_argument("--latent-method", choices=["pca", "umap", "tsne"], default="umap")
    parser.add_argument("--latent-point-size", type=float, default=3.0)
    parser.add_argument("--latent-sample-events", type=int, default=6, help="Random DiffAE samples to generate per latent prior.")
    parser.add_argument("--latent-sample-priors", nargs="+", choices=["empirical", "standard"], default=["empirical", "standard"])
    parser.add_argument("--latent-prior-samples", type=int, default=512, help="Shared or saved latents used to fit the empirical latent prior.")
    parser.add_argument("--latent-sample-temperature", type=float, default=1.0)
    parser.add_argument("--latent-sample-pbar", action="store_true")
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--lopsided-frac", type=float, default=0.5)
    parser.add_argument("--lopsided-sigma", type=float, default=10.0)
    parser.add_argument("--ss-ms-samples", type=int, default=3000, help="Events per class for SS-vs-MS probe.")
    parser.add_argument("--ss-ms-probe-epochs", type=int, default=1000)
    parser.add_argument("--ss-shift", type=int, default=0)
    parser.add_argument("--delta-max", type=int, default=50)
    parser.add_argument("--anomaly-events", type=int, default=500)
    parser.add_argument("--anomaly-perplexity", type=int, default=30)
    parser.add_argument("--node-chunk-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-recon", action="store_true")
    parser.add_argument("--skip-recon-compare", action="store_true")
    parser.add_argument("--skip-rq", action="store_true")
    parser.add_argument("--skip-latent", action="store_true")
    parser.add_argument("--skip-latent-sampling", action="store_true")
    parser.add_argument("--skip-node", action="store_true")
    parser.add_argument("--skip-ss-ms", action="store_true")
    parser.add_argument("--skip-anomaly", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_root = ensure_dir(os.path.join(ROOT, args.output_dir))
    summary_path = os.path.join(output_root, "summary.json")

    models = [
        load_ae_light(args.ae_latent_dim),
        load_diffae_light(args.diffae_latent_dim),
    ]

    print("Loaded light checkpoints:")
    for model in models:
        print(f"  {model.display_name}: {model.checkpoint_path}  (epoch {model.epoch}, z={model.latent_dim})")

    shared_store = create_shared_sample_store(
        models=models,
        output_root=output_root,
        n_samples=args.shared_samples,
        batch_size=args.batch_size,
        diffae_samples=args.diffae_samples,
        regenerate=args.regenerate,
    )

    summary: Dict[str, Any] = {
        "output_root": os.path.abspath(output_root),
        "seed": args.seed,
        "shared_sample_store": {
            "path": os.path.abspath(shared_store.path),
            "n_samples": shared_store.n_samples,
            "n_nodes": shared_store.n_nodes,
            "n_channels": shared_store.n_channels,
            "n_time_points": shared_store.n_time_points,
            "diffae_samples": shared_store.diffae_samples,
        },
        "models": {
            model.key: {
                "display_name": model.display_name,
                "latent_dim": model.latent_dim,
                "checkpoint_dir": os.path.abspath(model.checkpoint_dir),
                "checkpoint_path": os.path.abspath(model.checkpoint_path),
                "epoch": model.epoch,
            }
            for model in models
        },
        "experiments": {},
        "skipped_legacy_experiments": skipped_legacy_experiments(),
    }
    write_json(summary_path, summary)

    if not args.skip_recon:
        out_dir = os.path.join(output_root, "reconstruction_eval")
        run_experiment(
            summary,
            summary_path,
            "reconstruction_eval",
            out_dir,
            lambda: run_reconstruction_eval(
                models=models,
                shared_store=shared_store,
                out_dir=out_dir,
                n_events=args.recon_events,
            ),
        )

    if not args.skip_recon_compare:
        out_dir = os.path.join(output_root, "reconstruction_examples_full_pmt")
        run_experiment(
            summary,
            summary_path,
            "reconstruction_examples_full_pmt",
            out_dir,
            lambda: run_full_pmt_reconstruction_examples(
                models=models,
                shared_store=shared_store,
                out_dir=out_dir,
                n_examples=args.recon_compare_examples,
                seed=args.seed,
            ),
        )

    if not args.skip_rq:
        out_dir = os.path.join(output_root, "rq_analysis")
        run_experiment(
            summary,
            summary_path,
            "rq_analysis",
            out_dir,
            lambda: run_rq_analysis(
                models=models,
                shared_store=shared_store,
                out_dir=out_dir,
                n_samples=args.rq_samples,
                n_examples=args.rq_examples,
                seed=args.seed,
            ),
        )

    if not args.skip_latent:
        delta_dir = os.path.join(output_root, "latent_delta_mu")
        run_experiment(
            summary,
            summary_path,
            "latent_delta_mu",
            delta_dir,
            lambda: run_latent_delta_mu(
                models=models,
                shared_store=shared_store,
                out_dir=delta_dir,
                n_samples=args.latent_samples,
                batch_size=args.batch_size,
                method=args.latent_method,
                point_size=args.latent_point_size,
                knn_k=args.knn_k,
                seed=args.seed,
            ),
        )

        roughness_dir = os.path.join(output_root, "latent_roughness")
        run_experiment(
            summary,
            summary_path,
            "latent_roughness",
            roughness_dir,
            lambda: run_latent_roughness(
                models=models,
                shared_store=shared_store,
                out_dir=roughness_dir,
                n_samples=args.latent_samples,
                batch_size=args.batch_size,
                method=args.latent_method,
                point_size=args.latent_point_size,
                knn_k=args.knn_k,
                seed=args.seed,
            ),
        )

        lopsided_dir = os.path.join(output_root, "latent_lopsided")
        run_experiment(
            summary,
            summary_path,
            "latent_lopsided",
            lopsided_dir,
            lambda: run_latent_lopsided(
                models=models,
                shared_store=shared_store,
                out_dir=lopsided_dir,
                n_samples=args.latent_samples,
                batch_size=args.batch_size,
                method=args.latent_method,
                point_size=args.latent_point_size,
                seed=args.seed,
                frac=args.lopsided_frac,
                sigma=args.lopsided_sigma,
            ),
        )

    if not args.skip_latent_sampling:
        out_dir = os.path.join(output_root, "latent_sampling")
        run_experiment(
            summary,
            summary_path,
            "latent_sampling",
            out_dir,
            lambda: run_latent_sampling(
                models=models,
                shared_store=shared_store,
                out_dir=out_dir,
                priors=args.latent_sample_priors,
                n_samples=args.latent_sample_events,
                prior_samples=args.latent_prior_samples,
                batch_size=args.batch_size,
                latent_temperature=args.latent_sample_temperature,
                seed=args.seed,
                pbar=args.latent_sample_pbar,
            ),
        )

    if not args.skip_node:
        out_dir = os.path.join(output_root, "node_distribution_stochasticity")
        run_experiment(
            summary,
            summary_path,
            "node_distribution_stochasticity",
            out_dir,
            lambda: run_node_distribution_stochasticity(
                models=models,
                shared_store=shared_store,
                out_dir=out_dir,
                node_chunk_size=args.node_chunk_size,
            ),
        )

    if not args.skip_ss_ms:
        out_dir = os.path.join(output_root, "ss_ms_probe")
        run_experiment(
            summary,
            summary_path,
            "ss_ms_probe",
            out_dir,
            lambda: run_ss_ms_probe(
                models=models,
                out_dir=out_dir,
                n_samples=args.ss_ms_samples,
                probe_epochs=args.ss_ms_probe_epochs,
                method=args.latent_method,
                point_size=args.latent_point_size,
                seed=args.seed,
                ss_shift=args.ss_shift,
                delta_max=args.delta_max,
            ),
        )

    if not args.skip_anomaly:
        out_dir = os.path.join(output_root, "anomaly_probe")
        run_experiment(
            summary,
            summary_path,
            "anomaly_probe",
            out_dir,
            lambda: run_anomaly_probe(
                models=models,
                out_dir=out_dir,
                n_events=args.anomaly_events,
                batch_size=args.batch_size,
                perplexity=args.anomaly_perplexity,
                seed=args.seed,
            ),
        )

    failed_experiments = [
        (name, info)
        for name, info in summary["experiments"].items()
        if info.get("status") == "error"
    ]
    if failed_experiments:
        banner = "!" * 72
        print(f"\n{banner}", flush=True)
        print(f"COMPLETED WITH {len(failed_experiments)} FAILED EXPERIMENT(S)", flush=True)
        for name, info in failed_experiments:
            print(f"  - {name}: {info.get('error', 'unknown error')}", flush=True)
        print(f"See {summary_path} for full details.", flush=True)
        print(f"{banner}", flush=True)
    else:
        print("\nAll requested experiments completed successfully.", flush=True)

    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
