"""
SS vs MS discrimination task.

Both classes are constructed by co-adding two real SS events in the channel-summed
(single time-profile) sense:

  SS: two SS events co-added with the SAME fixed time shift for every event in the class
      (default delta=0, i.e. coincident pulses → looks like a single-scatter peak)

  MS: two SS events co-added where each event independently draws a random time shift
      from Uniform[-delta_max, +delta_max] bins (default ±50)
      → varying double-peak separation per event

The resulting waveforms are encoded with the frozen AE and DiffAE encoders, then:
  1. Dimensionality-reduced (UMAP by default) and scatter-plotted, coloured by SS/MS
  2. Linear probe trained on frozen latents to classify SS vs MS

Usage:
    python diagnose/probe_ss_ms.py
    python diagnose/probe_ss_ms.py --ss-shift 0 --delta-max 50 --method umap
    python diagnose/probe_ss_ms.py --ss-shift 15 --n-samples 4000 --method pca
"""
import argparse
import os
import sys
import copy
import h5py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import default_config, get_config
from lz_data_loader import shift_waveform_2d
from plot_style import apply_style, COLORS

apply_style()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _build_encoder_ctx(model_type: str, cfg):
    cfg = copy.deepcopy(cfg)
    if model_type == "ae":
        from ae import AEContext
        probe_ctx = AEContext.build(cfg, for_training=True, verbose=False)
        ckpt_path = probe_ctx.latest_checkpoint()
        if ckpt_path is None:
            return None, None
        chk = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        enc_sd = chk.get("encoder", {})
        for key in ("mlp.0.weight", "backbone.0.weight"):
            if key in enc_sd:
                cfg.encoder.hidden_dim = enc_sd[key].shape[0]
                break
        ctx = AEContext.build(cfg, for_training=True, verbose=False)
    else:
        from diffae import DiffAEContext
        probe_ctx = DiffAEContext.build(cfg, for_training=True, verbose=False)
        ckpt_path = probe_ctx.latest_checkpoint()
        if ckpt_path is None:
            return None, None
        chk = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        enc_sd = chk.get("encoder", {})
        for key in ("backbone.0.weight", "mlp.0.weight"):
            if key in enc_sd:
                cfg.encoder.hidden_dim = enc_sd[key].shape[0]
                break
        ctx = DiffAEContext.build(cfg, for_training=True, verbose=False)

    ctx.load_checkpoint(ckpt_path, load_optim=False)
    enc = ctx.ema_encoder if ctx.ema_encoder is not None else ctx.encoder
    enc.eval()
    print(f"  [{model_type}] checkpoint: {ckpt_path}")
    return ctx, enc


def _get_ss_loader(loader):
    """Unwrap OnlineMSBatcher → TritiumSSDataLoader if needed."""
    return loader.ss_loader if hasattr(loader, "ss_loader") else loader


@torch.no_grad()
def _encode_chunk(enc, ctx, batch_np: np.ndarray, model_type: str) -> np.ndarray:
    bs = batch_np.shape[0]
    x = torch.from_numpy(ctx.data_stats.normalize(batch_np).astype(np.float32)).to(ctx.device)
    x_flat = x.view(bs * ctx.n_nodes, 1)
    if model_type == "ae":
        z, _ = enc(x_flat, ctx.A_sparse, ctx.pos, batch_size=bs)
    else:
        z, _, _ = enc(x_flat, ctx.A_sparse, ctx.pos, batch_size=bs)
    return z.cpu().numpy()


@torch.no_grad()
def encode_all(enc, ctx, batch_np: np.ndarray, model_type: str, chunk: int = 128) -> np.ndarray:
    return np.concatenate(
        [_encode_chunk(enc, ctx, batch_np[i:i + chunk], model_type)
         for i in range(0, len(batch_np), chunk)],
        axis=0,
    )


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def _draw_raw_pairs(ss_loader, n: int, rng: np.random.Generator):
    """Draw n non-duplicate (wf1, wf2) pairs as (C, T) numpy arrays."""
    n_samples   = ss_loader.n_samples
    n_channels  = ss_loader.n_channels
    n_time      = ss_loader.n_time_points

    idx1 = rng.integers(0, n_samples, size=n)
    idx2 = rng.integers(0, n_samples, size=n)
    dup = idx1 == idx2
    while dup.any():
        idx2[dup] = rng.integers(0, n_samples, size=dup.sum())
        dup = idx1 == idx2

    all_idx = np.unique(np.concatenate([idx1, idx2]))
    with h5py.File(ss_loader.h5_file_path, "r") as f:
        wf_all = f["waveforms"][all_idx].astype(np.float32)   # (U, C, T)

    idx_map = {int(v): i for i, v in enumerate(all_idx)}
    wf1 = np.stack([wf_all[idx_map[int(i)]] for i in idx1])  # (n, C, T)
    wf2 = np.stack([wf_all[idx_map[int(i)]] for i in idx2])  # (n, C, T)
    return wf1, wf2, n_channels, n_time


def _pairs_to_flat(ms_wf: np.ndarray) -> np.ndarray:
    """(B, C, T) → (B, T*C, 1) layer-major order."""
    B, C, T = ms_wf.shape
    return np.transpose(ms_wf, (0, 2, 1)).reshape(B, T * C, 1).astype(np.float32)


def generate_ss_coadded(ss_loader, n: int, ss_shift: int, rng: np.random.Generator) -> np.ndarray:
    """n events: two SS co-added with a FIXED time shift (same for every event).

    All events in this class have the identical delta → single-scatter-like pulse shape.
    Returns (n, T*C, 1) in layer-major order.
    """
    wf1, wf2, _, _ = _draw_raw_pairs(ss_loader, n, rng)
    ms_wf = np.zeros_like(wf1)
    for b in range(n):
        ms_wf[b] = wf1[b] + shift_waveform_2d(wf2[b], ss_shift)
    return _pairs_to_flat(ms_wf)


def generate_ms_coadded(ss_loader, n: int, delta_max: int, rng: np.random.Generator) -> np.ndarray:
    """n events: two SS co-added with a DIFFERENT random shift per event from U[-delta_max, +delta_max].

    Each event has a distinct time separation → multi-scatter-like.
    Returns (n, T*C, 1) in layer-major order.
    """
    wf1, wf2, _, _ = _draw_raw_pairs(ss_loader, n, rng)
    shifts = rng.integers(-delta_max, delta_max + 1, size=n)
    ms_wf = np.zeros_like(wf1)
    for b in range(n):
        ms_wf[b] = wf1[b] + shift_waveform_2d(wf2[b], int(shifts[b]))
    return _pairs_to_flat(ms_wf)


# ---------------------------------------------------------------------------
# Linear probe
# ---------------------------------------------------------------------------

def train_probe(z_tr, y_tr, z_te, y_te, epochs: int = 1000):
    probe = nn.Linear(z_tr.shape[1], 2)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    for ep in range(epochs):
        loss = F.cross_entropy(probe(z_tr), y_tr)
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % max(1, epochs // 4) == 0:
            print(f"    epoch {ep + 1:4d}  loss={loss.item():.4f}")
    with torch.no_grad():
        tr_acc = (probe(z_tr).argmax(1) == y_tr).float().mean().item()
        te_acc = (probe(z_te).argmax(1) == y_te).float().mean().item()
    return tr_acc, te_acc


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

def reduce(latents: np.ndarray, method: str, seed: int = 42, **kw) -> np.ndarray:
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=seed).fit_transform(latents)
    elif method == "umap":
        from umap.umap_ import UMAP
        return UMAP(
            n_neighbors=kw.get("n_neighbors", 15),
            min_dist=kw.get("min_dist", 0.1),
            random_state=seed,
        ).fit_transform(latents)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(
            n_components=2, perplexity=kw.get("perplexity", 30),
            learning_rate="auto", init="pca", random_state=seed,
        ).fit_transform(latents)
    else:
        raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

SS_COLOR = COLORS["ae"]      # blue  — SS (coincident co-addition)
MS_COLOR = COLORS["diffae"]  # red   — MS (variable-shift co-addition)


def plot_scatter(panels, ss_shift, delta_max, method_label, out_path, point_size=3.0):
    """One scatter panel per model, coloured by SS (blue) vs MS (red)."""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n + 0.5, 5.5), squeeze=False)
    axes = axes[0]

    for ax, (name, emb, labels, te_acc) in zip(axes, panels):
        ss_m = labels == 0
        ms_m = labels == 1
        ax.scatter(emb[ss_m, 0], emb[ss_m, 1], c=SS_COLOR,
                   s=point_size, alpha=0.5, edgecolors="none", rasterized=True,
                   label="SS", zorder=2)
        ax.scatter(emb[ms_m, 0], emb[ms_m, 1], c=MS_COLOR,
                   s=point_size, alpha=0.5, edgecolors="none", rasterized=True,
                   label="MS", zorder=3)
        ax.set_title(
            f"{name}\nprobe acc = {te_acc:.3f}",
            fontweight="bold",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_box_aspect(1)
        ax.grid(False)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=SS_COLOR,
               markersize=7, label=f"SS  (δ = {ss_shift} bins, fixed)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=MS_COLOR,
               markersize=7, label=f"MS  (δ ~ U[−{delta_max}, +{delta_max}])"),
    ]
    axes[-1].legend(handles=legend_elements, loc="upper right", fontsize=8,
                    frameon=True, edgecolor="0.8")

    fig.suptitle(
        f"{method_label} of encoded latents — SS vs MS  "
        f"(SS δ={ss_shift}, MS δ~U[±{delta_max}])",
        fontweight="bold",
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved scatter: {out_path}")


def plot_probe_bar(results, out_path):
    """Bar chart comparing AE vs DiffAE probe accuracy."""
    models = [r[0] for r in results]
    tr_accs = [r[1] for r in results]
    te_accs = [r[2] for r in results]
    colors  = [COLORS["ae"] if "AE" in m and "Diff" not in m else COLORS["diffae"]
               for m in models]

    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(max(4, len(models) * 2), 4))
    ax.bar(x - 0.18, tr_accs, width=0.34, color=colors, alpha=0.5, label="Train")
    ax.bar(x + 0.18, te_accs, width=0.34, color=colors, alpha=0.85, label="Test")
    ax.axhline(0.5, color="#888", linestyle="--", linewidth=0.8, label="Chance")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Linear probe accuracy")
    ax.set_ylim(0.4, 1.05)
    ax.set_title("SS vs MS — linear probe accuracy", fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved probe bar: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SS vs MS discrimination: linear probe + UMAP")
    p.add_argument("--ss-shift",    type=int,   default=0,
                   help="Fixed time shift (bins) for the SS co-addition class (default: 0)")
    p.add_argument("--delta-max",   type=int,   default=50,
                   help="MS shift drawn from U[−delta_max, +delta_max] bins (default: 50)")
    p.add_argument("--n-samples",   type=int,   default=3000,
                   help="Events per class; total dataset = 2×n_samples (default: 3000)")
    p.add_argument("--method",      type=str,   default="umap",
                   choices=["pca", "umap", "tsne"],
                   help="Dimensionality reduction method (default: umap)")
    p.add_argument("--probe-epochs", type=int,  default=1000)
    p.add_argument("--output-dir",  type=str,   default="diagnose_ss_ms")
    p.add_argument("--point-size",  type=float, default=3.0)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--latent-dim",  type=int,   default=None,
                   help="Override latent dim from config")
    p.add_argument("--n-neighbors", type=int,   default=15,  help="UMAP n_neighbors")
    p.add_argument("--min-dist",    type=float, default=0.1, help="UMAP min_dist")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = get_config(latent_dim=args.latent_dim) if args.latent_dim else default_config
    method_label = {"pca": "PCA", "umap": "UMAP", "tsne": "t-SNE"}[args.method]
    os.makedirs(args.output_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # ---- Load models --------------------------------------------------------
    print("Loading AE...")
    ae_ctx,  ae_enc  = _build_encoder_ctx("ae",     cfg)
    print("Loading DiffAE...")
    dae_ctx, dae_enc = _build_encoder_ctx("diffae", cfg)

    if ae_ctx is None and dae_ctx is None:
        print("No checkpoints found. Aborting.")
        return

    ref_ctx = ae_ctx if ae_ctx is not None else dae_ctx
    ss_loader = _get_ss_loader(ref_ctx.loader)

    # ---- Generate data ------------------------------------------------------
    n = args.n_samples
    print(f"\nGenerating {n} SS events  (δ = {args.ss_shift} bins, fixed)...")
    ss_events = generate_ss_coadded(ss_loader, n, args.ss_shift, rng)

    print(f"Generating {n} MS events  (δ ~ U[−{args.delta_max}, +{args.delta_max}])...")
    ms_events = generate_ms_coadded(ss_loader, n, args.delta_max, rng)

    all_events = np.concatenate([ss_events, ms_events], axis=0)   # (2n, N, 1)
    labels     = np.array([0] * n + [1] * n, dtype=np.int64)

    perm           = np.random.RandomState(args.seed).permutation(2 * n)
    all_events_p   = all_events[perm]
    labels_p       = labels[perm]
    split          = (2 * n) * 3 // 4
    y_tr = torch.tensor(labels_p[:split], dtype=torch.long)
    y_te = torch.tensor(labels_p[split:], dtype=torch.long)

    # ---- Encode, probe, reduce per model ------------------------------------
    scatter_panels = []
    probe_results  = []

    for model_type, model_name, ctx, enc in [
        ("ae",     "AE",     ae_ctx,  ae_enc),
        ("diffae", "DiffAE", dae_ctx, dae_enc),
    ]:
        if ctx is None:
            print(f"\n  {model_name}: no checkpoint, skipping")
            continue

        print(f"\n[{model_name}] Encoding {2 * n} events...")
        z_all  = encode_all(enc, ctx, all_events_p, model_type)
        z_tr   = torch.from_numpy(z_all[:split].astype(np.float32))
        z_te   = torch.from_numpy(z_all[split:].astype(np.float32))

        print(f"[{model_name}] Training linear probe ({args.probe_epochs} epochs)...")
        tr_acc, te_acc = train_probe(z_tr, y_tr, z_te, y_te, epochs=args.probe_epochs)
        print(f"  {model_name}  train={tr_acc:.4f}  test={te_acc:.4f}  (chance=0.50)")
        probe_results.append((model_name, tr_acc, te_acc))

        print(f"[{model_name}] Running {method_label}...")
        emb = reduce(z_all, args.method, seed=args.seed,
                     n_neighbors=args.n_neighbors, min_dist=args.min_dist)
        scatter_panels.append((model_name, emb, labels_p, te_acc))

    # ---- Plots --------------------------------------------------------------
    if scatter_panels:
        out_scatter = os.path.join(
            args.output_dir,
            f"ss_ms_scatter_{args.method}_ss{args.ss_shift}_ms{args.delta_max}.png",
        )
        plot_scatter(scatter_panels, args.ss_shift, args.delta_max,
                     method_label, out_scatter, args.point_size)

    if probe_results:
        out_bar = os.path.join(args.output_dir, "ss_ms_probe_bar.png")
        plot_probe_bar(probe_results, out_bar)

    print("\n" + "=" * 60)
    print("SUMMARY — SS vs MS Linear Probe Accuracy")
    print(f"  SS: δ = {args.ss_shift} bins (fixed)")
    print(f"  MS: δ ~ U[−{args.delta_max}, +{args.delta_max}] bins")
    print("=" * 60)
    print(f"{'Model':>8s}  {'Train':>8s}  {'Test':>8s}")
    print("-" * 30)
    for name, tr, te in probe_results:
        print(f"{name:>8s}  {tr:>8.4f}  {te:>8.4f}")
    print("(chance = 0.50)")


if __name__ == "__main__":
    main()
