"""
Plot dimensionality-reduced projections of AE and DiffAE encoded latents.

Default mode: color by |delta_mu|.
--lopsided mode: generate events on-the-fly with lopsided augmentation,
                 encode them, and color by augmentation side
                 (red=left, blue=right, grey=none).

Supported methods: PCA, UMAP, t-SNE.

Usage:
    python plot_umap.py --latent-dim 64
    python plot_umap.py --latent-dim 64 --method tsne
    python plot_umap.py --latent-dim 64 --lopsided
    python plot_umap.py --latent-dim 64 --lopsided --lopsided-frac 0.3
"""
import os
import argparse

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from config import default_config, get_config

METHODS = ["pca", "umap", "tsne"]


def load_latents(h5_path: str, n_samples: int = 10000):
    with h5py.File(h5_path, "r") as f:
        total = f["latents"].shape[0]
        n = min(n_samples, total)
        idx = np.sort(np.random.choice(total, size=n, replace=False))
        latents = f["latents"][idx]
        delta_mu = f["delta_mu"][idx] if "delta_mu" in f else None
    return latents, delta_mu


def _waveform_roughness(batch_np: np.ndarray, n_channels: int, n_time: int) -> np.ndarray:
    """Mean absolute first-difference of the z-profile for each event.

    Args:
        batch_np: (B, N, 1) raw waveforms in layer-major order
        n_channels: number of channels
        n_time: number of time bins

    Returns:
        (B,) array of roughness values
    """
    from compare_rqs import wf_to_z_profile
    B = batch_np.shape[0]
    roughness = np.empty(B, dtype=np.float32)
    for i in range(B):
        z = wf_to_z_profile(batch_np[i, :, 0], n_channels, n_time)
        roughness[i] = np.abs(np.diff(z)).mean()
    return roughness


@torch.no_grad()
def _build_encoder_ctx(model_type: str, cfg):
    """Load model checkpoint and return (ctx, encoder).

    Peeks at the checkpoint to infer encoder hidden_dim so the model
    architecture matches what was actually trained.
    """
    import copy
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
        if "cond_proj.0.weight" in chk.get("ddpm", {}):
            cfg.conditioning.cond_proj_dim = chk["ddpm"]["cond_proj.0.weight"].shape[0]
        ctx = DiffAEContext.build(cfg, for_training=True, verbose=False)

    ctx.load_checkpoint(ckpt_path, load_optim=False)
    enc = ctx.ema_encoder if ctx.ema_encoder is not None else ctx.encoder
    enc.eval()
    print(f"  Checkpoint: {ckpt_path}")
    return ctx, enc


@torch.no_grad()
def _encode_batch(enc, ctx, batch_np, model_type):
    """Encode a single batch, return latents (B, latent_dim)."""
    bs = batch_np.shape[0]
    wf_norm = ctx.data_stats.normalize(batch_np)
    x = torch.from_numpy(wf_norm.astype(np.float32)).to(ctx.device)
    x_flat = x.view(bs * ctx.n_nodes, 1)
    if model_type == "ae":
        z, _ = enc(x_flat, ctx.A_sparse, ctx.pos, batch_size=bs)
    else:
        z, _, _ = enc(x_flat, ctx.A_sparse, ctx.pos, batch_size=bs)
    return z.cpu().numpy()


@torch.no_grad()
def encode_lopsided_batch(model_type: str, cfg, n_samples: int, frac: float, sigma: float):
    """Generate events with lopsided augmentation and encode them.

    Returns:
        latents: (N, latent_dim)
        sides: (N,) int array — 0=none, 1=left, 2=right
    """
    from scipy.ndimage import gaussian_filter1d

    result = _build_encoder_ctx(model_type, cfg)
    if result[0] is None:
        return None, None
    ctx, enc = result

    all_latents, all_sides = [], []
    batch_size = 128
    encoded = 0

    while encoded < n_samples:
        bs = min(batch_size, n_samples - encoded)
        batch_np, *_ = ctx.loader.get_batch(bs)

        sides = np.zeros(bs, dtype=np.int32)
        half = batch_np.shape[1] // 2
        n_aug = max(1, int(bs * frac))
        aug_idx = np.random.choice(bs, size=n_aug, replace=False)
        aug_sides = np.random.randint(1, 3, size=n_aug)
        for i, s in zip(aug_idx, aug_sides):
            if s == 1:
                batch_np[i, :half, 0] = gaussian_filter1d(batch_np[i, :half, 0], sigma=sigma)
            else:
                batch_np[i, half:, 0] = gaussian_filter1d(batch_np[i, half:, 0], sigma=sigma)
            sides[i] = s

        all_latents.append(_encode_batch(enc, ctx, batch_np, model_type))
        all_sides.append(sides)
        encoded += bs

    return np.concatenate(all_latents), np.concatenate(all_sides)


@torch.no_grad()
def encode_with_roughness(model_type: str, cfg, n_samples: int):
    """Encode events and compute per-event waveform roughness.

    Returns:
        latents: (N, latent_dim)
        roughness: (N,) float array — mean |diff| of z-profile
    """
    result = _build_encoder_ctx(model_type, cfg)
    if result[0] is None:
        return None, None
    ctx, enc = result
    n_channels = ctx.n_channels
    n_time = ctx.n_time_points

    all_latents, all_roughness = [], []
    batch_size = 128
    encoded = 0

    while encoded < n_samples:
        bs = min(batch_size, n_samples - encoded)
        batch_np, *_ = ctx.loader.get_batch(bs)

        all_roughness.append(_waveform_roughness(batch_np, n_channels, n_time))
        all_latents.append(_encode_batch(enc, ctx, batch_np, model_type))
        encoded += bs

    return np.concatenate(all_latents), np.concatenate(all_roughness)


def reduce_pca(latents: np.ndarray, **kwargs):
    from sklearn.decomposition import PCA
    return PCA(n_components=2, random_state=kwargs.get("seed", 42)).fit_transform(latents)


def reduce_umap(latents: np.ndarray, **kwargs):
    # Import UMAP class directly to avoid triggering optional parametric/tensorflow imports.
    from umap.umap_ import UMAP
    reducer = UMAP(
        n_neighbors=kwargs.get("n_neighbors", 15),
        min_dist=kwargs.get("min_dist", 0.1),
        random_state=kwargs.get("seed", 42),
    )
    try:
        return reducer.fit_transform(latents)
    except TypeError as e:
        if "force_all_finite" in str(e):
            raise RuntimeError(
                "UMAP/scikit-learn version mismatch detected. "
                "Pin scikit-learn to a compatible version (e.g. 1.5.2)."
            ) from e
        raise


def reduce_tsne(latents: np.ndarray, **kwargs):
    from sklearn.manifold import TSNE
    return TSNE(
        n_components=2,
        perplexity=kwargs.get("perplexity", 30),
        learning_rate=kwargs.get("learning_rate", "auto"),
        init="pca",
        random_state=kwargs.get("seed", 42),
    ).fit_transform(latents)


REDUCERS = {
    "pca": reduce_pca,
    "umap": reduce_umap,
    "tsne": reduce_tsne,
}


def knn_label_smoothness(latents: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    """Mean absolute label difference between each point and its k nearest neighbors in latent space."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(latents)
    _, indices = nn.kneighbors(latents)
    neighbor_idx = indices[:, 1:]
    diffs = np.abs(labels[:, None] - labels[neighbor_idx])
    return float(diffs.mean())


def plot_delta_mu(panels, axes, ldim, args, method_label):
    """Plot panels colored by |delta_mu|."""
    all_abs_dmu = np.concatenate([np.abs(p[2]) for p in panels if p[2] is not None])
    vmin, vmax = 0.0, np.percentile(all_abs_dmu, 98) if len(all_abs_dmu) > 0 else 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)

    for col, (name, emb, dmu, lat) in enumerate(panels):
        ax = axes[0, col]
        smoothness = knn_label_smoothness(lat, dmu, k=args.knn_k)
        print(f"  {name}: k-NN smoothness (k={args.knn_k}) = {smoothness:.2f} ns")

        sc = ax.scatter(
            emb[:, 0], emb[:, 1],
            c=np.abs(dmu), cmap="viridis", norm=norm,
            s=args.point_size, alpha=0.6, edgecolors="none", rasterized=True,
        )
        ax.set_title(
            f"{name}  (z={ldim})\nk-NN smoothness = {smoothness:.2f} ns",
            fontweight="bold", fontsize=12,
        )
        ax.set_box_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    cbar = axes[0, 0].figure.colorbar(sc, ax=axes[0, :].tolist(), shrink=0.8, pad=0.04)
    cbar.set_label(r"$|\Delta\mu|$ (ns)", fontsize=11)


def plot_noise(panels, axes, ldim, args, method_label):
    """Plot panels colored by waveform roughness (mean |diff| of z-profile)."""
    all_rough = np.concatenate([p[2] for p in panels])
    vmin, vmax = 0.0, np.percentile(all_rough, 98) if len(all_rough) > 0 else 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)

    for col, (name, emb, roughness, lat) in enumerate(panels):
        ax = axes[0, col]
        smoothness = knn_label_smoothness(lat, roughness, k=args.knn_k)
        print(f"  {name}: k-NN noise continuity (k={args.knn_k}) = {smoothness:.4f}")

        sc = ax.scatter(
            emb[:, 0], emb[:, 1],
            c=roughness, cmap="magma", norm=norm,
            s=args.point_size, alpha=0.6, edgecolors="none", rasterized=True,
        )
        ax.set_title(
            f"{name}  (z={ldim})\nnoise continuity = {smoothness:.4f}",
            fontweight="bold", fontsize=12,
        )
        ax.set_box_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    cbar = axes[0, 0].figure.colorbar(sc, ax=axes[0, :].tolist(), shrink=0.8, pad=0.04)
    cbar.set_label("Waveform roughness (mean |Δz|)", fontsize=10)


def plot_lopsided(panels, axes, ldim, args, method_label):
    """Plot panels colored by lopsided side: grey=none, red=left, blue=right."""
    COLORS = {0: "#aaaaaa", 1: "#d62728", 2: "#1f77b4"}

    for col, (name, emb, sides, lat) in enumerate(panels):
        ax = axes[0, col]

        for side_val, label in [(0, "None"), (1, "Left"), (2, "Right")]:
            mask = sides == side_val
            if not mask.any():
                continue
            ax.scatter(
                emb[mask, 0], emb[mask, 1],
                c=COLORS[side_val], s=args.point_size, alpha=0.6,
                edgecolors="none", rasterized=True, label=label,
                zorder=2 if side_val == 0 else 3,
            )

        n_left = (sides == 1).sum()
        n_right = (sides == 2).sum()
        n_none = (sides == 0).sum()
        ax.set_title(
            f"{name}  (z={ldim})\nL={n_left}  R={n_right}  none={n_none}",
            fontweight="bold", fontsize=12,
        )
        ax.set_box_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#aaaaaa",
               markersize=6, label="None"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728",
               markersize=6, label="Left"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4",
               markersize=6, label="Right"),
    ]
    axes[0, -1].legend(handles=legend_elements, loc="upper right", fontsize=8,
                       frameon=True, edgecolor="0.8")


def main():
    parser = argparse.ArgumentParser(
        description="Dimensionality-reduced projections of AE / DiffAE latents")
    parser.add_argument("--latent-dim", type=int, required=True, help="Latent dimension to load")
    parser.add_argument("--method", type=str, default="umap", choices=METHODS,
                        help="Reduction method: pca, umap, tsne")
    parser.add_argument("--n-samples", type=int, default=10000, help="Max points to embed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="umap_plots")
    parser.add_argument("--point-size", type=float, default=2.5, help="Scatter point size")

    umap_group = parser.add_argument_group("UMAP options")
    umap_group.add_argument("--n-neighbors", type=int, default=15)
    umap_group.add_argument("--min-dist", type=float, default=0.1)

    tsne_group = parser.add_argument_group("t-SNE options")
    tsne_group.add_argument("--perplexity", type=float, default=30)
    tsne_group.add_argument("--learning-rate", type=str, default="auto",
                            help="'auto' or a float")

    parser.add_argument("--knn-k", type=int, default=10,
                        help="k for k-NN smoothness metric (computed in latent space)")

    lop_group = parser.add_argument_group("Lopsided mode")
    lop_group.add_argument("--lopsided", action="store_true",
                           help="Encode fresh events with lopsided augmentation, color by side")
    lop_group.add_argument("--lopsided-frac", type=float, default=0.3,
                           help="Fraction of events to augment (default: 0.3)")
    lop_group.add_argument("--lopsided-sigma", type=float, default=10.0,
                           help="Gaussian kernel sigma")

    parser.add_argument("--model", type=str, default="both", choices=["ae", "diffae", "both"],
                        help="Which model(s) to plot: ae, diffae, or both")
    parser.add_argument("--noise", action="store_true",
                        help="Color by waveform roughness (mean |diff| of z-profile)")

    args = parser.parse_args()

    lr = args.learning_rate
    if lr != "auto":
        lr = float(lr)

    np.random.seed(args.seed)
    cfg = get_config(latent_dim=args.latent_dim)
    ldim = args.latent_dim
    method = args.method.lower()
    method_label = {"pca": "PCA", "umap": "UMAP", "tsne": "t-SNE"}[method]

    reduce_fn = REDUCERS[method]
    reduce_kwargs = dict(
        seed=args.seed,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        perplexity=args.perplexity,
        learning_rate=lr,
    )

    all_models = [("ae", "AE"), ("diffae", "DiffAE")]
    if args.model != "both":
        all_models = [(m, n) for m, n in all_models if m == args.model]

    if args.lopsided:
        print("Mode: loading models from checkpoint, encoding on-the-fly (lopsided augmentation).")
        panels = []
        for mtype, mname in all_models:
            print(f"{mname}: encoding {args.n_samples} events with lopsided augmentation...")
            lat, sides = encode_lopsided_batch(
                mtype, cfg, args.n_samples,
                frac=args.lopsided_frac, sigma=args.lopsided_sigma,
            )
            if lat is not None:
                print(f"  {mname}: {lat.shape[0]} samples — running {method_label}...")
                emb = reduce_fn(lat, **reduce_kwargs)
                panels.append((mname, emb, sides, lat))
            else:
                print(f"  {mname}: no checkpoint found, skipping")

        if not panels:
            print("No models loaded.")
            return

        n_panels = len(panels)
        panel_size = 5.5
        fig, axes = plt.subplots(1, n_panels,
                                 figsize=(panel_size * n_panels + 0.5, panel_size),
                                 squeeze=False)
        plot_lopsided(panels, axes, ldim, args, method_label)

        fig.suptitle(
            f"{method_label}  —  z={ldim}  —  lopsided (σ={args.lopsided_sigma}, frac={args.lopsided_frac})",
            fontsize=13, fontweight="bold")
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, f"{method}_z{ldim}_lopsided.png")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")

    elif args.noise:
        print("Mode: loading models from checkpoint, encoding on-the-fly (waveform roughness).")
        panels = []
        for mtype, mname in all_models:
            print(f"{mname}: encoding {args.n_samples} events and computing roughness...")
            lat, roughness = encode_with_roughness(mtype, cfg, args.n_samples)
            if lat is not None:
                print(f"  {mname}: {lat.shape[0]} samples — running {method_label}...")
                emb = reduce_fn(lat, **reduce_kwargs)
                panels.append((mname, emb, roughness, lat))
            else:
                print(f"  {mname}: no checkpoint found, skipping")

        if not panels:
            print("No models loaded.")
            return

        n_panels = len(panels)
        panel_size = 5.5
        fig, axes = plt.subplots(1, n_panels,
                                 figsize=(panel_size * n_panels + 1.2, panel_size),
                                 squeeze=False)
        plot_noise(panels, axes, ldim, args, method_label)

        fig.suptitle(
            f"{method_label}  —  z={ldim}  —  waveform roughness",
            fontsize=13, fontweight="bold")
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, f"{method}_z{ldim}_noise.png")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")

    else:
        print("Mode: loading pre-saved encodings from H5 (color by |delta_mu|).")
        ae_subdir = cfg.paths.ae_subdir.format(latent_dim=ldim)
        ae_h5 = os.path.join(cfg.paths.checkpoint_dir, ae_subdir, "ae_encoded_ms_latents.h5")
        dae_subdir = cfg.paths.diffae_subdir.format(latent_dim=ldim)
        dae_h5 = os.path.join(cfg.paths.checkpoint_dir, dae_subdir, "encoded_ms_latents.h5")

        panels = []
        if args.model in ("ae", "both"):
            if os.path.exists(ae_h5):
                lat, dmu = load_latents(ae_h5, args.n_samples)
                print(f"AE: {lat.shape[0]} samples from {ae_h5}  — running {method_label}...")
                emb = reduce_fn(lat, **reduce_kwargs)
                panels.append(("AE", emb, dmu, lat))
            else:
                print(f"AE latents not found: {ae_h5}")

        if args.model in ("diffae", "both"):
            if os.path.exists(dae_h5):
                lat, dmu = load_latents(dae_h5, args.n_samples)
                print(f"DiffAE: {lat.shape[0]} samples from {dae_h5}  — running {method_label}...")
                emb = reduce_fn(lat, **reduce_kwargs)
                panels.append(("DiffAE", emb, dmu, lat))
            else:
                print(f"DiffAE latents not found: {dae_h5}")

        if not panels:
            print("No encoded latents found. Run compare_latent_sizes.py first.")
            return

        n_panels = len(panels)
        panel_size = 5.5
        fig, axes = plt.subplots(1, n_panels,
                                 figsize=(panel_size * n_panels + 1.2, panel_size),
                                 squeeze=False)
        plot_delta_mu(panels, axes, ldim, args, method_label)

        fig.suptitle(f"{method_label} of Encoded Latents  —  latent dim = {ldim}",
                     fontsize=14, fontweight="bold")
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, f"{method}_z{ldim}.png")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
