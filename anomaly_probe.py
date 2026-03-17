#!/usr/bin/env python3
"""
anomaly_probe.py — Three scatter plots comparing anomaly separability.

For each of three spaces (RQ metrics, AE latent, DiffAE latent) the figure
shows:
  • N real SS events  — small grey dots (background distribution)
  • 1 prototype per anomaly type — large coloured markers, labelled

The spatial anomaly prototypes lie exactly on top of the base real event in
RQ space (by construction); temporal anomalies deviate according to how much
the z-profile changed.  Both model latent plots test whether the encoder
separates what the RQ metrics cannot.

Two anomaly families:

  SPATIAL  — z-profile unchanged → all 8 RQs identical to real event.
    uniform, peripheral, single_pmt, checkerboard

  TEMPORAL  — z-profile modified; real spatial weights preserved.
    Synthetic (clearly differ on some RQs):
      square_wave, smooth_gaussian
    Modification-of-real (nearly indistinguishable in RQ space):
      pmt_saturation, diffusion_smear, delayed_echo, stretched_tail

Usage:
  python anomaly_probe.py [--n-events 500] [--batch-size 8]
                          [--latent-dim 64] [--output-dir anomaly_results]
"""

import argparse
import copy
import glob
import os

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from config import default_config
from diffae import DiffAEContext
from graphae import GraphAEContext
from compare_rqs import collect_rqs, compute_rqs, wf_to_z_profile
from plot_style import apply_style

apply_style()

# ---------------------------------------------------------------------------
# Anomaly registry
# ---------------------------------------------------------------------------

ANOMALY_TYPES = [
    # (key,              display label,             colour,    family)
    ("uniform",          "Uniform spatial",         "#2166AC", "spatial"),
    ("peripheral",       "Peripheral PMTs",         "#D6604D", "spatial"),
    ("single_pmt",       "Single PMT",              "#1A9850", "spatial"),
    ("checkerboard",     "Checkerboard",            "#762A83", "spatial"),
    ("square_wave",      "Square wave",             "#B35806", "temporal"),
    ("smooth_gaussian",  "Smooth Gaussian",         "#4393C3", "temporal"),
    ("pmt_saturation",   "PMT saturation",          "#D01C8B", "temporal"),
    ("diffusion_smear",  "Diffusion smear",         "#74C476", "temporal"),
    ("delayed_echo",     "Delayed echo",            "#F4A460", "temporal"),
    ("stretched_tail",   "Stretched tail",          "#9370DB", "temporal"),
]

ANOMALY_LABELS = {t[0]: t[1] for t in ANOMALY_TYPES}
ANOMALY_COLORS = {t[0]: t[2] for t in ANOMALY_TYPES}
ANOMALY_FAMILY = {t[0]: t[3] for t in ANOMALY_TYPES}
SPATIAL_TYPES  = [t[0] for t in ANOMALY_TYPES if t[3] == "spatial"]
TEMPORAL_TYPES = [t[0] for t in ANOMALY_TYPES if t[3] == "temporal"]

PLOT_DPI = 300


# ---------------------------------------------------------------------------
# Spatial weight helpers
# ---------------------------------------------------------------------------

def compute_spatial_weights(
    channel_positions: np.ndarray,
    mode: str,
    xc: float = 0.0,
    yc: float = 0.0,
) -> np.ndarray:
    C = len(channel_positions)
    center = channel_positions.mean(axis=0)
    radii  = np.linalg.norm(channel_positions - center, axis=1)

    if mode == "uniform":
        w = np.ones(C, dtype=np.float32)

    elif mode == "peripheral":
        thr = np.percentile(radii, 75.0)
        w   = (radii >= thr).astype(np.float32)

    elif mode == "single_pmt":
        w = np.zeros(C, dtype=np.float32)
        w[int(np.argmin(radii))] = 1.0
        return w

    elif mode == "checkerboard":
        angles = np.arctan2(channel_positions[:, 1] - center[1],
                            channel_positions[:, 0] - center[0])
        idx = np.argsort(angles)
        w   = np.zeros(C, dtype=np.float32)
        w[idx[0::2]] = 1.0

    else:
        raise ValueError(f"Unknown spatial mode: {mode!r}")

    s = w.sum()
    return w / s if s > 1e-8 else np.ones(C, dtype=np.float32) / C


def make_spatial_anomaly(
    wf_ct: np.ndarray,
    channel_positions: np.ndarray,
    mode: str,
    xc: float = 0.0,
    yc: float = 0.0,
) -> np.ndarray:
    z = wf_ct.sum(axis=0)
    w = compute_spatial_weights(channel_positions, mode, xc, yc)
    return (w[:, None] * z[None, :]).astype(np.float32)


# ---------------------------------------------------------------------------
# Temporal z-profile constructors
# ---------------------------------------------------------------------------

def _make_temporal_zprofile(
    T: int,
    rqs: dict,
    mode: str,
    z_real: np.ndarray | None = None,
) -> np.ndarray:
    t        = np.arange(T, dtype=np.float64)
    pt       = float(rqs["peak_time"])
    amp      = float(rqs["peak_amplitude"])
    integral = float(rqs["total_integral"])
    fwhm     = max(float(rqs["fwhm"]), 4.0)

    if mode == "square_wave":
        half_w = fwhm / 2.0
        tau    = 1.5
        rise   = 1.0 / (1.0 + np.exp(-(t - (pt - half_w)) / tau))
        fall   = 1.0 / (1.0 + np.exp( (t - (pt + half_w)) / tau))
        z      = amp * rise * fall
        z = np.clip(z, 0.0, None)
        cur = float(np.trapz(z, t))
        if cur > 1e-8:
            z *= integral / cur

    elif mode == "smooth_gaussian":
        sigma = integral / (amp * np.sqrt(2.0 * np.pi) + 1e-8)
        sigma = max(sigma, 2.0)
        z     = amp * np.exp(-0.5 * ((t - pt) / sigma) ** 2)
        z = np.clip(z, 0.0, None)
        cur = float(np.trapz(z, t))
        if cur > 1e-8:
            z *= integral / cur

    elif mode == "pmt_saturation":
        assert z_real is not None
        sat_level = 0.85 * float(z_real.max())
        z = np.minimum(z_real.astype(np.float64), sat_level)
        z = np.clip(z, 0.0, None)
        cur = float(np.trapz(z, t))
        if cur > 1e-8:
            z *= integral / cur

    elif mode == "diffusion_smear":
        assert z_real is not None
        z = gaussian_filter1d(z_real.astype(np.float64), sigma=12.0)
        z = np.clip(z, 0.0, None)

    elif mode == "delayed_echo":
        assert z_real is not None
        delay     = 28
        echo_frac = 0.07
        echo      = np.zeros(T, dtype=np.float64)
        echo[delay:] = z_real[:-delay].astype(np.float64) * echo_frac
        z = z_real.astype(np.float64) + echo
        z = np.clip(z, 0.0, None)

    elif mode == "stretched_tail":
        assert z_real is not None
        t_arr    = np.arange(T, dtype=np.float64)
        peak_loc = int(np.argmax(z_real))
        peak_amp = float(z_real[peak_loc])
        tau_slow = 70.0
        tail     = np.where(
            t_arr > peak_loc,
            0.12 * peak_amp * np.exp(-(t_arr - peak_loc) / tau_slow),
            0.0,
        )
        z = z_real.astype(np.float64) + tail
        z = np.clip(z, 0.0, None)

    else:
        raise ValueError(f"Unknown temporal mode: {mode!r}")

    return z.astype(np.float32)


def make_temporal_anomaly(wf_ct: np.ndarray, mode: str, rqs: dict) -> np.ndarray:
    C, T   = wf_ct.shape
    total  = float(wf_ct.sum())
    ch_w   = wf_ct.sum(axis=1) / total if total > 1e-8 else np.ones(C) / C
    z_real = wf_ct.sum(axis=0).astype(np.float64)
    z_anom = _make_temporal_zprofile(T, rqs, mode, z_real=z_real)
    return (ch_w[:, None] * z_anom[None, :]).astype(np.float32)


def make_anomaly(
    wf_ct: np.ndarray,
    channel_positions: np.ndarray,
    mode: str,
    xc: float = 0.0,
    yc: float = 0.0,
    rqs: dict | None = None,
) -> np.ndarray:
    if mode in SPATIAL_TYPES:
        return make_spatial_anomaly(wf_ct, channel_positions, mode, xc, yc)
    assert rqs is not None
    return make_temporal_anomaly(wf_ct, mode, rqs)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def to_flat(wf_batch: np.ndarray) -> np.ndarray:
    """(B, C, T) → (B, T*C) layer-major."""
    B = wf_batch.shape[0]
    return np.transpose(wf_batch, (0, 2, 1)).reshape(B, -1)


@torch.no_grad()
def encode_batch(wf_batch: np.ndarray, ctx, batch_size: int = 8) -> np.ndarray:
    """
    (B, C, T) → (B, latent_dim).

    Accepts either DiffAEContext (has ctx.encoder / ctx.ema_encoder, returns
    z as first element of a 3-tuple) or GraphAEContext (has ctx.model /
    ctx.ema_model; uses model.encode which returns (z, pool_indices)).
    """
    B      = wf_batch.shape[0]
    wf_col = np.transpose(wf_batch, (0, 2, 1)).reshape(B, -1, 1).astype(np.float32)
    wf_col = ctx.data_stats.normalize(wf_col)

    # Resolve the callable and call convention for each context type.
    is_graphae = isinstance(ctx, GraphAEContext)
    if is_graphae:
        net = ctx.ema_model if ctx.ema_model is not None else ctx.model
    else:
        net = ctx.ema_encoder if ctx.ema_encoder is not None else ctx.encoder
    net.eval()

    all_z = []
    for start in range(0, B, batch_size):
        end = min(start + batch_size, B)
        bs  = end - start
        x   = torch.from_numpy(wf_col[start:end]).to(ctx.device)
        xf  = x.view(bs * ctx.n_nodes, 1)
        if is_graphae:
            z, _ = net.encode(xf, ctx.A_sparse, ctx.pos, batch_size=bs)
        else:
            z, _, _ = net(xf, ctx.A_sparse, ctx.pos, batch_size=bs)
        all_z.append(z.cpu().numpy())
    return np.concatenate(all_z, axis=0)


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

def fit_reduce(
    Z_real: np.ndarray,
    Z_protos: np.ndarray,
    perplexity: int = 30,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Embed Z_real with t-SNE, then position Z_protos by fitting a 1-NN
    look-up into the existing embedding (nearest neighbour in the original
    space, inherit its 2-D coordinate).  This avoids re-running t-SNE for
    the prototype points while still placing them correctly relative to the
    real-event cloud.
    Returns (Z_real_2d, Z_protos_2d).
    """
    tsne = TSNE(n_components=2, perplexity=perplexity,
                random_state=seed, init="pca", learning_rate="auto")
    Z_real_2d = tsne.fit_transform(Z_real)

    # Place each prototype at the 2-D position of its nearest real neighbour.
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1).fit(Z_real)
    _, idx = nn.kneighbors(Z_protos)
    Z_protos_2d = Z_real_2d[idx[:, 0]]
    return Z_real_2d, Z_protos_2d


# ---------------------------------------------------------------------------
# Scatter plot
# ---------------------------------------------------------------------------

def scatter_plot(
    Z_real_2d: np.ndarray,
    Z_protos_2d: np.ndarray,
    title: str,
    subtitle: str,
    path: str,
) -> None:
    """
    Grey in-distribution cloud + red anomaly dots (single "Anomaly" legend entry).
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(
        Z_real_2d[:, 0], Z_real_2d[:, 1],
        c="#888888", s=6, alpha=0.40, linewidths=0,
        zorder=1, label="In distribution",
    )
    ax.scatter(
        Z_protos_2d[:, 0], Z_protos_2d[:, 1],
        c="#D6231A", s=60, alpha=1.0, linewidths=0,
        zorder=3, label="Anomaly",
    )

    ax.set_xlabel("t-SNE 1", fontsize=10)
    ax.set_ylabel("t-SNE 2", fontsize=10)
    ax.set_title(f"{title}\n{subtitle}", fontsize=11)
    ax.legend(fontsize=9, markerscale=1.4, loc="best",
              handlelength=0.8, borderpad=0.5)

    plt.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  {os.path.basename(path)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Anomaly scatter plots: RQ space + AE latent + DiffAE latent"
    )
    parser.add_argument("--n-events",   type=int, default=500,
                        help="Number of real SS events for background cloud")
    parser.add_argument("--batch-size",      type=int, default=8)
    parser.add_argument("--ae-latent-dim",   type=int, default=64,
                        help="Latent dimension for the AE model (default 64)")
    parser.add_argument("--diffae-latent-dim", type=int, default=64,
                        help="Latent dimension for the DiffAE model (default 64)")
    parser.add_argument("--output-dir", type=str, default="anomaly_results")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--perplexity",  type=int, default=30,
                        help="t-SNE perplexity (default 30)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)

    cfg = copy.deepcopy(default_config)

    # -----------------------------------------------------------------------
    # 1. Load real SS events
    # -----------------------------------------------------------------------
    print(f"Loading {args.n_events} real SS events from {cfg.paths.tritium_h5} …")
    with h5py.File(cfg.paths.tritium_h5, "r") as f:
        n_total = int(f["waveforms"].shape[0])
        rng     = np.random.default_rng(args.seed)
        idx     = np.sort(rng.choice(n_total, size=args.n_events, replace=False))
        wf_real = np.array(f["waveforms"][idx], dtype=np.float32)   # (N, C, T)
        xc_real = np.array(f["xc"][idx], dtype=np.float32)
        yc_real = np.array(f["yc"][idx], dtype=np.float32)

    N, C, T = wf_real.shape
    print(f"  shape {wf_real.shape}")

    # Load channel positions from DiffAEContext (needed before building models)
    print("Building contexts to get graph / channel positions …")
    cfg_diffae = copy.deepcopy(default_config)
    cfg_diffae.encoder.latent_dim = args.diffae_latent_dim
    ctx_diffae = DiffAEContext.build(cfg_diffae, for_training=True,
                                     verbose=False, use_ms_data=False)
    channel_pos = ctx_diffae.loader.channel_positions   # (C, 2)

    # -----------------------------------------------------------------------
    # 2. Pick one prototype base event (closest to median integral)
    # -----------------------------------------------------------------------
    print("Selecting prototype base event …")
    wf_flat_all = to_flat(wf_real)
    integrals   = wf_flat_all.sum(axis=1)
    med_int     = float(np.median(integrals))
    base_idx    = int(np.argmin(np.abs(integrals - med_int)))
    wf_base     = wf_real[base_idx]         # (C, T)
    xc_base     = float(xc_real[base_idx])
    yc_base     = float(yc_real[base_idx])

    z_base      = wf_to_z_profile(wf_flat_all[base_idx], C, T)
    rqs_base    = compute_rqs(z_base)
    if rqs_base is None:
        rqs_base = {k: 0.0 for k in
                    ["peak_amplitude", "peak_time", "total_integral",
                     "rise_time", "fall_time", "fwhm", "width_10_90", "std_dev"]}
    print(f"  Base event #{base_idx}  integral={integrals[base_idx]:.1f}  "
          f"(dataset median={med_int:.1f})")

    # -----------------------------------------------------------------------
    # 3. Generate anomaly prototypes  (one per type, all from base event)
    # -----------------------------------------------------------------------
    print("Generating anomaly prototypes …")
    proto_wfs  = {}  # atype → (C, T)
    for atype, _, _, _ in ANOMALY_TYPES:
        proto_wfs[atype] = make_anomaly(
            wf_base, channel_pos, atype,
            xc=xc_base, yc=yc_base, rqs=rqs_base,
        )

    # Stack into (n_types, C, T) for batch encoding
    proto_types_list = [t[0] for t in ANOMALY_TYPES]
    wf_protos = np.stack([proto_wfs[k] for k in proto_types_list], axis=0)  # (10, C, T)

    # -----------------------------------------------------------------------
    # 4. Compute RQs for real events + prototypes
    # -----------------------------------------------------------------------
    print("Computing RQs …")
    rq_real_dict   = collect_rqs(wf_flat_all, C, T)
    rq_proto_dict  = collect_rqs(to_flat(wf_protos), C, T)

    rq_names = list(rq_real_dict.keys())

    def to_mat(d: dict) -> np.ndarray:
        return np.column_stack([d[k] for k in rq_names]).astype(np.float32)

    RQ_real   = to_mat(rq_real_dict)    # (N, 8)
    RQ_protos = to_mat(rq_proto_dict)   # (n_types, 8)

    # Impute rare NaNs with column median
    for col in range(RQ_real.shape[1]):
        bad = ~np.isfinite(RQ_real[:, col])
        if bad.any():
            RQ_real[bad, col] = float(np.nanmedian(RQ_real[:, col]))
    for col in range(RQ_protos.shape[1]):
        bad = ~np.isfinite(RQ_protos[:, col])
        if bad.any():
            RQ_protos[bad, col] = float(np.nanmedian(RQ_real[:, col]))

    # -----------------------------------------------------------------------
    # 5. Load DiffAE encoder and encode
    # -----------------------------------------------------------------------
    print(f"\nLoading DiffAE encoder (latent_dim={args.diffae_latent_dim}) …")
    ckpt_diffae = ctx_diffae.latest_checkpoint()
    if ckpt_diffae is None:
        print(f"  WARNING: no DiffAE checkpoint found in {ctx_diffae.checkpoint_dir}")
        Z_diffae_real   = None
        Z_diffae_protos = None
    else:
        ep = ctx_diffae.load_checkpoint(ckpt_diffae, load_optim=False)
        print(f"  Loaded {os.path.basename(ckpt_diffae)} (epoch {ep})")
        print("  Encoding real events …")
        Z_diffae_real   = encode_batch(wf_real,   ctx_diffae, args.batch_size)
        print("  Encoding prototypes …")
        Z_diffae_protos = encode_batch(wf_protos, ctx_diffae, args.batch_size)
        print(f"  z shape: {Z_diffae_real.shape}")

    # -----------------------------------------------------------------------
    # 6. Load GraphAE encoder and encode
    # -----------------------------------------------------------------------
    print(f"\nLoading GraphAE encoder (latent_dim={args.ae_latent_dim}) …")
    cfg_ae = copy.deepcopy(default_config)
    cfg_ae.encoder.latent_dim = args.ae_latent_dim

    # Find the checkpoint before building so we can infer hidden_dim from the
    # saved weights, which may differ from the current config default.
    _graphae_subdir = cfg_ae.paths.graph_ae_subdir.format(latent_dim=args.ae_latent_dim)
    _graphae_ckpt_dir = os.path.join(cfg_ae.paths.checkpoint_dir, _graphae_subdir)
    _graphae_files = glob.glob(os.path.join(_graphae_ckpt_dir, "graphae_epoch_*.pt"))

    if _graphae_files:
        def _epoch_num(p):
            try:
                return int(os.path.splitext(os.path.basename(p))[0].split("_")[-1])
            except (ValueError, IndexError):
                return -1
        _ckpt_peek = max(_graphae_files, key=_epoch_num)
        _state = torch.load(_ckpt_peek, map_location="cpu", weights_only=False)
        # encoder.in_proj.weight has shape [hidden_dim, in_dim=1]
        _hidden_dim = int(_state["model"]["encoder.in_proj.weight"].shape[0])
        cfg_ae.encoder.hidden_dim = _hidden_dim
        print(f"  Inferred hidden_dim={_hidden_dim} from checkpoint weights")

    ctx_ae = GraphAEContext.build(cfg_ae, for_training=True, verbose=False,
                                  use_ms_data=False)
    ckpt_ae = ctx_ae.latest_checkpoint()
    if ckpt_ae is None:
        print(f"  WARNING: no GraphAE checkpoint found in {ctx_ae.checkpoint_dir}")
        Z_ae_real   = None
        Z_ae_protos = None
    else:
        ep = ctx_ae.load_checkpoint(ckpt_ae, load_optim=False)
        print(f"  Loaded {os.path.basename(ckpt_ae)} (epoch {ep})")
        print("  Encoding real events …")
        Z_ae_real   = encode_batch(wf_real,   ctx_ae, args.batch_size)
        print("  Encoding prototypes …")
        Z_ae_protos = encode_batch(wf_protos, ctx_ae, args.batch_size)
        print(f"  z shape: {Z_ae_real.shape}")

    # -----------------------------------------------------------------------
    # 7. Dimensionally reduce + plot
    # -----------------------------------------------------------------------
    print("\nGenerating plots (t-SNE) …")

    # --- Plot 1: RQ space ---
    RQ_real_2d, RQ_protos_2d = fit_reduce(
        RQ_real, RQ_protos, perplexity=args.perplexity, seed=args.seed
    )
    scatter_plot(
        Z_real_2d   = RQ_real_2d,
        Z_protos_2d = RQ_protos_2d,
        title       = "RQ space",
        subtitle    = f"t-SNE of 8 pulse-shape metrics  ({N} real events)",
        path        = os.path.join(args.output_dir, "scatter_rq.png"),
    )

    # --- Plot 2: GraphAE latent ---
    if Z_ae_real is not None:
        ae_real_2d, ae_protos_2d = fit_reduce(
            Z_ae_real, Z_ae_protos, perplexity=args.perplexity, seed=args.seed
        )
        scatter_plot(
            Z_real_2d   = ae_real_2d,
            Z_protos_2d = ae_protos_2d,
            title       = f"GraphAE latent space  (z={args.ae_latent_dim})",
            subtitle    = (f"t-SNE of {Z_ae_real.shape[1]}-dim encoder output  "
                           f"({N} real events)"),
            path        = os.path.join(args.output_dir, "scatter_ae.png"),
        )

    # --- Plot 3: DiffAE latent ---
    if Z_diffae_real is not None:
        diffae_real_2d, diffae_protos_2d = fit_reduce(
            Z_diffae_real, Z_diffae_protos, perplexity=args.perplexity, seed=args.seed
        )
        scatter_plot(
            Z_real_2d   = diffae_real_2d,
            Z_protos_2d = diffae_protos_2d,
            title       = f"DiffAE latent space  (z={args.diffae_latent_dim})",
            subtitle    = (f"t-SNE of {Z_diffae_real.shape[1]}-dim encoder output  "
                           f"({N} real events)"),
            path        = os.path.join(args.output_dir, "scatter_diffae.png"),
        )

    print(f"\nDone → {args.output_dir}/")
    print("  scatter_rq.png     — RQ space (poor separability expected)")
    print("  scatter_ae.png     — GraphAE latent space")
    print("  scatter_diffae.png — DiffAE latent space")


if __name__ == "__main__":
    main()
