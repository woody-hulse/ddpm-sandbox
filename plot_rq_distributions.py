"""
Plot RQ distributional overlays comparing original events to DiffAE and GraphAE reconstructions.

Generates two figures:
  - diffae_rq_distributions.png  : original vs DiffAE per-RQ histogram overlays
  - graphae_rq_distributions.png : original vs GraphAE per-RQ histogram overlays

Usage:
    python plot_rq_distributions.py
    python plot_rq_distributions.py --n-samples 1000 --batch-size 16
    python plot_rq_distributions.py --output-dir results/rq_dist --latent-dim 128
"""

import os
import glob
import argparse
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import ks_2samp

from plot_style import apply_style, COLORS
from config import default_config, get_config
from graphae import GraphAEContext, reconstruct_graphae
from diffae import DiffAEContext, sample_diffae
from compare_rqs import collect_rqs, RQ_UNITS

apply_style()

PLOT_DPI = 300

RQ_DISPLAY_NAMES = {
    "peak_amplitude": "Peak Amplitude",
    "peak_time":      "Peak Time",
    "total_integral": "Total Integral",
    "rise_time":      "Rise Time",
    "fall_time":      "Fall Time",
    "fwhm":           "FWHM",
    "width_10_90":    "Width 10–90",
    "std_dev":        "Std Dev",
}

N_COLS = 4


# -------------------------------------------------------------------------
# Checkpoint helpers
# -------------------------------------------------------------------------

def _latest_ckpt(directory: str, prefix: str) -> Optional[str]:
    """Return checkpoint with highest epoch number in *directory*."""
    files = glob.glob(os.path.join(directory, f"{prefix}_epoch_*.pt"))
    if not files:
        return None
    def _epoch(p: str) -> int:
        try:
            return int(os.path.splitext(os.path.basename(p))[0].split("_")[-1])
        except Exception:
            return -1
    return max(files, key=_epoch)


def _enc_type_from_keys(keys) -> str:
    """Detect encoder type from state-dict keys."""
    if any(k.startswith("mlp.") for k in keys):
        return "mlp"
    # GraphEncoder (graphae.py) uses stage-based hierarchy
    if any(k.startswith("stages.") or k.startswith("pool") or k.startswith("global") for k in keys):
        return "graph"
    return "cnn"   # Conv1DEncoder uses 'backbone.*'


def _dec_type_from_keys(keys) -> str:
    """Detect decoder type from state-dict keys."""
    if any(k.startswith("mlp.") for k in keys):
        return "mlp"
    if any(k.startswith("stages.") for k in keys):
        return "graph"
    return "ddpm"   # GraphDDPMUNet


def _infer_graphae_cfg(ckpt_path: str) -> Tuple[int, int]:
    """Return (latent_dim, hidden_dim) from a GraphAE checkpoint."""
    chk = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = chk.get("ema_model") or chk["model"]
    # GraphAEEncoder ends with a linear to latent_dim
    linear_keys = [k for k, v in state.items() if k.endswith(".weight") and len(v.shape) == 2]
    # First linear in encoder: 'encoder.stages.0.blocks.0.lin1.weight' → hidden_dim
    # Last encoder linear: projects to latent_dim
    enc_linears = [k for k in linear_keys if k.startswith("encoder.")]
    latent_dim = int(state[enc_linears[-1]].shape[0]) if enc_linears else 64
    # hidden_dim: first hidden linear in encoder
    hidden_dim = int(state[enc_linears[0]].shape[0]) if enc_linears else 64
    return latent_dim, hidden_dim


def _infer_diffae_cfg(ckpt_path: str) -> Tuple[str, str, int]:
    """
    Return (encoder_type, decoder_type, latent_dim) from a DiffAE checkpoint.
    latent_dim is inferred from the encoder's to_latent layer or first encoder weight.
    """
    chk = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # encoder type and latent_dim
    enc = chk.get("ema_encoder") or chk["encoder"]
    encoder_type = _enc_type_from_keys(list(enc.keys()))

    # Infer latent_dim from encoder weights
    latent_dim = None
    for key in ("to_latent.weight", "to_mu.weight"):
        if key in enc:
            latent_dim = int(enc[key].shape[0])
            break
    if latent_dim is None:
        # Fall back: look for old latent_proj in checkpoint
        lp = chk.get("ema_latent_proj") or chk.get("latent_proj")
        if lp and "0.weight" in lp:
            latent_dim = int(lp["0.weight"].shape[1])
    if latent_dim is None:
        raise ValueError(f"Could not infer latent_dim from checkpoint {ckpt_path}")

    # decoder type
    dec = chk.get("ema_decoder") or chk["decoder"]
    decoder_type = _dec_type_from_keys(list(dec.keys()))

    return encoder_type, decoder_type, latent_dim


# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------

def _hist_range(a: np.ndarray, b: np.ndarray, clip_pct: float = 99.5):
    finite = np.concatenate([a, b])
    finite = finite[np.isfinite(finite)]
    if len(finite) < 2:
        return None
    lo = np.percentile(finite, 100 - clip_pct)
    hi = np.percentile(finite, clip_pct)
    margin = max((hi - lo) * 0.05, 1e-8)
    return (lo - margin, hi + margin)


def plot_distributions(
    rq_true: Dict[str, np.ndarray],
    rq_gen: Dict[str, np.ndarray],
    label_true: str,
    label_gen: str,
    color_true: str,
    color_gen: str,
    output_path: str,
    n_bins: int = 50,
    title: str = "",
):
    """Grid of per-RQ histogram overlays (probability density, semi-transparent fills)."""
    rq_names = list(rq_true.keys())
    n_rqs    = len(rq_names)
    n_rows   = (n_rqs + N_COLS - 1) // N_COLS

    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(4.5 * N_COLS, 3.5 * n_rows), squeeze=False)

    for idx, rq_name in enumerate(rq_names):
        row, col = divmod(idx, N_COLS)
        ax = axes[row][col]

        t = rq_true[rq_name]; t = t[np.isfinite(t)]
        g = rq_gen[rq_name];  g = g[np.isfinite(g)]

        if len(t) < 2 or len(g) < 2:
            ax.set_visible(False)
            continue

        rng = _hist_range(t, g)
        if rng is None:
            ax.set_visible(False)
            continue

        bins   = np.linspace(rng[0], rng[1], n_bins + 1)
        unit   = RQ_UNITS.get(rq_name, "")
        xlabel = RQ_DISPLAY_NAMES.get(rq_name, rq_name) + (f" [{unit}]" if unit else "")

        ax.hist(t, bins=bins, density=True,
                color=color_true, alpha=0.55, linewidth=0.5, edgecolor=color_true, label=label_true)
        ax.hist(g, bins=bins, density=True,
                color=color_gen,  alpha=0.55, linewidth=0.5, edgecolor=color_gen,  label=label_gen)

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Probability Density")
        ax.legend(loc="upper right", fontsize=8)

        ks_stat, _ = ks_2samp(t, g)
        ax.set_title(f"KS = {ks_stat:.3f}", fontsize=9)

    for idx in range(n_rqs, n_rows * N_COLS):
        row, col = divmod(idx, N_COLS)
        axes[row][col].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=0.99)

    if title:
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    else:
        fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

@torch.no_grad()
def main():
    cfg = default_config   # read latent_dim before argparse so we can use it as default

    parser = argparse.ArgumentParser(description="Plot RQ distributional overlays")
    parser.add_argument("--n-samples",  type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-bins",     type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="rq_dist_plots")
    parser.add_argument("--latent-dim", type=int, default=cfg.encoder.latent_dim,
                        help="Latent dim used to locate checkpoint subdirs (default: from config)")
    args = parser.parse_args()

    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device : {device}")
    print(f"Samples: {args.n_samples}")
    print(f"z-dim  : {args.latent_dim}")

    ckpt_root = cfg.paths.checkpoint_dir

    # ------------------------------------------------------------------ #
    #  Load GraphAE
    # ------------------------------------------------------------------ #
    has_graphae = False
    graphae_ctx: Optional[GraphAEContext] = None

    graphae_dir  = os.path.join(ckpt_root, cfg.paths.graph_ae_subdir.format(latent_dim=args.latent_dim))
    graphae_ckpt = _latest_ckpt(graphae_dir, "graphae")

    if graphae_ckpt is None:
        print(f"GraphAE: no checkpoint in {graphae_dir} — skipping")
    else:
        try:
            latent_dim, hidden_dim = _infer_graphae_cfg(graphae_ckpt)
            gae_cfg = get_config(latent_dim=latent_dim)
            gae_cfg.encoder.hidden_dim = hidden_dim
            print(f"GraphAE: {graphae_ckpt}  (z={latent_dim}, h={hidden_dim})")

            graphae_ctx = GraphAEContext.build(gae_cfg, for_training=False, verbose=False, use_ms_data=True)
            graphae_ctx.load_checkpoint(graphae_ckpt, load_optim=False)
            model = graphae_ctx.ema_model if graphae_ctx.ema_model is not None else graphae_ctx.model
            graphae_ctx.model = model
            graphae_ctx.model.eval()
            has_graphae = True
        except Exception as e:
            print(f"GraphAE: could not load — {e}")

    # ------------------------------------------------------------------ #
    #  Load DiffAE
    # ------------------------------------------------------------------ #
    has_diffae  = False
    diffae_ctx: Optional[DiffAEContext] = None
    diffae_cfg  = cfg   # will be overwritten below if checkpoint found

    diffae_dir  = os.path.join(ckpt_root, cfg.paths.diffae_subdir.format(latent_dim=args.latent_dim))
    diffae_ckpt = _latest_ckpt(diffae_dir, "diffae")

    if diffae_ckpt is None:
        print(f"DiffAE : no checkpoint in {diffae_dir} — skipping")
    else:
        try:
            enc_type, dec_type, latent_dim = _infer_diffae_cfg(diffae_ckpt)
            diffae_cfg = get_config(encoder_type=enc_type, decoder_type=dec_type, latent_dim=latent_dim)
            diffae_cfg.encoder.hidden_dim = max(diffae_cfg.encoder.hidden_dim, latent_dim)
            print(f"DiffAE : {diffae_ckpt}  (enc={enc_type}, dec={dec_type}, z={latent_dim})")

            diffae_ctx = DiffAEContext.build(diffae_cfg, for_training=False, verbose=False, use_ms_data=True)
            diffae_ctx.load_checkpoint(diffae_ckpt, load_optim=False)
            diffae_ctx.encoder = diffae_ctx.ema_encoder if diffae_ctx.ema_encoder is not None else diffae_ctx.encoder
            diffae_ctx.decoder = diffae_ctx.ema_decoder if diffae_ctx.ema_decoder is not None else diffae_ctx.decoder
            diffae_ctx.encoder.eval()
            diffae_ctx.decoder.eval()
            has_diffae = True
        except Exception as e:
            print(f"DiffAE : could not load — {e}")

    if not has_graphae and not has_diffae:
        print("ERROR: No models loaded. Check checkpoint paths or --latent-dim.")
        return

    ref_ctx = graphae_ctx if has_graphae else diffae_ctx
    assert ref_ctx is not None
    loader     = ref_ctx.loader
    n_channels = ref_ctx.n_channels
    n_time     = ref_ctx.n_time_points

    # ------------------------------------------------------------------ #
    #  Collect reconstructions
    # ------------------------------------------------------------------ #
    all_raw          = []
    all_graphae_rec  = []
    all_diffae_rec   = []

    n_batches = (args.n_samples + args.batch_size - 1) // args.batch_size
    collected = 0

    pbar = tqdm(range(n_batches), desc="Reconstructing", ncols=90)
    for _ in pbar:
        B = min(args.batch_size, args.n_samples - collected)
        if B <= 0:
            break

        wf_col, *_ = loader.get_batch(B)       # (B, N, 1)
        all_raw.append(wf_col[:, :, 0])        # (B, N) raw

        if has_graphae:
            assert graphae_ctx is not None
            wf_norm = graphae_ctx.data_stats.normalize(wf_col)
            x       = torch.from_numpy(wf_norm.astype(np.float32)).to(device)
            rec     = reconstruct_graphae(graphae_ctx.model, graphae_ctx.A_sparse, graphae_ctx.pos, x)
            rec_np  = graphae_ctx.data_stats.denormalize(rec.cpu().numpy())   # (B, 1, N)
            all_graphae_rec.append(np.clip(rec_np[:, 0, :], 0, None))

        if has_diffae:
            assert diffae_ctx is not None
            wf_norm  = diffae_ctx.data_stats.normalize(wf_col)
            x_diffae = torch.from_numpy(wf_norm.astype(np.float32)).to(device)
            rec      = sample_diffae(
                encoder         = diffae_ctx.encoder,
                decoder         = diffae_ctx.decoder,
                schedule        = diffae_ctx.schedule,
                A_sparse        = diffae_ctx.A_sparse,
                pos             = diffae_ctx.pos,
                time_dim        = diffae_cfg.conditioning.time_dim,
                x_ref           = x_diffae,
                parametrization = diffae_cfg.diffusion.parametrization,
                pbar            = False,
            )
            rec_np = diffae_ctx.data_stats.denormalize(rec.cpu().numpy())   # (B, 1, N)
            all_diffae_rec.append(np.clip(rec_np[:, 0, :], 0, None))

        collected += B
        pbar.set_postfix(n=collected)

    raw = np.concatenate(all_raw, axis=0)
    print(f"\nComputing RQs for {raw.shape[0]} events…")
    rq_true = collect_rqs(raw, n_channels, n_time)

    # ------------------------------------------------------------------ #
    #  Plots
    # ------------------------------------------------------------------ #
    if has_diffae:
        diffae_rec = np.concatenate(all_diffae_rec, axis=0)
        rq_diffae  = collect_rqs(diffae_rec, n_channels, n_time)
        plot_distributions(
            rq_true     = rq_true,
            rq_gen      = rq_diffae,
            label_true  = "Original",
            label_gen   = "DiffAE",
            color_true  = COLORS["truth"],
            color_gen   = COLORS["diffae"],
            output_path = os.path.join(args.output_dir, "diffae_rq_distributions.png"),
            n_bins      = args.n_bins,
            title       = "RQ Distributions — Original vs DiffAE Reconstructions",
        )

    if has_graphae:
        graphae_rec = np.concatenate(all_graphae_rec, axis=0)
        rq_graphae  = collect_rqs(graphae_rec, n_channels, n_time)
        plot_distributions(
            rq_true     = rq_true,
            rq_gen      = rq_graphae,
            label_true  = "Original",
            label_gen   = "GraphAE",
            color_true  = COLORS["truth"],
            color_gen   = COLORS["ae"],
            output_path = os.path.join(args.output_dir, "graphae_rq_distributions.png"),
            n_bins      = args.n_bins,
            title       = "RQ Distributions — Original vs GraphAE Reconstructions",
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
