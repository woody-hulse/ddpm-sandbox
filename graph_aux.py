"""
graph_aux.py — Graph encoder auxiliary task comparison.

Compares three approaches to predicting delta_mu (event time separation, ns) on MS events:

  Branch A — MLP on pre-encoded GraphAE latents:
    Load h5 file produced by graphae training (encode_dataset_every), train a small
    MLP to predict delta_mu from the frozen GraphAE latents.

  Branch A2 — MLP on pre-encoded DiffAE latents:
    Same MLP setup but using latents from DiffAE training (encoded_ms_latents.h5).

  Branch B — GraphEncoder + regression head (end-to-end):
    Attach a regression head directly to a GraphAEEncoder and train end-to-end on
    raw MS waveforms generated online. Encoder weights are optionally warm-started
    from the most recent graphae checkpoint (EMA weights).

Usage:
    # auto-find all paths from config defaults
    python graph_aux.py

    # explicit paths
    python graph_aux.py \\
        --latents checkpoints/graph_ae_z64/graphae_encoded_ms_latents.h5 \\
        --diffae-latents checkpoints/diffae_z64/encoded_ms_latents.h5 \\
        --graphae-ckpt checkpoints/graph_ae_z64/graphae_epoch_0500.pt

    # skip branches
    python graph_aux.py --skip-diffae-mlp --skip-encoder
    python graph_aux.py --skip-graphae-mlp --skip-diffae-mlp  # encoder only

    # keep encoder frozen (linear-probe mode)
    python graph_aux.py --freeze-encoder
"""

import argparse
import glob
import os
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from aux import (
    AuxTrainingResult,
    EncodedMSDataset,
    MLP,
    evaluate_aux_mlp_on_latents,
    train_aux_mlp_on_latents,
)
from config import Config, default_config, get_config
from graphae import GraphAEContext, GraphAEEncoder  # type: ignore[import]
from lz_data_loader import OnlineMSBatcher


# ---------------------------------------------------------------------------
# Model: GraphAEEncoder + scalar regression head
# ---------------------------------------------------------------------------

class GraphEncoderWithHead(nn.Module):
    """GraphAEEncoder + MLP regression head for end-to-end delta_mu prediction."""

    def __init__(
        self,
        encoder: GraphAEEncoder,
        latent_dim: int,
        hidden_dims: Tuple[int, ...] = (128, 64),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.head = MLP(latent_dim, list(hidden_dims), out_dim=1, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        pos: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Returns scalar predictions (B,)."""
        z, _ = self.encoder(x, adj, pos, batch_size=batch_size)
        return self.head(z).squeeze(-1)


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def find_latest_graphae_ckpt(cfg: Config) -> Optional[str]:
    subdir = cfg.paths.graph_ae_subdir.format(latent_dim=cfg.encoder.latent_dim)
    ckpt_dir = os.path.join(cfg.paths.checkpoint_dir, subdir)
    files = glob.glob(os.path.join(ckpt_dir, "graphae_epoch_*.pt"))
    if not files:
        return None

    def _epoch(p: str) -> int:
        try:
            return int(os.path.splitext(os.path.basename(p))[0].split("_")[-1])
        except (ValueError, IndexError):
            return -1

    return max(files, key=_epoch)


def find_latest_latents_h5(cfg: Config) -> Optional[str]:
    subdir = cfg.paths.graph_ae_subdir.format(latent_dim=cfg.encoder.latent_dim)
    path = os.path.join(cfg.paths.checkpoint_dir, subdir, "graphae_encoded_ms_latents.h5")
    return path if os.path.exists(path) else None


def find_diffae_latents_h5(cfg: Config) -> Optional[str]:
    subdir = cfg.paths.diffae_subdir.format(latent_dim=cfg.encoder.latent_dim)
    path = os.path.join(cfg.paths.checkpoint_dir, subdir, "encoded_ms_latents.h5")
    return path if os.path.exists(path) else None


def transfer_encoder_weights(
    encoder: GraphAEEncoder,
    ckpt_path: str,
    device: torch.device,
    verbose: bool = True,
) -> bool:
    """Load encoder weights from a graphae checkpoint (EMA model preferred)."""
    chk = torch.load(ckpt_path, map_location=device)
    graphae_state = chk.get("ema_model", chk.get("model"))
    if graphae_state is None:
        if verbose:
            print("  Warning: checkpoint has no 'model' or 'ema_model' key")
        return False

    # GraphAutoencoder state dict: keys start with "encoder." or "decoder."
    enc_state = {
        k[len("encoder."):]: v
        for k, v in graphae_state.items()
        if k.startswith("encoder.")
    }
    if not enc_state:
        if verbose:
            print("  Warning: no encoder.* keys found in checkpoint")
        return False

    missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
    if verbose:
        print(f"  Transferred {len(enc_state)} encoder tensors from {os.path.basename(ckpt_path)}")
        if missing:
            print(f"  Missing keys: {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")
    return True


# ---------------------------------------------------------------------------
# Branch A: MLP on pre-encoded latents
# ---------------------------------------------------------------------------

def run_mlp_branch(
    h5_path: str,
    cfg: Config,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 512,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    verbose: bool = True,
) -> AuxTrainingResult:
    print("\n" + "=" * 60)
    print("Branch A: MLP on pre-encoded GraphAE latents")
    print("=" * 60)

    dataset = EncodedMSDataset(h5_path)
    latent_dim = dataset.latent_dim
    n = len(dataset)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    n_train = n - n_val - n_test
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    if verbose:
        print(f"  Dataset: {n} samples, latent_dim={latent_dim}")
        print(f"  Split: train={n_train}, val={n_val}, test={n_test}")

    mlp = MLP(
        in_dim=latent_dim,
        hidden_dims=list(cfg.aux_task.hidden_dims),
        out_dim=1,
        dropout=cfg.aux_task.dropout,
    ).to(device)

    train_losses, val_losses, val_times = train_aux_mlp_on_latents(
        mlp, train_loader, val_loader, device,
        epochs=epochs, lr=lr,
    )

    mae, rmse, preds, targets = evaluate_aux_mlp_on_latents(mlp, test_loader, device)
    print(f"\n  Test MAE:  {mae:.2f} ns")
    print(f"  Test RMSE: {rmse:.2f} ns")

    return AuxTrainingResult(
        model_name="MLP (frozen GraphAE latents)",
        train_losses=train_losses,
        val_losses=val_losses,
        test_mae=mae,
        test_rmse=rmse,
        predictions=preds,
        targets=targets,
        val_times=val_times,
    )


# ---------------------------------------------------------------------------
# Branch B: GraphEncoder + regression head, end-to-end
# ---------------------------------------------------------------------------

def _generate_test_set(
    loader: OnlineMSBatcher,
    data_stats,
    device: torch.device,
    n: int = 2000,
    batch_size: int = 64,
) -> Tuple[List[torch.Tensor], List[np.ndarray]]:
    """Pre-generate a fixed test set of normalised waveform tensors + delta_mu."""
    xs, delta_mus = [], []
    sampled = 0
    while sampled < n:
        bsz = min(batch_size, n - sampled)
        wf_col, cond, _ = loader.get_batch(bsz)
        delta_mu = cond[:, 4]
        wf_norm = data_stats.normalize(wf_col).astype(np.float32)
        # (B, N, 1) already in column-major order from OnlineMSBatcher
        B, N, C = wf_norm.shape
        x = torch.from_numpy(wf_norm).view(B * N, C).to(device)
        xs.append(x.cpu())
        delta_mus.append(delta_mu)
        sampled += bsz
    return xs, delta_mus


def run_encoder_branch(
    cfg: Config,
    device: torch.device,
    graphae_ckpt: Optional[str],
    freeze_encoder: bool = False,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 16,
    steps_per_epoch: int = 64,
    n_test: int = 2000,
    verbose: bool = True,
) -> AuxTrainingResult:
    print("\n" + "=" * 60)
    print("Branch B: GraphEncoder + regression head (end-to-end)")
    print("=" * 60)

    # Build graph context for adjacency + data stats
    ctx = GraphAEContext.build(cfg, for_training=False, verbose=verbose, use_ms_data=True)
    adj = ctx.A_sparse
    pos = ctx.pos
    data_stats = ctx.data_stats

    # Build encoder
    encoder = GraphAEEncoder(
        in_dim=1,
        hidden_dim=cfg.model.hidden_dim,
        latent_dim=cfg.encoder.latent_dim,
        depth=cfg.encoder.depth,
        blocks_per_stage=cfg.encoder.blocks_per_stage,
        pool_ratio=cfg.encoder.pool_ratio,
        dropout=cfg.encoder.dropout,
        pos_dim=cfg.model.pos_dim,
    ).to(device)

    # Transfer weights from graphae checkpoint
    transferred = False
    if graphae_ckpt is not None and os.path.exists(graphae_ckpt):
        print(f"  Loading encoder weights from: {graphae_ckpt}")
        transferred = transfer_encoder_weights(encoder, graphae_ckpt, device, verbose=verbose)
    else:
        print("  No graphae checkpoint found — encoder initialised randomly")

    if freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad_(False)
        print("  Encoder frozen (linear probe mode)")

    model = GraphEncoderWithHead(
        encoder=encoder,
        latent_dim=cfg.encoder.latent_dim,
        hidden_dims=cfg.aux_task.hidden_dims,
        dropout=cfg.aux_task.dropout,
    ).to(device)

    loader = ctx.loader  # OnlineMSBatcher

    # Pre-generate fixed test set
    if verbose:
        print(f"  Generating {n_test} fixed test events...")
    test_xs, test_dmus = _generate_test_set(
        loader, data_stats, device, n=n_test, batch_size=batch_size * 4
    )

    params = model.parameters() if not freeze_encoder else model.head.parameters()
    optim = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(
            range(steps_per_epoch),
            desc=f"Epoch {epoch+1}/{epochs}",
            ncols=110,
            disable=not verbose,
        )
        for _ in pbar:
            wf_col, cond, _ = loader.get_batch(batch_size)
            delta_mu = torch.from_numpy(cond[:, 4]).to(device)
            wf_norm = data_stats.normalize(wf_col).astype(np.float32)
            B, N, C = wf_norm.shape
            x = torch.from_numpy(wf_norm).view(B * N, C).to(device)

            pred = model(x, adj, pos, batch_size=B)
            loss = F.mse_loss(pred, delta_mu)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            optim.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        train_losses.append(epoch_loss / steps_per_epoch)

        # Validation on a fresh mini-batch
        model.eval()
        with torch.no_grad():
            wf_val, cond_val, _ = loader.get_batch(batch_size * 4)
            dmu_val = torch.from_numpy(cond_val[:, 4]).to(device)
            wf_norm_val = data_stats.normalize(wf_val).astype(np.float32)
            Bv, Nv, Cv = wf_norm_val.shape
            xv = torch.from_numpy(wf_norm_val).view(Bv * Nv, Cv).to(device)
            val_pred = model(xv, adj, pos, batch_size=Bv)
            val_loss = F.mse_loss(val_pred, dmu_val).item()
        val_losses.append(val_loss)

        if verbose:
            print(f"  Epoch {epoch+1}/{epochs}: train={train_losses[-1]:.4f}  val={val_loss:.4f}")

    # Evaluate on pre-generated fixed test set
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x_cpu, dmu in zip(test_xs, test_dmus):
            B_t = len(dmu)
            x_t = x_cpu.to(device)
            pred_t = model(x_t, adj, pos, batch_size=B_t).cpu().numpy()
            all_preds.append(pred_t)
            all_targets.append(dmu)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    mae = float(np.mean(np.abs(all_preds - all_targets)))
    rmse = float(np.sqrt(np.mean((all_preds - all_targets) ** 2)))

    print(f"\n  Test MAE:  {mae:.2f} ns")
    print(f"  Test RMSE: {rmse:.2f} ns")
    name = "GraphEncoder+head"
    if transferred:
        name += " (graphae init)"
    if freeze_encoder:
        name += " [frozen]"

    return AuxTrainingResult(
        model_name=name,
        train_losses=train_losses,
        val_losses=val_losses,
        test_mae=mae,
        test_rmse=rmse,
        predictions=all_preds,
        targets=all_targets,
    )


# ---------------------------------------------------------------------------
# Comparison plots
# ---------------------------------------------------------------------------

def plot_results(results: List[AuxTrainingResult], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]

    # --- loss curves ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for r, col in zip(results, colors):
        axes[0].plot(r.train_losses, label=r.model_name, color=col, linewidth=1.2)
        axes[1].plot(r.val_losses, label=r.model_name, color=col, linewidth=1.2)
    for ax, title in zip(axes, ["Train MSE loss", "Val MSE loss"]):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE (ns²)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "loss_curves.png"), dpi=150)
    plt.close(fig)

    # --- scatter: pred vs true ---
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5), squeeze=False)
    for ax, r, col in zip(axes[0], results, colors):
        lim = max(np.abs(r.targets).max(), np.abs(r.predictions).max()) * 1.05
        ax.scatter(r.targets, r.predictions, s=2, alpha=0.3, color=col, rasterized=True)
        ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.8, label="ideal")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("True δμ (ns)")
        ax.set_ylabel("Predicted δμ (ns)")
        ax.set_title(f"{r.model_name}\nMAE={r.test_mae:.1f} ns  RMSE={r.test_rmse:.1f} ns")
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "scatter.png"), dpi=150)
    plt.close(fig)

    # --- residuals ---
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), squeeze=False)
    for ax, r, col in zip(axes[0], results, colors):
        resid = r.predictions - r.targets
        ax.hist(resid, bins=80, color=col, alpha=0.7, density=True)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Residual (ns)")
        ax.set_ylabel("Density")
        ax.set_title(f"{r.model_name}\nMAE={r.test_mae:.1f} ns")
        ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "residuals.png"), dpi=150)
    plt.close(fig)


def print_table(results: List[AuxTrainingResult]) -> None:
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    w = max(len(r.model_name) for r in results) + 2
    print(f"{'Model':<{w}}  {'MAE (ns)':>10}  {'RMSE (ns)':>10}")
    print("-" * (w + 26))
    for r in results:
        print(f"{r.model_name:<{w}}  {r.test_mae:>10.2f}  {r.test_rmse:>10.2f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Graph encoder aux task comparison")
    parser.add_argument("--latents", type=str, default=None,
                        help="Path to GraphAE pre-encoded MS latents h5 (Branch A)")
    parser.add_argument("--diffae-latents", type=str, default=None,
                        help="Path to DiffAE pre-encoded MS latents h5 (Branch A2)")
    parser.add_argument("--graphae-ckpt", type=str, default=None,
                        help="Path to graphae checkpoint for weight transfer (Branch B)")
    parser.add_argument("--output-dir", type=str, default="graph_aux_results",
                        help="Directory to save plots and results")
    parser.add_argument("--epochs-mlp", type=int, default=None,
                        help="Epochs for MLP branch (default: cfg.aux_task.epochs)")
    parser.add_argument("--epochs-enc", type=int, default=100,
                        help="Epochs for encoder branch (default: 100)")
    parser.add_argument("--steps-per-epoch", type=int, default=64,
                        help="Steps per epoch for encoder branch")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size for encoder branch (default: cfg.training.batch_size)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="Freeze encoder weights; only train the head (linear probe)")
    parser.add_argument("--skip-graphae-mlp", action="store_true", help="Skip GraphAE MLP branch")
    parser.add_argument("--skip-diffae-mlp", action="store_true", help="Skip DiffAE MLP branch")
    parser.add_argument("--skip-encoder", action="store_true", help="Skip encoder branch")
    parser.add_argument("--latent-dim", type=int, default=None,
                        help="Override cfg.encoder.latent_dim")
    args = parser.parse_args()

    cfg = default_config if args.latent_dim is None else get_config(latent_dim=args.latent_dim)
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}  |  latent_dim: {cfg.encoder.latent_dim}")

    # Auto-detect paths
    latents_path = args.latents or find_latest_latents_h5(cfg)
    diffae_latents_path = args.diffae_latents or find_diffae_latents_h5(cfg)
    graphae_ckpt = args.graphae_ckpt or find_latest_graphae_ckpt(cfg)

    if latents_path:
        print(f"GraphAE latents:  {latents_path}")
    if diffae_latents_path:
        print(f"DiffAE latents:   {diffae_latents_path}")
    if graphae_ckpt:
        print(f"GraphAE ckpt:     {graphae_ckpt}")

    epochs_mlp = args.epochs_mlp or cfg.aux_task.epochs
    batch_size = args.batch_size or cfg.training.batch_size

    results: List[AuxTrainingResult] = []

    if not args.skip_graphae_mlp:
        if latents_path is None:
            print("WARNING: no GraphAE latents h5 found — skipping GraphAE MLP branch. "
                  "Run graphae training with encode_dataset_every>0, or pass --latents.")
        else:
            r = run_mlp_branch(
                h5_path=latents_path,
                cfg=cfg,
                device=device,
                epochs=epochs_mlp,
                lr=args.lr,
                batch_size=cfg.aux_task.batch_size,
                verbose=True,
            )
            r.model_name = "MLP (GraphAE latents)"
            results.append(r)

    if not args.skip_diffae_mlp:
        if diffae_latents_path is None:
            print("WARNING: no DiffAE latents h5 found — skipping DiffAE MLP branch. "
                  "Run diffae training with encode_dataset_every>0, or pass --diffae-latents.")
        else:
            r = run_mlp_branch(
                h5_path=diffae_latents_path,
                cfg=cfg,
                device=device,
                epochs=epochs_mlp,
                lr=args.lr,
                batch_size=cfg.aux_task.batch_size,
                verbose=True,
            )
            r.model_name = "MLP (DiffAE latents)"
            results.append(r)

    if not args.skip_encoder:
        results.append(run_encoder_branch(
            cfg=cfg,
            device=device,
            graphae_ckpt=graphae_ckpt,
            freeze_encoder=args.freeze_encoder,
            epochs=args.epochs_enc,
            lr=args.lr,
            batch_size=batch_size,
            steps_per_epoch=args.steps_per_epoch,
            verbose=True,
        ))

    if not results:
        print("Nothing to do — both branches skipped.")
        return

    print_table(results)
    plot_results(results, args.output_dir)
    print(f"\nPlots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
