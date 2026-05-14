"""
Configuration for DiffAE/AE training and auxiliary tasks.

This file contains all hyperparameters and settings. Edit values here to 
change training behavior without modifying code.

Usage:
    from config import default_config, Config
    
    # Use default config
    cfg = default_config
    
    # Or create with overrides
    cfg = get_config(latent_dim=128, lr=1e-4)
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import os
from typing import Any, Optional


_SECTION_NAMES = (
    "training",
    "encoder",
    "model",
    "diffusion",
    "graph",
    "ms_data",
    "aux_task",
    "paths",
    "conditioning",
)

_OVERRIDE_ALIASES = {
    "hidden_dim": ("model", "hidden_dim"),
    "model_hidden_dim": ("model", "hidden_dim"),
    "unet_hidden_dim": ("model", "hidden_dim"),
    "encoder_hidden_dim": ("encoder", "hidden_dim"),
    "graph_encoder_hidden_dim": ("encoder", "hidden_dim"),
}


# =============================================================================
# Model Architecture
# =============================================================================

@dataclass
class ModelConfig:
    """Graph U-Net decoder architecture parameters.
    
    Used by the diffusion decoder (GraphDDPMUNet) in DiffAE.
    """
    in_dim: int = 1                 # Input feature dimension per node
    out_dim: int = 1                # Output feature dimension per node
    hidden_dim: int = 64            # Diffusion graph U-Net node width
    depth: int = 3                  # Number of pooling/unpooling stages
    blocks_per_stage: int = 2       # Residual blocks per stage
    pool_ratio: float = 0.5         # Fraction of nodes to keep per pooling
    dropout: float = 0.05           # Dropout rate
    pos_dim: int = 3                # Position embedding dimension (x, y, z)
    pos_dropout: float = 0.0        # Dropout on position embeddings
    cache_norm_top: bool = True     # Cache normalization at top level
    skip_scale: float = 1.0         # Scale factor for U-Net skip connections


@dataclass
class EncoderConfig:
    """Graph encoder parameters for DiffAE and Graph AE.

    The encoder maps input graphs to latent representations.
    use_stochastic and kl_weight apply only to DiffAE's encoder.
    decoder_type applies only to AE: "graph", "cnn", or "mlp".
    """
    latent_dim: int = 1024            # Latent representation dimension
    hidden_dim: int = 64            # Graph encoder node width
    latent_head_dim: Optional[int] = None  # None => max(2 * latent_dim, graph readout dim)
    depth: int = 4                  # Number of pooling stages
    blocks_per_stage: int = 2       # Residual blocks per stage
    pool_ratio: float = 0.5         # Pooling ratio per stage
    dropout: float = 0.05           # Dropout rate
    use_stochastic: bool = False    # DiffAE encoder: stochastic (VAE-style) encoding
    kl_weight: float = 0.001        # DiffAE: KL weight when use_stochastic
    encoder_type: str = "graph"   # Encoder: "cnn", "mlp", or "graph"
    decoder_type: str = "graph"     # AE decoder: "graph" (SimpleGraphDecoder), "cnn", or "mlp"
    conv_channels: tuple = (32, 64, 128)  # CNN encoder/decoder channel widths
    conv_kernel_size: int = 7       # CNN encoder/decoder temporal kernel size
    conv_pool_size: int = 4         # CNN encoder/decoder pooling/upsampling factor
    mlp_hidden_dim: int = 128       # MLP encoder/decoder hidden width
    mlp_decoder_hidden_dim: Optional[int] = None  # None => mirror MLP encoder geometry
    mlp_encoder_layers: int = 3     # MLP encoder: number of hidden layers (only if encoder_type="mlp")
    mlp_decoder_layers: int = 3     # MLP decoder: number of hidden layers (only if decoder_type="mlp")
    regressive_hidden_dim: int = 64 # Optional regressive graph/MLP decoder width
    latent_anchor_dim: Optional[int] = None  # None => latent_dim; 0 disables node-level anchor readout
    latent_anchor_count: int = 128     # Geometric anchors per encoder scale for latent readout
    latent_anchor_value_dim: int = 4   # Features pooled per anchor before source projection
    use_regressive_head: bool = True   # DiffAE: add a second decoder head with regressive (MSE) loss
    regressive_head_weight: float = 0.5 # DiffAE: weight for the regressive head loss


@dataclass
class ConditioningConfig:
    """Conditioning network parameters for DiffAE.

    Controls how latent codes condition the diffusion process.
    """
    cond_in_dim: int = 5            # Raw condition input dimension
    time_dim: int = 64              # Sinusoidal time embedding dimension


# =============================================================================
# Diffusion Process
# =============================================================================

@dataclass
class DiffusionConfig:
    """Diffusion process parameters.
    
    Controls the forward/reverse diffusion for DiffAE training.
    """
    timesteps: int = 250            # Number of diffusion timesteps
    parametrization: str = "v"      # Prediction target: "v" (velocity) or "eps" (noise)
    p2_gamma: float = 0.0           # P2 loss weighting gamma (0 = uniform, >0 upweights high-noise steps)
    p2_k: float = 1.0               # P2 loss weighting k
    t_min_frac: float = 0.0         # Optional minimum timestep fraction for fine-tuning probes

# =============================================================================
# Graph Construction  
# =============================================================================

@dataclass 
class GraphConfig:
    """Graph construction parameters for 3D spatio-temporal graphs.

    Defines how nodes (channel × time) are connected.
    """
    radius: float = 15.0            # Spatial radius for within-layer adjacency (cm)
    z_sep: float = 5.0             # Z-spacing between time layers
    z_hops: int = 5                 # Cross-layer connectivity distance
    weighted_edges: bool = True     # Gaussian distance-weighted edges (vs binary)
    lpe_dim: int = 16               # Laplacian positional encoding dimension (0 = disabled)


# =============================================================================
# Multi-Scatter Data Generation
# =============================================================================

@dataclass
class MSDataConfig:
    """Online multi-scatter (MS) event generation parameters.
    
    MS events are created by co-adding pairs of single-scatter (SS) events
    with random time shifts. This creates training data for learning to
    separate overlapping events.
    
    The time shift (delta) is measured in bins, where each bin = ns_per_bin nanoseconds.
    """
    delta_min: int = -50            # Minimum time shift (bins), negative = SS2 before SS1
    delta_max: int = 50             # Maximum time shift (bins), positive = SS2 after SS1
    ns_per_bin: float = 10.0        # Nanoseconds per time bin (for delta_mu calculation)
    seed: Optional[int] = None      # Random seed (None = different each run)


# =============================================================================
# Training
# =============================================================================

@dataclass
class TrainingConfig:
    """Training loop parameters.
    Controls optimization, checkpointing, and dataset encoding.
    """
    # Optimization
    epochs: int = 20_000            # Total training epochs
    batch_size: int = 8             # Batch size
    steps_per_epoch: Optional[int] = None  # None => use all available batches from the loader
    lr: float = 5e-4                # Learning rate
    lr_schedule: str = "cosine"     # LR schedule: "constant" or "cosine"
    warmup_epochs: int = 0        # Linear warmup epochs (0 = no warmup)
    weight_decay: float = 0         # AdamW weight decay
    ema_decay: float = 0.999        # Exponential moving average decay
    grad_clip: float = 5.0          # Gradient clipping norm
    use_amp: bool = True            # Mixed precision (bfloat16) — halves activation memory

    # AE-specific regularization (applies in train_ae)
    ae_denoising: bool = False      # Optional denoising mode (corrupt input, reconstruct clean)
    ae_input_noise_std: float = 0.0  # Gaussian input corruption std (normalized space)
    ae_mask_prob: float = 0.0      # Random feature masking probability for denoising AE
    ae_latent_l1_weight: float = 0.0  # Optional L1 sparsity penalty on latent activations
    
    # Augmentation
    lopsided_aug: bool = False       # Apply lopsided Gaussian blur to fraction of events
    lopsided_frac: float = 0.5     # Fraction of events to augment (default 50%)
    lopsided_sigma: float = 10.0    # Gaussian kernel sigma for lopsided augmentation

    # Checkpointing
    checkpoint_every: int = 25     # Save checkpoint every N epochs
    visualize_every: int = 25      # Generate visualizations every N epochs
    
    # Encoded dataset export (for aux task)
    encode_dataset_every: int = 0    # Export encoded latents every N epochs (0 = disable)
    encode_n_samples: int = 500_000       # Number of MS samples to encode and save

    def resolved_steps_per_epoch(self, n_samples: int) -> int:
        if self.steps_per_epoch is not None:
            return max(1, int(self.steps_per_epoch))
        return max(1, int(n_samples) // int(self.batch_size))


@dataclass
class AuxTaskConfig:
    """Auxiliary task training parameters.
    
    The aux task trains MLPs to predict delta_mu from encoded latents,
    evaluating how well the encoder preserves timing information.
    """
    epochs: int = 20               # MLP training epochs
    batch_size: int = 512            # Batch size for aux training
    lr: float = 1e-3                # Learning rate
    hidden_dims: tuple = (128, 64)  # MLP hidden layer dimensions
    dropout: float = 0.1            # Dropout rate
    output_dir: str = "figures/aux_results" # Directory for aux task outputs


# =============================================================================
# File Paths
# =============================================================================

@dataclass
class PathConfig:
    """File paths for data and outputs.
    
    Checkpoint/plot subdirs use {latent_dim} placeholder for organization.
    """
    # Input data
    tritium_h5: str = "data/tritium_ss_42.h5"
    channel_positions: str = "data/pmt_xy_42.h5"
    
    # Output directories
    checkpoint_dir: str = "checkpoints"
    plot_dir: str = "figures/plots"
    
    # Subdirectory templates (use .format(latent_dim=N))
    diffae_subdir: str = "diffae_z{latent_dim}"
    ae_subdir: str = "ae_z{latent_dim}"
    ae_light_subdir: str = "ae_light_z{latent_dim}"
    diffae_light_subdir: str = "diffae_light_z{latent_dim}"
    graph_ae_subdir: str = "graph_ae_z{latent_dim}"

    # Encoded latent filenames (written by encode_dataset_every)
    ae_latents_file:      str = "ae_encoded_ms_latents.h5"
    graphae_latents_file: str = "graphae_encoded_ms_latents.h5"
    diffae_latents_file:  str = "encoded_ms_latents.h5"

    def run_subdir(self, subdir_attr: str, latent_dim: int) -> str:
        return getattr(self, subdir_attr).format(latent_dim=latent_dim)

    def run_dirs(self, subdir_attr: str, latent_dim: int) -> tuple[str, str]:
        subdir = self.run_subdir(subdir_attr, latent_dim)
        return os.path.join(self.checkpoint_dir, subdir), os.path.join(self.plot_dir, subdir)


# =============================================================================
# Complete Configuration
# =============================================================================

@dataclass
class Config:
    """Complete configuration container.
    
    Groups all config sections. Access via:
        cfg.model.hidden_dim
        cfg.training.lr
        cfg.encoder.latent_dim
        etc.
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    ms_data: MSDataConfig = field(default_factory=MSDataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    aux_task: AuxTaskConfig = field(default_factory=AuxTaskConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    # Runtime flags
    device: Optional[str] = None    # Device override (None = auto-detect)
    resume: bool = True             # Resume from checkpoint if available
    visualize: bool = True          # Generate visualizations during training

    def copy(self) -> "Config":
        return deepcopy(self)

    def update(self, **overrides: Any) -> "Config":
        for key, value in overrides.items():
            _apply_override(self, key, value)
        return self

    def updated(self, **overrides: Any) -> "Config":
        return self.copy().update(**overrides)


# =============================================================================
# Helpers
# =============================================================================

default_config = Config()


def _apply_override(cfg: Config, key: str, value: Any) -> None:
    if key in _OVERRIDE_ALIASES:
        section_name, attr_name = _OVERRIDE_ALIASES[key]
        setattr(getattr(cfg, section_name), attr_name, value)
        return

    if hasattr(cfg, key):
        setattr(cfg, key, value)
        return

    for section_name in _SECTION_NAMES:
        section = getattr(cfg, section_name)
        if hasattr(section, key):
            setattr(section, key, value)
            return

    raise KeyError(f"Unknown config override: {key}")


def get_config(**overrides) -> Config:
    """Create config with optional overrides.
    
    Searches for keys in nested configs:
        get_config(lr=1e-4)           # sets training.lr
        get_config(latent_dim=128)    # sets encoder.latent_dim
        get_config(model_hidden_dim=64)   # sets model.hidden_dim
        get_config(encoder_hidden_dim=64) # sets encoder.hidden_dim
    """
    return Config().update(**overrides)


def print_config(cfg: Config, include_encoder: bool = False, include_ms: bool = False) -> None:
    """Print configuration summary."""
    print("=" * 50)
    print("Configuration")
    print("=" * 50)
    
    print(f"\nModel (diffusion graph U-Net):")
    print(f"  hidden_dim: {cfg.model.hidden_dim}")
    print(f"  depth: {cfg.model.depth}")
    print(f"  blocks_per_stage: {cfg.model.blocks_per_stage}")
    print(f"  pool_ratio: {cfg.model.pool_ratio}")
    
    print(f"\nDiffusion:")
    print(f"  timesteps: {cfg.diffusion.timesteps}")
    print(f"  parametrization: {cfg.diffusion.parametrization}")
    
    if include_encoder:
        print(f"\nEncoder:")
        print(f"  latent_dim: {cfg.encoder.latent_dim}")
        print(f"  graph_hidden_dim: {cfg.encoder.hidden_dim}")
        print(f"  latent_head_dim: {cfg.encoder.latent_head_dim or 'auto'}")
        print(f"  conv_channels: {cfg.encoder.conv_channels}")
        print(f"  mlp_hidden_dim: {cfg.encoder.mlp_hidden_dim}")
        print(f"  mlp_decoder_hidden_dim: {cfg.encoder.mlp_decoder_hidden_dim or 'auto'}")
        print(f"  regressive_hidden_dim: {cfg.encoder.regressive_hidden_dim}")
        print(f"  latent_anchor_dim: {cfg.encoder.latent_anchor_dim or 'auto'}")
        print(f"  latent_anchor_count: {cfg.encoder.latent_anchor_count}")
        print(f"  latent_anchor_value_dim: {cfg.encoder.latent_anchor_value_dim}")
        print(f"  depth: {cfg.encoder.depth}")
        print(f"  encoder_type: {cfg.encoder.encoder_type}")
        print(f"  decoder_type: {cfg.encoder.decoder_type}")
        print(f"  stochastic: {cfg.encoder.use_stochastic}")
    
    if include_ms:
        print(f"\nMS Data:")
        print(f"  delta_range: [{cfg.ms_data.delta_min}, {cfg.ms_data.delta_max}] bins")
        print(f"  ns_per_bin: {cfg.ms_data.ns_per_bin}")
    
    print(f"\nTraining:")
    print(f"  lr: {cfg.training.lr}")
    print(f"  batch_size: {cfg.training.batch_size}")
    print(f"  epochs: {cfg.training.epochs}")
    print(f"  encode_every: {cfg.training.encode_dataset_every} epochs")
    print(f"  encode_n_samples: {cfg.training.encode_n_samples}")
    
    print(f"\nGraph:")
    print(f"  radius: {cfg.graph.radius}")
    print(f"  z_hops: {cfg.graph.z_hops}")
    print("=" * 50)

 
