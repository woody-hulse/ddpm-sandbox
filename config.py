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
from dataclasses import dataclass, field
from typing import Optional


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
    hidden_dim: int = 32            # Hidden dimension in graph convolutions
    depth: int = 3                  # Number of pooling/unpooling stages
    blocks_per_stage: int = 2       # Residual blocks per stage
    pool_ratio: float = 0.7         # Fraction of nodes to keep per pooling
    dropout: float = 0.0            # Dropout rate
    pos_dim: int = 3                # Position embedding dimension (x, y, z)
    pos_dropout: float = 0.0        # Dropout on position embeddings
    cache_norm_top: bool = True     # Cache normalization at top level


@dataclass
class EncoderConfig:
    """Graph encoder parameters for DiffAE and Graph AE.

    The encoder maps input graphs to latent representations.
    use_stochastic and kl_weight apply only to DiffAE's encoder.
    decoder_type applies only to Graph AE: "graph" or "mlp".
    """
    latent_dim: int = 64            # Latent representation dimension
    hidden_dim: int = 32            # Hidden dimension in encoder layers
    depth: int = 4                  # Number of pooling stages
    blocks_per_stage: int = 2       # Residual blocks per stage
    pool_ratio: float = 0.5         # Pooling ratio per stage
    dropout: float = 0.0            # Dropout rate
    use_stochastic: bool = False    # DiffAE encoder: stochastic (VAE-style) encoding
    kl_weight: float = 0.001        # DiffAE: KL weight when use_stochastic
    encoder_type: str = "mlp"     # Encoder: "graph" or "mlp"
    decoder_type: str = "mlp"     # AE decoder: "graph" (SimpleGraphDecoder) or "mlp"
    mlp_encoder_layers: int = 3     # MLP encoder: number of hidden layers (only if encoder_type="mlp")
    mlp_decoder_layers: int = 3     # MLP decoder: number of hidden layers (only if decoder_type="mlp")
    use_regressive_head: bool = True   # DiffAE: add a second decoder head with regressive (MSE) loss
    regressive_head_weight: float = 1.0 # DiffAE: weight for the regressive head loss


@dataclass
class ConditioningConfig:
    """Conditioning network parameters for DiffAE.
    
    Controls how latent codes condition the diffusion process.
    """
    cond_in_dim: int = 5            # Raw condition input dimension
    cond_proj_dim: int = 64         # Projected condition dimension
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
    p2_gamma: float = 0.5           # P2 loss weighting gamma
    p2_k: float = 1.0               # P2 loss weighting k


# =============================================================================
# Graph Construction  
# =============================================================================

@dataclass 
class GraphConfig:
    """Graph construction parameters for 3D spatio-temporal graphs.
    
    Defines how nodes (channel × time) are connected.
    """
    radius: float = 16.0            # Spatial radius for within-layer adjacency (cm)
    z_sep: float = 20.0             # Z-spacing between time layers
    z_hops: int = 4                 # Cross-layer connectivity distance


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
    epochs: int = 10_000            # Total training epochs
    batch_size: int = 8             # Batch size
    steps_per_epoch: int = 64       # Gradient steps per epoch
    lr: float = 1e-3                # Learning rate
    weight_decay: float = 0         # AdamW weight decay
    ema_decay: float = 0.999        # Exponential moving average decay
    grad_clip: float = 1.0          # Gradient clipping norm
    
    # Checkpointing
    checkpoint_every: int = 100     # Save checkpoint every N epochs
    visualize_every: int = 100      # Generate visualizations every N epochs
    
    # Encoded dataset export (for aux task)
    encode_dataset_every: int = 500    # Export encoded latents every N epochs (0 = disable)
    encode_n_samples: int = 500_000       # Number of MS samples to encode and save


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
    output_dir: str = "aux_results" # Directory for aux task outputs


# =============================================================================
# File Paths
# =============================================================================

@dataclass
class PathConfig:
    """File paths for data and outputs.
    
    Checkpoint/plot subdirs use {latent_dim} placeholder for organization.
    """
    # Input data
    tritium_h5: str = "data/tritium_ss_single_node.h5"
    channel_positions: str = "data/pmt_xy_single_node.h5"
    
    # Output directories
    checkpoint_dir: str = "checkpoints"
    plot_dir: str = "plots"
    
    # Subdirectory templates (use .format(latent_dim=N))
    diffae_subdir: str = "diffae_z{latent_dim}"
    ae_subdir: str = "ae_z{latent_dim}"
    graph_ae_subdir: str = "graph_ae_z{latent_dim}"


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
    resume: bool = False             # Resume from checkpoint if available
    visualize: bool = True          # Generate visualizations during training


# =============================================================================
# Helpers
# =============================================================================

default_config = Config()


def get_config(**overrides) -> Config:
    """Create config with optional overrides.
    
    Searches for keys in nested configs:
        get_config(lr=1e-4)           # sets training.lr
        get_config(latent_dim=128)    # sets encoder.latent_dim
        get_config(hidden_dim=64)     # sets model.hidden_dim
    """
    cfg = Config()
    
    for key, value in overrides.items():
        # Check top-level
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        # Check nested configs
        elif hasattr(cfg.training, key):
            setattr(cfg.training, key, value)
        elif hasattr(cfg.encoder, key):
            setattr(cfg.encoder, key, value)
        elif hasattr(cfg.model, key):
            setattr(cfg.model, key, value)
        elif hasattr(cfg.diffusion, key):
            setattr(cfg.diffusion, key, value)
        elif hasattr(cfg.ms_data, key):
            setattr(cfg.ms_data, key, value)
        elif hasattr(cfg.aux_task, key):
            setattr(cfg.aux_task, key, value)
    
    return cfg


def print_config(cfg: Config, include_encoder: bool = False, include_ms: bool = False) -> None:
    """Print configuration summary."""
    print("=" * 50)
    print("Configuration")
    print("=" * 50)
    
    print(f"\nModel (decoder):")
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
        print(f"  hidden_dim: {cfg.encoder.hidden_dim}")
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
