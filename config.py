"""
Hyperparameter configuration for DDPM training.
Edit values here to change training behavior.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Graph U-Net architecture parameters."""
    in_dim: int = 1
    out_dim: int = 1
    hidden_dim: int = 64
    depth: int = 2
    blocks_per_stage: int = 2
    pool_ratio: float = 0.35
    dropout: float = 0.00
    pos_dim: int = 3
    pos_dropout: float = 0.0
    cache_norm_top: bool = True


@dataclass
class DiffusionConfig:
    """Diffusion process parameters."""
    timesteps: int = 250
    parametrization: str = "v"  # "v" or "eps"
    p2_gamma: float = 0.0       # P2 loss weighting (0 = disabled)
    p2_k: float = 1.0


@dataclass
class ConditioningConfig:
    """Conditioning network parameters."""
    cond_in_dim: int = 5        # Input condition dimension
    cond_proj_dim: int = 32     # Projected condition dimension  
    time_dim: int = 64          # Time embedding dimension


@dataclass 
class GraphConfig:
    """Graph construction parameters."""
    radius: float = 20.0        # Spatial radius for adjacency
    z_sep: float = 5.0          # Z separation between time layers
    z_hops: int = 2             # Number of z-hops for connectivity


@dataclass
class TrainingConfig:
    """Training loop parameters."""
    epochs: int = 10_000
    batch_size: int = 8
    steps_per_epoch: int = 64   # Optimizer steps per epoch
    lr: float = 5e-5
    weight_decay: float = 1e-4
    ema_decay: float = 0.9995
    grad_clip: float = 1.0
    
    # Checkpointing
    checkpoint_every: int = 100
    visualize_every: int = 100


@dataclass
class PathConfig:
    """File paths."""
    tritium_h5: str = "data/tritium_ss_42.h5"
    channel_positions: str = "data/pmt_xy_42.h5"
    checkpoint_dir: str = "checkpoints"
    plot_dir: str = "plots"


@dataclass
class Config:
    """Complete configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    # Runtime
    device: Optional[str] = None
    resume: bool = True
    visualize: bool = True


# Default configuration instance
default_config = Config()


def get_config(**overrides) -> Config:
    """Get config with optional overrides."""
    cfg = Config()
    
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        elif hasattr(cfg.training, key):
            setattr(cfg.training, key, value)
        elif hasattr(cfg.model, key):
            setattr(cfg.model, key, value)
        elif hasattr(cfg.diffusion, key):
            setattr(cfg.diffusion, key, value)
    
    return cfg


def print_config(cfg: Config) -> None:
    """Print configuration summary."""
    print("=" * 50)
    print("Configuration")
    print("=" * 50)
    print(f"\nModel:")
    print(f"  hidden_dim: {cfg.model.hidden_dim}")
    print(f"  depth: {cfg.model.depth}")
    print(f"  blocks_per_stage: {cfg.model.blocks_per_stage}")
    print(f"  pool_ratio: {cfg.model.pool_ratio}")
    print(f"  dropout: {cfg.model.dropout}")
    
    print(f"\nDiffusion:")
    print(f"  timesteps: {cfg.diffusion.timesteps}")
    print(f"  parametrization: {cfg.diffusion.parametrization}")
    print(f"  p2_gamma: {cfg.diffusion.p2_gamma}")
    
    print(f"\nTraining:")
    print(f"  lr: {cfg.training.lr}")
    print(f"  batch_size: {cfg.training.batch_size}")
    print(f"  ema_decay: {cfg.training.ema_decay}")
    print(f"  epochs: {cfg.training.epochs}")
    
    print(f"\nGraph:")
    print(f"  radius: {cfg.graph.radius}")
    print(f"  z_hops: {cfg.graph.z_hops}")
    print("=" * 50)
