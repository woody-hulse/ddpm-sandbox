import os
from pathlib import Path

import pytest

from config import get_config
from utils.run_paths import (
    epoch_plot_dir,
    latest_checkpoint,
    latest_checkpoint_across_runs,
    resolve_model_run_dirs,
)


def test_get_config_supports_nested_path_graph_and_conditioning_overrides() -> None:
    cfg = get_config(
        latent_dim=16,
        plot_dir="custom-plots",
        lpe_dim=8,
        time_dim=32,
        ae_light_subdir="ae_light_custom_z{latent_dim}",
    )

    assert cfg.encoder.latent_dim == 16
    assert cfg.paths.plot_dir == "custom-plots"
    assert cfg.graph.lpe_dim == 8
    assert cfg.conditioning.time_dim == 32
    assert cfg.paths.ae_light_subdir == "ae_light_custom_z{latent_dim}"


def test_config_updated_returns_independent_copy() -> None:
    base = get_config(latent_dim=32)
    updated = base.updated(latent_dim=8, skip_scale=0.5)

    assert base.encoder.latent_dim == 32
    assert updated.encoder.latent_dim == 8
    assert base.model.skip_scale == 1.0
    assert updated.model.skip_scale == 0.5


def test_get_config_rejects_unknown_overrides() -> None:
    with pytest.raises(KeyError, match="Unknown config override"):
        get_config(not_a_real_option=123)


def test_training_steps_resolve_from_dataset_or_override() -> None:
    cfg = get_config(batch_size=8)
    assert cfg.training.resolved_steps_per_epoch(100) == 12

    cfg.training.steps_per_epoch = 7
    assert cfg.training.resolved_steps_per_epoch(100) == 7


def test_resolve_model_run_dirs_uses_config_templates(tmp_path: Path) -> None:
    cfg = get_config(latent_dim=32)
    cfg.paths.checkpoint_dir = str(tmp_path / "checkpoints")
    cfg.paths.plot_dir = str(tmp_path / "plots")

    checkpoint_dir, plot_dir = resolve_model_run_dirs(cfg, "diffae_light_subdir", create=True)

    assert checkpoint_dir == str(tmp_path / "checkpoints" / "diffae_light_z32")
    assert plot_dir == str(tmp_path / "plots" / "diffae_light_z32")
    assert Path(checkpoint_dir).is_dir()
    assert Path(plot_dir).is_dir()


def test_latest_checkpoint_helpers_pick_highest_epoch(tmp_path: Path) -> None:
    run_dir = tmp_path / "checkpoints" / "ae_z64"
    run_dir.mkdir(parents=True)
    for epoch in (1, 25, 3):
        (run_dir / f"ae_epoch_{epoch:04d}.pt").write_text("x")

    assert latest_checkpoint(str(run_dir), "ae_epoch_*.pt") == str(run_dir / "ae_epoch_0025.pt")


def test_latest_checkpoint_across_runs_prefers_most_recent_file(tmp_path: Path) -> None:
    parent = tmp_path / "checkpoints"
    older = parent / "ae_light_z32"
    newer = parent / "ae_light_z64"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    older_file = older / "ae_light_epoch_0005.pt"
    newer_file = newer / "ae_light_epoch_0002.pt"
    older_file.write_text("old")
    newer_file.write_text("new")
    os.utime(older_file, (1, 1))
    os.utime(newer_file, (2, 2))

    result = latest_checkpoint_across_runs(
        str(parent / "ae_light_z16"),
        "ae_light_z",
        "ae_light_epoch_*.pt",
    )

    assert result == str(newer_file)


def test_epoch_plot_dir_creates_epoch_subdirectory(tmp_path: Path) -> None:
    path = epoch_plot_dir(str(tmp_path / "plots"), 50)
    assert path == str(tmp_path / "plots" / "epoch_50")
    assert Path(path).is_dir()
