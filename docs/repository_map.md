# Repository Map

## Main code paths

- `config.py`: central configuration for model architecture, training, graph construction, and output paths.
- `data_loader.py`: dataset access, graph construction, and online multi-scatter batching.
- `ae.py`: baseline autoencoder training, checkpointing, reconstructions, and latent export.
- `diffae.py`: main DiffAE training and sampling path.
- `graphae.py`: graph autoencoder variant and GraphAE checkpoint/export flow.
- `ae_light.py`, `diffae_light.py`: lighter graph pyramid variants with their own training and latent export paths.
- `models/graph_unet.py`: core graph U-Net, pooling, and sparse graph layers used across the graph models.
- `diffusion/schedule.py`: cosine schedule and sinusoidal embedding utilities.
- `tools/`: runnable workflows organized by purpose:
  - `tools/analysis/`: evaluation, probing, and comparison scripts
  - `tools/visualization/`: plots, event viewers, and figure generation
  - `tools/data/`: dataset reshaping and synthetic-data utilities
  - `tools/experiments/`: one-off experimental training workflows

## Evaluation and analysis

- `tools/analysis/eval_recon.py`: reconstruction-quality evaluation and summary plots.
- `tools/analysis/light_eval.py`: consolidated inference-only evaluation flow for `ae_light.py` and `diffae_light.py`.
- `tools/analysis/compare_latent_sizes.py`: latent-size sweep for AE vs DiffAE.
- `tools/analysis/compare_rqs.py` and `tools/visualization/plot_rq_distributions.py`: reconstructed-quantity extraction and distribution overlays.
- `tools/analysis/graph_aux.py` and `aux.py`: downstream latent probes and auxiliary regression tasks.
- `tools/analysis/anomaly_probe.py`, `tools/visualization/plot_umap.py`, and `diagnose/`: probing, embedding analysis, and targeted debugging scripts.

## Plotting code and figure outputs

- Plotting and event-viewer scripts now live under `tools/visualization/`.
- Shared styling is centralized in `plot_style.py`.
- Generated figures now default under `figures/`, with each workflow writing to its own subdirectory.

## Stylistic refactor notes

- Output directory templates are now centralized in `config.py`, including the light-model variants.
- Shared checkpoint/plot path logic now lives in `utils/run_paths.py` instead of being duplicated across model entry points.
- `event_database/` now behaves like an actual package, with module-safe imports and path handling that does not depend on the current working directory.

## Remaining cleanup candidates

- Output artifacts are still committed alongside source in several top-level result directories. If the repo should become code-only, the next step would be moving these under a dedicated `artifacts/` root or untracking them.
