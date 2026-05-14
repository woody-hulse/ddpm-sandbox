# SGDA

Graph-based autoencoder and diffusion experiments for event-waveform modeling, reconstruction, latent analysis, and visualization.

The repo contains:
- core model/training code for `AE`, `DiffAE`, `GraphAE`, `AE Light`, and `DiffAE Light`
- analysis and evaluation scripts for reconstruction quality, latent structure, anomaly probing, and auxiliary tasks
- visualization utilities for events, manifolds, reduced quantities, and scaling plots
- data utilities for waveform compression, synthetic event generation, and event-database benchmarking

## Repository layout

- [config.py](/Users/woodyhulse/Documents/sgda/config.py): central configuration and default paths
- [ae.py](/Users/woodyhulse/Documents/sgda/ae.py), [diffae.py](/Users/woodyhulse/Documents/sgda/diffae.py), [graphae.py](/Users/woodyhulse/Documents/sgda/graphae.py), [ae_light.py](/Users/woodyhulse/Documents/sgda/ae_light.py), [diffae_light.py](/Users/woodyhulse/Documents/sgda/diffae_light.py): main training code
- [data_loader.py](/Users/woodyhulse/Documents/sgda/data_loader.py), [data.py](/Users/woodyhulse/Documents/sgda/data.py): data loading and graph construction
- [models](/Users/woodyhulse/Documents/sgda/models), [diffusion](/Users/woodyhulse/Documents/sgda/diffusion), [utils](/Users/woodyhulse/Documents/sgda/utils): reusable model and utility code
- [tools/analysis](/Users/woodyhulse/Documents/sgda/tools/analysis): evaluation, probing, and comparison scripts
- [tools/visualization](/Users/woodyhulse/Documents/sgda/tools/visualization): plotting and event-viewer scripts
- [tools/data](/Users/woodyhulse/Documents/sgda/tools/data): data transformation and synthetic-data scripts
- [tools/experiments](/Users/woodyhulse/Documents/sgda/tools/experiments): one-off experimental workflows
- [diagnose](/Users/woodyhulse/Documents/sgda/diagnose): targeted debugging and probing scripts

## Data and outputs

Default input paths are defined in [config.py](/Users/woodyhulse/Documents/sgda/config.py):
- `data/tritium_ss_42.h5`
- `data/pmt_xy_42.h5`

Common output directories:
- `checkpoints/`: model checkpoints and encoded latent datasets
- `figures/`: generated figures, evaluation artifacts, and experiment plots

Run commands from the repository root so relative data/output paths resolve correctly.

## Environment

There is no committed `requirements.txt` or `pyproject.toml`. A typical environment needs:
- Python 3.12
- `torch`
- `numpy`
- `scipy`
- `matplotlib`
- `h5py`
- `tqdm`

Some tools also need optional packages:
- `scikit-learn`, `umap-learn` for manifold analysis
- `lmdb` for `event_database/`
- `psutil` for scaling measurements
- `seaborn`, `Pillow` for specific plotting/animation paths

## How to run code

### Train models

These scripts train immediately when run:

```bash
python ae.py
python diffae.py
python graphae.py
python ae_light.py
python diffae_light.py
```

The defaults come from [config.py](/Users/woodyhulse/Documents/sgda/config.py). For code-driven runs, use `get_config(...)` to override fields.

### Run evaluation and analysis

Representative entry points:

```bash
python -m tools.analysis.eval_recon --n-events 1024 --output-dir figures/eval_recon
python -m tools.analysis.light_eval
python -m tools.analysis.compare_latent_sizes --latent-dims 4 8 16 32 64
python -m tools.analysis.compare_rqs --n-samples 500
python -m tools.analysis.graph_aux --latent-dim 64
python -m tools.analysis.anomaly_probe --n-events 512
```

Most analysis and visualization commands now default to a subdirectory of `figures/`.

### Visualize events and latent structure

```bash
python -m tools.visualization.view_events --view --n 8
python -m tools.visualization.view_events --compress 3 --model both
python -m tools.visualization.plot_umap --latent-dim 64
python -m tools.visualization.plot_rq_distributions --latent-dim 64
python -m tools.visualization.plot_real_event_3d --help
python -m tools.visualization.plot_tpc_graph_3d --help
python -m tools.visualization.plot_diffae_scaling
python -m tools.visualization.synthetic_visualizer --help
```

### Data utilities

```bash
python -m tools.data.compress --help
python -m tools.data.generate_synthetic --help
python -m event_database.build_db
python -m event_database.benchmark
```

### Experiments

```bash
python -m tools.experiments.fewshot_triple_scatter --help
python -m tools.experiments.train_time_reversal --help
```

## Tests

Focused tests currently used for verification:

```bash
pytest tests/test_run_paths.py tests/test_graclus.py
```

## Notes

- The repository is script-oriented rather than packaged as an installable library.
- Most analysis and visualization entry points are now under `tools/` and should be run with `python -m ...`.
- A short structural guide is also available in [docs/repository_map.md](/Users/woodyhulse/Documents/sgda/docs/repository_map.md).
