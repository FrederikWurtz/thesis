Master package refactor
=======================

This folder has been reorganized into a small Python package to make the
codebase easier to import and maintain. The structure under
`Speciale/master/` is now:

- `configs/`          — (placeholder) configuration files
- `data/`             — data generation utilities (DEM, renderer wrappers)
- `render/`           — physics rendering components (DEM, Camera, Hapke, Renderer)
- `models/`           — (placeholder) model definitions (future)
- `train/`            — training modules (split into `trainer_core`, `checkpoints`, `cli`)
- `scripts/`          — CLI wrappers (run_train.py, gen_data.py)
- `utils/`            — plotting and helper utilities
- `__init__.py`       — package root

Quick usage
-----------

Run a fast smoke training step (no heavy compute):

```bash
python Speciale/master/scripts/run_train.py --smoke
```

Generate a small dataset (example):

```bash
python Speciale/master/scripts/gen_data.py --outdir ./data_small --n 20
```

Tests
-----

Run the package tests (pytest required):

```bash
python -m pytest Speciale/master/tests -q
```

Next steps
----------
- Split the larger scripts (`train_unet_mps.py`, `create_training_data.py`) fully
  into the package modules and remove the legacy top-level shims once tests
  and smoke runs are green.
- Add `models/unet.py` and `train/checkpoints.py` refinements.
- Add a top-level `pyproject.toml` or `requirements.txt` to capture deps.

If you want, I can continue the migration: move the remaining logic into the
package, add CI, and clean up the legacy files.
