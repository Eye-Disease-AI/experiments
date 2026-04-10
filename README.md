# Experiments

ML experiments repo. Results are tracked in MLflow.

## Structure

- `data/` - datasets, not committed to git
- `lib/` - shared code (preprocessing, metrics, utils)
- `experiments/` - one directory per experiment, single `.py` script per experiment optionally an exploration notebook.

## Conventions

- An experiment is considered done when it has a `run.py` that can be run end-to-end without manual steps.
- Notebooks are for exploration only.
