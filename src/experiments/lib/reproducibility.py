import subprocess
import lightning as L

from experiments.lib.mlflow_setup import Experiment


def global_seed_rng(seed):
    L.seed_everything(seed)


def get_git_sha() -> str:
    sha = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    return f"{sha}-dirty" if dirty else sha


def get_trial_vals(
    exp: Experiment,
    study_name: str,
    only_crossvalidation_trials=False,
    only_parent_run=True,
) -> list[dict[str, float]]:
    mlflow_exp = exp.client.get_experiment_by_name(exp.experiment_name)
    filter_string = f"tags.optuna_study = '{study_name}'"
    if only_crossvalidation_trials:
        filter_string += " and tags.validation_sample = 'true'"
    if only_parent_run:
        filter_string += " and tags.best_trial_number != ''"
    runs = exp.client.search_runs(
        experiment_ids=[mlflow_exp.experiment_id],
        filter_string=filter_string,
    )
    print(f"Found {len(runs)} validation-sample runs for study '{study_name}'.")
    result = []
    for r in runs:
        if r.data.metrics:
            result.append(r.data.metrics)
        else:
            print(f"WARNING: Run {r.info.run_id} is empty!")
    return result
