import numpy as np
import mlflow
from lib.mlflow_setup import Experiment

def get_trial_val_losses(exp: Experiment, study_name: str) -> list[float]:
    mlflow_exp = exp.client.get_experiment_by_name(exp.experiment_name)
    runs = exp.client.search_runs(
        experiment_ids=[mlflow_exp.experiment_id],
        filter_string=f"tags.optuna_study = '{study_name}'",
        order_by=["metrics.val_loss ASC"],
    )
    return [
        r.data.metrics["val_loss"]
        for r in runs
        if "val_loss" in r.data.metrics
        and r.data.tags.get("pruned") != "true"  # filter in Python instead
    ]


def bootstrap_ci(values: list[float], n_bootstrap=10_000, ci=0.95) -> dict:
    arr = np.array(values)
    means = [np.mean(np.random.choice(arr, size=len(arr), replace=True)) for _ in range(n_bootstrap)]
    alpha = (1 - ci) / 2
    return {
        "mean":    np.mean(arr),
        "median":  np.median(arr),
        "std":     np.std(arr),
        "ci_low":  np.percentile(means, alpha * 100),
        "ci_high": np.percentile(means, (1 - alpha) * 100),
        "n":       len(arr),
    }


if __name__ == "__main__":
    EXPERIMENT_NAME = "nuclear-cataract-test"
    STUDY_NAME = f"{EXPERIMENT_NAME}/lr-search4"
    exp = Experiment(EXPERIMENT_NAME)
    mlflow.set_tracking_uri("http://localhost:5000")
    losses = get_trial_val_losses(exp, STUDY_NAME)

    if not losses:
        print("No completed trials found.")
    else:
        ci = bootstrap_ci(losses)
        print(f"Trials:        {ci['n']}")
        print(f"Mean val_loss: {ci['mean']:.4f}")
        print(f"Median:        {ci['median']:.4f}")
        print(f"Std:           {ci['std']:.4f}")
        print(f"95% CI:        [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]")