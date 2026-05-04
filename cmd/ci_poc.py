import argparse

import mlflow
import numpy as np
from scipy import stats

from lib.mlflow_setup import Experiment


def get_trial_vals(
    exp: Experiment, study_name: str, metric_name: str = "val_acc"
) -> list[float]:
    mlflow_exp = exp.client.get_experiment_by_name(exp.experiment_name)
    runs = exp.client.search_runs(
        experiment_ids=[mlflow_exp.experiment_id],  # pyright: ignore
        filter_string=f"tags.optuna_study = '{study_name}'",
    )
    return [
        r.data.metrics[metric_name]
        for r in runs
        if metric_name in r.data.metrics
        and r.data.tags.get("pruned") != "true"  # filter in Python instead
    ]


def bootstrap_ci(values: list[float], n_bootstrap=10_000, ci=0.95) -> dict:
    arr = np.array(values)
    mean = np.mean(arr)
    alpha = (1 - ci) / 2
    standard_error = stats.sem(arr)

    # bootstrap simulation,
    if len(values) < n_bootstrap:
        bstrap_vals = [
            np.mean(np.random.choice(arr, size=len(arr), replace=True))
            for _ in range(n_bootstrap)
        ]
    else:
        bstrap_vals = arr
    bstrap_ci_low = np.percentile(bstrap_vals, alpha * 100)
    bstrap_ci_high = np.percentile(bstrap_vals, (1 - alpha) * 100)

    # t-student

    t_st_ci_low, t_st_ci_high = stats.t.interval(
        ci, df=len(arr) - 1, loc=mean, scale=standard_error
    )
    # z-score
    z_st_ci_low, z_st_ci_high = stats.norm.interval(ci, loc=mean, scale=standard_error)

    return {
        "mean": np.mean(arr),
        "median": np.median(arr),
        "std": np.std(arr),
        "t_st_ci_low": t_st_ci_low,
        "t_st_ci_high": t_st_ci_high,
        "z_st_ci_low": z_st_ci_low,
        "z_st_ci_high": z_st_ci_high,
        "bstrap_ci_low": bstrap_ci_low,
        "bstrap_ci_high": bstrap_ci_high,
        "n": len(arr),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    EXPERIMENT_NAME = "nuclear-cataract-test"
    STUDY_NAME = f"{EXPERIMENT_NAME}/lr-search5"
    exp = Experiment(EXPERIMENT_NAME)
    mlflow.set_tracking_uri("http://localhost:5000")
    losses = get_trial_vals(exp, STUDY_NAME)

    if not losses:
        print("No completed trials found.")
    else:
        ci = bootstrap_ci(losses)
        rows = [
            ("Values:", f"{[f'{x:.4f}' for x in losses[:10]]}..."),
            ("Trials:", f"{ci['n']}"),
            ("Mean val_acc:", f"{ci['mean']:.4f}"),
            ("Median:", f"{ci['median']:.4f}"),
            ("Std:", f"{ci['std']:.4f}"),
            (
                "Bootstrap 95% CI:",
                f"[{ci['bstrap_ci_low']:.4f}, {ci['bstrap_ci_high']:.4f}]",
            ),
            ("Z-score 95% CI:", f"[{ci['z_st_ci_low']:.4f}, {ci['z_st_ci_high']:.4f}]"),
            (
                "T-student 95% CI:",
                f"[{ci['t_st_ci_low']:.4f}, {ci['t_st_ci_high']:.4f}]",
            ),
        ]

        w = max(len(r[0]) for r in rows)
        for label, value in rows:
            print(f"{label:{w}}  {value}")
