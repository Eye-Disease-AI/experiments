import argparse

import mlflow
import numpy as np
from scipy import stats
from experiments.lib.mlflow_setup import Experiment
from experiments.lib.reproducibility import get_trial_vals


def calculate_ci(
    values: list[float], do_bootstrap_simulation=False, n_bootstrap=10_000, ci=0.95
) -> dict:
    arr = np.array(values)
    mean = np.mean(arr)
    median = np.median(arr)
    alpha = (1 - ci) / 2
    standard_error = stats.sem(arr)
    std = np.std(arr, ddof=1)

    # bootstrap simulation,
    # It randomly draws the values of metrics with return
    # (like how we would do with the DATA SAMPLES in a bootstrapping study)
    # and simulates multiple validation runs that way to estimate the distribution.
    # Not really a scientifically backed up solution, just felt like it could
    # give some interesting results
    if do_bootstrap_simulation:
        if len(values) < n_bootstrap:
            rng = np.random.default_rng(0)
            bstrap_vals = [
                np.mean(rng.choice(arr, size=len(arr), replace=True))
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
    out = {
        "mean": mean,
        "median": median,
        "std": std,
        "t_st_ci_low": t_st_ci_low,
        "t_st_ci_high": t_st_ci_high,
        "t_st_ci_width": t_st_ci_high - t_st_ci_low,
        "z_st_ci_low": z_st_ci_low,
        "z_st_ci_high": z_st_ci_high,
        "z_st_ci_width": z_st_ci_high - z_st_ci_low,
    }
    if do_bootstrap_simulation:
        out |= {
            "bstrap_ci_low": bstrap_ci_low,
            "bstrap_ci_high": bstrap_ci_high,
            "bstrap_ci_width": bstrap_ci_high - bstrap_ci_low,
        }
    out |= {"n": len(arr)}
    return out


def print_ci(vals, metric_name, optuna_metric, do_bootstrap_simulation=False):
    ci = calculate_ci(vals, do_bootstrap_simulation)
    marker = " (The optimized metric)" if metric_name == optuna_metric else ""
    rows = [
        ("Values:", f"{[f'{x:.4f}' for x in vals[:10]]}..."),
        ("Trials:", f"{ci['n']}"),
        ("Mean:", f"{ci['mean']:.4f}"),
        ("Median:", f"{ci['median']:.4f}"),
        ("Std:", f"{ci['std']:.4f}"),
        (
            "Z-score 95% CI:",
            f"[{ci['z_st_ci_low']:.4f}, {ci['z_st_ci_high']:.4f}], +- {ci['z_st_ci_width'] / 2:.4f}",
        ),
        (
            "T-student 95% CI:",
            f"[{ci['t_st_ci_low']:.4f}, {ci['t_st_ci_high']:.4f}], +- {ci['t_st_ci_width'] / 2:.4f}",
        ),
    ]
    if do_bootstrap_simulation:
        rows.append(
            (
                "Bootstrap 95% CI:",
                f"[{ci['bstrap_ci_low']:.4f}, {ci['bstrap_ci_high']:.4f}], +- {ci['bstrap_ci_width'] / 2:.4f}",
            )
        )
    print(f"\n--- {metric_name}{marker} ---")
    w = max(len(r[0]) for r in rows)
    for label, value in rows:
        print(f"{label:{w}}  {value}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--run_to_verify", type=str)
    parser.add_argument("--optuna_metric", type=str)
    return parser.parse_args()


def main():
    args = parse_args()

    assert args.experiment_name is not None, "experiment_name is required"
    assert args.run_to_verify is not None, "run_to_verify is required"
    assert args.optuna_metric is not None, "optuna_metric is required"

    exp = Experiment(args.experiment_name)
    mlflow.set_tracking_uri("http://localhost:5000")
    trial_metrics = get_trial_vals(exp, args.run_to_verify, only_parent_run=False)

    if not trial_metrics:
        print("No completed trials found.")
    else:
        print(f"Optimized metric: {args.optuna_metric}")
        all_metric_names = sorted({k for m in trial_metrics for k in m})
        for metric_name in all_metric_names:
            vals = [m[metric_name] for m in trial_metrics if metric_name in m]
            print_ci(vals, metric_name, args.optuna_metric)


if __name__ == "__main__":
    main()
