import mlflow
import numpy as np
from scipy import stats
import sys
from lib.mlflow_setup import Experiment
from experiment import common_config

def get_trial_vals(exp: Experiment, study_name: str) -> list[dict[str, float]]:
    mlflow_exp = exp.client.get_experiment_by_name(exp.experiment_name)
    runs = exp.client.search_runs(
        experiment_ids=[mlflow_exp.experiment_id],  # pyright: ignore
        filter_string=(
            f"tags.optuna_study = '{study_name}' "
            f"and tags.validation_sample = 'true'"
        ),
    )
    print(f"Found {len(runs)} validation-sample runs for study '{study_name}'.")
    result = []
    for r in runs:
        sample = {}
        for k, v in r.data.metrics.items():
            if k.startswith("best_val_"):
                sample[k[len("best_"):]] = v
        if sample:
            result.append(sample)
        else:
            print(f"WARNING: Run {r.info.run_id} is empty!")
    return result


def calculate_ci(values: list[float], do_bootstrap_simulation=False, n_bootstrap=10_000, ci=0.95) -> dict:
    arr = np.array(values)
    mean = np.mean(arr)
    alpha = (1 - ci) / 2
    standard_error = stats.sem(arr)

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
        "mean": np.mean(arr),
        "median": np.median(arr),
        "std": np.std(arr),
        "t_st_ci_low": t_st_ci_low,
        "t_st_ci_high": t_st_ci_high,
        "z_st_ci_low": z_st_ci_low,
        "z_st_ci_high": z_st_ci_high,
    }
    if do_bootstrap_simulation:
        out |= {
            "bstrap_ci_low": bstrap_ci_low,
            "bstrap_ci_high": bstrap_ci_high,
        }
    out |= {"n": len(arr)}
    return out

def print_ci(vals, do_bootstrap_simulation=False):
    ci = calculate_ci(vals, do_bootstrap_simulation)
    marker = " (The optimized metric)" if metric_name == common_config.OPTUNA_METRIC else ""
    rows = [
        ("Values:", f"{[f'{x:.4f}' for x in vals[:10]]}..."),
        ("Trials:", f"{ci['n']}"),
        ("Mean:", f"{ci['mean']:.4f}"),
        ("Median:", f"{ci['median']:.4f}"),
        ("Std:", f"{ci['std']:.4f}"),
        ("Z-score 95% CI:", f"[{ci['z_st_ci_low']:.4f}, {ci['z_st_ci_high']:.4f}]"),
        ("T-student 95% CI:", f"[{ci['t_st_ci_low']:.4f}, {ci['t_st_ci_high']:.4f}]"),
    ]
    if do_bootstrap_simulation:
        rows.append(("Bootstrap 95% CI:", f"[{ci['bstrap_ci_low']:.4f}, {ci['bstrap_ci_high']:.4f}]"))
    print(f"\n--- {metric_name}{marker} ---")
    w = max(len(r[0]) for r in rows)
    for label, value in rows:
        print(f"{label:{w}}  {value}")

if __name__ == "__main__":
    STUDY_NAME = f"{common_config.EXPERIMENT_NAME}/lr-search5"
    if len(sys.argv) > 1:
        STUDY_NAME=sys.argv[1]
    exp = Experiment(common_config.EXPERIMENT_NAME)
    mlflow.set_tracking_uri("http://localhost:5000")
    trial_metrics = get_trial_vals(exp, STUDY_NAME)

    if not trial_metrics:
        print("No completed trials found.")
    else:
        print(f"Optimized metric: {common_config.OPTUNA_METRIC}")
        all_metric_names = sorted({k for m in trial_metrics for k in m})
        for metric_name in all_metric_names:
            vals = [m[metric_name] for m in trial_metrics if metric_name in m]
            print_ci(vals)


