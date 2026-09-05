import argparse
import re
import mlflow
import pandas as pd
from experiments.cmd.run_ci_poc import calculate_ci
from experiments.lib.mlflow_setup import Experiment
from experiments.lib.reproducibility import get_trial_vals

RUN_NAME_RE = re.compile(r"_L(\d+)_")
OUTPUT_CSV = "gain2_layers_ci.csv"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment_name", type=str, required=True)
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--output_csv", type=str, default=OUTPUT_CSV)
    p.add_argument("--metric_prefix", type=str, default="best_val_")
    return p.parse_args()


def sort_key(layers: str) -> tuple[int, str]:
    return len(layers), layers


def find_study_runs(experiment_name: str, revision: str | None) -> dict[str, str]:
    runs = mlflow.search_runs(
        experiment_names=[experiment_name], output_format="pandas"
    )
    studies: dict[str, str] = {}
    for name in runs["tags.mlflow.runName"].dropna().unique():
        if "/" in name:  # validation parent runs, not studies
            continue
        if revision and not name.endswith(f"_{revision}"):
            continue
        m = RUN_NAME_RE.search(name)
        if m:
            studies.setdefault(m.group(1), name)
    return studies


def main():
    args = parse_args()
    exp = Experiment(args.experiment_name)

    studies = find_study_runs(args.experiment_name, args.revision)
    rows = []
    for layers in sorted(studies, key=sort_key):
        trial_metrics = get_trial_vals(
            exp,
            f"{studies[layers]}/validation",
            only_crossvalidation_trials=True,
            only_parent_run=False,
        )
        metric_names = sorted(
            {k for m in trial_metrics for k in m if k.startswith(args.metric_prefix)}
        )
        for metric in metric_names:
            vals = [m[metric] for m in trial_metrics if metric in m]
            if len(vals) < 2:  # calculate_ci needs a spread
                print(
                    f"WARNING: layers {layers} metric {metric} has {len(vals)} samples"
                )
                continue
            ci = calculate_ci(vals)
            rows.append(
                {
                    "layers": layers,
                    "metric": metric,
                    "n": ci["n"],
                    "mean": ci["mean"],
                    "median": ci["median"],
                    "std": ci["std"],
                    "t_ci_low": ci["t_st_ci_low"],
                    "t_ci_high": ci["t_st_ci_high"],
                    "t_ci_halfwidth": ci["t_st_ci_width"] / 2,
                    "z_ci_low": ci["z_st_ci_low"],
                    "z_ci_high": ci["z_st_ci_high"],
                    "z_ci_halfwidth": ci["z_st_ci_width"] / 2,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)

    if df.empty:
        print("No validation samples found, wrote an empty csv")
    else:
        print(f"{df['layers'].nunique()} combinations, {len(df)} metric rows")


if __name__ == "__main__":
    main()
