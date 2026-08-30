import argparse
import mlflow
import pandas as pd
from experiments.lib.mlflow_setup import Experiment

X_AXIS_PARAM = "params.model_config.warmup_epochs"
OUTPUT_CSV = "gain6_runs.csv"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment_name", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    Experiment(args.experiment_name)
    mlflow.set_tracking_uri("http://localhost:5000")
    runs = mlflow.search_runs(
        experiment_names=[args.experiment_name], output_format="pandas"
    )
    runs = runs.assign(
        run_name=runs["tags.mlflow.runName"],
        ratio=pd.to_numeric(runs[X_AXIS_PARAM], errors="coerce"),
    )
    runs = runs[runs["ratio"].notna()]
    metric_cols = [c for c in runs.columns if c.startswith("metrics.")]
    df = runs[["run_id", "run_name", "ratio", *metric_cols]].rename(
        columns={c: c.removeprefix("metrics.") for c in metric_cols}
    )
    df = df.dropna(axis=1, how="all").sort_values("ratio").reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"{len(df)} runs, {df['ratio'].nunique()} ratios")


if __name__ == "__main__":
    main()
