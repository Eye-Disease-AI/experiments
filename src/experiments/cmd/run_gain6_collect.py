import argparse
import re
import mlflow

RUN_NAME_RE = re.compile(r"_warmup_([0-9]+)%_")
OUTPUT_CSV = "gain6_runs.csv"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment_name", type=str, required=True)
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--output_csv", type=str, default=OUTPUT_CSV)
    return p.parse_args()


def main():
    args = parse_args()
    runs = mlflow.search_runs(
        experiment_names=[args.experiment_name], output_format="pandas"
    )
    names = runs["tags.mlflow.runName"].fillna("")
    runs["warmup_percent"] = names.str.extract(RUN_NAME_RE)[0].astype("Int64")
    runs["duration_s"] = (runs["end_time"] - runs["start_time"]).dt.total_seconds()
    runs = runs[runs["warmup_percent"].notna()]
    if args.revision:
        runs = runs[names.str.contains(f"_{args.revision}")]

    cols = ["tags.mlflow.runName", "warmup_percent", "duration_s"] + [
        c for c in runs.columns if (c.startswith("metrics.retrain_val"))
    ]
    runs[cols].to_csv(args.output_csv, index=False)
    print(f"{len(runs)} runs")


if __name__ == "__main__":
    main()
