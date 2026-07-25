import json
from dataclasses import dataclass
from pathlib import Path

import mlflow
import torch
from PIL import Image

from experiments.experiment.augmentors.augmentor import Augmentor, AugmentorConfig
from experiments.lib.config_serializing import ClassConfig
from experiments.lib.mlflow_setup import Experiment


@dataclass(frozen=True, kw_only=True)
class StyleganAugmentorConfig(AugmentorConfig):
    @staticmethod
    def get_configured_class():
        return StyleganAugmentor


class StyleganAugmentor(Augmentor):
    def __init__(self, config: StyleganAugmentorConfig):
        self.config = config

    @staticmethod
    def upload_results_to_mlflow(train_run_dir_path: Path):
        training_options_path = train_run_dir_path / "training_options.json"

        with open(training_options_path) as training_options_file:
            training_options = json.load(training_options_file)

        training_options_flat = ClassConfig.flatten_dict(training_options)

        validation_interval_steps = (
            training_options_flat["image_snapshot_ticks"]
            * training_options_flat["kimg_per_tick"]
            * 1000  # 1000 steps, because kimg = 1000 images
        )

        stats_interval_steps = training_options_flat["kimg_per_tick"] * 1000

        exp = Experiment("stylegan3")

        with mlflow.start_run() as run:
            for key in training_options_flat:
                exp.client.log_param(run.info.run_id, key, training_options_flat[key])

            for path in train_run_dir_path.iterdir():
                if (
                    path.is_file()
                    and path.name.startswith("metric-")
                    and path.name.endswith(".jsonl")
                ):
                    with open(path) as metric_file:
                        cur_step = 0
                        for line_json in metric_file:
                            line = json.loads(line_json)
                            metric_name = line["metric"]
                            metric_value = line["results"][metric_name]
                            exp.client.log_metric(
                                run_id=run.info.run_id,
                                key=metric_name,
                                value=metric_value,
                                timestamp=int(line["timestamp"]),
                                step=cur_step,
                            )
                            cur_step += validation_interval_steps

            for path in train_run_dir_path.iterdir():
                if (
                    path.is_file()
                    and path.name.startswith("fakes")
                    and path.name.endswith(".png")
                ):
                    try:
                        image_step = int(
                            path.name.removeprefix("fakes").removesuffix(".png")
                        )
                        exp.client.log_image(
                            run_id=run.info.run_id,
                            image=Image.open(path),
                            key="fake_samples",
                            step=image_step,
                        )
                    except ValueError:
                        print(
                            f"Skipping logging image of name {path.name}, because it can't be correlated with step number"
                        )

            reals_path = train_run_dir_path / "reals.png"
            if reals_path.exists():
                exp.client.log_image(
                    run_id=run.info.run_id,
                    image=Image.open(reals_path),
                    key="reals",
                )
            else:
                print("Warning: No reals.png found")

            stats_path = train_run_dir_path / "stats.jsonl"
            if stats_path.exists():
                cur_step = 0
                with open(stats_path) as stats_file:
                    for line_json in stats_file:
                        line = json.loads(line_json)
                        timestamp = int(line["timestamp"])

                        for stat in line:
                            if stat == "timestamp":
                                continue

                            mean = line[stat]["mean"]
                            std = line[stat]["std"]

                            exp.client.log_metric(
                                run_id=run.info.run_id,
                                key=f"{stat}/mean",
                                value=mean,
                                step=cur_step,
                                timestamp=timestamp,
                            )
                            exp.client.log_metric(
                                run_id=run.info.run_id,
                                key=f"{stat}/std",
                                value=std,
                                step=cur_step,
                                timestamp=timestamp,
                            )

                        cur_step += stats_interval_steps
            else:
                print("Warning: No stats.jsonl found")

    def generate(self, labels: torch.Tensor) -> torch.Tensor:
        return torch.Tensor()
