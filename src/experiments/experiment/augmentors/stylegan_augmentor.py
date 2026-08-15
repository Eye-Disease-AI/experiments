import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import mlflow.entities
import torch
from mlflow.entities import Metric, Param
from PIL import Image
from torchvision.io import read_image

from experiments.experiment.augmentors.augmentor import Augmentor, AugmentorConfig
from experiments.experiment.augmentors.common import find_run
from experiments.lib.config_serializing import ClassConfig
from experiments.lib.mlflow_setup import Experiment

REPO_ROOT = Path(__file__).resolve().parents[4]
STYLEGAN3_PATH = REPO_ROOT / "packages" / "stylegan3"
MAX_PARAMS_PER_BATCH = 100
MAX_METRICS_PER_BATCH = 1000
IMAGE_UPLOAD_WORKERS = 8


def chunked(items: list, size: int):
    items_iter = iter(items)
    while chunk := list(islice(items_iter, size)):
        yield chunk


@dataclass(frozen=True, kw_only=True)
class StyleganAugmentorConfig(AugmentorConfig):
    kimg: int
    gpus: int
    batch: int
    gamma: float
    batch_gpu: int
    snap: int
    metrics: str
    cond: bool
    cfg: str
    # Not a command line argument, but important
    resolution: int

    @staticmethod
    def known_config_r() -> StyleganAugmentorConfig:
        return StyleganAugmentorConfig(
            kimg=1200,
            gpus=1,
            batch=12,
            gamma=2,
            batch_gpu=12,
            snap=10,
            metrics="fid20k_full",
            cond=True,
            cfg="stylegan3-r",
            resolution=256,
        )

    @staticmethod
    def known_config_t() -> StyleganAugmentorConfig:
        return StyleganAugmentorConfig(
            kimg=1200,
            gpus=1,
            batch=12,
            gamma=2,
            batch_gpu=12,
            snap=10,
            metrics="fid20k_full",
            cond=True,
            cfg="stylegan3-t",
            resolution=256,
        )

    def query_params(self) -> list[tuple[str, str]]:
        if self.cfg == "stylegan3-r":
            use_radial_filters = True
        elif self.cfg == "stylegan3-t":
            use_radial_filters = False
        else:
            raise RuntimeError(f"Unknown stylegan architecture (cfg = {self.cfg})")

        return [
            ("total_kimg", str(self.kimg)),
            ("num_gpus", str(self.gpus)),
            ("batch_size", str(self.batch)),
            ("batch_gpu", str(self.batch_gpu)),
            # Gamma float is not really safe to compare, but that is what stylegan gives us
            ("loss_kwargs.r1_gamma", f"{self.gamma:.1f}"),
            # I think that image_snapshot_ticks and network_snapshot_ticks are the same value
            ("image_snapshot_ticks", str(self.snap)),
            # I don't know if it will work for multiple metrics:
            ("metrics", f"['{self.metrics}']"),
            ("training_set_kwargs.use_labels", "True" if self.cond else "False"),
            ("G_kwargs.use_radial_filters", "True" if use_radial_filters else "False"),
            ("training_set_kwargs.resolution", str(self.resolution)),
        ]

    @staticmethod
    def get_configured_class():
        return StyleganAugmentor


class StyleganAugmentor(Augmentor):
    EXPERIMENT_NAME = "stylegan3"

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

        exp = Experiment(StyleganAugmentor.EXPERIMENT_NAME)

        with mlflow.start_run() as run:
            params = [
                Param(key, str(value)) for key, value in training_options_flat.items()
            ]
            for params_batch in chunked(params, MAX_PARAMS_PER_BATCH):
                exp.client.log_batch(run.info.run_id, params=params_batch)

            metrics: list[Metric] = []
            found_fid20k = False
            best_fid20k = 0.0
            best_snapshot = ""

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
                            snapshot_pkl = line["snapshot_pkl"]

                            if metric_name == "fid20k_full" and (
                                metric_value < best_fid20k or not found_fid20k
                            ):
                                found_fid20k = True
                                best_snapshot = snapshot_pkl
                                best_fid20k = metric_value

                            metrics.append(
                                Metric(
                                    key=metric_name,
                                    value=metric_value,
                                    timestamp=int(line["timestamp"]),
                                    step=cur_step,
                                )
                            )
                            cur_step += validation_interval_steps

            print("Logging best snapshot")
            best_snapshot_path = train_run_dir_path / best_snapshot
            exp.client.log_artifact(run.info.run_id, best_snapshot_path)
            print("Best snapshot logged")

            # Images are uploaded one artifact request each, so do them in parallel
            with ThreadPoolExecutor(max_workers=IMAGE_UPLOAD_WORKERS) as executor:
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
                            executor.submit(
                                exp.client.log_image,
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
                    executor.submit(
                        exp.client.log_image,
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

                            metrics.append(
                                Metric(
                                    key=f"{stat}/mean",
                                    value=mean,
                                    timestamp=timestamp,
                                    step=cur_step,
                                )
                            )
                            metrics.append(
                                Metric(
                                    key=f"{stat}/std",
                                    value=std,
                                    timestamp=timestamp,
                                    step=cur_step,
                                )
                            )

                        cur_step += stats_interval_steps
            else:
                print("Warning: No stats.jsonl found")

            for metrics_batch in chunked(metrics, MAX_METRICS_PER_BATCH):
                exp.client.log_batch(run.info.run_id, metrics=metrics_batch)

    @staticmethod
    def generate_cmdline(
        weights_path: str, seeds: list[int], cls: int, out_dir: str
    ) -> list[str]:
        seeds_str: list[str] = [str(s) for s in seeds]
        seeds_joined = ",".join(seeds_str)
        return [
            "uv",
            "run",
            f"{STYLEGAN3_PATH}/gen_images.py",
            "--network",
            f"{weights_path}",
            "--seeds",
            f"{seeds_joined}",
            "--class",
            f"{cls}",
            "--outdir",
            f"{out_dir}",
        ]

    def generate(self, seeds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        super().generate(seeds, labels)
        # Activate experiment
        exp = Experiment(self.EXPERIMENT_NAME, set_active=False)
        query_params = self.config.query_params()
        run: mlflow.entities.Run = find_run(query_params, exp)
        print(f"Found run with matching parameters: {run.info.run_name}")

        best_snapshot_path = None
        for art in exp.client.list_artifacts(run.info.run_id):
            path = Path(art.path)
            if path.name.startswith("network-snapshot-") and path.name.endswith(".pkl"):
                best_snapshot_path = path

        if best_snapshot_path:
            print(f"Found best snapshot path: {best_snapshot_path}")
        else:
            raise RuntimeError("Could not found best snapshot in generate")

        with TemporaryDirectory(prefix="stylegan3") as tmp_dir:
            mlflow.artifacts.download_artifacts(
                run_id=run.info.run_id,
                artifact_path=str(best_snapshot_path),
                dst_path=tmp_dir,
            )
            tmp_dir_best_snapshot_path = Path(tmp_dir) / best_snapshot_path.name
            generations_dir = Path(tmp_dir) / "generations"
            generations_dir.mkdir()

            seeds_list = [int(seed) for seed in seeds.tolist()]
            labels_seeds: dict[int, list[int]] = {}

            for seed, label in zip(seeds_list, labels.tolist()):
                if label in labels_seeds:
                    labels_seeds[label].append(seed)
                else:
                    labels_seeds[label] = [seed]

            for label, seeds in labels_seeds.items():
                cmd_line = self.generate_cmdline(
                    weights_path=str(tmp_dir_best_snapshot_path),
                    seeds=seeds,
                    cls=label,
                    out_dir=str(generations_dir),
                )
                subprocess.run(cmd_line, check=True)

            generations_paths = []
            for seed in seeds_list:
                generations_paths.append(generations_dir / f"seed{seed:04d}.png")

            generated_images = []
            for path in generations_paths:
                generated_images.append(read_image(path))

        return torch.stack(generated_images)
