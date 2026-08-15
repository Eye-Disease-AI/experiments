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
TACGAN_PATH = REPO_ROOT / "packages" / "tacgan"
RESULTS_PATH = TACGAN_PATH / "Biggan_result"
MAX_PARAMS_PER_BATCH = 100
MAX_METRICS_PER_BATCH = 1000
IMAGE_UPLOAD_WORKERS = 8


def chunked(items: list, size: int):
    items_iter = iter(items)
    while chunk := list(islice(items_iter, size)):
        yield chunk


@dataclass(frozen=True, kw_only=True)
class TacganAugmentorConfig(AugmentorConfig):
    loss_type: str
    ac_weight: float
    batch_size: int
    dataset: str
    seed: int
    resolution: int

    @staticmethod
    def known_config_ac() -> TacganAugmentorConfig:
        return TacganAugmentorConfig(
            loss_type="AC",
            ac_weight=1.0,
            batch_size=2,
            dataset="NuclearCataract",
            seed=2018,
            resolution=256,
        )

    @staticmethod
    def known_config_tac1() -> TacganAugmentorConfig:
        return TacganAugmentorConfig(
            loss_type="Twin_AC",
            ac_weight=1.0,
            batch_size=2,
            dataset="NuclearCataract",
            seed=2018,
            resolution=256,
        )

    @staticmethod
    def known_config_tac2() -> TacganAugmentorConfig:
        return TacganAugmentorConfig(
            loss_type="Twin_AC",
            ac_weight=2.0,
            batch_size=16,
            dataset="NuclearCataract",
            seed=2018,
            resolution=256,
        )

    @staticmethod
    def known_config_tac3() -> TacganAugmentorConfig:
        return TacganAugmentorConfig(
            loss_type="Twin_AC",
            ac_weight=2.0,
            batch_size=16,
            dataset="NuclearCataractDominate",
            seed=2018,
            resolution=256,
        )

    def query_params(self) -> list[tuple[str, str]]:
        return [
            ("loss_type", self.loss_type),
            ("AC_weight", f"{self.ac_weight:.1f}"),
            ("batch_size", str(self.batch_size)),
            ("dataset", self.dataset),
            ("seed", str(self.seed)),
            ("resolution", str(self.resolution)),
        ]

    @staticmethod
    def get_configured_class():
        return TacganAugmentor


class TacganAugmentor(Augmentor):
    EXPERIMENT_NAME = "tacgan"

    def __init__(self, config: TacganAugmentorConfig):
        self.config = config

    @staticmethod
    def query_params_from_config_dict(
        config_dict: dict[str, str],
    ) -> list[tuple[str, str]]:
        applicable_keys = [
            "loss_type",
            "AC_weight",
            "batch_size",
            "dataset",
            "seed",
            "resolution",
        ]
        return [(k, str(config_dict[k])) for k in applicable_keys]

    @staticmethod
    def upload_results_to_mlflow(train_run_name: str, *, skip_weights: bool = False):
        state_dict_path = RESULTS_PATH / "weights" / train_run_name / "state_dict.pth"
        state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=False)
        config_dict = state_dict["config"]
        config_flat = ClassConfig.flatten_dict(config_dict)

        exp = Experiment(TacganAugmentor.EXPERIMENT_NAME)
        query_params = TacganAugmentor.query_params_from_config_dict(config_flat)
        if find_run(query_params, exp, fail_if_not_exists=False):
            print("Run with matching parameters is already uploaded, skipping")
            return

        with mlflow.start_run() as run:
            params = [Param(key, str(value)) for key, value in config_flat.items()]
            for params_batch in chunked(params, MAX_PARAMS_PER_BATCH):
                exp.client.log_batch(run.info.run_id, params=params_batch)

        metrics: list[Metric] = []
        metrics_log_path = RESULTS_PATH / "logs" / (train_run_name + "_log.jsonl")

        with open(metrics_log_path) as metrics_log_file:
            for line_json in metrics_log_file:
                line = json.loads(line_json)
                itr = line.pop("itr")
                timestamp = line.pop("_stamp")

                for metric_name, metric_value in line.items():
                    metrics.append(
                        Metric(
                            key=metric_name,
                            value=metric_value,
                            timestamp=int(timestamp),
                            step=itr,
                        )
                    )

        if not skip_weights:
            print("Logging weights")
            weights_dir_path = RESULTS_PATH / "weights" / train_run_name
            exp.client.log_artifact(run.info.run_id, weights_dir_path)
            print("Logged weights")

        for metrics_batch in chunked(metrics, MAX_METRICS_PER_BATCH):
            exp.client.log_batch(run.info.run_id, metrics=metrics_batch)

        samples_dir_path = RESULTS_PATH / "samples" / train_run_name
        samples_uploads = []

        for path in samples_dir_path.iterdir():
            if (
                path.is_file()
                and path.name.startswith("fixed_samples")
                and path.name.endswith(".jpg")
            ):
                try:
                    image_step = int(
                        path.name.removeprefix("fixed_samples").removesuffix(".jpg")
                    )
                    samples_uploads.append((image_step, path))
                except ValueError:
                    print(
                        f"Skipping logging image of name {path.name}, because it can't be correlated with step number"
                    )

        with ThreadPoolExecutor(max_workers=IMAGE_UPLOAD_WORKERS) as executor:
            for upload in samples_uploads:
                executor.submit(
                    exp.client.log_image,
                    run_id=run.info.run_id,
                    image=Image.open(upload[1]),
                    key="fake_samples",
                    step=upload[0],
                )

    @staticmethod
    def generate_cmdline(
        config: TacganAugmentorConfig,
        classes: list[int],
        seeds: list[int],
        weights_root: str,
        samples_root: str,
    ) -> list[str]:
        assert len(classes) == len(seeds)
        classes_str = [str(c) for c in classes]
        seeds_str = [str(s) for s in seeds]

        return [
            "uv",
            "run",
            str(TACGAN_PATH / "TAC-BigGAN" / "sample.py"),
            "--base_root",
            "",
            "--weights_root",
            weights_root,
            "--samples_root",
            samples_root,
            "--loss_type",
            config.loss_type,
            "--AC",
            "--AC_weight",
            f"{config.ac_weight:.1f}",
            "--shuffle",
            "--batch_size",
            str(config.batch_size),
            "--parallel",
            "--num_G_accumulations",
            "1",
            "--num_D_accumulations",
            "1",
            "--num_epochs",
            "2",
            "--num_D_steps",
            "2",
            "--num_G_steps",
            "1",
            "--G_lr",
            "2e-4",
            "--D_lr",
            "2e-4",
            "--dataset",
            config.dataset,
            "--G_ortho",
            "0.0",
            "--G_attn",
            "0",
            "--D_attn",
            "0",
            "--G_init",
            "N02",
            "--D_init",
            "N02",
            "--save_every",
            "10",
            "--num_best_copies",
            "5",
            "--num_save_copies",
            "2",
            # this is seed used for training
            "--seed",
            str(config.seed),
            "--ema",
            "--use_ema",
            "--load_weights",
            "best0",
            "--ema_start",
            "10000",
            "--num_workers",
            "0",
            # here are classes and seeds used for sampling
            "--sample_classes",
            *classes_str,
            "--sample_seeds",
            *seeds_str,
            "--sample_gen",
            "--G_eval_mode",
        ]

    def generate(self, seeds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        super().generate(seeds, labels)
        exp = Experiment(self.EXPERIMENT_NAME, set_active=False)
        query_params = self.config.query_params()
        run: mlflow.entities.Run = find_run(query_params, exp)
        print(f"Found run with matching parameters: {run.info.run_name}")

        required_weights = {"state_dict_best0.pth", "G_ema_best0.pth"}
        weights_artifacts = []
        for artifact in exp.client.list_artifacts(run.info.run_id):
            if artifact.is_dir:
                children = exp.client.list_artifacts(run.info.run_id, artifact.path)
                if required_weights <= {Path(child.path).name for child in children}:
                    weights_artifacts.append(artifact)

        if len(weights_artifacts) != 1:
            raise RuntimeError(
                f"Expected one weights artifact directory, found {len(weights_artifacts)}"
            )

        weights_artifact = weights_artifacts[0]
        with TemporaryDirectory(prefix="tacgan") as tmp_dir:
            weights_root = Path(tmp_dir) / "weights"
            model_weights_dir = weights_root / Path(weights_artifact.path).name
            model_weights_dir.mkdir(parents=True)

            for filename in required_weights:
                mlflow.artifacts.download_artifacts(
                    run_id=run.info.run_id,
                    artifact_path=str(Path(weights_artifact.path) / filename),
                    dst_path=str(model_weights_dir),
                )

            print("Downloaded temporary files:")
            for path in Path(tmp_dir).rglob("*"):
                print(path)

            classes = [int(label) for label in labels.tolist()]
            seeds = [int(seed) for seed in seeds.tolist()]
            samples_root = Path(tmp_dir) / "samples"
            samples_root.mkdir()
            cmd_line = self.generate_cmdline(
                self.config,
                classes,
                seeds,
                str(weights_root),
                str(samples_root),
            )
            subprocess.run(cmd_line, check=True, cwd=TACGAN_PATH / "TAC-BigGAN")

            print("Generated temporary files:")
            for path in samples_root.rglob("*"):
                print(path)

            generations_dir = samples_root / model_weights_dir.name / "-1"
            generations_paths = []
            for class_id, seed in zip(classes, seeds):
                generations_paths.append(
                    generations_dir / f"class_{class_id}_seed_{seed}.jpg"
                )

            generated_images = []
            for path in generations_paths:
                generated_images.append(read_image(path))

        return torch.stack(generated_images)
