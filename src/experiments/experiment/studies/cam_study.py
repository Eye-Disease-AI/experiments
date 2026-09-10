import shutil
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, override

import lightning as L
import mlflow.artifacts
import torch
from lightning.pytorch.loggers.mlflow import MLFlowLogger
from mlflow.entities import Run
from pytorch_grad_cam import LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision.transforms import v2

from experiments.experiment.datamodules.datamodule import DataModule, DataModuleConfig
from experiments.experiment.datamodules.nuclear_cataract_datamodule import (
    NuclearCataractDataModule,
)
from experiments.experiment.models.convnext import ConvNext, ConvNextConfig
from experiments.experiment.studies.study import Study, StudyConfig
from experiments.lib.reproducibility import Experiment

REPO_ROOT = Path(__file__).resolve().parents[4]
CACHE_DIR = REPO_ROOT / ".cache"


@dataclass(frozen=True, kw_only=True)
class CAMStudyConfig(StudyConfig):
    experiment_name: str = "cam"
    datamodule_config: DataModuleConfig
    model_config: ConvNextConfig
    # Experiment name and run name
    weights_run_name: str | None
    weights_run_id: str | None
    weights_art_name: str | None
    max_trials: int = 1
    retrain_best: bool = False

    @staticmethod
    def from_classifier_study(
        experiment_name: str, run_name: str, with_weights: bool = True
    ):
        exp = Experiment(experiment_name=experiment_name, set_active=False)
        run = CAMStudyConfig.find_run(exp, run_name)
        snp = CAMStudyConfig.find_snapshot(exp, run)
        cfg = CAMStudyConfig.find_config(exp, run)

        if (not snp) or (not cfg):
            raise RuntimeError("Could not find either weights snapshot or study config")

        with TemporaryDirectory(prefix="camstudy") as temp_dir:
            cfg_file_path = mlflow.artifacts.download_artifacts(
                run_id=run.info.run_id,
                artifact_path=str(cfg),
                dst_path=temp_dir,
            )

            with open(cfg_file_path) as cfg_file:
                cfg_json = cfg_file.read()

            study_config = StudyConfig.from_json(cfg_json)

        return CAMStudyConfig(
            datamodule_config=study_config.datamodule_config,  # type: ignore
            model_config=study_config.model_config,  # type: ignore
            weights_run_id=run.info.run_id,
            weights_art_name=str(snp / "best_retrain.ckpt"),
            weights_run_name=run_name,
        )

    @staticmethod
    def find_run(exp: Experiment, name: str):
        for run in exp.client.search_runs(
            [exp.mlflow_experiment.experiment_id], max_results=9999
        ):
            if (
                "optuna_study" in run.data.tags
                and run.data.tags["optuna_study"] == name
                and "mlflow.parentRunId" not in run.data.tags
            ):
                return run

    @staticmethod
    def find_snapshot(exp: Experiment, run: Run) -> Path | None:
        for art in exp.client.list_artifacts(run.info.run_id):
            if art.path == "model":
                return Path(art.path)

        return None

    @staticmethod
    def find_config(exp: Experiment, run: Run) -> Path | None:
        for art in exp.client.list_artifacts(run.info.run_id):
            if art.path == "study_config.json":
                return Path(art.path)

        return None

    @override
    @staticmethod
    def get_configured_class():
        return CAMStudy


class CAMStudy(Study):
    _config: CAMStudyConfig

    def __init__(self, config: CAMStudyConfig):
        super().__init__(config)
        self._config = config

    @override
    def _init_datamodules(self) -> list[DataModule]:
        mod_config = replace(self._config.datamodule_config, return_paths=True)  # type: ignore
        self._datamodule = mod_config.build()
        return [self._datamodule]

    @override
    def _configure_datamodules(self, params: dict[str, Any]) -> None:
        config = self._set_config_optuna_params(params)
        self._datamodule.batch_size = config.datamodule_config.batch_size

    @override
    def _train(
        self,
        params: dict[str, Any],
        get_logger: Callable[[str], MLFlowLogger],
        callbacks,
    ) -> None:
        if not self._config.weights_run_id:
            raise RuntimeError("Running without weights not supported")

        model_ckpt = self.ckpt_from_cache()
        model = self.build_model(model_ckpt)

        data_loaders = [
            self._datamodule.train_dataloader(),
            self._datamodule.val_dataloader(),
        ]
        logger = get_logger("cam")

        for loader in data_loaders:
            cams = self.generate_cams(model, loader)
            self.log_cams(logger, cams)

    def ckpt_from_cache(self) -> Path:
        ckpt_path = (
            CACHE_DIR
            / f"{self._config.weights_run_name}_{self._config.weights_run_id}.ckpt"
        )
        ckpt_path.parent.mkdir(exist_ok=True)

        print(f"Getting {ckpt_path} from cache...")

        if not ckpt_path.exists():
            print(f"{ckpt_path} not found in cache, downloading...")
            self.download_ckpt_from_mlflow(ckpt_path)

        return ckpt_path

    def download_ckpt_from_mlflow(self, dest_path: Path):
        with TemporaryDirectory() as temp_dir:
            print(f"Downloading @ {temp_dir}")

            mlflow.artifacts.download_artifacts(
                run_id=self._config.weights_run_id,
                artifact_path=self._config.weights_art_name,
                dst_path=temp_dir,
            )
            downloaded_ckpt_path = Path(temp_dir) / "best_retrain.ckpt"
            print(f"Moving {downloaded_ckpt_path} -> {dest_path}")
            shutil.move(downloaded_ckpt_path, dest_path)

    def build_model(self, model_ckpt: Path):
        return ConvNext.load_from_checkpoint(
            model_ckpt,
            config=self._config.model_config,
            strict=False,
        )

    def generate_cams(self, model: ConvNext, data_loader) -> dict:
        last_conv_layer = model.model.features[-1][-1].block[0]  # type: ignore
        target_layers = [last_conv_layer]
        targets = [ClassifierOutputTarget(0), ClassifierOutputTarget(1)]
        cam_type = LayerCAM
        result = {}

        for imgs, labels, paths in data_loader:
            for ix, (img, path) in enumerate(zip(imgs, paths)):
                input_tensor = img.unsqueeze(0)
                with cam_type(model=model, target_layers=target_layers) as cam:
                    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)  # type: ignore
                    grayscale_cam = grayscale_cam[0, :]

                    if self._config.datamodule_config.normalize:  # type: ignore
                        inv_normalize = v2.Normalize(
                            mean=[
                                -m / s
                                for m, s in zip(
                                    NuclearCataractDataModule.DATASET_MEAN,
                                    NuclearCataractDataModule.DATASET_STD,
                                )
                            ],
                            std=[1 / s for s in NuclearCataractDataModule.DATASET_STD],
                        )
                        showable_img = torch.clamp(inv_normalize(img), 0, 1)
                        showable_img = showable_img.permute(1, 2, 0).numpy()
                    else:
                        showable_img = img.permute(1, 2, 0).numpy()

                    visualization = show_cam_on_image(
                        showable_img,
                        grayscale_cam,
                        use_rgb=True,
                    )
                    result[path] = visualization
                    print("G", end="", flush=True)

        return result

    def log_cams(self, logger: MLFlowLogger, cams):
        client, run_id = logger.experiment, logger.run_id

        with ThreadPoolExecutor(16) as pool:
            futures = []
            for path, cam in cams.items():
                path = path.rsplit(".", 1)[0] + ".png"
                futures.append(pool.submit(client.log_image, run_id, cam, path))

            for future in as_completed(futures):
                future.result()
                print("U", end="", flush=True)

    @override
    def _parse_optuna_study(self, optuna_study):
        return optuna_study.best_trial, {}, 0

    @override
    def _retrain(
        self, get_logger, best_params: dict, best_epoch: int, callbacks: list
    ) -> tuple[L.Trainer, Any]:
        raise NotImplementedError
