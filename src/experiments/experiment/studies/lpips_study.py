from dataclasses import dataclass
from typing import Any, Literal, override

import lightning as L
import mlflow
import optuna
import torch
from optuna.trial import TrialState
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.transforms import v2 as transforms

from experiments.experiment.augmentors.augmentor import Augmentor, AugmentorConfig
from experiments.experiment.datamodules.datamodule import DataModule
from experiments.experiment.studies.study import Study, StudyConfig
from experiments.lib.mlflow_setup import Experiment
from experiments.lib.reproducibility import global_seed_rng


@dataclass(frozen=True, kw_only=True)
class LPIPSStudyConfig(StudyConfig):
    experiment_name: str = "lpips"
    augmentor_config: AugmentorConfig | None = None
    n_samples: int = 1000
    batch_size: int = 16
    n_classes: int = 2
    net_type: Literal["alex", "squeeze", "vgg"] = "alex"
    max_trials: int = 1
    optuna_metric: str = "lpips"
    optuna_direction: Literal["min", "max"] = "max"
    retrain_best: bool = False

    @override
    def post_init_checks(self):
        super().post_init_checks()
        if self.augmentor_config is None:
            raise ValueError("LPIPS requires augmentor_config")
        if self.n_samples <= 0:
            raise ValueError("LPIPS requires n_samples > 0")
        if self.batch_size <= 0:
            raise ValueError("LPIPS requires batch_size > 0")
        if self.n_classes <= 0:
            raise ValueError("LPIPS requires n_classes > 0")

    @override
    @staticmethod
    def get_configured_class():
        return LPIPSStudy


class LPIPSStudy(Study):
    _config: LPIPSStudyConfig
    _results: dict[str, float]

    def __init__(
        self,
        config: LPIPSStudyConfig,
    ):
        super().__init__(config)
        self._config = config

    @override
    def _init_datamodules(self) -> list[DataModule]:
        return []

    @override
    def _configure_datamodules(self, params: dict[str, Any]) -> None:
        pass

    @override
    def _train(self, params: dict[str, Any], get_logger, callbacks) -> None:
        self._results = self._calculate()
        mlflow.log_metrics(self._results)

    @override
    def _retrain(
        self,
        get_logger,
        best_params: dict,
        best_epoch: int,
        callbacks: list,
    ) -> tuple[L.Trainer, Any]:
        raise NotImplementedError("LPIPSStudy does not retrain a model")

    def _device(self) -> torch.device:
        if self._config.device == "gpu":
            if not torch.cuda.is_available():
                raise RuntimeError("LPIPS device='gpu' requires CUDA")
            return torch.device("cuda")
        if self._config.device == "auto" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @staticmethod
    def _prepare_images(images: torch.Tensor, device: torch.device) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                f"LPIPS expects images shaped (N, 3, H, W), got {tuple(images.shape)}"
            )
        images = transforms.ToDtype(torch.float32, scale=True)(images)
        images = images.clamp(0, 1).mul(2).sub(1)
        return images.to(device)

    def _calculate(self) -> dict[str, float]:
        config = self._config
        assert config.augmentor_config is not None
        global_seed_rng(config.seed)
        augmentor: Augmentor = config.augmentor_config.build()
        device = self._device()
        metric = LearnedPerceptualImagePatchSimilarity(
            net_type=config.net_type, reduction="none", normalize=False
        ).to(device)

        labels = torch.arange(config.n_samples, dtype=torch.long) % config.n_classes
        generated_images = augmentor.generate(torch.cat([labels, labels]))
        first_images, second_images = generated_images.split(config.n_samples)

        distance_batches = []
        for start in range(0, config.n_samples, config.batch_size):
            end = min(start + config.batch_size, config.n_samples)
            first = self._prepare_images(first_images[start:end], device)
            second = self._prepare_images(second_images[start:end], device)
            distance_batches.append(metric(first, second).reshape(-1).detach().cpu())

        distances = torch.cat(distance_batches)
        results = {"lpips": distances.mean().item()}
        for class_index in range(config.n_classes):
            results[f"lpips_class_{class_index}"] = (
                distances[labels == class_index].mean().item()
            )

        return results

    @override
    def run(self) -> tuple[Experiment, optuna.Study]:
        config = self._config
        mlflow_experiment, optuna_study, parent_run, already_complete = (
            self._setup_optuna_study(max_trials=config.max_trials)
        )
        if already_complete:
            return mlflow_experiment, optuna_study

        try:
            n_finished = sum(
                trial.state in (TrialState.COMPLETE, TrialState.PRUNED)
                for trial in optuna_study.trials
            )
            for _ in range(config.max_trials - n_finished):
                trial = optuna_study.ask()
                try:
                    with mlflow.start_run(
                        run_name=f"trial-{trial.number}", nested=True
                    ):
                        mlflow.set_tag("optuna_study", self.name)
                        mlflow.set_tag("optuna_trial", trial.number)
                        mlflow.log_params(config.serialize_config())
                        self._train({}, None, [])
                        trial.set_user_attr("metrics", self._results)
                    optuna_study.tell(trial, self._results[config.optuna_metric])
                except Exception:
                    optuna_study.tell(trial, state=TrialState.FAIL)
                    raise
        finally:
            mlflow.end_run()

        return mlflow_experiment, optuna_study
