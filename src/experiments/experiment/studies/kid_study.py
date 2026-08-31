from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, override

import lightning as L
import torch
from lightning.pytorch.loggers.mlflow import MLFlowLogger
from torch.utils.data.dataloader import DataLoader
from torchmetrics.image.kid import KernelInceptionDistance
from torchvision.transforms import v2

from experiments.experiment.callbacks import OptunaMLflowCallback
from experiments.experiment.datamodules.datamodule import DataModule, DataModuleConfig
from experiments.experiment.studies.study import Study, StudyConfig


@dataclass(frozen=True, kw_only=True)
class KIDStudyConfig(StudyConfig):
    experiment_name: str = "kid"
    real_datamodule_config: DataModuleConfig
    fake_datamodule_config: DataModuleConfig
    max_trials: int = 1
    optuna_metric: str = "kid"
    optuna_direction: Literal["min", "max"] = "min"
    retrain_best: bool = False

    @override
    @staticmethod
    def get_configured_class():
        return KIDStudy


class KIDStudy(Study):
    _config: KIDStudyConfig

    def __init__(self, config: KIDStudyConfig):
        super().__init__(config)
        self._config = config

    @override
    def _init_datamodules(self) -> list[DataModule]:
        self._real_datamodule = self._config.real_datamodule_config.build()
        self._fake_datamodule = self._config.fake_datamodule_config.build()
        return [self._real_datamodule, self._fake_datamodule]

    @override
    def _configure_datamodules(self, params: dict[str, Any]) -> None:
        pass

    @override
    def _train(
        self,
        params: dict[str, Any],
        get_logger: Callable[[str], MLFlowLogger],
        callbacks,
    ) -> None:
        logger = get_logger("kid")
        full_real_dataset = self.load_full_real_dataset()
        full_fake_dataset = self.load_full_fake_dataset()
        kid_means, kid_stds = self.calculate_kids(full_real_dataset, full_fake_dataset)
        self.log_kid_values(logger, kid_means, kid_stds)

        optuna_cb = next(c for c in callbacks if isinstance(c, OptunaMLflowCallback))
        kid = torch.stack(kid_means).mean().item()
        optuna_cb.best_value = kid
        optuna_cb.trial.report(kid, 0)

    def load_full_real_dataset(self):
        return self.collect_datamodule_by_class(self._real_datamodule)

    def load_full_fake_dataset(self):
        return self.collect_datamodule_by_class(self._fake_datamodule)

    def calculate_kids(
        self, real_imgs_by_cls: list[torch.Tensor], fake_imgs_by_cls: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        kid_means = []
        kid_stds = []

        for real_imgs, fake_imgs in zip(real_imgs_by_cls, fake_imgs_by_cls):
            kid = KernelInceptionDistance(subset_size=300)
            kid.update(real_imgs, real=True)
            kid.update(fake_imgs, real=False)
            kid_mean, kid_std = kid.compute()
            kid_means.append(kid_mean)
            kid_stds.append(kid_std)

        return kid_means, kid_stds

    def log_kid_values(self, logger: MLFlowLogger, kid_means, kid_stds):
        metrics_map = {}

        for label, (kid_mean, kid_std) in enumerate(zip(kid_means, kid_stds)):
            metrics_map[f"kid_class_{label}_mean"] = kid_mean
            metrics_map[f"kid_class_{label}_std"] = kid_std

        logger.log_metrics(metrics_map)

    def collect_datamodule_by_class(self, datamodule: DataModule) -> list[torch.Tensor]:
        train_collected = self.collect_dataloader_by_class(
            datamodule.n_classes,
            datamodule.train_dataloader(),
        )
        val_collected = self.collect_dataloader_by_class(
            datamodule.n_classes,
            datamodule.val_dataloader(),
        )

        result = []
        for train_tensor, val_tensor in zip(train_collected, val_collected):
            result.append(torch.cat((train_tensor, val_tensor)))

        return result

    def collect_dataloader_by_class(
        self, n_classes: int, dataloader: DataLoader
    ) -> list[torch.Tensor]:
        imgs_by_class: list[list] = [[] for _ in range(n_classes)]

        for imgs, labels in dataloader:
            for img, label in zip(imgs, labels):
                conv_to_uint = v2.ConvertImageDtype(dtype=torch.uint8)
                label_int = label.item()
                imgs_by_class[label_int].append(conv_to_uint(img))

        return [torch.stack(tensor_list) for tensor_list in imgs_by_class]

    @override
    def _retrain(
        self, get_logger, best_params: dict, best_epoch: int, callbacks: list
    ) -> tuple[L.Trainer, Any]:
        raise NotImplementedError
