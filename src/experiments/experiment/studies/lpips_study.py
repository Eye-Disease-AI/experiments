from dataclasses import dataclass, replace
from random import choice
from typing import Any, Literal, cast, override

import lightning as L
import torch
from dataset.hard_policy import HardPolicy
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import v2 as transforms

from experiments.experiment.datamodules.datamodule import DataModule, DataModuleConfig
from experiments.experiment.datamodules.nuclear_cataract_datamodule import (
    NuclearCataractDataModuleConfig,
)
from experiments.experiment.models.lpips import LPIPSModule, LPIPSModuleConfig
from experiments.experiment.studies.study import Study, StudyConfig
from experiments.lib.reproducibility import global_seed_rng

REAL_DATAMODULE_CONFIG = NuclearCataractDataModuleConfig(
    batch_size=64,
    return_paths=False,
    cache=True,
    hard_policy=HardPolicy.DOMINATE,
    image_size=224,
    normalize=False,
    augment_rot_angle=0,
)


@dataclass(frozen=True, kw_only=True)
class LPIPSStudyConfig(StudyConfig):
    experiment_name: str = "lpips"
    first_datamodule_config: DataModuleConfig = REAL_DATAMODULE_CONFIG
    second_datamodule_config: DataModuleConfig = REAL_DATAMODULE_CONFIG
    model_config: LPIPSModuleConfig = LPIPSModuleConfig()
    n_samples: int = 1000
    batch_size: int = 16
    image_size: int = 224
    max_trials: int = 1
    optuna_metric: str = "lpips"
    optuna_direction: Literal["min", "max"] = "min"
    retrain_best: bool = False

    @override
    def post_init_checks(self):
        super().post_init_checks()
        if self.n_samples <= 0:
            raise ValueError("LPIPS requires n_samples > 0")
        if self.batch_size <= 0:
            raise ValueError("LPIPS requires batch_size > 0")

    @override
    @staticmethod
    def get_configured_class():
        return LPIPSStudy


class LPIPSStudy(Study):
    _config: LPIPSStudyConfig

    def __init__(self, config: LPIPSStudyConfig):
        super().__init__(config)
        self._config = config

    @override
    def _init_datamodules(self) -> list[DataModule]:
        self._first_datamodule: DataModule = (
            self._config.first_datamodule_config.build()
        )
        self._second_datamodule: DataModule = (
            self._config.second_datamodule_config.build()
        )
        return [self._first_datamodule, self._second_datamodule]

    @override
    def _configure_datamodules(self, params: dict[str, Any]) -> None:
        pass

    @override
    def _train(self, params: dict[str, Any], get_logger, callbacks) -> None:
        self._evaluate(get_logger, callbacks)

    @override
    def _retrain(
        self, get_logger, best_params: dict, best_epoch: int, callbacks: list
    ) -> tuple[L.Trainer, Any]:
        return self._evaluate(get_logger, callbacks)

    def _images_by_class(self, datamodule: DataModule) -> list[list[torch.Tensor]]:
        result: list[list[torch.Tensor]] = [[] for _ in range(datamodule.n_classes)]

        for images, labels, *_ in datamodule.val_dataloader():
            for image, label in zip(images, labels, strict=True):
                result[label.item()].append(image)

        return result

    def _image_pairs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._first_datamodule.n_classes != self._second_datamodule.n_classes:
            raise ValueError("LPIPS datamodules must have the same number of classes")

        first_by_class = self._images_by_class(self._first_datamodule)
        second_by_class = self._images_by_class(self._second_datamodule)

        # Check if all classes are present in the datasets
        for class_index in range(self._first_datamodule.n_classes):
            assert first_by_class[class_index], (
                f"Not enough images in first dataset class {class_index}"
            )
            assert second_by_class[class_index], (
                f"No images in second dataset class {class_index}"
            )

        labels = torch.arange(self._config.n_samples) % self._first_datamodule.n_classes
        pairs = []
        for label in labels.tolist():
            pairs.append(
                (choice(first_by_class[label]), choice(second_by_class[label]))
            )
        first, second = zip(*pairs, strict=True)
        return torch.stack(first), torch.stack(second), labels

    def _evaluate(self, get_logger, callbacks) -> tuple[L.Trainer, Any]:
        global_seed_rng(self._config.seed)
        first, second, labels = self._image_pairs()
        transform = transforms.Compose(
            [
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Resize((self._config.image_size, self._config.image_size)),
            ]
        )
        first = transform(first).clamp(0, 1)
        second = transform(second).clamp(0, 1)
        dataloader = DataLoader(
            TensorDataset(first, second, labels),
            batch_size=self._config.batch_size,
        )
        model_config = replace(
            self._config.model_config,
            n_classes=self._first_datamodule.n_classes,
        )
        model: LPIPSModule = model_config.build()
        trainer = L.Trainer(
            accelerator=self._config.device,
            logger=get_logger(model_config.class_name()),
            callbacks=callbacks,
            enable_progress_bar=True,
            enable_model_summary=False,
            enable_checkpointing=False,
            precision=cast(Any, self._config.gpu_precision),
            deterministic=True,
        )
        metrics = trainer.validate(model, dataloaders=dataloader)
        return trainer, metrics
