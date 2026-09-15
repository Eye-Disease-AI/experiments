from dataclasses import dataclass
from typing import override

import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.experiment.augmentors.augmentor import Augmentor, AugmentorConfig
from experiments.experiment.datamodules.datamodule import DataModule, DataModuleConfig


@dataclass(frozen=True, kw_only=True)
class GeneratedDataModuleConfig(DataModuleConfig):
    batch_size: int = 64
    augmentor_config: AugmentorConfig | None = None
    n_samples: int = 1000
    n_classes: int = 2

    @override
    def validate_config(self):
        super().validate_config()
        if self.augmentor_config is None:
            raise ValueError("GeneratedDataModule requires augmentor_config")
        if self.n_samples <= 0:
            raise ValueError("GeneratedDataModule requires n_samples > 0")
        if self.n_classes <= 0:
            raise ValueError("GeneratedDataModule requires n_classes > 0")

    @override
    @staticmethod
    def get_configured_class():
        return GeneratedDataModule


class GeneratedDataModule(DataModule):
    def __init__(self, config: GeneratedDataModuleConfig):
        super().__init__()
        self._config = config
        self._batch_size = config.batch_size

    @override
    def setup(self, stage: str | None = None) -> None:
        if hasattr(self, "dataset"):
            return
        assert self._config.augmentor_config is not None
        augmentor: Augmentor = self._config.augmentor_config.build()
        labels = torch.arange(self._config.n_samples) % self._config.n_classes
        generator = torch.Generator().manual_seed(self._config.augmentor_seed)
        seeds = torch.randint(0, 2**31, [1], generator=generator) + torch.arange(
            self._config.n_samples
        )
        self.dataset = TensorDataset(augmentor.generate(seeds, labels), labels)
        del augmentor

    def _dataloader(self):
        return DataLoader(self.dataset, batch_size=self._batch_size)

    @override
    def train_dataloader(self):
        return self._dataloader()

    @override
    def val_dataloader(self):
        return self._dataloader()

    @override
    def test_dataloader(self):
        return self._dataloader()

    @property
    @override
    def config(self) -> GeneratedDataModuleConfig:
        return self._config

    @property
    @override
    def n_classes(self) -> int:
        return self._config.n_classes

    @property
    @override
    def class_names(self) -> list[str]:
        return [str(index) for index in range(self.n_classes)]

    @property
    @override
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    @override
    def batch_size(self, value: int):
        self._batch_size = value
