from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import override

import lightning as L
import torch
from torch.utils.data.dataloader import DataLoader

from experiments.experiment.augmentors.augmentor import AugmentorConfig
from experiments.lib.config_serializing import ClassConfig, OptunaOptimised


@dataclass(frozen=True, kw_only=True)
class DataModuleConfig(ClassConfig):
    batch_size: int | OptunaOptimised = 64
    augmentor_config: AugmentorConfig | None = None
    n_augment: int | None = None
    cas_mode: bool = False

    @override
    def post_init_checks(self):
        super().post_init_checks()
        if self.cas_mode and not (self.augmentor_config and self.n_augment):
            raise ValueError("cas_mode requires augmentor_config and n_augment")


class DataModule(ABC, L.LightningDataModule):
    @property
    @abstractmethod
    def config(self) -> DataModuleConfig: ...

    @property
    @abstractmethod
    def n_classes(self) -> int: ...

    @property
    @abstractmethod
    def class_names(self) -> list[str]: ...

    @property
    @abstractmethod
    def batch_size(self) -> int | OptunaOptimised: ...
    @batch_size.setter
    @abstractmethod
    def batch_size(self, v: int): ...

    # Require defining LightningDataModule methods
    @abstractmethod
    def setup(self: DataModule, stage: str | None = None) -> None: ...

    @abstractmethod
    def train_dataloader(self: DataModule) -> DataLoader: ...

    @abstractmethod
    def val_dataloader(self: DataModule) -> DataLoader: ...

    @abstractmethod
    def test_dataloader(self: DataModule) -> DataLoader: ...

    @property
    def class_weights(self) -> torch.Tensor | None:
        return None
