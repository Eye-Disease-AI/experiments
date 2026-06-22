from abc import abstractmethod, ABC
from dataclasses import dataclass
import lightning as L
import torch
from torch.utils.data.dataloader import DataLoader

from experiments.lib.config_serializing import ClassConfig


@dataclass(frozen=True, kw_only=True)
class DataModuleConfig(ClassConfig):
    batch_size: int = 64


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
    def batch_size(self) -> int: ...
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

    def class_weights(self) -> torch.Tensor | None:
        return None
