from abc import abstractmethod, ABC
from dataclasses import dataclass
from dataset import hard_policy
import torch
import lightning as L
from torch.utils.data.dataloader import DataLoader
from ._names import DataModuleType

@dataclass(frozen=True, kw_only=True)
class DataModuleConfig():
    batch_size: int
    # name of the datamodule used for serializing its type to allow reconstruction
    name: DataModuleType
    hard_policy: hard_policy.HardPolicyType

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

    # Require defining LightningDataModule methods
    @abstractmethod
    def setup(self: DataModule, stage: str | None = None) -> None: ...

    @abstractmethod
    def train_dataloader(self: DataModule) -> DataLoader: ...

    @abstractmethod
    def val_dataloader(self: DataModule) -> DataLoader: ...

    @abstractmethod
    def test_dataloader(self: DataModule) -> DataLoader: ...