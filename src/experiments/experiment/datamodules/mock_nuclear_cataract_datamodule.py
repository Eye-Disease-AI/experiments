from dataclasses import dataclass, field
from typing import override
import torch
from torch.utils.data.dataloader import DataLoader

from experiments.experiment.datamodules.datamodule import DataModule, DataModuleConfig


@dataclass(frozen=True, kw_only=True)
class MockNuclearCataractDatamoduleConfig(DataModuleConfig):
    data_shape: tuple = (3, 224, 224)
    n_classes: int = 2
    class_names: list[str] = field(default_factory=lambda: ["mock_1", "mock_2"])
    dataset_len: int = 100

    @staticmethod
    @override
    def get_configured_class():
        return MockNuclearCataractDataModule


class MockNuclearCataractDataset(torch.utils.data.Dataset):
    def __init__(self, config: MockNuclearCataractDatamoduleConfig):
        self._config = config

    def __len__(self) -> int:
        return self._config.dataset_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image = torch.rand(self._config.data_shape)
        label = int(torch.randint(0, self._config.n_classes, (1,)).item())
        return image, label


class MockNuclearCataractDataModule(DataModule):
    def __init__(self, config: MockNuclearCataractDatamoduleConfig):
        super().__init__()
        self._config = config
        self._batch_size = self.config.batch_size

    @property
    @override
    def config(self) -> DataModuleConfig:
        return self._config

    @property
    @override
    def n_classes(self) -> int:
        return self._config.n_classes

    @property
    @override
    def class_names(self) -> list[str]:
        return ["mock_1", "mock_2"]

    @property
    @override
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    @override
    def batch_size(self, v: int):
        self._batch_size = v

    # Require defining LightningDataModule methods
    @override
    def setup(self: DataModule, stage: str | None = None) -> None:
        pass

    @override
    def train_dataloader(self: DataModule) -> DataLoader:
        return DataLoader(MockNuclearCataractDataset(self._config), shuffle=True)

    @override
    def val_dataloader(self: DataModule) -> DataLoader:
        return DataLoader(MockNuclearCataractDataset(self._config))

    @override
    def test_dataloader(self: DataModule) -> DataLoader:
        return DataLoader(MockNuclearCataractDataset(self._config))
