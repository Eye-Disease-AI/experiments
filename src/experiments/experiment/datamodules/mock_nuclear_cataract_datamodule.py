from dataclasses import dataclass, field
from typing import override

import torch
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms.v2 import Compose, Resize, ToDtype

from experiments.experiment.augmentors.fake_dataset import FakeDataset
from experiments.experiment.augmentors.mock_nuclear_cataract_augmentor import (
    MockNuclearCataractAugmentor,
    MockNuclearCataractAugmentorConfig,
)
from experiments.experiment.datamodules.datamodule import DataModule, DataModuleConfig
from experiments.lib.config_serializing import OptunaOptimised


@dataclass(frozen=True, kw_only=True)
class MockNuclearCataractDatamoduleConfig(DataModuleConfig):
    data_shape: tuple = (3, 224, 224)
    n_classes: int = 2
    class_names: list[str] = field(default_factory=lambda: ["mock_1", "mock_2"])
    dataset_len: int = 100
    augmentor_config: MockNuclearCataractAugmentorConfig | None
    n_augment: int | None

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
    def batch_size(self) -> int | OptunaOptimised:
        return self._batch_size

    @batch_size.setter
    @override
    def batch_size(self, v: int):
        self._batch_size = v

    # Require defining LightningDataModule methods
    @override
    def setup(self, stage: str | None = None) -> None:
        if not hasattr(self, "train_set"):
            train_set = MockNuclearCataractDataset(self._config)
            self.val_set = MockNuclearCataractDataset(self._config)
            self.test_set = MockNuclearCataractDataset(self._config)

            if self._config.augmentor_config and self._config.n_augment:
                augmentor: MockNuclearCataractAugmentor = (
                    self._config.augmentor_config.build()
                )
                fake_labels = torch.zeros((self._config.n_augment), dtype=torch.long)
                fake_images = augmentor.generate(fake_labels)
                fake_set = FakeDataset(
                    fake_images,
                    fake_labels,
                    Compose(
                        [
                            Resize(self._config.data_shape[1:]),
                            ToDtype(torch.float32),
                        ]
                    ),
                )
                train_concat = torch.utils.data.ConcatDataset([train_set, fake_set])
                self.train_set = train_concat
            else:
                self.train_set = train_set

    def setup_fold(self, _fold: int, _num_folds: int) -> None:
        pass

    @override
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, shuffle=True)

    @override
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set)

    @override
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set)
