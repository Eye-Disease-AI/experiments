from dataclasses import dataclass
from random import choice
from typing import override

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import v2 as transforms

from experiments.experiment.datamodules.datamodule import DataModule, DataModuleConfig


@dataclass(frozen=True, kw_only=True)
class LPIPSDataModuleConfig(DataModuleConfig):
    first_config: DataModuleConfig
    second_config: DataModuleConfig
    n_samples: int = 1000
    batch_size: int = 16
    image_size: int = 224

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
        return LPIPSDataModule


class LPIPSDataModule(DataModule):
    def __init__(self, config: LPIPSDataModuleConfig):
        super().__init__()
        self._config = config
        self._batch_size = config.batch_size
        self._first: DataModule = config.first_config.build()
        self._second: DataModule = config.second_config.build()

    @override
    def setup(self, stage: str | None = None) -> None:
        if hasattr(self, "dataset"):
            return

        self._first.setup(stage)
        self._second.setup(stage)

        if self._first.n_classes != self._second.n_classes:
            raise ValueError("LPIPS datamodules must have the same number of classes")

        first_by_class = self._images_by_class(self._first)
        second_by_class = self._images_by_class(self._second)

        # Check if all classes are present in the datasets
        for class_index in range(self.n_classes):
            assert first_by_class[class_index], (
                f"No images in first dataset class {class_index}"
            )
            assert second_by_class[class_index], (
                f"No images in second dataset class {class_index}"
            )

        labels = torch.arange(self._config.n_samples) % self.n_classes
        pairs = [
            (choice(first_by_class[label]), choice(second_by_class[label]))
            for label in labels.tolist()
        ]
        first, second = zip(*pairs, strict=True)
        transform = transforms.Compose(
            [
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Resize((self._config.image_size, self._config.image_size)),
            ]
        )
        self.dataset = TensorDataset(
            transform(torch.stack(first)).clamp(0, 1),
            transform(torch.stack(second)).clamp(0, 1),
            labels,
        )

    @staticmethod
    def _images_by_class(datamodule: DataModule) -> list[list[torch.Tensor]]:
        result: list[list[torch.Tensor]] = [[] for _ in range(datamodule.n_classes)]

        for images, labels, *_ in datamodule.val_dataloader():
            for image, label in zip(images, labels, strict=True):
                result[label.item()].append(image)

        return result

    @override
    def train_dataloader(self):
        raise NotImplementedError("LPIPS image pairs are validation only")

    @override
    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self._batch_size)

    @override
    def test_dataloader(self):
        raise NotImplementedError("LPIPS image pairs are validation only")

    @property
    @override
    def config(self) -> LPIPSDataModuleConfig:
        return self._config

    @property
    @override
    def n_classes(self) -> int:
        return self._first.n_classes

    @property
    @override
    def class_names(self) -> list[str]:
        return self._first.class_names

    @property
    @override
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    @override
    def batch_size(self, value: int):
        self._batch_size = value
