from dataclasses import dataclass
from typing import override

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as transforms

from experiments.experiment.datamodules.datamodule import DataModule, DataModuleConfig


@dataclass(frozen=True, kw_only=True)
class LPIPSDataModuleConfig(DataModuleConfig):
    first_config: DataModuleConfig
    second_config: DataModuleConfig
    equal_datamodules: bool = False
    batch_size: int = 16
    image_size: int = 224

    @override
    def validate_config(self):
        super().validate_config()
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

        self.dataset = _PairDataset(
            first_by_class,
            second_by_class,
            self._config.image_size,
            self._config.equal_datamodules,
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


class _PairDataset(Dataset):
    def __init__(
        self,
        first_by_class: list[list[torch.Tensor]],
        second_by_class: list[list[torch.Tensor]],
        image_size: int,
        equal_datamodules: bool,
    ):
        self._first = first_by_class
        self._second = second_by_class

        self._transform = transforms.Compose(
            [
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Resize((image_size, image_size)),
            ]
        )

        index = []
        for label, (first, second) in enumerate(
            zip(first_by_class, second_by_class, strict=True)
        ):
            pairs = torch.cartesian_prod(
                torch.arange(len(first)), torch.arange(len(second))
            )
            if equal_datamodules:
                pairs = pairs[pairs[:, 0] != pairs[:, 1]]
            labels = torch.full((len(pairs), 1), label)
            index.append(torch.cat([labels, pairs], dim=1))
        self._index = torch.cat(index)
        assert len(self._index) > 0, "No image pairs to compare"

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx: int):
        label, first, second = self._index[idx].tolist()
        return (
            self._transform(self._first[label][first]).clamp(0, 1),
            self._transform(self._second[label][second]).clamp(0, 1),
            label,
        )
