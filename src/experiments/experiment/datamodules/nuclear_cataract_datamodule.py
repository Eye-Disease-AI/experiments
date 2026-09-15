import math
import os
from dataclasses import dataclass, replace
from typing import override

import numpy as np
import torch
from dataset.hard_policy import HardPolicyType
from dataset.loader import HardPolicy, NuclearCataractDataset
from torchvision.transforms import v2 as transformsv2

from experiments.experiment.augmentors.augmentor import Augmentor
from experiments.experiment.augmentors.common import augmentor_name
from experiments.experiment.augmentors.dataset_utils import (
    AugmentedDataset,
    CASDataset,
    FakeDataset,
    simple_stats,
)
from experiments.experiment.datamodules.datamodule import DataModule, DataModuleConfig
from experiments.lib.config_serializing import OptunaOptimised

data_path = "data"
if not os.path.exists(data_path):
    os.mkdir(data_path)

dataset_name = "Nuclear_Cataract_2025_12_28"
dataset_zip_name = f"{dataset_name}.zip"
dataset_zip_path = os.path.join(data_path, dataset_zip_name)
inner_dir_name = " ".join(dataset_name.split("_")[:-3])
inner_path = f"{inner_dir_name}/"

DATASET_PATH = os.path.join(data_path, dataset_name)


@dataclass(frozen=True, kw_only=True)
class NuclearCataractDataModuleConfig(DataModuleConfig):
    batch_size: int | OptunaOptimised = 64
    return_paths: bool = False
    cache: bool = True
    hard_policy: HardPolicyType = HardPolicy.PASSTHROUGH
    image_size: int = 224
    normalize: bool = False
    augment_rot_angle: float = 15
    return_bboxes: bool = False
    bboxes_ratio: float = 1.0

    @staticmethod
    @override
    def get_configured_class():
        return NuclearCataractDataModule


class NuclearCataractDataModule(DataModule):
    DATASET_STD: list[float] = [0.281245, 0.243682, 0.220464]
    DATASET_MEAN: list[float] = [0.229015, 0.1663, 0.106812]
    is_augmented: bool = False

    def __init__(
        self,
        config: NuclearCataractDataModuleConfig,
    ):
        super().__init__()
        self._dir = DATASET_PATH
        self._image_channels = 3
        self._config = config
        self._input_size = (
            self._config.image_size * self._config.image_size * self._image_channels
        )
        self.train_class_weights = None
        self._batch_size = self._config.batch_size
        self._setup_transforms()

    def _cache_size(self) -> int | None:
        if not self._config.cache:
            return None
        max_rad = math.radians(self._config.augment_rot_angle)
        return math.ceil(
            self._config.image_size * (math.sin(max_rad) + math.cos(max_rad))
        )

    def _select_bbox_drops(self, ratio, subset) -> set[int] | None:
        n_total = len(subset)
        box_idxs = [i for i, b in enumerate(subset.bboxes) if b.numel() > 0]
        n_boxes = len(box_idxs)
        target = round(ratio * n_total)
        n_drop = max(0, n_boxes - target)
        if n_drop == 0:
            return None
        perm = torch.randperm(n_boxes)
        return {box_idxs[j] for j in perm[:n_drop].tolist()}

    @staticmethod
    def boxes_ratio(train_set: _SubsetTransformer) -> float:
        """Fraction of samples still carrying a bbox"""
        bboxes = train_set.subset.bboxes
        dropped = train_set.drop_bbox or set()
        have = sum(b.numel() > 0 and i not in dropped for i, b in enumerate(bboxes))
        return have / len(bboxes)

    def _make_train_set(self, train) -> _SubsetTransformer:
        drop = (
            self._select_bbox_drops(self._config.bboxes_ratio, train)
            if self._config.return_bboxes
            else None
        )
        return _SubsetTransformer(train, transform=self.transform, drop_bbox=drop)

    def _augment_dataset_if_needed(self, dataset):
        if (
            ("G" in self.config.augment_kind or self.config.cas_mode)
            and self.config.augmentor_config
            and self.config.n_augment
        ):
            augmentor: Augmentor = self.config.augmentor_config.build()
            n_classes = self.dataset.n_classes
            generator = torch.Generator().manual_seed(self.config.augmentor_seed)
            fake_seeds = torch.randint(
                0, 2**31, [1], generator=generator
            ) + torch.arange(self.config.n_augment)
            fake_labels = torch.randint(
                0, n_classes, [self.config.n_augment], generator=generator
            )
            fake_images = augmentor.generate(fake_seeds, fake_labels)
            print(
                f"Generated fake images shape={fake_images.shape} min={fake_images.min()} max={fake_images.max()} dtype={fake_images.dtype}"
            )
            paths = None
            if self.config.return_paths:
                name = augmentor_name(self.config.augmentor_config)
                paths = [f"augm_{name}_{seed}.png" for seed in fake_seeds.tolist()]
            # We don't need transforms here, because it will be transformed by the same transforms as the real images
            fake_set = FakeDataset(
                fake_images,
                fake_labels,
                paths=paths,
            )
            print(simple_stats(fake_set, "FakeDataset stats"))
            augm = AugmentedDataset([dataset, fake_set])
            print(
                f"Size of dataset (or subset) after augmentation: orig={len(dataset)} augm={len(augm)}"
            )
            return augm

        return dataset

    def _setup_transforms(self):
        val_transforms = [
            transformsv2.Resize((self._config.image_size, self._config.image_size)),
            transformsv2.ConvertImageDtype(),
        ]
        if self._config.normalize:
            val_transforms.append(
                transformsv2.Normalize(mean=self.DATASET_MEAN, std=self.DATASET_STD)
            )
        self.val_transform = transformsv2.Compose(val_transforms)

        if "A" in self.config.augment_kind:
            train_transforms = [
                transformsv2.Resize(
                    (
                        int(np.ceil(self._config.image_size * 1.5)),
                        int(np.ceil(self._config.image_size * 1.5)),
                    )
                ),
                transformsv2.RandomHorizontalFlip(0.5),
                transformsv2.RandomRotation(self._config.augment_rot_angle),
                transformsv2.Resize((self._config.image_size, self._config.image_size)),
                transformsv2.ConvertImageDtype(),
            ]
            if self._config.normalize:
                train_transforms.append(
                    transformsv2.Normalize(mean=self.DATASET_MEAN, std=self.DATASET_STD)
                )
        # This is "R" or "RG" case here
        else:
            # Copy the list
            train_transforms = list(val_transforms)

        self.transform = transformsv2.Compose(train_transforms)

    @override
    def setup(self, stage: str | None = None):
        if not hasattr(self, "dataset"):
            dataset = NuclearCataractDataset(
                NuclearCataractDataset.TrainValMode(0.8, 0.2),
                self._cache_size(),
                self._config.return_paths,
                self._config.return_bboxes,
                hard_policy=self._config.hard_policy,
            )
            self.dataset = CASDataset(dataset) if self._config.cas_mode else dataset

        if not hasattr(self, "test_dataset"):
            self.test_dataset = NuclearCataractDataset(
                NuclearCataractDataset.TestMode(),
                self._cache_size(),
                self._config.return_paths,
                self._config.return_bboxes,
                hard_policy=self._config.hard_policy,
            )

        if not hasattr(self, "train_set"):
            real_train = self.dataset.train_set()
            train = self._augment_dataset_if_needed(real_train)
            weights_from = train if self._config.cas_mode else real_train
            self.train_class_weights = weights_from.class_weights()
            val = self.dataset.val_set()
            test = self.test_dataset.test_set()

            self.train_set = self._make_train_set(train)
            print(simple_stats(self.train_set, "_SubsetTransformer stats"))
            self.val_set = _SubsetTransformer(val, transform=self.val_transform)
            self.test_set = _SubsetTransformer(test, transform=self.val_transform)

        self.dataLoaderCommon = lambda dataset: torch.utils.data.DataLoader(
            dataset,
            batch_size=self._batch_size,
            num_workers=0,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )

    @staticmethod
    def _collate_fn(batch):
        # batch is a list of lists: [img, label, (path?), (bboxes?)]
        images = torch.stack([b[0] for b in batch], dim=0)
        labels = torch.as_tensor([b[1] for b in batch], dtype=torch.long)

        # Figure out what fields are present
        out = [images, labels]
        for i in range(2, len(batch[0])):
            out.append([b[i] for b in batch])
        return tuple(out)

    def setup_fold(self, fold: int, num_folds: int) -> None:
        if self._config.cas_mode:
            raise RuntimeError("setup_fold not supported in cas mode")

        self.dataset = NuclearCataractDataset(
            NuclearCataractDataset.KFoldCVMode(num_folds),
            self._cache_size(),
            self._config.return_paths,
            self._config.return_bboxes,
            hard_policy=self._config.hard_policy,
        )
        real_train = self.dataset.fold_train_set(fold)
        train = self._augment_dataset_if_needed(real_train)
        val = self.dataset.fold_val_set(fold)
        self.train_class_weights = real_train.class_weights()
        self.train_set = self._make_train_set(train)
        self.val_set = _SubsetTransformer(val, transform=self.val_transform)

    @override
    def train_dataloader(self):
        return self.dataLoaderCommon(self.train_set)

    @override
    def val_dataloader(self):
        return self.dataLoaderCommon(self.val_set)

    @override
    def test_dataloader(self):
        return self.dataLoaderCommon(self.test_set)

    @override
    @property
    def config(self) -> NuclearCataractDataModuleConfig:
        return self._config

    @override
    @property
    def n_classes(self) -> int:
        return self.dataset.n_classes

    @override
    @property
    def class_names(self) -> list[str]:
        idx_to_label = {v: k for k, v in self.dataset.label_to_idx.items()}
        return [idx_to_label[i] for i in range(len(idx_to_label))]

    @override
    @property
    def class_weights(self) -> torch.Tensor | None:
        return self.train_class_weights

    @override
    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, v: int):
        assert v > 0
        self._batch_size = v


class _SubsetTransformer(torch.utils.data.Dataset):
    """Wrapper for subset that allows applying different transforms on each dataset subset (train, val, test)"""

    def __init__(self, subset, transform=None, cache=True, drop_bbox=None):
        self.subset = subset
        self.transform = transform
        self.do_cache = cache
        self.drop_bbox = drop_bbox

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        data = self.subset[idx]
        sample = data[0]
        if self.transform is not None:
            sample = self.transform(sample)
        result = [sample]
        result += list(data[1:])
        if self.drop_bbox is not None and idx in self.drop_bbox:
            result[-1] = torch.zeros((0, 4))
        return tuple(result)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from experiments.lib.reproducibility import global_seed_rng

    seed = 2137
    global_seed_rng(seed)
    c = NuclearCataractDataModuleConfig(
        batch_size=16,
        return_paths=True,
        cache=False,
        hard_policy=HardPolicy.PASSTHROUGH,
        image_size=224,
        normalize=False,
        augment_rot_angle=15,
    )

    print("Verifying whether providing bbox ratio works correctly")
    d1: NuclearCataractDataModule = replace(c, return_bboxes=True).build()
    d1.prepare_data()
    d1.setup()
    max_ratio = NuclearCataractDataModule.boxes_ratio(d1.train_set)
    for p in [i / 10 for i in range(10, -1, -1)]:
        d: NuclearCataractDataModule = replace(
            c, return_bboxes=True, bboxes_ratio=p
        ).build()
        d.prepare_data()
        d.setup()
        ratio = NuclearCataractDataModule.boxes_ratio(d.train_set)
        print(ratio)
        assert abs(ratio - p) < 0.05 or (p >= max_ratio and ratio == max_ratio)

    datamodule = c.build()
    datamodule.prepare_data()
    datamodule.setup()

    loader = datamodule.train_dataloader()
    fig, ax = plt.subplots(3, 3)
    loader_iter = iter(loader)
    img, label, *path = next(loader_iter)

    for i, ax in enumerate(ax.flat):
        ax.imshow(img[i].permute(1, 2, 0))
        ax.axis("off")

        # Take just first element, because we also take first image from the batch
        print(label[i])

    plt.tight_layout()
    plt.show()

    print("Positive cases:")

    def print_ratio(name, set):
        p = sum([d[1] for d in set])
        set_len = len(set)
        print(f"{name}: {p}/{set_len}={p / set_len * 100:.4}%")

    print_ratio("train", datamodule.train_set)
    print_ratio("val", datamodule.val_set)
    print_ratio("test", datamodule.test_set)

    # Dataset normalization stats
    ds = NuclearCataractDataset(
        NuclearCataractDataset.TrainValMode(0.8, 0.2), cache_size=224
    )

    all_splits = ds.train_set() + ds.val_set()
    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sum_sq = torch.zeros(3, dtype=torch.float64)
    n_pixels = 0
    for img, _ in all_splits:
        img = img.double() / 255.0 if img.dtype == torch.uint8 else img.double()
        c, h, w = img.shape
        channel_sum += img.sum(dim=[1, 2])
        channel_sum_sq += (img**2).sum(dim=[1, 2])
        n_pixels += h * w
    mean = channel_sum / n_pixels
    std = (channel_sum_sq / n_pixels - mean**2).sqrt()
    print(f"NORMALIZE_MEAN = {[round(v, 6) for v in mean.tolist()]}")
    print(f"NORMALIZE_STD  = {[round(v, 6) for v in std.tolist()]}")
