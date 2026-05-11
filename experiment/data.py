import math
import os

import lightning as L
import torch
from torchvision.transforms import v2 as transformsv2

from dataset.loader import NuclearCataractDataset, HardPolicy
from experiment.common_config import NORMALIZE_MEAN, NORMALIZE_STD
from lib.seed import RNG
import numpy as np

data_path = "data"
if not os.path.exists(data_path):
    os.mkdir(data_path)

dataset_name = "Nuclear_Cataract_2025_12_28"
dataset_zip_name = f"{dataset_name}.zip"
dataset_zip_path = os.path.join(data_path, dataset_zip_name)
inner_dir_name = " ".join(dataset_name.split("_")[:-3])
inner_path = f"{inner_dir_name}/"

DATASET_PATH = os.path.join(data_path, dataset_name)


class SubsetTransformer(torch.utils.data.Dataset):
    """Wrapper for subset that allows applying different transforms on each dataset subset (train, val, test)"""

    def __init__(self, subset, transform=None, cache=True):
        self.subset = subset
        self.transform = transform
        self.do_cache = cache

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        data = self.subset[idx]
        sample = data[0]
        if self.transform is not None:
            sample = self.transform(sample)
        result = [sample]
        result += list(data[1:])
        return tuple(result)


class MyDataModule(L.LightningDataModule):
    """Splits to train/val/test subsets with custom transforms"""

    def __init__(self, rng: RNG, batch_size: int = 32, return_paths=False, cache=True, hard_policy=HardPolicy.PASSTHROUGH):
        super().__init__()
        self.dir = DATASET_PATH
        self.image_size = 224
        self.image_channels = 3
        self.input_size = self.image_size * self.image_size * self.image_channels
        self.return_paths = return_paths
        self.rng = rng
        self.batch_size = batch_size
        self.do_cache = cache
        self.hard_policy = hard_policy

    def setup(self, stage: str | None = None):
        max_angle = 15
        max_rad = math.radians(max_angle)
        pre_rot_size = int(
            math.ceil(self.image_size * (math.sin(max_rad) + math.cos(max_rad)))
        )

        if not hasattr(self, "dataset"):
            self.dataset = NuclearCataractDataset(
                NuclearCataractDataset.TrainValMode(0.8, 0.2),
                pre_rot_size if self.do_cache else None,
                self.return_paths,
                hard_policy=self.hard_policy
            )

        if not hasattr(self, "test_dataset"):
            self.test_dataset = NuclearCataractDataset(
                NuclearCataractDataset.TestMode(),
                pre_rot_size if self.do_cache else None,
                self.return_paths,
                hard_policy=self.hard_policy
            )

        self.transform = transformsv2.Compose(
            [
            transformsv2.Resize((int(np.ceil(self.image_size*1.5)), int(np.ceil(self.image_size*1.5)))),
            transformsv2.RandomHorizontalFlip(0.5),
            transformsv2.RandomRotation(15),
            transformsv2.Resize((self.image_size, self.image_size)),
            transformsv2.ConvertImageDtype(),
            ]
        )
        self.val_transform = transformsv2.Compose(
            [
                transformsv2.Resize((self.image_size, self.image_size)),
                transformsv2.ConvertImageDtype(),
                transformsv2.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
            ]
        )

        if not hasattr(self, "train_set"):
            train = self.dataset.train_set()
            self.train_class_weights = train.class_weights()
            val = self.dataset.val_set()
            test = self.test_dataset.test_set()

            self.train_set = SubsetTransformer(train, transform=self.transform)
            self.val_set = SubsetTransformer(val, transform=self.val_transform)
            self.test_set = SubsetTransformer(test, transform=self.val_transform)

        self.dataLoaderCommon = lambda dataset: torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, num_workers=0, pin_memory=True
        )

    def train_dataloader(self):
        return self.dataLoaderCommon(self.train_set)

    def val_dataloader(self):
        return self.dataLoaderCommon(self.val_set)

    def test_dataloader(self):
        return self.dataLoaderCommon(self.test_set)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rng = RNG()
    rng.set_seed(2137)

    datamodule = MyDataModule(rng, return_paths=True, cache=False)
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
    for img, _ in all_splits:  # pyright: ignore
        img = img.double() / 255.0 if img.dtype == torch.uint8 else img.double()
        c, h, w = img.shape
        channel_sum += img.sum(dim=[1, 2])
        channel_sum_sq += (img**2).sum(dim=[1, 2])
        n_pixels += h * w
    mean = channel_sum / n_pixels
    std = (channel_sum_sq / n_pixels - mean**2).sqrt()
    print(f"NORMALIZE_MEAN = {[round(v, 6) for v in mean.tolist()]}")
    print(f"NORMALIZE_STD  = {[round(v, 6) for v in std.tolist()]}")
