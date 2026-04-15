from lib.seed import RNG
import lightning as L
import os
import torchvision
from torchvision.transforms import v2 as transformsv2
import torch
import requests
from requests.auth import HTTPBasicAuth
from getpass import getpass
import zipfile
import shutil
from multiprocessing import cpu_count
import math
import json
import random

data_path = "data"
if not os.path.exists(data_path):
    os.mkdir(data_path)

dataset_name = "Nuclear_Cataract_2025_12_28"
dataset_zip_name = f"{dataset_name}.zip"
dataset_zip_path = os.path.join(data_path, dataset_zip_name)
inner_dir_name = " ".join(dataset_name.split("_")[:-3])
inner_path = f"{inner_dir_name}/"

DATASET_PATH=os.path.join(data_path, dataset_name)

# Data loading
class MyDataset(torch.utils.data.Dataset):
    """Class representing the dataset. Converts the dataset files into values usable to the model."""
    def __init__(self, dir, return_paths=False, split_subset="trainvalSet"):
        super().__init__()
        self.data_dir = dir
        self.images_dir = os.path.join(self.data_dir, "images")
        self.annotations_file = os.path.join(self.data_dir, "labels.json")
        self.split_file = os.path.join(self.data_dir, "split_with_hard.json")
        self.return_paths = return_paths

        with open(self.annotations_file, "r") as f:
            self.raw_annotations = json.load(f)

        with open(self.split_file, "r") as f:
            self.splits = json.load(f)

        self.packs = self.splits[split_subset]

        self.annotations = [(fname, label) for pack in self.packs for fname, label in pack]
        self.label_names = sorted({label for _, label in self.annotations}, reverse=True)
        
        self.n_classes = len(self.label_names)
        # map of labelname to idx
        self.label_to_idx = {name: i for i, name in enumerate(self.label_names)}
        # change annotations to contain label indices instead of names
        self.annotations = [
            [fname, self.label_to_idx[label]]
            for fname, label in self.annotations
        ]

        # For fast access we also create a map from paths to indexes
        self.path_to_index = {
            fname: i for i, (fname, _) in enumerate(self.annotations)
        }
        
    def __len__(self):
        """size of dataset"""
        return len(self.annotations)
    
    def __getitem__(self, idx):
        """get n'th item (img,label) in the dataset"""
        fname, label = self.annotations[idx]
        img = torchvision.io.decode_image(os.path.join(self.images_dir, fname))

        if self.return_paths:
            return img, label, fname
        else:
            return img, label

    def get_by_path(self, path: str):
        idx = self.path_to_index[path]
        return self[idx]

class SubsetTransformer(torch.utils.data.Dataset):
    """Wrapper for subset that allows applying different transforms on each dataset subset (train, val, test)"""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

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
    def __init__(self, rng: RNG, batch_size: int = 32, return_paths=False):
        super().__init__()
        self.dir = DATASET_PATH
        self.image_size = 224
        self.image_channels = 3
        self.input_size = self.image_size * self.image_size * self.image_channels
        self.return_paths = return_paths
        self.rng = rng
        self.batch_size = batch_size
    
    def prepare_data(self):
        """Download and unzip dataset. No state assignment here (Lightning may skip this on non-rank-0)."""
        if not os.path.exists(dataset_zip_path):
            print("Downloading the dataset")
            username = input("Username: ")
            password = getpass("Password: ")
            basic = HTTPBasicAuth(username, password)
            r = requests.get(f"https://zpbfileserver.krzyzanowski.dev/{dataset_zip_name}", auth=basic, stream=True)
            with open(dataset_zip_path, "wb") as dest_file:
                for chunk in r.iter_content(chunk_size=128):
                    dest_file.write(chunk)
        else:
            print(f"Skipping dataset download as {dataset_zip_path} is already present")

        if not os.path.isdir(DATASET_PATH):
            with zipfile.ZipFile(dataset_zip_path, "r") as zip_ref:
                for inner_file in zip_ref.namelist():
                    if inner_file.startswith(inner_path) and len(inner_file) > len(inner_path):
                        zip_ref.extract(inner_file, data_path + "/")
            shutil.move(os.path.join(data_path, inner_path), DATASET_PATH)
        else:
            print(f"Dataset already exists at {DATASET_PATH}")

    def setup(self, stage: str = None):
        max_angle = 15
        max_rad = math.radians(max_angle)
        pre_rot_size = int(math.ceil(self.image_size * (math.sin(max_rad) + math.cos(max_rad))))
        self.transform = transformsv2.Compose([
            transformsv2.Resize((pre_rot_size, pre_rot_size)),
            transformsv2.ConvertImageDtype(),
            transformsv2.RandomHorizontalFlip(0.5),
            transformsv2.RandomRotation(15),
            transformsv2.CenterCrop((self.image_size, self.image_size)),
            transformsv2.GaussianNoise(mean=0, sigma=0.01)
        ])
        self.val_transform = transformsv2.Compose([
            transformsv2.Resize((self.image_size, self.image_size)),
            transformsv2.ConvertImageDtype(),
        ])

        if not hasattr(self, 'dataset'):
            self.dataset = MyDataset(self.dir, return_paths=self.return_paths, split_subset="trainvalSet")

        if not hasattr(self, 'test_dataset'):
            self.test_dataset = MyDataset(self.dir, return_paths=self.return_paths, split_subset="testSet")

        if not hasattr(self, 'train_set'):
            train, val = self.split_dataset(self.dataset)
            self.train_set = SubsetTransformer(train, transform=self.transform)
            self.val_set   = SubsetTransformer(val,   transform=self.val_transform)
            self.test_set   = SubsetTransformer(self.test_dataset,   transform=self.val_transform)

        num_workers = 0 if torch.mps.is_available() else cpu_count() - 1
        self.dataLoaderCommon = lambda dataset: torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, num_workers=num_workers, pin_memory=True
        )

    def split_dataset(self, dataset: MyDataset, train_ratio=8/9, val_ratio=1/9):
        rng = random.Random(self.rng.seed)
        packages = list(dataset.packs)
        rng.shuffle(packages)

        total = len(packages)
        train_n = int(train_ratio * total)

        train_pkgs = packages[:train_n]
        val_pkgs   = packages[train_n:]

        def pkgs_to_indices(pkgs):
            return [
                dataset.path_to_index[fname]
                for pkg in pkgs
                for fname, label in pkg
            ]

        return (
            torch.utils.data.Subset(dataset, pkgs_to_indices(train_pkgs)),
            torch.utils.data.Subset(dataset, pkgs_to_indices(val_pkgs))
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

    datamodule = MyDataModule(rng, True)
    datamodule.prepare_data()
    datamodule.setup()

    loader = datamodule.val_dataloader()
    fig, ax = plt.subplots(3,3)
    loader_iter = iter(loader)
    img, label, *path = next(loader_iter)

    for i, ax in enumerate(ax.flat):
        ax.imshow(img[i].permute(1,2,0))
        ax.axis('off')
        
        # Take just first element, because we also take first image from the batch
        print(label[i])
        
        # Path is optional so we need to check
        # * means that it will be an array, if present it will have one element
        # is absent it won't have any elements
        if len(path) != 0:
            print(path[0][i])
            print([i["id"] for i in datamodule.dataset.raw_annotations if str(i["image"]).split('/')[-1] == path[0][0]])
    plt.tight_layout()
    plt.show()

    print("Positive cases:")
    def print_ratio(name, set):
        p = sum([d[1] for d in set])
        l = len(set)
        print(f"{name}: {p}/{l}={p/l*100:.4}%")
    print_ratio("train", datamodule.train_set)
    print_ratio("val", datamodule.val_set)
    print_ratio("test", datamodule.test_set)