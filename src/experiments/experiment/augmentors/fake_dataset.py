import torch
from torch.utils.data.dataset import Dataset


class FakeDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor, transforms=None):
        assert images.shape[0] == labels.shape[0]
        self.images = images
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx: int):
        img, label = self.images[idx], self.labels[idx]

        if self.transforms:
            img = self.transforms(img)

        return img, label
