from collections import defaultdict

import torch
from torch.utils.data.dataset import ConcatDataset, Dataset


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


class AugmentedDataset(ConcatDataset):
    def class_weights(self) -> torch.Tensor:
        classes_counts_dict: dict = defaultdict(lambda: 0)
        all_count = 0

        for _, label in self:  # type: ignore
            if isinstance(label, torch.Tensor):
                label_item = label.item()
            else:
                label_item = label

            classes_counts_dict[label_item] += 1
            all_count += 1

        classes_counts = []
        for _, v in sorted(classes_counts_dict.items()):
            classes_counts.append(v)
        classes_counts_tensor = torch.Tensor(classes_counts)

        return all_count / (len(classes_counts) * classes_counts_tensor)
