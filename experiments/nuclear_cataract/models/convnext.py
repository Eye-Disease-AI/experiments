import torch
import torch.nn as nn
import torchvision
from .base import ModelBase


class ConvNext(ModelBase):
    def __init__(self, n_classes: int, lr: float = 1e-3, weight_decay: float = 1e-4, dropout: float = 0.2):
        super().__init__(n_classes)
        self.save_hyperparameters()
        self.model = torchvision.models.convnext_base(weights='IMAGENET1K_V1')
        new_clf = list(self.model.classifier.children())[:-1]
        new_clf.append(nn.LazyLinear(n_classes))
        new_clf.append(nn.Dropout(dropout))
        self.model.classifier = nn.Sequential(*new_clf)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
