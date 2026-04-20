import torch
import torch.nn as nn
import torchvision
from .base import ModelBase


class ConvNext(ModelBase):
    def __init__(self, n_classes: int, lr: float = 1e-3, weight_decay: float = 1e-4, dropout: float = 0.2, class_weights: torch.Tensor | None = None):
        super().__init__(n_classes, class_weights=class_weights)
        self.save_hyperparameters()
        self.model = torchvision.models.convnext_base(weights='IMAGENET1K_V1')
        new_clf = list(self.model.classifier.children())[:-1]
        new_clf.append(nn.Dropout(dropout))
        new_clf.append(nn.LazyLinear(n_classes))
        self.model.classifier = nn.Sequential(*new_clf)

    def backbone_modules(self):
        return [self.model.features]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
