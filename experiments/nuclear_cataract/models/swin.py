import torch
import torch.nn as nn
import torchvision
from .base import ModelBase
from ..common_config import LOSS_FN


class Swin(ModelBase):
    def __init__(self, n_classes: int, lr: float = 1e-3, weight_decay: float = 1e-4, dropout: float = 0.2):
        super().__init__(n_classes)
        self.save_hyperparameters()
        self.model = torchvision.models.swin_b(weights='IMAGENET1K_V1')
        in_features = self.model.head.in_features  # 1024
        self.model.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, n_classes),
        )
        self.loss_fn = LOSS_FN()

    def backbone_modules(self):
        return [self.model.features, self.model.norm]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)