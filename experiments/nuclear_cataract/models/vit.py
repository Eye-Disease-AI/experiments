import torch
import torch.nn as nn
import torchvision
from .base import ModelBase
from ..common_config import LOSS_FN


class ViT(ModelBase):
    def __init__(self, n_classes: int, lr: float = 1e-3, weight_decay: float = 1e-4, dropout: float = 0.2):
        super().__init__(n_classes)
        self.save_hyperparameters()
        self.model = torchvision.models.vit_b_16(weights='IMAGENET1K_V1')
        in_features = self.model.heads.head.in_features  # 768
        self.model.heads = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, n_classes),
        )
        self.loss_fn = LOSS_FN()

    def backbone_modules(self):
        return [self.model.encoder]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)