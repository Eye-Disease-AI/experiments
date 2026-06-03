import torch
import torch.nn as nn
import torchvision

from .base import ModelBase


class ViT(ModelBase):
    def __init__(
        self,
        n_classes: int,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.2,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__(n_classes, class_weights=class_weights)
        self.save_hyperparameters()
        self.model = torchvision.models.vit_b_16(weights="IMAGENET1K_V1")
        in_features = self.model.heads.head.in_features  # type:ignore # 768
        self.model.heads = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, n_classes), # type: ignore
        )

    def backbone_modules(self):
        return [self.model.encoder]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
