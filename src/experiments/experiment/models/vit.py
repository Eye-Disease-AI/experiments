from dataclasses import dataclass
from typing import override

import torch
import torch.nn as nn
import torchvision

from experiments.experiment.models.base_classifier import (
    BaseClassifierModel,
    BaseClassifierModelConfig,
)


@dataclass(frozen=True, kw_only=True)
class ViTConfig(BaseClassifierModelConfig):
    dropout: float = 0.2

    @staticmethod
    @override
    def get_configured_class():
        return ViT


class ViT(BaseClassifierModel):
    def __init__(
        self,
        config: ViTConfig,
    ):
        super().__init__(config)
        self.model = torchvision.models.vit_b_16(weights="IMAGENET1K_V1")
        in_features = self.model.heads.head.in_features  # type:ignore # 768
        self.model.heads = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(in_features, config.n_classes),  # type: ignore
        )

    def backbone_modules(self):
        return [self.model.encoder]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
