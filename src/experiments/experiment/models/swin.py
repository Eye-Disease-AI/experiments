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
class SwinConfig(BaseClassifierModelConfig):
    dropout: float

    @staticmethod
    @override
    def get_configured_class():
        return Swin


class Swin(BaseClassifierModel):
    DEFAULT_CONFIG: SwinConfig

    def __init__(
        self,
        config: SwinConfig,
    ):
        super().__init__(config)
        self.save_hyperparameters()
        self.model = torchvision.models.swin_b(weights="IMAGENET1K_V1")
        in_features = self.model.head.in_features  # 1024
        self.model.head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(in_features, config.n_classes),
        )

    def backbone_modules(self):
        return [self.model.features, self.model.norm]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


Swin.DEFAULT_CONFIG = SwinConfig(
    **vars(BaseClassifierModel.DEFAULT_CONFIG), dropout=0.2
)
