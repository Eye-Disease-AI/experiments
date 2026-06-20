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
class ConvNextConfig(BaseClassifierModelConfig):
    dropout: float = 0.2

    @staticmethod
    @override
    def get_configured_class():
        return ConvNext


class ConvNext(BaseClassifierModel):
    def __init__(
        self,
        config: ConvNextConfig,
    ):
        super().__init__(config)
        self.save_hyperparameters()
        self.model = torchvision.models.convnext_tiny(weights="IMAGENET1K_V1")
        new_clf = list(self.model.classifier.children())[:-1]
        new_clf.append(nn.LazyLinear(config.n_classes))
        new_clf.append(nn.Dropout(config.dropout))
        self.model.classifier = nn.Sequential(*new_clf)

    def backbone_modules(self) -> list[nn.Module]:
        return [self.model.features]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
