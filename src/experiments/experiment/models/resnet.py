from dataclasses import dataclass
from typing import override

import torch
import torch.nn as nn
import torchvision

from experiments.experiment.models.base_classifier import (
    BaseClassifierModel,
    BaseClassifierModelConfig,
)
from experiments.lib.config_serializing import OptunaOptimised


@dataclass(frozen=True, kw_only=True)
class ResNetConfig(BaseClassifierModelConfig):
    dropout: float | OptunaOptimised = 0.2

    @staticmethod
    @override
    def get_configured_class():
        return ResNet


class ResNet(BaseClassifierModel):
    def __init__(
        self,
        config: ResNetConfig,
    ):
        super().__init__(config)
        self.model = torchvision.models.resnet50(
            weights="IMAGENET1K_V2",
            replace_stride_with_dilation=[
                False,
                True,
                True,
            ],  # increases feature map resolution
        )
        self.model.fc = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.LazyLinear(config.n_classes),
        )

    def backbone_modules(self) -> list[nn.Module]:
        m = self.model
        return [m.conv1, m.bn1, m.layer1, m.layer2, m.layer3, m.layer4]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
