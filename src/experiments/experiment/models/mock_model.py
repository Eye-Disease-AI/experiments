from dataclasses import dataclass
from typing import override

import torch
import torch.nn as nn

from experiments.experiment.models.base_classifier import (
    BaseClassifierModel,
)
from experiments.experiment.models.convnext import ConvNextConfig


@dataclass(frozen=True, kw_only=True)
class MockConvNextConfig(ConvNextConfig):
    @staticmethod
    @override
    def get_configured_class():
        return MockConvNext


class MockConvNext(BaseClassifierModel):
    def __init__(self, config: MockConvNextConfig):
        super().__init__(config)
        self.head = nn.Linear(1, config.n_classes)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x.flatten(1).mean(dim=1, keepdim=True))
