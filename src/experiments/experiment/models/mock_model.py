from dataclasses import dataclass
from typing import override

import torch
import torch.nn as nn

from experiments.experiment.models.base_classifier import BaseClassifierModel, BaseClassifierModelConfig

@dataclass(frozen=True, kw_only=True)
class MockModelConfig(BaseClassifierModelConfig):
    @staticmethod
    @override
    def get_configured_class():
        return MockModel


class MockModel(BaseClassifierModel):
    DEFAULT_CONFIG: MockModelConfig

    def __init__(
        self,
        config: MockModelConfig,
    ):
        super().__init__(config)

    def backbone_modules(self) -> list[nn.Module]:
        return [self.model.features]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(self.config.n_classses)


MockModel.DEFAULT_CONFIG = MockModelConfig(
    **vars(BaseClassifierModel.DEFAULT_CONFIG)
)
