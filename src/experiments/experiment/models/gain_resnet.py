from dataclasses import dataclass, field
from typing import override

from experiments.experiment.models.resnet import ResNet, ResNetConfig
from experiments.experiment.models.gain import GAINWrapper, GAINWrapperConfig


@dataclass(frozen=True, kw_only=True)
class GAINResNetConfig(GAINWrapperConfig, ResNetConfig):
    target_layers: list[tuple[str | int, ...]] = field(
        default_factory=lambda: [("model", "layer4", -1)]
    )

    @staticmethod
    @override
    def get_configured_class():
        return GAINResNet


class GAINResNet(GAINWrapper, ResNet):
    """The class only exists to join a model with GAINWrapper"""

    pass
