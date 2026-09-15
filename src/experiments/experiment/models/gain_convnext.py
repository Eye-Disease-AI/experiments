from dataclasses import dataclass, field
from typing import override

from experiments.experiment.models.convnext import ConvNext, ConvNextConfig
from experiments.experiment.models.gain import GAINWrapper, GAINWrapperConfig


@dataclass(frozen=True, kw_only=True)
class GAINConvNextConfig(GAINWrapperConfig, ConvNextConfig):
    target_layers: list[tuple[str | int, ...]] = field(
        default_factory=lambda: [
            (
                "model",
                "features",
                -3,
            )
        ]
    )

    @staticmethod
    @override
    def get_configured_class():
        return GAINConvNext


class GAINConvNext(GAINWrapper, ConvNext):
    """The class only exists to join a model with GAINWrapper"""

    pass
