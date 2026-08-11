from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from experiments.lib.config_serializing import ClassConfig


@dataclass(frozen=True, kw_only=True)
class AugmentorConfig(ClassConfig): ...


class Augmentor(ABC):
    @abstractmethod
    def generate(self, seeds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        assert len(seeds) == len(labels)


#    @abstractmethod
#    def command_line(self) -> str:
#        raise NotImplementedError
#
#    @abstractmethod
#    def load_from_cache(self) -> str:
#        raise NotImplementedError
