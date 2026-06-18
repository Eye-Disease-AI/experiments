from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass

@dataclass(frozen=True, kw_only=True)
class StudyConfig():
    experiment_name: str
    seed: int

class Study(ABC):
    @abstractmethod
    def __init__(self, config: StudyConfig): ...
    @abstractmethod
    def run(self): ...
