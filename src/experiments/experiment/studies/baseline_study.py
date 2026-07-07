from dataclasses import dataclass
from typing import override

from experiments.experiment.models.convnext import ConvNextConfig

from experiments.experiment.studies.baseline_search_study import (
    BaselineSearchStudy,
    BaselineSearchStudyConfig,
)
from experiments.experiment.studies.study import KFoldValidatable


@dataclass(frozen=True, kw_only=True)
class BaselineStudyConfig(BaselineSearchStudyConfig):
    # optimised values: baseline_model_search_convnext_4160e993df7b
    model_config: ConvNextConfig = ConvNextConfig(
        learning_rate=8.759615455801614e-05,
        weight_decay=0.565268242196509e-08,
        dropout=0.15595523137824943,
    )
    # no optuna trials, just a reproduction of the baseline
    max_trials: int = 1

    @override
    @staticmethod
    def get_configured_class():
        return BaselineSearchStudy


class BaselineStudy(BaselineSearchStudy, KFoldValidatable):
    pass
