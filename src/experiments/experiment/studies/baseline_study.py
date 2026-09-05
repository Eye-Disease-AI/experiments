from dataclasses import dataclass
from typing import override

from experiments.experiment.models.convnext import ConvNextConfig

from experiments.experiment.studies.baseline_search_study import (
    BaselineSearchStudy,
    BaselineSearchStudyConfig,
)
from experiments.experiment.studies.study import KFoldValidatable, SeedValidatable


@dataclass(frozen=True, kw_only=True)
class BaselineStudyConfig(BaselineSearchStudyConfig):
    # optimised values: baseline_model_search_convnext_4160e993df7b
    model_config: ConvNextConfig = ConvNextConfig(
        learning_rate=8.396571283771462e-05,
        weight_decay=0.020808622843020545,
        dropout=0.1504758475233118,
    )
    # no optuna trials, just a reproduction of the baseline
    max_trials: int = 1

    @override
    @staticmethod
    def get_configured_class():
        return BaselineStudy


class BaselineStudy(BaselineSearchStudy, KFoldValidatable, SeedValidatable):
    pass
