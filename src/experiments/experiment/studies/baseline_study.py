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
    # only override from parent
    # TODO use actual optuna-optimised values
    model_config: ConvNextConfig = ConvNextConfig(
        learning_rate=5e-5, weight_decay=1e-4, dropout=0.2
    )
    # no optuna trials, just a reproduction of the baseline
    max_trials: int = 1

    @override
    @staticmethod
    def get_configured_class():
        return BaselineSearchStudy


class BaselineStudy(BaselineSearchStudy, KFoldValidatable):
    pass
