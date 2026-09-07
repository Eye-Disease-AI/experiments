from dataclasses import dataclass
from typing import override

from experiments.experiment.models.gain_convnext import GAINConvNextConfig
from experiments.experiment.studies.baseline_study import (
    BaselineStudy,
    BaselineStudyConfig,
)
from experiments.experiment.studies.study import KFoldValidatable, SeedValidatable


@dataclass(frozen=True, kw_only=True)
class GainOptimisedStudyConfig(BaselineStudyConfig):
    # optimised values: gain1_optim_val_miou_8af2d05a683a/validation
    model_config: GAINConvNextConfig = GAINConvNextConfig(
        learning_rate=8.759615455801614e-05,
        weight_decay=0.565268242196509e-08,
        dropout=0.15595523137824943,
        am_loss_weight=0.08297327587849512,
        es_loss_weight=6.19856154535434,
        sigma_mask=0.988739867921397,
        omega_mask=59.00482132043015,
    )
    # no optuna trials, just a reproduction of the baseline
    max_trials: int = 1

    @override
    @staticmethod
    def get_configured_class():
        return GainOptimisedStudy


class GainOptimisedStudy(BaselineStudy, KFoldValidatable, SeedValidatable):
    pass
