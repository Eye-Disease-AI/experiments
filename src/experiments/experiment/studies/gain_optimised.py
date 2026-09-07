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
    # optimised values: gain1_optim_val_miou_e8eb6292e31f/validation
    model_config: GAINConvNextConfig = GAINConvNextConfig(
        learning_rate=8.396571283771462e-05,
        weight_decay=0.020808622843020545,
        dropout=0.1504758475233118,
        am_loss_weight=2.5912372766285934,
        es_loss_weight=9.732618849987137,
        sigma_mask=0.9861238615442142,
        omega_mask=199.67411980541792,
    )
    # no optuna trials, just a reproduction of the baseline
    max_trials: int = 1

    @override
    @staticmethod
    def get_configured_class():
        return GainOptimisedStudy


class GainOptimisedStudy(BaselineStudy, KFoldValidatable, SeedValidatable):
    pass
