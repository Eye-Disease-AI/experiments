# Extension of BaselineStudy that may be used to calculate CAS metric
# for generative models. CAS metric can be calculated by using only
# fake dataset to train the classification model and then running
# classification of the real dataset using such classifier. Resulting
# accuracy is the CAS. In the original paper authors mention
# the use of top-5 accuracy, but for our dataset, which has only 2 classes
# this is meaninless. We will thus stick to using top-1 accuracy.
# But should this really be an extension? Maybe adding one flag to the
# baseline study config would make it a lot simpler. E.g. calculate_cas?
# Then we would train the classifier as usual, but use real datamodule in
# test mode and only for calculating CAS, train on fake dataset entirely.
# Or maybe we should WRAP the baseline study, and not extend it!
# And then extend the abstract Study class... using things from
# the baseline study class?


from dataclasses import dataclass
from typing import override

from experiments.experiment.studies.baseline_study import (
    BaselineStudy,
    BaselineStudyConfig,
)


@dataclass(frozen=True, kw_only=True)
class CASStudyConfig(BaselineStudyConfig):
    optuna_metric: str = "val_acc"

    @override
    def validate_config(self):
        super().validate_config()
        if not self.datamodule_config.cas_mode:
            raise ValueError("CAS requires datamodule_config.cas_mode=True")
        if self.datamodule_config.augmentor_config is None:
            raise ValueError("CAS requires datamodule_config.augmentor_config")
        if self.datamodule_config.n_augment is None:
            raise ValueError("CAS requires datamodule_config.n_augment")
        if self.datamodule_config.n_augment <= 0:
            raise ValueError("CAS requires datamodule_config.n_augment > 0")

    @override
    @staticmethod
    def get_configured_class():
        return CASStudy


class CASStudy(BaselineStudy):
    pass
