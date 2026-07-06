import dataclasses

import torch

from dataset.hard_policy import HardPolicy
from experiments.experiment.datamodules.nuclear_cataract_datamodule import (
    NuclearCataractDataModuleConfig,
)
from experiments.experiment.models.gain_convnext import GAINConvNextConfig
from experiments.experiment.studies.baseline_study import BaselineStudyConfig

_baseline_model = BaselineStudyConfig().model_config

exp_name = "gain_test___004"
studies = [
    BaselineStudyConfig(
        experiment_name=f"{exp_name}",
        datamodule_config=NuclearCataractDataModuleConfig(
            hard_policy=HardPolicy.PASSTHROUGH,
            cache=False,
        ),
        model_config=GAINConvNextConfig(
            use_attention_mining=False,
            use_external_supervision=False,
        ),
        max_trials=1,
        max_epochs=10,
    ),
    BaselineStudyConfig(
        experiment_name=f"{exp_name}_ES",
        datamodule_config=NuclearCataractDataModuleConfig(
            hard_policy=HardPolicy.PASSTHROUGH,
            cache=False,
        ),
        model_config=GAINConvNextConfig(
            use_attention_mining=False,
            use_external_supervision=True,
        ),
        max_trials=1,
        max_epochs=10,
    ),
    BaselineStudyConfig(
        experiment_name=f"{exp_name}_AM",
        datamodule_config=NuclearCataractDataModuleConfig(
            hard_policy=HardPolicy.PASSTHROUGH,
            cache=False,
        ),
        model_config=GAINConvNextConfig(
            use_attention_mining=True,
            use_external_supervision=False,
        ),
        max_trials=1,
        max_epochs=10,
    ),
    BaselineStudyConfig(
        experiment_name=f"{exp_name}_AM_ES",
        datamodule_config=NuclearCataractDataModuleConfig(
            hard_policy=HardPolicy.PASSTHROUGH,
            cache=False,
        ),
        model_config=GAINConvNextConfig(
            use_attention_mining=True,
            use_external_supervision=True,
        ),
        max_trials=1,
        max_epochs=10,
    ),
]


def main():
    torch.set_float32_matmul_precision("high")

    for study in studies:
        study = study.build()

        print("Running baseline study...")
        study.run()


if __name__ == "__main__":
    main()
