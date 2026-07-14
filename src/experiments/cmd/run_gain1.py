import torch

from dataset.hard_policy import HardPolicy
from experiments.experiment.datamodules.nuclear_cataract_datamodule import (
    NuclearCataractDataModuleConfig,
)
from experiments.experiment.models.gain_convnext import GAINConvNextConfig
from experiments.experiment.studies.baseline_search_study import (
    BaselineSearchStudyConfig,
)

exp_name = "gain_test___007"
studies = [
    BaselineSearchStudyConfig(
        experiment_name=exp_name,
        study_suffix="baseline",
        datamodule_config=NuclearCataractDataModuleConfig(
            hard_policy=HardPolicy.DOMINATE,
            return_bboxes=True,
        ),
        model_config=GAINConvNextConfig(
            use_attention_mining=False,
            use_external_supervision=False,
        ),
        max_trials=1,
    ),
    BaselineSearchStudyConfig(
        experiment_name=exp_name,
        study_suffix="ES",
        datamodule_config=NuclearCataractDataModuleConfig(
            hard_policy=HardPolicy.DOMINATE,
            return_bboxes=True,
        ),
        model_config=GAINConvNextConfig(
            use_attention_mining=False,
            use_external_supervision=True,
        ),
        max_trials=1,
    ),
    BaselineSearchStudyConfig(
        experiment_name=exp_name,
        study_suffix="AM",
        datamodule_config=NuclearCataractDataModuleConfig(
            hard_policy=HardPolicy.DOMINATE,
            return_bboxes=True,
        ),
        model_config=GAINConvNextConfig(
            use_attention_mining=True,
            use_external_supervision=False,
        ),
        max_trials=1,
    ),
    BaselineSearchStudyConfig(
        experiment_name=exp_name,
        study_suffix="AM_ES",
        datamodule_config=NuclearCataractDataModuleConfig(
            hard_policy=HardPolicy.DOMINATE,
            return_bboxes=True,
        ),
        model_config=GAINConvNextConfig(
            use_attention_mining=True,
            use_external_supervision=True,
        ),
        max_trials=1,
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
