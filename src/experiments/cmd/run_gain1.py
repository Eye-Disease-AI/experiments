from experiments.experiment.datamodules.nuclear_cataract_datamodule import (
    NuclearCataractDataModuleConfig,
)
from experiments.experiment.models.gain_convnext import GAINConvNextConfig
from experiments.experiment.studies.baseline_study import (
    BaselineStudyConfig,
)

def dmc():
    return NuclearCataractDataModuleConfig.from_other(
        BaselineStudyConfig().datamodule_config,
        return_bboxes=True,
    )

exp_name = "gain_test_0008"
studies = [
    BaselineStudyConfig(
        experiment_name=exp_name,
        study_suffix="baseline",
        datamodule_config=dmc(),
        model_config=GAINConvNextConfig.from_other(
            BaselineStudyConfig().model_config,
            use_attention_mining=False,
            use_external_supervision=False,
        ),
    ),
    BaselineStudyConfig(
        experiment_name=exp_name,
        study_suffix="ES",
        datamodule_config=dmc(),
        model_config=GAINConvNextConfig.from_other(
            BaselineStudyConfig().model_config,
            use_attention_mining=False,
            use_external_supervision=True,
        ),
    ),
    BaselineStudyConfig(
        experiment_name=exp_name,
        study_suffix="AM",
        datamodule_config=dmc(),
        model_config=GAINConvNextConfig.from_other(
            BaselineStudyConfig().model_config,
            use_attention_mining=True,
            use_external_supervision=False,
        ),
    ),
    BaselineStudyConfig(
        experiment_name=exp_name,
        study_suffix="AM_ES",
        datamodule_config=dmc(),
        model_config=GAINConvNextConfig.from_other(
            BaselineStudyConfig().model_config,
            use_attention_mining=True,
            use_external_supervision=True,
        ),
    ),
]


def main():
    for study in studies:
        study = study.build()

        print("Running baseline study...")
        study.run()


if __name__ == "__main__":
    main()
