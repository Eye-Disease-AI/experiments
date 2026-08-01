from dataclasses import replace

from experiments.experiment.datamodules.nuclear_cataract_datamodule import (
    NuclearCataractDataModuleConfig,
)
from experiments.experiment.models.resnet import ResNetConfig
from experiments.experiment.models.gain_resnet import GAINResNetConfig
from experiments.experiment.studies.baseline_search_study import (
    BaselineSearchStudyConfig,
)
from experiments.experiment.studies.baseline_study import BaselineStudyConfig


exp_name = "gain2.1_layers"


def mc():
    return ResNetConfig.from_other(BaselineSearchStudyConfig().model_config)


def main():
    # hypertuning a resnet
    search = BaselineSearchStudyConfig(
        experiment_name=exp_name,
        study_suffix="resnet_search",
        model_config=mc(),
    ).build()
    _, optuna_study = search.run()

    best_params = {
        k.removeprefix("model_config."): v
        for k, v in optuna_study.best_params.items()
        if k.startswith("model_config.")
    }
    tuned_model = replace(mc(), **best_params)  # OptunaOptimised -> best floats

    # test with GAIN
    gain = BaselineStudyConfig(
        experiment_name=exp_name,
        study_suffix="resnet_gain",
        datamodule_config=NuclearCataractDataModuleConfig.from_other(
            BaselineStudyConfig().datamodule_config,
            return_bboxes=True,
        ),
        model_config=GAINResNetConfig.from_other(
            tuned_model,
            use_attention_mining=True,
            use_external_supervision=True,
        ),
    ).build()
    gain.run()


if __name__ == "__main__":
    main()
