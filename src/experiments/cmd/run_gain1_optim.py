# Gain optimisation
# Find optimal hiperparams for default GAIN AM+ES

from experiments.experiment.datamodules.nuclear_cataract_datamodule import (
    NuclearCataractDataModuleConfig,
)
from experiments.experiment.models.gain_convnext import GAINConvNextConfig
from experiments.experiment.studies.baseline_study import (
    BaselineStudyConfig,
)
from experiments.lib.config_serializing import OptunaOptimised


def dmc():
    return NuclearCataractDataModuleConfig.from_other(
        BaselineStudyConfig().datamodule_config,
        return_bboxes=True,
    )


exp_name = "gain1_optim"
studies = [
    BaselineStudyConfig(
        experiment_name=exp_name,
        study_suffix="baseline",
        datamodule_config=dmc(),
        model_config=GAINConvNextConfig(
            use_attention_mining=True,
            use_external_supervision=True,
            am_loss_weight=OptunaOptimised(
                "float", {"low": 0, "high": 10, "log": False}
            ),
            es_loss_weight=OptunaOptimised(
                "float", {"low": 0, "high": 10, "log": False}
            ),
            sigma_mask=OptunaOptimised("float", {"low": 0.1, "high": 1, "log": False}),
            omega_mask=OptunaOptimised("float", {"low": 1, "high": 200, "log": False}),
        ),
        max_trials=100,
        optuna_metric="val_map_50",
    ),
]


def main():
    for study in studies:
        study = study.build()
        study.run()


if __name__ == "__main__":
    main()
