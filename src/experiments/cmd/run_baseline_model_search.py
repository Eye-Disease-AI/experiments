from experiments.experiment.models.convnext import ConvNextConfig
from experiments.experiment.models.swin import SwinConfig
from experiments.experiment.models.vit import ViTConfig
from experiments.experiment.studies.baseline_search_study import (
    BaselineSearchStudyConfig,
)
from experiments.lib.config_serializing import OptunaOptimised


def hparams():
    return dict(
        learning_rate=OptunaOptimised(
            "float", {"low": 1e-6, "high": 1e-4, "log": True}
        ),
        weight_decay=OptunaOptimised("float", {"low": 1e-8, "high": 5e-2, "log": True}),
        dropout=OptunaOptimised("float", {"low": 0.0, "high": 0.5}),
    )


exp_name = "baseline_model_search"
studies = [
    BaselineSearchStudyConfig(
        experiment_name=exp_name,
        study_suffix="convnext",
        model_config=ConvNextConfig(**hparams()),
    ),
    BaselineSearchStudyConfig(
        experiment_name=exp_name,
        study_suffix="vit",
        model_config=ViTConfig(**hparams()),
    ),
    BaselineSearchStudyConfig(
        experiment_name=exp_name,
        study_suffix="swin",
        model_config=SwinConfig(**hparams()),
    ),
]


def main():
    for config in studies:
        study = config.build()

        print(f"Running baseline search study: {config.study_suffix}...")
        study.run()


if __name__ == "__main__":
    main()
