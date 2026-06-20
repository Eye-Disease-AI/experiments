from experiments.experiment import common_config
from experiments.experiment.datamodules import init_datamodule
from experiments.experiment.models.convnext import ConvNext
from experiments.experiment.run_study import run_study
from experiments.experiment.studies.baseline_study import BaselineStudy
from experiments.lib.mlflow_setup import Experiment
from experiments.lib.reproducibility import global_seed_rng, get_git_sha

SHA = get_git_sha()

MODELS = [(ConvNext, f"{common_config.EXPERIMENT_NAME}/convnext-search_{SHA}")]


def main():
    global_seed_rng(common_config.SEED)
    exp = Experiment(common_config.EXPERIMENT_NAME)

    datamodule = init_datamodule(BaselineStudy.default_config)
    datamodule.prepare_data()

    for ModelClass, study_name in MODELS:
        print(f"\n{'=' * 60}")
        print(f"Model: {ModelClass.__name__}, Study: {study_name}")
        print(f"{'=' * 60}\n")
        run_study(
            ModelClass,
            study_name,
            exp,
            common_config.SEED,
            datamodule,
            common_config.OPTUNA_METRIC,
            common_config.OPTUNA_DIRECTION,
        )


if __name__ == "__main__":
    main()
