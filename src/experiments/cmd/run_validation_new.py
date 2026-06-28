"""
Description of what this entrypoint should achieve:
It should perform some kind of verification. I can see that
there are two modes:

* seeds with --num-seeds option
* kfold with -K option

It creates separete validation study with the first component
matching the original name. Study is run as part of the same
MLFlow experiment.

1. We load best study params
2. Set up data modules

In the legacy code we set up Nuclear Cataract datamodule using
the params, but now we are leaving decision about any needed modules to the
study class. So either study class should expose publicly method for getting its
datamodules OR we can consturct them ourselves based on the config OR we can
move all validation logic to be part of study class.

3. MLFlow orchestration
4. Look for parameters with `best_` prefix in the selected study, start a run
and log these values there
5. Start nested run named `retrain` and do the actual training there using earlier
loaded and logged best parameters
6. Run either kfold or seeds validation based on CLI options. These modes also
start nested runs and log parameters which are mode-specific and run names
contain information about which seed or which fold was used
"""

import argparse
import os
from tempfile import TemporaryDirectory
from experiments.experiment.studies.study import Study
from experiments.lib.config_serializing import ClassConfig
from experiments.lib.mlflow_setup import Experiment
import mlflow
import mlflow.entities
import optuna


class StudyValidator:
    def __init__(
        self,
        experiment_name: str,
        study_name: str,
    ):
        self.experiment_name = experiment_name
        self.study_name = study_name

    def _find_parent_run(self) -> mlflow.entities.Run | None:
        client = Experiment().client
        mlflow_exp = client.get_experiment_by_name(self.experiment_name)

        if not mlflow_exp:
            return None

        for run in client.search_runs([mlflow_exp.experiment_id], max_results=9999):
            if (
                run.data.tags["optuna_study"] == self.study_name
                and "mlflow.parentRunId" not in run.data.tags
            ):
                return run

        return None

    def _find_best_params(self):
        exp = Experiment(self.experiment_name)
        study = optuna.load_study(study_name=self.study_name, storage=exp.storage)
        return study.best_params

    def _recreate_study(self):
        exp = Experiment(self.experiment_name)
        run = self._find_parent_run()
        assert run is not None

        with TemporaryDirectory() as tmp_dir:
            config_file_path = exp.client.download_artifacts(
                run.info.run_id, Study.MLFLOW_STUDY_CONFIG_FILE_PATH, tmp_dir
            )
            with open(config_file_path, "r") as config_tmp_file:
                config_file_json = config_tmp_file.read()
                class_config = ClassConfig.from_json(config_file_json)
                return class_config.build()

    def _find_best_epoch(self):
        exp = Experiment(self.experiment_name)
        study = optuna.load_study(study_name=self.study_name, storage=exp.storage)
        _agg = min if study.direction == "min" else max
        return _agg(
            study.best_trial.intermediate_values,
            key=study.best_trial.intermediate_values.get,  # pyright: ignore[reportArgumentType]
        )  # type: ignore

    def _run_validation(self):
        validation_study_name = f"{self.study_name}/validation"
        best_params = self._find_best_params()
        best_epoch = self._find_best_epoch()
        study = self._recreate_study()

        with mlflow.start_run(run_name=validation_study_name):
            with mlflow.start_run(run_name="retrain", nested=True) as retrain_run:
                study._retrain(
                    get_logger=study._get_logger_func(
                        retrain_run.info.run_id, mlflow.get_tracking_uri()
                    ),
                    best_params=best_params,
                    best_epoch=best_epoch,
                )

    @staticmethod
    def list_all_runs(parent_only=True):
        client = Experiment().client
        for experiment in client.search_experiments():
            runs = client.search_runs([experiment.experiment_id], max_results=5000)

            if parent_only:
                runs = [
                    run for run in runs if "mlflow.parentRunId" not in run.data.tags
                ]

            if len(runs) > 0:
                print(f"Experiment {experiment.name} runs:")
                for run in runs:
                    this_script_path = os.path.relpath(__file__)
                    print(
                        f"\tuv run {this_script_path} --experiment_name {experiment.name} --run_to_verify {run.data.tags['optuna_study']}"
                    )


class KFoldValidator:
    def __init__(self, k: int):
        pass


class SeedsValidator:
    def __init__(self, num_seeds: int):
        pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--run_to_verify", type=str)
    parser.add_argument(
        "--mode", type=str, choices=["kfold", "seeds", "list"], default="list"
    )
    parser.add_argument("-K", type=int, default=5)
    parser.add_argument("--num_seeds", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()

    match args.mode:
        case "list":
            StudyValidator.list_all_runs()
        case "kfold" | "seeds":
            validator = StudyValidator(args.experiment_name, args.run_to_verify)
            validator._run_validation()


if __name__ == "__main__":
    main()
