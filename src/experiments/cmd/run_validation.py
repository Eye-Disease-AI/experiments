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
from dataclasses import replace
from tempfile import TemporaryDirectory
from experiments.experiment.studies.study import KFoldValidatable, Study, StudyConfig
from experiments.lib.config_serializing import ClassConfig
from experiments.lib.mlflow_setup import Experiment
import mlflow
import mlflow.entities
import optuna
from typing_extensions import override


class StudyValidator:
    def __init__(
        self,
        experiment_name: str,
        study_name: str,
    ):
        self.experiment_name = experiment_name
        self.study_name = study_name
        self.experiment = Experiment(self.experiment_name)

    def _find_parent_run(self) -> mlflow.entities.Run | None:
        mlflow_exp = self.experiment.mlflow_experiment

        for run in self.experiment.client.search_runs(
            [mlflow_exp.experiment_id], max_results=9999
        ):
            if (
                "optuna_study" in run.data.tags
                and run.data.tags["optuna_study"] == self.study_name
                and "mlflow.parentRunId" not in run.data.tags
            ):
                return run

        return None

    def _get_study(self):
        return optuna.load_study(
            study_name=self.study_name, storage=self.experiment.storage
        )

    def _find_best_params(self):
        study = self._get_study()
        if study is None:
            return None
        return study.best_params

    def _recreate_study(self, seed: int | None = None):
        run = self._find_parent_run()
        assert run is not None

        with TemporaryDirectory() as tmp_dir:
            config_file_path = self.experiment.client.download_artifacts(
                run.info.run_id, Study.MLFLOW_STUDY_CONFIG_FILE_PATH, tmp_dir
            )
            with open(config_file_path, "r") as config_tmp_file:
                config_file_json = config_tmp_file.read()
                class_config = ClassConfig.from_json(config_file_json)
                if seed is not None:
                    assert isinstance(class_config, StudyConfig)
                    class_config = replace(class_config, seed=seed)
                return class_config.build()

    def _find_best_epoch(self):
        study = self._get_study()
        _agg = min if study.direction == "min" else max
        return _agg(
            study.best_trial.intermediate_values,
            key=study.best_trial.intermediate_values.get,  # pyright: ignore[reportArgumentType]
        )  # type: ignore

    def _post_retrain(
        self, _best_params, _best_epoch, _study: Study, _validation_study_name
    ):
        pass

    def _log_top_level(self, best_params, best_epoch, study):
        mlflow.set_tag("study_name", self.study_name)
        mlflow.set_tag("validator", self.__class__.__name__)
        mlflow.log_params(best_params)
        mlflow.log_param("seed", study.seed)
        mlflow.log_metric("best_epoch", best_epoch)

        optuna_study = self._get_study()
        if optuna_study is not None:
            best_trial_run_id = optuna_study.best_trial.user_attrs.get("mlflow_run_id")
            if best_trial_run_id:
                best_run = self.experiment.client.get_run(best_trial_run_id)
                for k, v in best_run.data.metrics.items():
                    if k.startswith("best_val_"):
                        mlflow.log_metric(k, v)

    def run(self):
        validation_study_name = f"{self.study_name}/validation"
        best_params = self._find_best_params()
        best_epoch = self._find_best_epoch()
        study = self._recreate_study()

        with mlflow.start_run(run_name=validation_study_name):
            self._log_top_level(best_params, best_epoch, study)
            with mlflow.start_run(run_name="retrain", nested=True) as retrain_run:
                study._retrain(
                    get_logger=study._get_logger_func(
                        retrain_run.info.run_id, mlflow.get_tracking_uri()
                    ),
                    best_params=best_params,
                    best_epoch=best_epoch,
                )
            self._post_retrain(best_params, best_epoch, study, validation_study_name)

    @staticmethod
    def list_all_runs(parent_only=True):
        client = Experiment().client
        for experiment in client.search_experiments():
            runs = client.search_runs([experiment.experiment_id], max_results=5000)
            runs = filter(lambda run: "optuna_study" in run.data.tags, runs)

            if parent_only:
                runs = filter(
                    lambda run: "mlflow.parentRunId" not in run.data.tags, runs
                )

            runs = list(runs)
            if len(runs) > 0:
                print(f"Experiment {experiment.name} runs:")
                for run in runs:
                    this_script_path = os.path.relpath(__file__)
                    print(
                        f"\tuv run {this_script_path} --experiment_name {experiment.name} --run_to_verify {run.data.tags['optuna_study']}"
                    )


class KFoldValidator(StudyValidator):
    def __init__(
        self,
        experiment_name: str,
        study_name: str,
        k: int,
    ):
        super().__init__(experiment_name, study_name)
        self.k = k

    def _log_top_level(self, best_params, best_epoch, study):
        super()._log_top_level(best_params, best_epoch, study)
        mlflow.log_param("K", self.k)

    @override
    def _post_retrain(
        self, best_params, best_epoch, study: Study, validation_study_name
    ):
        assert isinstance(study, KFoldValidatable)

        results = []

        for i in range(self.k):
            print(f"\n--- Fold {i + 1}/{self.k}")
            with mlflow.start_run(run_name=f"fold-{i}", nested=True) as child_run:
                mlflow.set_tag("fold", i)
                mlflow.set_tag("optuna_study", validation_study_name)
                mlflow.set_tag("validation_sample", "true")
                result = study.validate_fold(
                    fold=i,
                    num_folds=self.k,
                    best_params=best_params,
                    best_epoch=best_epoch,
                    get_logger=study._get_logger_func(
                        child_run.info.run_id, mlflow.get_tracking_uri()
                    ),
                )
                results.append(result)

        return results


class SeedsValidator(StudyValidator):
    def __init__(
        self,
        experiment_name: str,
        study_name: str,
        num_seeds: int,
    ):
        super().__init__(experiment_name, study_name)
        self.num_seeds = num_seeds

    def _log_top_level(self, best_params, best_epoch, study):
        super()._log_top_level(best_params, best_epoch, study)
        mlflow.log_param("n_seeds", self.num_seeds)

    @override
    def _post_retrain(
        self, best_params, best_epoch, _study: Study, validation_study_name
    ):
        results = []
        seeds = list(range(self.num_seeds))
        for i, s in enumerate(seeds):
            print(f"\n--- Seed {s} ({i + 1}/{len(seeds)}) ---")
            with mlflow.start_run(run_name=f"seed-{s}", nested=True) as seed_run:
                mlflow.set_tag("seed", s)
                mlflow.set_tag("optuna_study", validation_study_name)
                mlflow.set_tag("validation_sample", "true")
                study = self._recreate_study(s)
                _, validation_metrics = study._retrain(
                    get_logger=study._get_logger_func(
                        seed_run.info.run_id, mlflow.get_tracking_uri()
                    ),
                    best_params=best_params,
                    best_epoch=best_epoch,
                )
                results.append(validation_metrics)

        return results


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
        case "kfold":
            validator = KFoldValidator(args.experiment_name, args.run_to_verify, args.K)
            validator.run()
        case "seeds":
            validator = SeedsValidator(
                args.experiment_name, args.run_to_verify, args.num_seeds
            )
            validator.run()


if __name__ == "__main__":
    main()
