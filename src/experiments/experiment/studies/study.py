from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
import os
import tempfile
from typing import Any, Literal, Protocol, override, runtime_checkable
import lightning as L
from matplotlib import pyplot as plt
from optuna.visualization import plot_optimization_history, plot_param_importances
from experiments.experiment.datamodules.datamodule import DataModule
import mlflow
import optuna
from optuna import create_study
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

from experiments.experiment.callbacks import BestSnapshotCallback, OptunaMLflowCallback
from lightning.pytorch.loggers import MLFlowLogger
from experiments.lib.log_silencer import stop_logs
from experiments.lib.reproducibility import get_git_sha, global_seed_rng
from experiments.lib.mlflow_setup import Experiment
from experiments.lib.config_serializing import ClassConfig, OptunaOptimised


@dataclass(frozen=True, kw_only=True)
class StudyConfig(ClassConfig):
    experiment_name: str | None = None
    seed: int = 2137
    max_trials: int = 100
    log_every_n_epochs: int = 1
    optuna_metric: str = "val_auroc"
    optuna_direction: Literal["min", "max"] = "max"
    gpu_precision: str = "bf16-mixed"
    retrain_best: bool = True
    checkpoint_dir: str = "checkpoints"
    device: Literal["auto", "gpu", "cpu"] = "auto"

    @override
    def post_init_checks(self):
        if not self.experiment_name:
            raise Exception(f"experiment_name is {self.experiment_name}")


class Study(ABC):
    MLFLOW_STUDY_CONFIG_FILE_PATH: str = "study_config.json"

    def __init__(
        self,
        config: StudyConfig,
    ) -> None:
        self._config = config
        self.name = f"{config.experiment_name}_{get_git_sha()}"
        stop_logs()

    @property
    def seed(self) -> int:
        return self._config.seed

    def _suggest_params(
        self, trial: optuna.Trial, config: Any = None, prefix: str = ""
    ) -> dict[str, Any]:
        """
        Suggests a value for every OptunaOptimised config field and returns in dict.
        `config` param allows for it to work recursively in subconfigs.
        """
        config = self._config if config is None else config
        out: dict[str, Any] = {}
        for f in fields(config):
            value = getattr(config, f.name)
            path = f"{prefix}{f.name}"
            if isinstance(value, OptunaOptimised):
                suggest = getattr(trial, f"suggest_{value.kind}")
                out[path] = suggest(path, **value.kwargs)
            elif is_dataclass(value) and not isinstance(value, type):
                out.update(self._suggest_params(trial, config=value, prefix=f"{path}."))
        return out

    def _set_config_optuna_params(
        self,
        optuna_values: dict[str, Any],
        config: Any = None,
        prefix: str = "",
    ):
        """Return a copy of the config with every OptunaOptimised field
        replaced by a value from `optuna_values` parameter.
        `config` param allows for it to work recursively in subconfigs.
        """
        config = self._config if config is None else config
        overrides: dict[str, Any] = {}
        for f in fields(config):
            value = getattr(config, f.name)
            path = f"{prefix}{f.name}"
            if isinstance(value, OptunaOptimised):
                overrides[f.name] = optuna_values[path]
            elif is_dataclass(value) and not isinstance(value, type):
                resolved = self._set_config_optuna_params(
                    optuna_values, value, f"{path}."
                )
                if resolved is not value:
                    overrides[f.name] = resolved
        return replace(config, **overrides) if overrides else config

    # This method should set all needed data modules as class fields and return list of them.
    @abstractmethod
    def _init_datamodules(self) -> list[DataModule]: ...

    @abstractmethod
    def _configure_datamodules(self, params: dict[str, Any]) -> None: ...

    @abstractmethod
    def _train(
        self,
        params: dict[str, Any],
        get_logger,
        callbacks,
    ) -> None: ...

    @abstractmethod
    def _retrain(
        self,
        get_logger,
        best_params: dict,
        best_epoch: int,
        callbacks: list,
    ) -> tuple[L.Trainer, Any]: ...

    def run(self) -> tuple[Experiment, optuna.Study]:
        self._prepare_datamodules()
        mlflow_experiment, optuna_study, parent_run, already_complete = (
            self._setup_optuna_study(max_trials=self._config.max_trials)
        )
        if already_complete:
            return mlflow_experiment, optuna_study
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                config_path = os.path.join(tmp_dir, Study.MLFLOW_STUDY_CONFIG_FILE_PATH)
                with open(config_path, "w") as study_config:
                    study_config.write(self._config.to_json(True))
                    study_config.flush()
                    mlflow.log_artifact(config_path)

            optuna_study.optimize(
                self._objective,
                n_trials=self._config.max_trials * 2,
                callbacks=[
                    MaxTrialsCallback(
                        self._config.max_trials,
                        states=(TrialState.COMPLETE, TrialState.PRUNED),
                    )
                ],
            )
            if not any(
                t for t in optuna_study.trials if t.state == TrialState.COMPLETE
            ):
                print("No completed trials. Skipping retrain and plots.")
                return mlflow_experiment, optuna_study

            best_trial, best_params, best_epoch = self._parse_optuna_study(optuna_study)

            if self._config.retrain_best:
                print("\n#### Retraining\n")
                best_snapshot = BestSnapshotCallback(
                    metric=self._config.optuna_metric,
                    direction=self._config.optuna_direction,
                    prefix="retrain_",
                )
                trainer, _metrics = self._retrain(
                    get_logger=self._get_logger_func(
                        parent_run.info.run_id, mlflow.get_tracking_uri()
                    ),
                    best_params=best_params,
                    best_epoch=best_epoch,
                    callbacks=[best_snapshot],
                )
                best_model_path = (
                    f"checkpoints/{optuna_study.study_name}/best_retrain.ckpt"
                )
                trainer.save_checkpoint(best_model_path)
                mlflow.log_artifact(best_model_path, artifact_path="model")

        finally:
            mlflow.end_run()
        return mlflow_experiment, optuna_study

    def _parse_optuna_study(self, optuna_study: optuna.Study):
        best_trial = optuna_study.best_trial
        best_params = optuna_study.best_params

        mlflow.log_params(best_trial.params)
        mlflow.log_metric(f"best_{self._config.optuna_metric}", best_trial.value)
        mlflow.set_tag("best_trial_number", best_trial.number)
        agg = min if self._config.optuna_direction == "min" else max
        best_epoch = agg(
            best_trial.intermediate_values,
            key=best_trial.intermediate_values.get,
        )
        mlflow.log_metric("best_epoch", best_epoch)
        try:
            figure = plot_optimization_history(optuna_study)
            mlflow.log_figure(figure, "optimization_history.png")
            plt.close()
        except Exception as e:
            print("WARN failed to log optimization_history")
            print(e)

        try:
            figure = plot_param_importances(optuna_study)
            mlflow.log_figure(figure, "param_importances.png")
            plt.close()
        except Exception as e:
            print("WARN failed to log param_importances")
            print(e)

        return best_trial, best_params, best_epoch

    def _objective(self, trial: optuna.Trial) -> float:
        global_seed_rng(self._config.seed)
        params = self._suggest_params(trial)
        self._configure_datamodules(params)
        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True) as run:
            trial.set_user_attr("mlflow_run_id", run.info.run_id)
            mlflow.set_tag("optuna_study", self.name)
            mlflow.set_tag("optuna_trial", trial.number)
            mlflow.log_params({**params, "seed": self._config.seed})

            optuna_callback = OptunaMLflowCallback(
                trial=trial,
                exp=Experiment(self._config.experiment_name),
                run_id=run.info.run_id,
                log_every_n_epochs=self._config.log_every_n_epochs,
                optuna_metric=self._config.optuna_metric,
                optuna_direction=self._config.optuna_direction,
            )
            best_snapshot = BestSnapshotCallback(
                metric=self._config.optuna_metric,
                direction=self._config.optuna_direction,
                prefix="best_",
            )
            self._train(
                params,
                get_logger=self._get_logger_func(
                    run.info.run_id, mlflow.get_tracking_uri()
                ),
                callbacks=[optuna_callback, best_snapshot],
            )

        best_value = optuna_callback.best_value
        if best_value is None:
            return (
                float("-inf")
                if self._config.optuna_direction == "max"
                else float("inf")
            )

        return best_value

    def _get_logger_func(self, run_id, tracking_uri):
        return lambda prefix: MLFlowLogger(
            run_id=run_id,
            tracking_uri=tracking_uri,
            prefix=prefix,
        )

    def _prepare_datamodules(self):
        global_seed_rng(self._config.seed)
        data_modules = self._init_datamodules()

        for data_module in data_modules:
            data_module.prepare_data()
            data_module.setup("fit")

    def _setup_optuna_study(
        self, max_trials: int, study_name: str | None = None
    ) -> tuple[Experiment, optuna.study.Study, mlflow.ActiveRun, bool]:
        """Setup and return Optuna study with MLflow integration."""
        mlflow_experiment = Experiment(self._config.experiment_name)
        actual_study_name: str = study_name or self.name

        optuna_study = create_study(
            study_name=actual_study_name,
            direction={"min": "minimize", "max": "maximize"}[
                self._config.optuna_direction
            ],
            storage=mlflow_experiment.storage,
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        )

        n_finished = sum(
            1
            for t in optuna_study.trials
            if t.state in (TrialState.COMPLETE, TrialState.PRUNED)
        )

        print(
            f"Study '{actual_study_name}': {n_finished}/{max_trials} finished trials."
        )

        parent_run_id = optuna_study.user_attrs.get("mlflow_parent_run_id")
        if parent_run_id:
            parent_run = mlflow.start_run(run_id=parent_run_id)
        else:
            parent_run = mlflow.start_run(run_name=actual_study_name)
            optuna_study.set_user_attr("mlflow_parent_run_id", parent_run.info.run_id)
            mlflow.set_tag("optuna_study", actual_study_name)
            mlflow.log_params(self._config.serialize_config())

        if n_finished >= max_trials:
            print(
                f"Study already complete. Best: {optuna_study.best_params}, {self._config.optuna_metric}: {optuna_study.best_value:.4f}"
            )
            mlflow.end_run()
            return mlflow_experiment, optuna_study, parent_run, True

        return mlflow_experiment, optuna_study, parent_run, False


@runtime_checkable
class KFoldValidatable(Protocol):
    def validate_fold(
        self,
        *,
        fold: int,
        num_folds: int,
        best_params: dict,
        best_epoch: int,
        get_logger,
    ) -> Mapping[str, float]: ...
