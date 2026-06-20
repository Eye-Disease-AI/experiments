from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, override

from experiments.experiment.datamodules.datamodule import DataModule
import mlflow
import optuna
from optuna import create_study
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

from experiments.experiment.objective import OptunaMLflowCallback
from experiments.lib.reproducibility import get_git_sha, global_seed_rng
from experiments.lib.mlflow_setup import Experiment
from experiments.lib.config_serializing import ClassConfig

@dataclass(frozen=True, kw_only=True)
class StudyConfig(ClassConfig):
    experiment_name: str | None = None
    seed: int = 2137
    max_trials: int = 100
    log_every_n_epochs: int = 1
    optuna_metric: str = "val_auroc"
    optuna_direction: str = "max"
    gpu_precision: str = "bf16-mixed"

    @override
    def post_init_checks(self):
        if not self.experiment_name:
            raise Exception(f"experiment_name is {self.experiment_name}")


class Study(ABC):
    DEFAULT_CONFIG: StudyConfig

    def __init__(
        self,
        config: StudyConfig,
    ) -> None:
        self._config = config
        self.name = f"{config.experiment_name}_{get_git_sha()}"

    @abstractmethod
    def _suggest_params(self, trial: optuna.Trial) -> dict[str, Any]: ...

    # This method should set all needed data modules as class fields and return list of them.
    @abstractmethod
    def _init_datamodules(self) -> list[DataModule]: ...

    @abstractmethod
    def _configure_datamodules(self, params: dict[str, Any]) -> None: ...

    @abstractmethod
    def _train(
        self,
        params: dict[str, Any],
        run: mlflow.ActiveRun,
        optuna_callback: OptunaMLflowCallback,
    ) -> None: ...

    def run(self) -> None:
        self._prepare_datamodules()
        optuna_study, _, already_complete = self._setup_optuna_study(
            max_trials=self._config.max_trials
        )

        if already_complete:
            return
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
        mlflow.end_run()

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
            self._train(params, run, optuna_callback)

        best_value = optuna_callback.best_value
        if best_value is None:
            return (
                float("-inf")
                if self._config.optuna_direction == "max"
                else float("inf")
            )

        return best_value

    def _prepare_datamodules(self):
        global_seed_rng(self._config.seed)
        data_modules = self._init_datamodules()

        for data_module in data_modules:
            data_module.prepare_data()
            data_module.setup("fit")

    def _setup_optuna_study(
        self, max_trials: int, study_name: str | None = None
    ) -> tuple[optuna.study.Study, mlflow.ActiveRun, bool]:
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
            return optuna_study, parent_run, True

        return optuna_study, parent_run, False
