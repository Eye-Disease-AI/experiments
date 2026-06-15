from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass

from experiments.experiment.data import DataModule
import mlflow
import optuna
from optuna import create_study
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

from experiments.experiment.objective import OptunaMLflowCallback
from experiments.lib.reproducibility import get_git_sha
from experiments.lib.reproducibility import RNG
from experiments.lib.mlflow_setup import Experiment


class BaseStudy(ABC):
    @abstractmethod
    def __init__(
        self,
        config: BaseStudyConfig,
    ) -> None:
        self._config = config
        self.name = f"{config.experiment_name}_{get_git_sha()}"
        self._rng = RNG()

    @abstractmethod
    def _suggest_params(self, trial: optuna.Trial) -> dict[str, Any]: ...

    @abstractmethod
    def _configure_data_modules(self, params: dict[str, Any]) -> None: ...

    @abstractmethod
    def _train(
        self,
        params: dict[str, Any],
        run: mlflow.ActiveRun,
        optuna_callback: OptunaMLflowCallback,
    ) -> None: ...

    def run(self) -> None:
        self._prepare_data_modules()
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
        self._rng.set_seed(self._config.seed)

        params = self._suggest_params(trial)
        self._configure_data_modules(params)
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

    # This method should set all needed data modules as class fields and return list of them.
    @abstractmethod
    def _init_data_modules(self) -> list[DataModule]: ...

    def _prepare_data_modules(self):
        self._rng.set_seed(self._config.seed)
        data_modules = self._init_data_modules()

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

        if n_finished >= max_trials:
            print(
                f"Study already complete. Best: {optuna_study.best_params}, {self._config.optuna_metric}: {optuna_study.best_value:.4f}"
            )
            mlflow.end_run()
            return optuna_study, parent_run, True

        return optuna_study, parent_run, False


@dataclass
class BaseStudyConfig(ABC):
    experiment_name: str
    seed: int
    max_trials: int
    log_every_n_epochs: int
    optuna_direction: str
    optuna_metric: str
