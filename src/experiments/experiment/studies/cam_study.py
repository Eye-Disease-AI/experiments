from experiments.experiment.studies.study import Study, StudyConfig
from dataclasses import dataclass
from experiments.experiment.models.convnext import ConvNextConfig
from typing import override, Any
from collections.abc import Callable
from experiments.lib.reproducibility import Experiment
from pathlib import Path
import mlflow.artifacts
from tempfile import TemporaryDirectory
from experiments.experiment.datamodules.datamodule import DataModule, DataModuleConfig
from mlflow.entities import Run
from lightning.pytorch.loggers.mlflow import MLFlowLogger
import lightning as L


@dataclass(frozen=True, kw_only=True)
class CAMStudyConfig(StudyConfig):
    experiment_name: str = "cam"
    datamodule_config: DataModuleConfig
    model_config: ConvNextConfig
    # Experiment name and run name
    weights_source: tuple[str, str] | None
    max_trials: int = 1
    retrain_best: bool = False

    @staticmethod
    def from_classifier_study(
        experiment_name: str, run_name: str, with_weights: bool = True
    ):
        exp = Experiment(experiment_name=experiment_name, set_active=False)
        run = CAMStudyConfig.find_run(exp, run_name)
        snp = CAMStudyConfig.find_snapshot(exp, run)
        cfg = CAMStudyConfig.find_config(exp, run)

        if (not snp) or (not cfg):
            raise RuntimeError("Could not find either weights snapshot or study config")

        with TemporaryDirectory(prefix="camstudy") as temp_dir:
            cfg_file_path = mlflow.artifacts.download_artifacts(
                run_id=run.info.run_id,
                artifact_path=str(cfg),
                dst_path=temp_dir,
            )

            with open(cfg_file_path) as cfg_file:
                cfg_json = cfg_file.read()

            study_config = StudyConfig.from_json(cfg_json)

        model_config = study_config.model_config  # type: ignore
        datamodule_config = study_config.datamodule_config  # type: ignore

        if with_weights:
            weights_source = (experiment_name, run_name)
        else:
            weights_source = None

        return CAMStudyConfig(
            datamodule_config=datamodule_config,
            model_config=model_config,
            weights_source=weights_source,
        )

    @staticmethod
    def find_run(exp: Experiment, name: str):
        for run in exp.client.search_runs(
            [exp.mlflow_experiment.experiment_id], max_results=9999
        ):
            if (
                "optuna_study" in run.data.tags
                and run.data.tags["optuna_study"] == name
                and "mlflow.parentRunId" not in run.data.tags
            ):
                return run

    @staticmethod
    def find_snapshot(exp: Experiment, run: Run) -> Path | None:
        for art in exp.client.list_artifacts(run.info.run_id):
            if art.path == "model":
                return Path(art.path)

        return None

    @staticmethod
    def find_config(exp: Experiment, run: Run) -> Path | None:
        for art in exp.client.list_artifacts(run.info.run_id):
            if art.path == "study_config.json":
                return Path(art.path)

        return None

    @override
    @staticmethod
    def get_configured_class():
        return CAMStudy


class CAMStudy(Study):
    _config: CAMStudyConfig

    def __init__(self, config: CAMStudyConfig):
        super().__init__(config)
        self._config = config

    @override
    def _init_datamodules(self) -> list[DataModule]:
        self._datamodule = self._config.datamodule_config.build()
        return [self._datamodule]

    @override
    def _configure_datamodules(self, params: dict[str, Any]) -> None:
        pass

    @override
    def _train(
        self,
        params: dict[str, Any],
        get_logger: Callable[[str], MLFlowLogger],
        callbacks,
    ) -> None:
        # model_ckpt = self.load_model_ckpt(self._config.weights_source)
        # model = self.build_model(self._config.model_config, model_ckpt)
        # data_loader = self.get_data_loader(self._config.datamodule_config)
        # logger = get_logger("cam")
        # cams = self.generate_cams(model, data_loader)
        # self.log_cams(cams)
        pass

    @override
    def _retrain(
        self, get_logger, best_params: dict, best_epoch: int, callbacks: list
    ) -> tuple[L.Trainer, Any]:
        raise NotImplementedError
