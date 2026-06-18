from dataclasses import dataclass
from typing import Literal, override

from dataset.hard_policy import HardPolicy
import lightning as L
import mlflow
import optuna
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger

from experiments.experiment import common_config
from experiments.experiment.datamodules import init_datamodule
from experiments.experiment.datamodules.default_nuclear_cataract import NuclearCataractDataModuleConfig
from experiments.experiment.objective import (
    BackboneFreezeCallback,
    OptunaMLflowCallback,
)
from experiments.experiment.datamodules.datamodule import DataModule, DataModuleConfig
from experiments.experiment.models.convnext import ConvNext
from experiments.experiment.studies.study import Study, StudyConfig


@dataclass(frozen=True, kw_only=True)
class ConvNextStudyConfig(StudyConfig):
    name: str = "ConvnextStudy"
    max_epochs: int
    datamodule_config: DataModuleConfig
    early_stopping_patience: int
    backbone_unfreeze_patience: int
    use_early_stopping: bool
    use_freezing: bool

_OptunaParams = Literal["lr", "weight_decay", "dropout", "batch_size"]

class ConvNextStudy(Study):
    def __init__(
        self,
        config: ConvNextStudyConfig,
    ):
        super().__init__(config)
        self._config = config
        self._datamodule = init_datamodule(self._config.datamodule_config)

    def _suggest_params(self, trial: optuna.Trial) -> dict[_OptunaParams]:
        d: dict[_OptunaParams] = {
            "lr": trial.suggest_float("lr", 1e-6, 1e-4, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-8, 5e-2, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "batch_size": trial.suggest_categorical("batch_size", [64])
        }
        return d

    def _init_datamodules(self) -> list[DataModule]:
        self._datamodule = init_datamodule(self._config.datamodule_config)
        return [self._datamodule]

    def _configure_datamodules(self, params: dict[_OptunaParams]) -> None:
        # TODO: Is this best way to do that?
        # I would say it is better to manually create dataset and reuse it to create dataloaders.
        # In other words: just pass dataset to the datamodule class, so it will reuse caches.
        # self._datamodule.property = params["property"]
        self._datamodule.batch_size = params["batch_size"]
        pass

    def _train(
        self, params: dict, run: mlflow.ActiveRun, optuna_callback: OptunaMLflowCallback
    ) -> None:
        model = ConvNext(
            n_classes=self._datamodule.n_classes,
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            dropout=params["dropout"],
            class_weights=self._datamodule.train_class_weights,
        )

        callbacks: list[L.Callback] = [optuna_callback]

        mlf_logger = MLFlowLogger(
            run_id=run.info.run_id,
            tracking_uri=mlflow.get_tracking_uri(),
            prefix="ConvNext",
        )

        if self._config.use_early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=self._config.early_stopping_patience,
                    mode="min",
                )
            )

        if self._config.use_freezing:
            callbacks.append(
                BackboneFreezeCallback(
                    monitor="val_loss",
                    patience=self._config.backbone_unfreeze_patience,
                    mode="min",
                )
            )

        trainer = L.Trainer(
            max_epochs=self._config.max_epochs,
            accelerator="auto",
            logger=mlf_logger,
            callbacks=callbacks,
            enable_progress_bar=True,
            enable_model_summary=False,
            enable_checkpointing=False,
            log_every_n_steps=1,
            precision="bf16-mixed",
            deterministic=True,
        )

        trainer.fit(model, datamodule=self._datamodule)


    DEFAULT_CONFIG = ConvNextStudyConfig(
        experiment_name=common_config.EXPERIMENT_NAME,
        seed=common_config.SEED,
        max_trials=common_config.MAX_TRIALS,
        log_every_n_epochs=common_config.LOG_EVERY_N_EPOCHS,
        optuna_direction=common_config.OPTUNA_DIRECTION,
        optuna_metric=common_config.OPTUNA_METRIC,
        max_epochs=common_config.EPOCHS,
        datamodule_config=NuclearCataractDataModuleConfig(
            seed=common_config.SEED,
            batch_size=32,
            return_paths=False,
            cache=common_config.CACHE_SIZE is not None,
            hard_policy=HardPolicy.PASSTHROUGH,
            image_size=common_config.CACHE_SIZE,
            normalize=common_config.NORMALIZE,
            normalize_std=common_config.NORMALIZE_STD,
            normalize_mean=common_config.NORMALIZE_MEAN,
        ),
        early_stopping_patience=5,
        backbone_unfreeze_patience=5,
        use_early_stopping=True,
        use_freezing=False,
    )
