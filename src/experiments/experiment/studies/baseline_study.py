from dataclasses import dataclass, replace
from typing import Literal, override

from dataset.hard_policy import HardPolicy
import lightning as L
import mlflow
import optuna
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger

from experiments.experiment.datamodules.nuclear_cataract_datamodule import (
    NuclearCataractDataModuleConfig,
)
from experiments.experiment.models.base_classifier import BaseClassifierModelConfig
from experiments.experiment.models.convnext import ConvNextConfig
from experiments.experiment.callbacks import (
    BackboneFreezeCallback,
    OptunaMLflowCallback,
)
from experiments.experiment.datamodules.datamodule import DataModule, DataModuleConfig
from experiments.experiment.studies.study import Study, StudyConfig


@dataclass(frozen=True, kw_only=True)
class BaselineStudyConfig(StudyConfig):
    max_epochs: int = 100
    datamodule_config: DataModuleConfig = (
        NuclearCataractDataModuleConfig(
            batch_size=64,
            return_paths=False,
            cache=True,
            hard_policy=HardPolicy.PASSTHROUGH,
            image_size=224,
            normalize=True,
            augment_rot_angle=15,
        ),
    )
    model_config: BaseClassifierModelConfig = ConvNextConfig()
    early_stopping_patience: int = 5
    backbone_unfreeze_mode: Literal["patience", "const_epochs"] = "const_epochs"
    backbone_unfreeze_num_epochs: int = 5
    use_early_stopping: bool = True
    use_freezing: bool = False

    @override
    @staticmethod
    def get_configured_class():
        return BaselineStudy


_OptunaParams = Literal["lr", "weight_decay", "dropout", "batch_size"]


class BaselineStudy(Study):
    def __init__(
        self,
        config: BaselineStudyConfig,
    ):
        super().__init__(config)
        self._config = config

    def _suggest_params(self, trial: optuna.Trial) -> dict[_OptunaParams]:
        d: dict[_OptunaParams] = {
            "lr": trial.suggest_float("lr", 1e-6, 1e-4, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-8, 5e-2, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "batch_size": trial.suggest_categorical("batch_size", [64]),
        }
        return d

    def _init_datamodules(self) -> list[DataModule]:
        self._datamodule = self._config.datamodule_config.build()
        return [self._datamodule]

    def _configure_datamodules(self, params: dict[_OptunaParams]) -> None:
        # TODO: Is this best way to do that?
        # I would say it is better to manually create dataset and reuse it to create dataloaders.
        # In other words: just pass dataset to the datamodule class, so it will reuse caches.
        self._datamodule.batch_size = params["batch_size"]

    def create_trainer(
        self, params: dict, run: mlflow.ActiveRun, optuna_callback: OptunaMLflowCallback
    ):
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
                    mode=self._config.backbone_unfreeze_mode,
                    direction="min",
                    num_epochs=self._config.backbone_unfreeze_num_epochs,
                )
            )

        return L.Trainer(
            max_epochs=self._config.max_epochs,
            accelerator="auto",
            logger=mlf_logger,
            callbacks=callbacks,
            enable_progress_bar=True,
            enable_model_summary=False,
            enable_checkpointing=False,
            log_every_n_steps=1,
            precision=self._config.gpu_precision,
            deterministic=True,
        )

    def _train(
        self, params: dict, run: mlflow.ActiveRun, optuna_callback: OptunaMLflowCallback
    ) -> None:
        model_config = replace(
            self._config.model_config,
            learning_rate=params["lr"],
            weight_decay=params["weight_decay"],
            dropout=params["dropout"],
        )
        model = model_config.build()
        trainer = self.create_trainer(params, run, optuna_callback)
        trainer.fit(model, datamodule=self._datamodule)
