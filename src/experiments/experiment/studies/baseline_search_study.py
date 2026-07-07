from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal, override

from dataset.hard_policy import HardPolicy
import lightning as L
import mlflow
from lightning.pytorch.callbacks import EarlyStopping

from experiments.experiment.datamodules.nuclear_cataract_datamodule import (
    NuclearCataractDataModule,
    NuclearCataractDataModuleConfig,
)
from experiments.experiment.models.convnext import ConvNextConfig
from experiments.experiment.callbacks import (
    BackboneFreezeCallback,
)
from experiments.experiment.datamodules.datamodule import DataModule, DataModuleConfig
from experiments.experiment.studies.study import KFoldValidatable, Study, StudyConfig
from experiments.lib.config_serializing import OptunaOptimised
from experiments.lib.reproducibility import global_seed_rng


@dataclass(frozen=True, kw_only=True)
class BaselineSearchStudyConfig(StudyConfig):
    max_epochs: int = 100
    datamodule_config: DataModuleConfig = NuclearCataractDataModuleConfig(
        batch_size=OptunaOptimised("categorical", {"choices": [64]}),
        return_paths=False,
        cache=True,
        hard_policy=HardPolicy.PASSTHROUGH,
        image_size=224,
        normalize=True,
        augment_rot_angle=15,
    )
    model_config: ConvNextConfig = ConvNextConfig(
        learning_rate=OptunaOptimised(
            "float", {"low": 1e-6, "high": 1e-4, "log": True}
        ),
        weight_decay=OptunaOptimised("float", {"low": 1e-8, "high": 5e-2, "log": True}),
        dropout=OptunaOptimised("float", {"low": 0.0, "high": 0.5}),
    )
    early_stopping_patience: int = 5
    backbone_unfreeze_mode: Literal["patience", "const_epochs"] = "const_epochs"
    backbone_unfreeze_num_epochs: int = 5
    use_early_stopping: bool = True
    use_freezing: bool = False
    use_class_weights = True

    @override
    @staticmethod
    def get_configured_class():
        return BaselineSearchStudy


class BaselineSearchStudy(Study, KFoldValidatable):
    _config: BaselineSearchStudyConfig

    def __init__(
        self,
        config: BaselineSearchStudyConfig,
    ):
        super().__init__(config)
        self._config = config

    @override
    def _init_datamodules(self) -> list[DataModule]:
        self._datamodule: NuclearCataractDataModule = (
            self._config.datamodule_config.build()
        )
        return [self._datamodule]

    @override
    def _configure_datamodules(self, params: dict[str, Any]) -> None:
        # TODO: Is this best way to do that?
        # I would say it is better to manually create dataset and reuse it to create dataloaders.
        # In other words: just pass dataset to the datamodule class, so it will reuse caches.
        config = self._set_config_optuna_params(params)
        self._datamodule.batch_size = config.datamodule_config.batch_size

    def _bake_model_config(self, config: BaselineSearchStudyConfig) -> ConvNextConfig:
        # Inject datamodule-derived fields (only known at runtime) into the
        # resolved model config before building.
        return replace(
            config.model_config,
            class_weights=self._datamodule.class_weights,
            n_classes=self._datamodule.n_classes,
        )

    def create_trainer(
        self,
        logger,
        max_epochs: int | None = None,
        callbacks: list[L.Callback] | None = None,
    ):
        callbacks = list(callbacks or [])
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
            max_epochs=max_epochs or self._config.max_epochs,
            accelerator=self._config.device,
            logger=logger,
            callbacks=callbacks,
            enable_progress_bar=True,
            enable_model_summary=False,
            enable_checkpointing=False,
            log_every_n_steps=1,
            precision=self._config.gpu_precision,  # type: ignore
            deterministic=True,
        )

    @override
    def _train(
        self, params: dict[str, Any], get_logger, callbacks: list | None = None
    ) -> None:
        config = self._set_config_optuna_params(params)
        model = self._bake_model_config(config).build()
        trainer = self.create_trainer(
            logger=get_logger(self._config.model_config.class_name()),
            callbacks=callbacks,
        )
        global_seed_rng(self._config.seed)
        trainer.fit(model, datamodule=self._datamodule)

    @override
    def _retrain(
        self,
        get_logger,
        best_params: dict,
        best_epoch: int,
        callbacks: list | None = None,
    ) -> tuple[L.Trainer, Any]:
        global_seed_rng(self._config.seed)
        self._init_datamodules()
        config = self._set_config_optuna_params(best_params)
        self._datamodule.batch_size = config.datamodule_config.batch_size
        self._datamodule.setup(stage="fit")
        best_model = self._bake_model_config(config).build()
        trainer = self.create_trainer(
            logger=get_logger(self._config.model_config.class_name()),
            # We train for best_epoch+1, because best_epoch is 0-indexed.
            # e.g. if we want to train up to epoch 2, we need to train for 3 epochs (0, 1, 2).
            max_epochs=best_epoch + 1,
            callbacks=callbacks,
        )
        global_seed_rng(self._config.seed)
        trainer.fit(best_model, datamodule=self._datamodule)
        validation_metrics = trainer.validate(best_model, datamodule=self._datamodule)
        retrain_metric = validation_metrics[0][self._config.optuna_metric]
        mlflow.log_metric(f"retrain_{self._config.optuna_metric}", retrain_metric)
        return trainer, validation_metrics

    def validate_fold(
        self,
        *,
        fold: int,
        num_folds: int,
        best_params: dict,
        best_epoch: int,
        get_logger,
        callbacks: list[L.Callback] | None = None,
    ) -> Mapping[str, float]:
        global_seed_rng(self._config.seed)
        self._init_datamodules()

        config = self._set_config_optuna_params(best_params)
        self._datamodule.batch_size = config.datamodule_config.batch_size
        self._datamodule.setup(stage="fit")
        self._datamodule.setup_fold(fold, num_folds)

        model = self._bake_model_config(config).build()
        trainer = self.create_trainer(
            logger=get_logger(self._config.model_config.class_name()),
            max_epochs=best_epoch + 1,
            callbacks=([] if callbacks is None else callbacks),
        )
        global_seed_rng(self._config.seed)
        trainer.fit(model, datamodule=self._datamodule)
        validation_metrics = trainer.validate(model, datamodule=self._datamodule)
        return validation_metrics[0]
