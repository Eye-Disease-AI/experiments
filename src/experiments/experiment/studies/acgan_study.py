from dataclasses import dataclass, replace
from typing import Any

import lightning as L
from dataset.hard_policy import HardPolicy
from typing_extensions import override

from experiments.experiment.datamodules.datamodule import DataModule, DataModuleConfig
from experiments.experiment.datamodules.nuclear_cataract_datamodule import (
    NuclearCataractDataModule,
    NuclearCataractDataModuleConfig,
)
from experiments.experiment.models.acgan import ACGANModuleConfig
from experiments.experiment.models.convnext import ConvNextConfig
from experiments.experiment.studies.study import Study, StudyConfig
from experiments.lib.config_serializing import OptunaOptimised
from experiments.lib.reproducibility import global_seed_rng


@dataclass(frozen=True, kw_only=True)
class ACGANStudyConfig(StudyConfig):
    datamodule_config: DataModuleConfig = NuclearCataractDataModuleConfig(
        batch_size=8,
        return_paths=False,
        cache=True,
        hard_policy=HardPolicy.PASSTHROUGH,
        image_size=224,
        normalize=True,
        augment_rot_angle=15,
    )
    clf_model_config: ConvNextConfig = ConvNextConfig(
        learning_rate=1e-4,
        weight_decay=1e-8,
        dropout=0.3,
    )
    gen_model_config: ACGANModuleConfig = ACGANModuleConfig(
        # I think we should support some way of saying a kw is undefined!
        n_classes=-1,
        latent_dim=OptunaOptimised("int", {"low": 50, "high": 200, "log": True}),
        img_size=224,
        num_channels=3,
    )
    max_clf_epochs: int = 100
    max_gen_epochs: int = 100

    @override
    @staticmethod
    def get_configured_class():
        return ACGANStudy


class ACGANStudy(Study):
    _config: ACGANStudyConfig

    def __init__(self, config: ACGANStudyConfig):
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
        # Nothing to configure
        pass

    def _bake_clf_model_config(self, config: ACGANStudyConfig) -> ConvNextConfig:
        return replace(
            config.clf_model_config,
            class_weights=self._datamodule.class_weights,
            n_classes=self._datamodule.n_classes,
        )

    def _bake_gen_model_config(self, config: ACGANStudyConfig) -> ACGANModuleConfig:
        return replace(
            config.gen_model_config,
            n_classes=self._datamodule.n_classes,
        )

    def _create_clf_trainer(
        self,
        logger,
    ):
        return L.Trainer(
            max_epochs=self._config.max_clf_epochs,
            accelerator=self._config.device,
            logger=logger,
            enable_progress_bar=True,
            enable_model_summary=False,
            enable_checkpointing=False,
            log_every_n_steps=1,
            precision=self._config.gpu_precision,  # type: ignore
            deterministic=True,
        )

    def _create_gen_trainer(
        self,
        logger,
    ):
        return L.Trainer(
            max_epochs=self._config.max_gen_epochs,
            accelerator=self._config.device,
            logger=logger,
            enable_progress_bar=True,
            enable_model_summary=False,
            enable_checkpointing=False,
            log_every_n_steps=1,
            precision=self._config.gpu_precision,  # type: ignore
            deterministic=True,
        )

    @override
    def _train(self, params: dict[str, Any], get_logger, callbacks) -> None:
        config: ACGANStudyConfig = self._set_config_optuna_params(params)
        gen_model = self._bake_gen_model_config(config).build()
        gen_trainer = self._create_gen_trainer(
            logger=get_logger(self._config.gen_model_config.class_name())
        )

        global_seed_rng(self._config.seed)
        gen_trainer.fit(gen_model, datamodule=self._datamodule)

    @override
    def _retrain(
        self, get_logger, best_params: dict, best_epoch: int, callbacks: list
    ) -> tuple[L.Trainer, Any]:
        raise NotImplementedError
