from dataclasses import dataclass, replace
from typing import Any, Literal, cast, override

import lightning as L
from dataset.hard_policy import HardPolicy

from experiments.experiment.datamodules.datamodule import DataModule
from experiments.experiment.datamodules.lpips_datamodule import (
    LPIPSDataModule,
    LPIPSDataModuleConfig,
)
from experiments.experiment.datamodules.nuclear_cataract_datamodule import (
    NuclearCataractDataModuleConfig,
)
from experiments.experiment.models.lpips import LPIPSModule, LPIPSModuleConfig
from experiments.experiment.studies.study import Study, StudyConfig

REAL_DATAMODULE_CONFIG = NuclearCataractDataModuleConfig(
    batch_size=64,
    return_paths=False,
    cache=True,
    hard_policy=HardPolicy.DOMINATE,
    image_size=224,
    normalize=False,
    augment_rot_angle=0,
)


@dataclass(frozen=True, kw_only=True)
class LPIPSStudyConfig(StudyConfig):
    experiment_name: str = "lpips"
    datamodule_config: LPIPSDataModuleConfig
    model_config: LPIPSModuleConfig = LPIPSModuleConfig()
    max_trials: int = 1
    optuna_metric: str = "lpips"
    optuna_direction: Literal["min", "max"] = "min"
    retrain_best: bool = False

    @override
    @staticmethod
    def get_configured_class():
        return LPIPSStudy


class LPIPSStudy(Study):
    _config: LPIPSStudyConfig

    def __init__(self, config: LPIPSStudyConfig):
        super().__init__(config)
        self._config = config

    @override
    def _init_datamodules(self) -> list[DataModule]:
        self._datamodule: LPIPSDataModule = self._config.datamodule_config.build()
        return [self._datamodule]

    @override
    def _configure_datamodules(self, params: dict[str, Any]) -> None:
        pass

    @override
    def _train(self, params: dict[str, Any], get_logger, callbacks) -> None:
        self._evaluate(get_logger, callbacks)

    @override
    def _retrain(
        self, get_logger, best_params: dict, best_epoch: int, callbacks: list
    ) -> tuple[L.Trainer, Any]:
        return self._evaluate(get_logger, callbacks)

    def _evaluate(self, get_logger, callbacks) -> tuple[L.Trainer, Any]:
        model_config = replace(
            self._config.model_config,
            n_classes=self._datamodule.n_classes,
        )
        model: LPIPSModule = model_config.build()
        trainer = L.Trainer(
            accelerator=self._config.device,
            logger=get_logger(model_config.class_name()),
            callbacks=callbacks,
            enable_progress_bar=True,
            enable_model_summary=False,
            enable_checkpointing=False,
            precision=cast(Any, self._config.gpu_precision),
            deterministic=True,
        )
        metrics = trainer.validate(model, datamodule=self._datamodule)
        return trainer, metrics
