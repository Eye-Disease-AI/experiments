from experiments.experiment.studies.study import StudyConfig
from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class CAMStudyConfig(StudyConfig):
    experiment_name: str = "cam"
    datamodule_config: DataModuleConfig
    max_trials: int = 1
    retrain_best: bool = False

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
        logger = get_logger("cam")

    @override
    def _retrain(
        self, get_logger, best_params: dict, best_epoch: int, callbacks: list
    ) -> tuple[L.Trainer, Any]:
        raise NotImplementedError
