from dataset.hard_policy import HardPolicy
from experiments.experiment.data import (
    DataModuleEntry,
    MyDataModule,
    MyDataModuleConfig,
)
from experiments.experiment.studies.convnext_study import ConvNextStudyConfig
import experiments.experiment.common_config as common_config
from experiments.lib.reproducibility import RNG

DEFAULT_CONVNEXT_CONFIG = ConvNextStudyConfig(
    experiment_name=common_config.EXPERIMENT_NAME,
    seed=common_config.SEED,
    max_trials=common_config.MAX_TRIALS,
    log_every_n_epochs=common_config.LOG_EVERY_N_EPOCHS,
    optuna_direction=common_config.OPTUNA_DIRECTION,
    optuna_metric=common_config.OPTUNA_METRIC,
    max_epochs=common_config.EPOCHS,
    data_module=DataModuleEntry(
        data_module_type=MyDataModule,
        data_module_config=MyDataModuleConfig(
            rng=RNG(),
            batch_size=32,
            return_paths=False,
            cache=common_config.CACHE_SIZE is not None,
            hard_policy=HardPolicy.PASSTHROUGH,
            image_size=common_config.CACHE_SIZE,
            normalize=common_config.NORMALIZE,
            normalize_std=common_config.NORMALIZE_STD,
            normalize_mean=common_config.NORMALIZE_MEAN,
        ),
    ),
    early_stopping_patience=5,
    backbone_unfreeze_patience=5,
    use_early_stopping=True,
    use_freezing=False,
)
