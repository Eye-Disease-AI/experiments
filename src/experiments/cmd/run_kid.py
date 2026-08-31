from experiments.cmd.run_augmentations import CONDITIONS
from experiments.experiment.augmentors.common import create_augmentor
from experiments.experiment.datamodules.generated_datamodule import (
    GeneratedDataModuleConfig,
)
from experiments.experiment.studies.kid_study import KIDStudyConfig
from experiments.experiment.studies.lpips_study import REAL_DATAMODULE_CONFIG

CONFIGS: dict[str, str] = {
    condition: augmentor
    for condition, (_, augmentor) in CONDITIONS.items()
    if augmentor is not None
}


def main():
    for cfg_name, augm_name in CONFIGS.items():
        fake_datamodule_config = GeneratedDataModuleConfig(
            augmentor_config=create_augmentor(augm_name),
            n_samples=3000,
        )
        cfg = KIDStudyConfig(
            study_suffix=cfg_name,
            fake_datamodule_config=fake_datamodule_config,
            real_datamodule_config=REAL_DATAMODULE_CONFIG,
        )
        sdy = cfg.build()
        sdy.run()


if __name__ == "__main__":
    main()
