from experiments.experiment.augmentors.tacgan_augmentor import TacganAugmentorConfig
from experiments.experiment.datamodules.nuclear_cataract_datamodule import (
    NuclearCataractDataModule,
    NuclearCataractDataModuleConfig,
)


def main():
    # To see augmented dataset stats
    nc = NuclearCataractDataModule(
        NuclearCataractDataModuleConfig(
            augmentor_config=TacganAugmentorConfig.known_config_tac1(),
            n_augment=100,
            cas_mode=True,
        ),
    )
    nc.setup("train")

    # To see non augmented dataset stats
    nc = NuclearCataractDataModule(NuclearCataractDataModuleConfig())
    nc.setup("train")


if __name__ == "__main__":
    main()
