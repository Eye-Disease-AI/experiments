import argparse
from dataclasses import replace

from experiments.experiment.augmentors.augmentor import AugmentorConfig
from experiments.experiment.augmentors.stylegan_augmentor import (
    StyleganAugmentorConfig,
)
from experiments.experiment.augmentors.tacgan_augmentor import (
    TacganAugmentorConfig,
)
from experiments.experiment.studies.cas_study import CASStudyConfig


def create_augmentor(config: str) -> AugmentorConfig:
    if config == "stylegan-r":
        return StyleganAugmentorConfig.known_config_r()
    elif config == "stylegan-t":
        return StyleganAugmentorConfig.known_config_t()
    elif config == "tacgan-ac":
        return TacganAugmentorConfig.known_config_ac()
    elif config == "tacgan-tac1":
        return TacganAugmentorConfig.known_config_tac1()
    elif config == "tacgan-tac2":
        return TacganAugmentorConfig.known_config_tac2()
    else:
        raise RuntimeError("unknown config")


def run_experiments(config: str, n_augment: int):
    if config == "all":
        configs = [
            "stylegan-r",
            "stylegan-t",
            "tacgan-ac",
            "tacgan-tac1",
            "tacgan-tac2",
        ]
    else:
        configs = [config]

    for c in configs:
        study_config = CASStudyConfig(
            experiment_name="cas",
            study_suffix=c,
            datamodule_config=replace(
                CASStudyConfig.datamodule_config,
                cas_mode=True,
                augmentor_config=create_augmentor(c),
                n_augment=n_augment,
            ),
        )
        print(f"Running CAS study (config {c})...")
        study_config.build().run()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        choices=[
            "stylegan-r",
            "stylegan-t",
            "tacgan-ac",
            "tacgan-tac1",
            "tacgan-tac2",
            "all",
        ],
        required=True,
    )
    parser.add_argument(
        "--n-augment",
        type=int,
        default=1000,
        help="Number of generated training samples (default: 1000)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n_augment <= 0:
        raise ValueError("--n-augment must be greater than zero")
    run_experiments(args.config, args.n_augment)


if __name__ == "__main__":
    main()
