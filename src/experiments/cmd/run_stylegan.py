import argparse
from dataclasses import replace

from experiments.experiment.augmentors.stylegan_augmentor import (
    StyleganAugmentorConfig,
)
from experiments.experiment.studies.baseline_study import BaselineStudyConfig


def main():
    args = parse_args()
    baseline_study_config = BaselineStudyConfig(
        experiment_name="baseline_with_stylegan"
    )
    experiments = []

    if args.variant == "r" or args.variant == "all":
        experiments.append(
            replace(
                baseline_study_config,
                study_suffix="R",
                datamodule_config=replace(
                    baseline_study_config.datamodule_config,
                    augmentor_config=StyleganAugmentorConfig.known_config_r(),
                    n_augment=1000,
                ),
            ),
        )

    if args.variant == "t" or args.variant == "all":
        experiments.append(
            replace(
                baseline_study_config,
                study_suffix="T",
                datamodule_config=replace(
                    baseline_study_config.datamodule_config,
                    augmentor_config=StyleganAugmentorConfig.known_config_t(),
                    n_augment=1000,
                ),
            ),
        )

    for config in experiments:
        augmented_study = config.build()
        print(f"Running stylegan study (variant {config.study_suffix})...")
        augmented_study.run()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, choices=["t", "r", "all"], default="all")
    return parser.parse_args()


if __name__ == "__main__":
    main()
