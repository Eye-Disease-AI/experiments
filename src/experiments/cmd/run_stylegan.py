import argparse
from dataclasses import replace

import matplotlib.pyplot as plt
import torch

from experiments.experiment.augmentors.augmentor import Augmentor
from experiments.experiment.augmentors.stylegan_augmentor import (
    StyleganAugmentorConfig,
)
from experiments.experiment.studies.baseline_study import BaselineStudyConfig


def run_experiments(variant: str):
    baseline_study_config = BaselineStudyConfig(
        experiment_name="baseline_with_stylegan"
    )
    experiments = []

    if variant == "r" or variant == "all":
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

    if variant == "t" or variant == "all":
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


def sample_augmentor(augmentor: Augmentor, variant: str):
    gens = augmentor.generate(torch.Tensor([0, 1]).to(dtype=torch.long))
    for i, gen in enumerate(gens):
        gen_perm = gen.permute(1, 2, 0)
        plt.imshow(gen_perm)
        plt.savefig(f"stylegan_{variant}_class{i}.png")


def sample(variant: str):
    if variant in ("t", "all"):
        t_config = StyleganAugmentorConfig.known_config_t()
        t_augmen: Augmentor = t_config.build()
        sample_augmentor(t_augmen, "t")

    if variant in ("r", "all"):
        r_config = StyleganAugmentorConfig.known_config_r()
        r_augmen: Augmentor = r_config.build()
        sample_augmentor(r_augmen, "r")


def main():
    args = parse_args()

    if args.sample:
        sample(args.variant)
    else:
        run_experiments(args.variant)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, choices=["t", "r", "all"], default="all")
    parser.add_argument("--sample", type=bool)
    return parser.parse_args()


if __name__ == "__main__":
    main()
