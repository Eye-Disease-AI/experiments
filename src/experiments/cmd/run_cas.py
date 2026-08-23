import argparse
from dataclasses import replace

from experiments.cmd.run_augmentations import CONDITIONS
from experiments.cmd.run_validation import SeedsValidator
from experiments.experiment.augmentors.common import create_augmentor
from experiments.experiment.studies.cas_study import CASStudyConfig

CONFIGS: dict[str, str] = {
    condition: augmentor
    for condition, (_, augmentor) in CONDITIONS.items()
    if augmentor is not None
}


def run_experiments(condition: str, num_seeds: int, n_augment: int):
    if condition == "all":
        conditions = list(CONFIGS)
    else:
        conditions = [condition]

    for c in conditions:
        study_config = CASStudyConfig(
            experiment_name="cas",
            study_suffix=c,
            datamodule_config=replace(
                CASStudyConfig.datamodule_config,
                cas_mode=True,
                augmentor_config=create_augmentor(CONFIGS[c]),
                n_augment=n_augment,
            ),
        )
        study = study_config.build()
        print(f"Running CAS study (condition {c})...")
        study.run()

        print(f"Validating condition {c} on {num_seeds} seeds...")
        SeedsValidator("cas", study.name, num_seeds).run()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition",
        type=str,
        choices=[*CONFIGS, "all"],
        default="all",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=5,
        help="Number of seeds each condition is repeated with (default: 5)",
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
    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be greater than zero")
    if args.n_augment <= 0:
        raise ValueError("--n-augment must be greater than zero")
    run_experiments(args.condition, args.num_seeds, args.n_augment)


if __name__ == "__main__":
    main()
