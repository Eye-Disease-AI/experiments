import argparse
from dataclasses import replace

from experiments.cmd.run_validation import SeedsValidator
from experiments.experiment.augmentors.common import create_augmentor
from experiments.experiment.studies.baseline_study import BaselineStudyConfig

CONDITIONS: dict[str, tuple[str, str | None]] = {
    "R": ("R", None),
    "RA": ("RA", None),
    "TACGAN": ("RAG", "tacgan-tac3"),
}


def run_experiments(condition: str, num_seeds: int, n_augment: int):
    if condition == "all":
        conditions = list(CONDITIONS)
    else:
        conditions = [condition]

    for c in conditions:
        augment_kind, augmentor = CONDITIONS[c]
        study_config = BaselineStudyConfig(
            experiment_name="augmentations",
            study_suffix=c,
            datamodule_config=replace(
                BaselineStudyConfig.datamodule_config,
                augment_kind=augment_kind,  # type: ignore[arg-type]
                augmentor_config=create_augmentor(augmentor) if augmentor else None,
                n_augment=n_augment if augmentor else None,
            ),
        )
        study = study_config.build()
        print(f"Running augmentations study (condition {c})...")
        study.run()

        print(f"Validating condition {c} on {num_seeds} seeds...")
        SeedsValidator("augmentations", study.name, num_seeds).run()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition",
        type=str,
        choices=[*CONDITIONS, "all"],
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
    if args.condition in ["all", "RAG"] and args.n_augment <= 0:
        raise ValueError("--n-augment must be greater than zero")
    run_experiments(args.condition, args.num_seeds, args.n_augment)


if __name__ == "__main__":
    main()
