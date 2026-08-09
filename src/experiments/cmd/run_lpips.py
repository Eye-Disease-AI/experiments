import argparse

from experiments.cmd.run_cas import create_augmentor
from experiments.experiment.studies.lpips_study import LPIPSStudyConfig


CONFIGS = [
    "stylegan-r",
    "stylegan-t",
    "tacgan-ac",
    "tacgan-tac1",
    "tacgan-tac2",
]


def run_experiments(config: str, n_samples: int, batch_size: int):
    configs = CONFIGS if config == "all" else [config]
    for current in configs:
        study_config = LPIPSStudyConfig(
            study_suffix=current,
            augmentor_config=create_augmentor(current),
            n_samples=n_samples,
            batch_size=batch_size,
        )
        print(f"Running LPIPS study (config {current})...")
        _, study = study_config.build().run()
        results = study.best_trial.user_attrs["metrics"]
        print(", ".join(f"{key}={value:.4f}" for key, value in results.items()))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=[*CONFIGS, "all"], required=True)
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of generated image pairs (default: 1000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="LPIPS inference batch size (default: 16)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_experiments(args.config, args.n_samples, args.batch_size)


if __name__ == "__main__":
    main()
