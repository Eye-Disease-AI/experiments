import argparse
from typing import Literal

from experiments.experiment.augmentors.common import create_augmentor
from experiments.experiment.datamodules.datamodule import DataModuleConfig
from experiments.experiment.datamodules.generated_datamodule import (
    GeneratedDataModuleConfig,
)
from experiments.experiment.studies.lpips_study import (
    REAL_DATAMODULE_CONFIG,
    LPIPSStudyConfig,
)

CONFIGS = [
    "stylegan-r",
    "stylegan-t",
    "tacgan-ac",
    "tacgan-tac1",
    "tacgan-tac2",
]


def run_study(
    name: str,
    first: DataModuleConfig,
    second: DataModuleConfig,
    direction: Literal["min", "max"],
    n_samples: int,
    batch_size: int,
):
    config = LPIPSStudyConfig(
        study_suffix=name,
        first_datamodule_config=first,
        second_datamodule_config=second,
        n_samples=n_samples,
        batch_size=batch_size,
        optuna_direction=direction,
    )
    print(f"Running LPIPS study ({name})...")
    _, study = config.build().run()
    print(f"lpips={study.best_value:.4f}")


def run_experiments(config: str, n_samples: int, batch_size: int):
    configs = CONFIGS if config == "all" else [config]
    for current in configs:
        fake = GeneratedDataModuleConfig(
            augmentor_config=create_augmentor(current),
            n_samples=n_samples,
        )
        run_study(f"{current}-fake-fake", fake, fake, "max", n_samples, batch_size)
        run_study(
            f"{current}-fake-real",
            fake,
            REAL_DATAMODULE_CONFIG,
            "min",
            n_samples,
            batch_size,
        )
    run_study(
        "real-real",
        REAL_DATAMODULE_CONFIG,
        REAL_DATAMODULE_CONFIG,
        "max",
        n_samples,
        batch_size,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=[*CONFIGS, "all"], required=True)
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of generated image pairs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="LPIPS inference batch size",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_experiments(args.config, args.n_samples, args.batch_size)


if __name__ == "__main__":
    main()
