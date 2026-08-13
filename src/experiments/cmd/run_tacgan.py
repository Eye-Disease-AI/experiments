import argparse
from dataclasses import replace

from experiments.experiment.augmentors.common import sample_augmentor
from experiments.experiment.augmentors.tacgan_augmentor import (
    TacganAugmentor,
    TacganAugmentorConfig,
)
from experiments.experiment.studies.baseline_study import BaselineStudyConfig


def upload(config: str):
    if config in ["tac1", "all"]:
        TacganAugmentor.upload_results_to_mlflow(
            "Twin_AC_AC_weight1.0_BigGAN_NuclearCataract_seed2018_Gch64_Dch64_bs2_nDs2_Glr2.0e-04_Dlr2.0e-04_Gnlrelu_Dnlrelu_GinitN02_DinitN02_ema",
        )

    if config in ["tac2", "all"]:
        TacganAugmentor.upload_results_to_mlflow(
            "Twin_AC_AC_weight2.0_BigGAN_NuclearCataract_seed2018_Gch64_Dch64_bs16_nDs2_Glr2.0e-04_Dlr2.0e-04_Gnlrelu_Dnlrelu_GinitN02_DinitN02_ema",
        )

    if config in ["ac", "all"]:
        TacganAugmentor.upload_results_to_mlflow(
            "AC_AC_weight1.0_BigGAN_NuclearCataract_seed2018_Gch64_Dch64_bs2_nDs2_Glr2.0e-04_Dlr2.0e-04_Gnlrelu_Dnlrelu_GinitN02_DinitN02_ema",
        )

    if config in ["tac3", "all"]:
        TacganAugmentor.upload_results_to_mlflow(
            "Twin_AC_AC_weight2.0_BigGAN_NuclearCataractDominate_seed2018_Gch64_Dch64_bs16_nDs2_Glr2.0e-04_Dlr2.0e-04_Gnlrelu_Dnlrelu_GinitN02_DinitN02_ema",
        )


def create_augmentor(config: str):
    if config == "ac":
        return TacganAugmentorConfig.known_config_ac().build()
    elif config == "tac1":
        return TacganAugmentorConfig.known_config_tac1().build()
    elif config == "tac2":
        return TacganAugmentorConfig.known_config_tac2().build()
    elif config == "tac3":
        return TacganAugmentorConfig.known_config_tac3().build()
    else:
        raise RuntimeError("unknown config")


def sample(config: str):
    if config == "all":
        configs = ["ac", "tac1", "tac2"]
    else:
        configs = [config]

    for c in configs:
        augm = create_augmentor(c)
        sample_augmentor(augm, c)


def run_experiments(config: str):
    baseline_study_config = BaselineStudyConfig(experiment_name="baseline_with_tacgan")
    experiments = []

    if config in ["ac", "all"]:
        experiments.append(
            replace(
                baseline_study_config,
                study_suffix="ac",
                datamodule_config=replace(
                    baseline_study_config.datamodule_config,
                    augmentor_config=TacganAugmentorConfig.known_config_ac(),
                    n_augment=1000,
                ),
            )
        )

    if config in ["tac1", "all"]:
        experiments.append(
            replace(
                baseline_study_config,
                study_suffix="tac1",
                datamodule_config=replace(
                    baseline_study_config.datamodule_config,
                    augmentor_config=TacganAugmentorConfig.known_config_tac1(),
                    n_augment=1000,
                ),
            )
        )

    if config in ["tac2", "all"]:
        experiments.append(
            replace(
                baseline_study_config,
                study_suffix="tac2",
                datamodule_config=replace(
                    baseline_study_config.datamodule_config,
                    augmentor_config=TacganAugmentorConfig.known_config_tac2(),
                    n_augment=1000,
                ),
            )
        )

    if config in ["tac3", "all"]:
        experiments.append(
            replace(
                baseline_study_config,
                study_suffix="tac3",
                datamodule_config=replace(
                    baseline_study_config.datamodule_config,
                    augmentor_config=TacganAugmentorConfig.known_config_tac3(),
                    n_augment=1000,
                ),
            )
        )

    for exp_config in experiments:
        augmented_study = exp_config.build()
        print(f"Running tacgan study (config {exp_config.study_suffix})...")
        augmented_study.run()


def main():
    args = parse_args()
    assert args.upload ^ args.sample ^ args.experiments, (
        "please use exactly one of --upload or --sample or --experiments"
    )

    if args.upload:
        upload(args.config)
    elif args.sample:
        sample(args.config)
    elif args.experiments:
        run_experiments(args.config)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true", default=False)
    parser.add_argument(
        "--config",
        type=str,
        choices=["ac", "tac1", "tac2", "tac3", "all"],
        required=True,
    )
    parser.add_argument("--sample", action="store_true", default=False)
    parser.add_argument("--experiments", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
