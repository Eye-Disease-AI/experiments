from experiments.experiment.datamodules.nuclear_cataract_datamodule import (
    NuclearCataractDataModuleConfig,
)
from experiments.experiment.studies.study import StudyConfig
import torch

from dataset.hard_policy import HardPolicy
from experiments.experiment.studies.baseline_study import BaselineStudyConfig


def main():
    EXPERIMENT_NAME = "baseline-hard-policy-test-debug"

    STUDIES: list[StudyConfig] = [
        BaselineStudyConfig(
            experiment_name=EXPERIMENT_NAME + "_dominate",
            datamodule_config=NuclearCataractDataModuleConfig(
                hard_policy=HardPolicy.DOMINATE,
            ),
        ),
        BaselineStudyConfig(
            experiment_name=EXPERIMENT_NAME + "_no_hard",
            datamodule_config=NuclearCataractDataModuleConfig(
                hard_policy=HardPolicy.NO_HARD,
            ),
        ),
        BaselineStudyConfig(
            experiment_name=EXPERIMENT_NAME + "_only_hard",
            datamodule_config=NuclearCataractDataModuleConfig(
                hard_policy=HardPolicy.ONLY_HARD,
            ),
        ),
        BaselineStudyConfig(
            experiment_name=EXPERIMENT_NAME + "_passthrough",
            datamodule_config=NuclearCataractDataModuleConfig(
                hard_policy=HardPolicy.PASSTHROUGH,
            ),
        ),
    ]

    torch.set_float32_matmul_precision("high")

    for study_config in STUDIES:
        study = study_config.build()
        print(f"\n{'=' * 60}")
        print(f"Study: {study.name}")
        print(f"{'=' * 60}\n")
        study.run()


if __name__ == "__main__":
    main()
