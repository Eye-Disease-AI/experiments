from experiments.experiment.datamodules.nuclear_cataract_datamodule import (
    NuclearCataractDataModuleConfig,
)
from experiments.experiment.studies.study import StudyConfig
import torch

from dataset.hard_policy import HardPolicy
from experiments.experiment.studies.baseline_study import BaselineStudyConfig


def main():
    EXPERIMENT_NAME = "baseline-hard-policy-test-debug-5"

    STUDIES: list[StudyConfig] = [
        BaselineStudyConfig(
            experiment_name=EXPERIMENT_NAME + "_dominate",
            datamodule_config=NuclearCataractDataModuleConfig(
                hard_policy=HardPolicy.DOMINATE,
            ),
            max_epochs=3,
            max_trials=2,
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
