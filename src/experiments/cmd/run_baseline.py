from experiments.experiment.datamodules.nuclear_cataract_datamodule import (
    NuclearCataractDataModuleConfig,
)
import torch

from dataset.hard_policy import HardPolicy
from experiments.experiment.studies.baseline_study import BaselineStudyConfig


def main():
    torch.set_float32_matmul_precision("high")

    baseline_study_config = BaselineStudyConfig(
        experiment_name="baseline",
        datamodule_config=NuclearCataractDataModuleConfig(
            hard_policy=HardPolicy.PASSTHROUGH,
        ),
    )
    baseline_study = baseline_study_config.build()

    print("Running baseline study...")
    baseline_study.run()


if __name__ == "__main__":
    main()
