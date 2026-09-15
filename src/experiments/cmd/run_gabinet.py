from dataset.datasets import DatasetKind

from experiments.experiment.datamodules.nuclear_cataract_datamodule import (
    NuclearCataractDataModuleConfig,
)
from experiments.experiment.studies.baseline_study import BaselineStudyConfig


def main():
    study_config = BaselineStudyConfig(
        experiment_name="gabinet",
        study_suffix="test1",
        datamodule_config=NuclearCataractDataModuleConfig.from_other(
            BaselineStudyConfig().datamodule_config,
            test_dataset_kind=DatasetKind.GABINET,
        ),
        unsafe_validate_on_test=True,
        max_epochs=5,
    )

    print("Running gabinet baseline study...")
    study_config.build().run()


if __name__ == "__main__":
    main()
