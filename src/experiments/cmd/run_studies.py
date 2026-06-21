from dataclasses import replace

from experiments.experiment.studies.study import StudyConfig
import torch

from dataset.hard_policy import HardPolicy
from experiments.experiment.studies.baseline_study import BaselineStudyConfig

EXPERIMENT_NAME = "baseline-hard-policy-test"

STUDIES: list[StudyConfig] = [
    replace(
        BaselineStudyConfig(),
        experiment_name=EXPERIMENT_NAME + "_dominate",
        datamodule_config=replace(
            BaselineStudyConfig().datamodule_config,
            hard_policy=HardPolicy.DOMINATE,
        ),
    ),
    replace(
        BaselineStudyConfig(),
        experiment_name=EXPERIMENT_NAME + "_no_hard",
        datamodule_config=replace(
            BaselineStudyConfig().datamodule_config,
            hard_policy=HardPolicy.NO_HARD,
        ),
    ),
    replace(
        BaselineStudyConfig(),
        experiment_name=EXPERIMENT_NAME + "_only_hard",
        datamodule_config=replace(
            BaselineStudyConfig().datamodule_config,
            hard_policy=HardPolicy.ONLY_HARD,
        ),
    ),
    replace(
        BaselineStudyConfig(),
        experiment_name=EXPERIMENT_NAME + "_passthrough",
        datamodule_config=replace(
            BaselineStudyConfig().datamodule_config,
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
