from dataclasses import replace

from experiments.experiment.studies.study import StudyConfig
import torch

from dataset.hard_policy import HardPolicy
from experiments.experiment.studies.baseline_study import BaselineStudy

EXPERIMENT_NAME = "baseline-hard-policy-test"

STUDIES: list[StudyConfig] = [
    replace(
        BaselineStudy.DEFAULT_CONFIG,
        experiment_name=EXPERIMENT_NAME + "_dominate",
        datamodule_config=replace(
            BaselineStudy.DEFAULT_CONFIG.datamodule_config,
            hard_policy=HardPolicy.DOMINATE,
        ),
    ),
    replace(
        BaselineStudy.DEFAULT_CONFIG,
        experiment_name=EXPERIMENT_NAME + "_no_hard",
        datamodule_config=replace(
            BaselineStudy.DEFAULT_CONFIG.datamodule_config,
            hard_policy=HardPolicy.NO_HARD,
        ),
    ),
    replace(
        BaselineStudy.DEFAULT_CONFIG,
        experiment_name=EXPERIMENT_NAME + "_only_hard",
        datamodule_config=replace(
            BaselineStudy.DEFAULT_CONFIG.datamodule_config,
            hard_policy=HardPolicy.ONLY_HARD,
        ),
    ),
    replace(
        BaselineStudy.DEFAULT_CONFIG,
        experiment_name=EXPERIMENT_NAME + "_passthrough",
        datamodule_config=replace(
            BaselineStudy.DEFAULT_CONFIG.datamodule_config,
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
