from dataclasses import replace

from experiments.experiment.studies.study import StudyConfig
from src.experiments.experiment.models.mock_model import MockModel
import torch
from experiments.experiment.studies.baseline_study import BaselineStudy, BaselineStudyConfig

EXPERIMENT_NAME = "baseline-hard-policy-test"

STUDIES: list[StudyConfig] = [
        BaselineStudyConfig(**BaselineStudy.DEFAULT_CONFIG_KWARGS,
        max_epochs=1,
        max_trials=1,
        experiment_name=EXPERIMENT_NAME + "_dominate",
        datamodule_config=replace(
            MockModel.DEFAULT_CONFIG.datamodule_config,
            cache=False,
        ),
        model_config = MockModel.DEFAULT_CONFIG,
    ),
]

torch.set_float32_matmul_precision("high")

for study_config in STUDIES:
    study = study_config.build()
    print(f"\n{'=' * 60}")
    print(f"Study: {study.name}")
    print(f"{'=' * 60}\n")
    study.run()
