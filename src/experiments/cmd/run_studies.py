from dataclasses import replace

from experiments.experiment.studies import init_study
from experiments.experiment.studies._serializing import flatten_dict, study_config_to_dict
from experiments.experiment.studies.study import StudyConfig
import torch

from dataset.hard_policy import HardPolicy
from experiments.experiment.studies.convnext_study import ConvNextStudy

STUDIES: list[StudyConfig] = [
    replace(
        ConvNextStudy.DEFAULT_CONFIG,
        experiment_name=ConvNextStudy.DEFAULT_CONFIG.experiment_name + "_dominate",
        datamodule_config=replace(
            ConvNextStudy.DEFAULT_CONFIG.datamodule_config,
            hard_policy=HardPolicy.DOMINATE,
        ),
    ),
    replace(
        ConvNextStudy.DEFAULT_CONFIG,
        experiment_name=ConvNextStudy.DEFAULT_CONFIG.experiment_name + "_no_hard",
        datamodule_config=replace(
            ConvNextStudy.DEFAULT_CONFIG.datamodule_config,
            hard_policy=HardPolicy.NO_HARD,
        ),
    ),
    replace(
        ConvNextStudy.DEFAULT_CONFIG,
        experiment_name=ConvNextStudy.DEFAULT_CONFIG.experiment_name + "_only_hard",
        datamodule_config=replace(
            ConvNextStudy.DEFAULT_CONFIG.datamodule_config,
            hard_policy=HardPolicy.ONLY_HARD,
        ),
    ),
    replace(
        ConvNextStudy.DEFAULT_CONFIG,
        experiment_name=ConvNextStudy.DEFAULT_CONFIG.experiment_name + "_passthrough",
        datamodule_config=replace(
            ConvNextStudy.DEFAULT_CONFIG.datamodule_config,
            hard_policy=HardPolicy.PASSTHROUGH,
        ),
    ),
]

torch.set_float32_matmul_precision("medium")  # to use tensor cores on newer gpus

for study_config in STUDIES:
    study = init_study(study_config)
    print(f"\n{'=' * 60}")
    print(f"Study: {study.name}")
    print(f"{'=' * 60}\n")
    study.run()
