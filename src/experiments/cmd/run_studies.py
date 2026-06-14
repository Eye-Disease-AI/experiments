import torch

from dataset.hard_policy import HardPolicy
from experiments.experiment.data import MyDataModule
from experiments.experiment.studies.convnext_study import ConvNextStudy
from experiments.lib.reproducibility import RNG
import experiments.experiment.common_config as common_config

rng = RNG()

STUDIES = [
    ConvNextStudy(
        common_config.EXPERIMENT_NAME + "_dominate",
        common_config.SEED,
        common_config.MAX_TRIALS,
        MyDataModule(rng, hard_policy=HardPolicy.DOMINATE),
        common_config.EPOCHS,
    ),
    ConvNextStudy(
        common_config.EXPERIMENT_NAME + "_only_hard",
        common_config.SEED,
        common_config.MAX_TRIALS,
        MyDataModule(rng, hard_policy=HardPolicy.ONLY_HARD),
        common_config.EPOCHS,
    ),
    ConvNextStudy(
        common_config.EXPERIMENT_NAME + "_no_hard",
        common_config.SEED,
        common_config.MAX_TRIALS,
        MyDataModule(rng, hard_policy=HardPolicy.NO_HARD),
        common_config.EPOCHS,
    ),
    ConvNextStudy(
        common_config.EXPERIMENT_NAME + "_passthrough",
        common_config.SEED,
        common_config.MAX_TRIALS,
        MyDataModule(rng, hard_policy=HardPolicy.PASSTHROUGH),
        common_config.EPOCHS,
    ),
]

torch.set_float32_matmul_precision("medium")  # to use tensor cores on newer gpus

for study in STUDIES:
    print(f"\n{'=' * 60}")
    print(f"Study: {study.name}")
    print(f"{'=' * 60}\n")
    study.run()
