from dataclasses import replace

from experiments.cmd.run_configs import DEFAULT_CONVNEXT_CONFIG, quick_dev_test
import torch

from dataset.hard_policy import HardPolicy
from experiments.experiment.studies.convnext_study import ConvNextStudy

STUDIES = [
    ConvNextStudy(
        quick_dev_test(
            replace(
                DEFAULT_CONVNEXT_CONFIG,
                experiment_name=DEFAULT_CONVNEXT_CONFIG.experiment_name + "_dominate",
                data_module=replace(
                    DEFAULT_CONVNEXT_CONFIG.data_module,
                    data_module_config=replace(
                        DEFAULT_CONVNEXT_CONFIG.data_module.data_module_config,
                        hard_policy=HardPolicy.DOMINATE,
                    ),
                ),
            )
        )
    ),
    ConvNextStudy(
        quick_dev_test(
            replace(
                DEFAULT_CONVNEXT_CONFIG,
                experiment_name=DEFAULT_CONVNEXT_CONFIG.experiment_name + "_no_hard",
                data_module=replace(
                    DEFAULT_CONVNEXT_CONFIG.data_module,
                    data_module_config=replace(
                        DEFAULT_CONVNEXT_CONFIG.data_module.data_module_config,
                        hard_policy=HardPolicy.NO_HARD,
                    ),
                ),
            )
        )
    ),
    ConvNextStudy(
        quick_dev_test(
            replace(
                DEFAULT_CONVNEXT_CONFIG,
                experiment_name=DEFAULT_CONVNEXT_CONFIG.experiment_name + "_only_hard",
                data_module=replace(
                    DEFAULT_CONVNEXT_CONFIG.data_module,
                    data_module_config=replace(
                        DEFAULT_CONVNEXT_CONFIG.data_module.data_module_config,
                        hard_policy=HardPolicy.ONLY_HARD,
                    ),
                ),
            )
        )
    ),
    ConvNextStudy(
        quick_dev_test(
            replace(
                DEFAULT_CONVNEXT_CONFIG,
                experiment_name=DEFAULT_CONVNEXT_CONFIG.experiment_name
                + "_passthrough",
                data_module=replace(
                    DEFAULT_CONVNEXT_CONFIG.data_module,
                    data_module_config=replace(
                        DEFAULT_CONVNEXT_CONFIG.data_module.data_module_config,
                        hard_policy=HardPolicy.PASSTHROUGH,
                    ),
                ),
            )
        )
    ),
]

torch.set_float32_matmul_precision("medium")  # to use tensor cores on newer gpus

for study in STUDIES:
    print(f"\n{'=' * 60}")
    print(f"Study: {study.name}")
    print(f"{'=' * 60}\n")
    study.run()
