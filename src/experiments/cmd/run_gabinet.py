from experiments.experiment.augmentors.stylegan_augmentor import StyleganAugmentorConfig
from experiments.experiment.datamodules.nuclear_cataract_datamodule import (
    NuclearCataractDataModuleConfig,
)
from dataset.datasets import DatasetKind
from experiments.experiment.models.gain_convnext import GAINConvNextConfig

from experiments.experiment.studies.gain_optimised import GainOptimisedStudyConfig


def dmc():
    return NuclearCataractDataModuleConfig.from_other(
        GainOptimisedStudyConfig().datamodule_config,
        return_bboxes=True,
        augmentor_config=StyleganAugmentorConfig.known_config_r(),
        n_augment=1000,
        augment_kind="RAG",
        test_dataset_kind=DatasetKind.GABINET,
    )


exp_name = "gain_genaug_gabinet_trial1"
studies = [
    GainOptimisedStudyConfig(
        experiment_name=exp_name,
        datamodule_config=dmc(),
        model_config=GAINConvNextConfig.from_other(
            GainOptimisedStudyConfig().model_config,
            use_attention_mining=True,
            use_external_supervision=True,
            target_layers=[("model", "features", 5), ("model", "features", 7)],
            warmup_epochs=15,
            visualization_heatmap_methods=["gradcam"],
            loss_heatmap_method="gradcam",
        ),
        unsafe_validate_on_test=True,
    ),
]


def main():
    for study in studies:
        study = study.build()
        study.run()


if __name__ == "__main__":
    main()
