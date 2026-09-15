from experiments.experiment.datamodules.nuclear_cataract_datamodule import (
    NuclearCataractDataModuleConfig,
)
from experiments.experiment.models.gain_convnext import GAINConvNextConfig
from experiments.experiment.studies.baseline_study import (
    BaselineStudyConfig,
)


def dmc():
    return NuclearCataractDataModuleConfig.from_other(
        BaselineStudyConfig().datamodule_config,
        return_bboxes=True,
    )


exp_name = "gain5_cam_variants"


def generate_studies():
    studies = []
    combos = [
        ("gradcam", [("model", "features", 7)], "L7"),
        ("gradcam", [("model", "features", 5), ("model", "features", 7)], "L57"),
        ("gradcam_pp", [("model", "features", 7)], "L7"),
        ("gradcam_pp", [("model", "features", 5), ("model", "features", 7)], "L57"),
        ("layercam", [("model", "features", 7)], "L7"),
        ("layercam", [("model", "features", 5), ("model", "features", 7)], "L57"),
    ]
    for cam, layers, layers_name in combos:
        studies.append(
            BaselineStudyConfig(
                experiment_name=exp_name,
                study_suffix=f"{cam}_{layers_name}",
                datamodule_config=dmc(),
                model_config=GAINConvNextConfig.from_other(
                    BaselineStudyConfig().model_config,
                    use_attention_mining=True,
                    use_external_supervision=True,
                    loss_heatmap_method=cam,
                    visualization_heatmap_methods=[cam],
                    target_layers=layers,
                ),
                use_early_stopping=False,
            )
        )
    return studies


studies = generate_studies()


def main():
    for study in studies:
        # print(study.study_suffix)
        study = study.build()
        study.run()


if __name__ == "__main__":
    main()
