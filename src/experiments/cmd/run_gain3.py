from experiments.experiment.datamodules.nuclear_cataract_datamodule import (
    NuclearCataractDataModule,
    NuclearCataractDataModuleConfig,
)
from experiments.experiment.models.gain_convnext import GAINConvNextConfig
from experiments.experiment.studies.baseline_study import (
    BaselineStudyConfig,
)


def dmc(ratio):
    return NuclearCataractDataModuleConfig.from_other(
        BaselineStudyConfig().datamodule_config, return_bboxes=True, bboxes_ratio=ratio
    )


exp_name = "gain3_bbox_percent"


def generate_studies():
    d: NuclearCataractDataModule = NuclearCataractDataModuleConfig(
        cache=False, return_bboxes=True, bboxes_ratio=1
    ).build()
    d.prepare_data()
    d.setup()
    max_ratio = NuclearCataractDataModule.boxes_ratio(d.train_set)

    ratios = [i / 100 for i in range(0, int(max_ratio * 100 + 1))]
    studies = []
    for ratio in ratios:
        studies.append(
            BaselineStudyConfig(
                experiment_name=exp_name,
                study_suffix=f"ratio_{ratio}",
                datamodule_config=dmc(ratio),
                model_config=GAINConvNextConfig.from_other(
                    BaselineStudyConfig().model_config,
                    use_attention_mining=True,
                    use_external_supervision=True,
                ),
            )
        )
    return studies


studies = generate_studies()


def main():
    for study in studies:
        study = study.build()
        study.run()


if __name__ == "__main__":
    main()
