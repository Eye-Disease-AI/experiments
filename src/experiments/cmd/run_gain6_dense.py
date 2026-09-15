from experiments.experiment.datamodules.nuclear_cataract_datamodule import (
    NuclearCataractDataModuleConfig,
)
from experiments.experiment.models.gain_convnext import GAINConvNextConfig
from experiments.experiment.studies.baseline_study import (
    BaselineStudyConfig,
)
from experiments.cmd.run_gain6 import scale


def dmc():
    return NuclearCataractDataModuleConfig.from_other(
        BaselineStudyConfig().datamodule_config,
        return_bboxes=True,
    )


exp_name = "gain6_pretrain_percent"


def generate_studies():
    max_epochs = BaselineStudyConfig().max_epochs
    sparse = set(scale(max_epochs))
    studies = []
    for warmup_epochs in range(1, 30):
        if warmup_epochs in sparse:  # already covered by run_gain6.py
            continue
        studies.append(
            BaselineStudyConfig(
                experiment_name=exp_name,
                study_suffix=f"warmup_{warmup_epochs / max_epochs * 100:.0f}%",
                datamodule_config=dmc(),
                model_config=GAINConvNextConfig.from_other(
                    BaselineStudyConfig().model_config,
                    use_attention_mining=True,
                    use_external_supervision=True,
                    warmup_epochs=warmup_epochs,
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
