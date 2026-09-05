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


exp_name = "gain6_pretrain_percent"


def scale(largest):
    # Numbers from 1 to largest, high density near ends, low in middle
    def biggest_round_step(limit):
        # The largest round number (1, 5, 10, 50, 100 ...) that fits
        step = 1
        growth = 5
        while step * growth <= limit:
            step *= growth
            growth = 2 if growth == 5 else 5
        return step

    points = {largest}
    v = 1
    while 2 * v <= largest:
        points.add(v)
        points.add(largest - v)
        v += biggest_round_step(v / 2)
    return sorted(points)


def generate_studies():
    max_epochs = BaselineStudyConfig().max_epochs
    studies = []
    for warmup_epochs in scale(max_epochs):
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
        study = study.build()
        study.run()


if __name__ == "__main__":
    main()
