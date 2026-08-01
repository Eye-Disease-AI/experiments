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


exp_name = "gain2_layers"
possible_layers = [
    ("model", "features", 1),  # 56x56
    ("model", "features", 3),  # 28x28
    ("model", "features", 5),  # 14x14
    ("model", "features", 7),  # 7x7
]


def generate_layer_combos():
    from itertools import combinations

    n = len(possible_layers)
    return [
        (
            [possible_layers[i] for i in idxs],
            "L" + "".join([str(possible_layers[j][2]) for j in idxs]),
        )
        for size in range(1, n + 1)
        for idxs in combinations(range(n), size)
    ]


def generate_studies():
    combos = generate_layer_combos()
    studies = []
    for combo in combos:
        layers, name = combo
        studies.append(
            BaselineStudyConfig(
                experiment_name=exp_name,
                study_suffix=name,
                datamodule_config=dmc(),
                model_config=GAINConvNextConfig.from_other(
                    BaselineStudyConfig().model_config,
                    use_attention_mining=True,
                    use_external_supervision=True,
                    target_layers=layers,
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
