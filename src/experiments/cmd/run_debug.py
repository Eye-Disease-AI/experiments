import pprint

from experiments.experiment.datamodules.mock_nuclear_cataract_datamodule import (
    MockNuclearCataractDatamoduleConfig,
)
from experiments.experiment.models.mock_model import MockConvNextConfig
import torch
from experiments.experiment.studies.baseline_study import (
    BaselineStudy,
    BaselineStudyConfig,
)
from experiments.lib.reproducibility import get_trial_vals

EXPERIMENT_NAME = "sanity-check-debug1236"
study_config = BaselineStudyConfig(
    max_epochs=1,
    max_trials=1,
    experiment_name=EXPERIMENT_NAME,
    datamodule_config=MockNuclearCataractDatamoduleConfig(
        batch_size=10, dataset_len=10
    ),
    model_config=MockConvNextConfig(),
)


def main():
    torch.set_float32_matmul_precision("high")

    study: BaselineStudy = study_config.build()
    print(f"\n{'=' * 60}")
    print(f"Study: {study.name}")
    print(f"{'=' * 60}\n")
    experiment, optuna_study = study.run()

    vals = get_trial_vals(
        experiment, optuna_study.study_name, only_crossvalidation_trials=False
    )
    if len(vals) != 1:
        raise Exception(f"amount of matching trials is {len(vals)}, should be 1")

    if len([key for key in vals[0].keys() if "retrain_" in key]) == 0:
        raise Exception(f"No 'retrain_' metrics found, metrics: {vals[0].items()}")

    metrics = vals[0]
    pprint.pprint(metrics)
    for key, val in metrics.items():
        if "retrain_" in key:
            bkey = key[len("retrain_") :]
            rkey = "retrain_" + key
            bval = metrics[bkey] if bkey in metrics else None
            rval = metrics[bkey] if bkey in metrics else None
            if bval and rval:
                print(
                    f"metrics[{bkey}]={metrics[bkey]}, metrics[{rkey}]={metrics[rkey]}"
                )
                if not metrics[rkey] == metrics[rkey]:
                    raise Exception(
                        f"best value != retrain: metrics[{bkey}]={metrics[bkey]}, metrics[{rkey}]={metrics[rkey]}"
                    )


if __name__ == "__main__":
    main()
