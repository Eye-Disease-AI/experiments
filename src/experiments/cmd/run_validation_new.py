"""
Description of what this entrypoint should achieve:
It should perform some kind of verification. I can see that
there are two modes:

* seeds with --num-seeds option
* kfold with -K option

It creates separete validation study with the first component
matching the original name. Study is run as part of the same
MLFlow experiment.

1. We load best study params
2. Set up data modules

In the legacy code we set up Nuclear Cataract datamodule using
the params, but now we are leaving decision about any needed modules to the
study class. So either study class should expose publicly method for getting its
datamodules OR we can consturct them ourselves based on the config OR we can
move all validation logic to be part of study class.

3. MLFlow orchestration
4. Look for parameters with `best_` prefix in the selected study, start a run
and log these values there
5. Start nested run named `retrain` and do the actual training there using earlier
loaded and logged best parameters
6. Run either kfold or seeds validation based on CLI options. These modes also
start nested runs and log parameters which are mode-specific and run names
contain information about which seed or which fold was used
"""

import argparse
from experiments.lib.mlflow_setup import Experiment
import optuna


class StudyValidator:
    def __init__(
        self,
        experiment_name: str,
        study_name: str,
    ):
        self.experiment_name = experiment_name
        self.study_name = study_name

    def _find_best_params(self):
        exp = Experiment(self.experiment_name)
        study = optuna.load_study(study_name=self.study_name, storage=exp.storage)
        print(study.best_params)

    @staticmethod
    def list_studies():
        storage = Experiment().storage
        for summary in optuna.get_all_study_summaries(storage=storage):
            print(summary.study_name)


class KFoldValidator:
    def __init__(self, k: int):
        pass


class SeedsValidator:
    def __init__(self, num_seeds: int):
        pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--study_to_verify", type=str)
    parser.add_argument(
        "--mode", type=str, choices=["kfold", "seeds", "list"], default="list"
    )
    parser.add_argument("-K", type=int, default=5)
    parser.add_argument("--num_seeds", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()

    match args.mode:
        case "list":
            StudyValidator.list_studies()
        case "kfold" | "seeds":
            validator = StudyValidator(args.experiment_name, args.study_to_verify)
            print(validator._find_best_params())


if __name__ == "__main__":
    main()
