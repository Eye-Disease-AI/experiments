from dataset.hard_policy import HardPolicy
from experiments.experiment import common_config
from experiments.experiment.data import MyDataModule
from experiments.experiment.models.convnext import ConvNext
from experiments.experiment.run_study import run_study
from experiments.lib.mlflow_setup import Experiment
from experiments.lib.reproducibility import RNG, get_git_sha

SHA = get_git_sha()

MODELS = [
    (ConvNext, f"{common_config.EXPERIMENT_NAME}/convnext-search_{SHA}")
]

def main():
    rng = RNG()
    rng.set_seed(common_config.SEED)
    exp = Experiment(common_config.EXPERIMENT_NAME)

    datamodule = MyDataModule(rng, hard_policy=HardPolicy.DOMINATE)
    datamodule.prepare_data()

    for ModelClass, study_name in MODELS:
        print(f"\n{'=' * 60}")
        print(f"Model: {ModelClass.__name__}, Study: {study_name}")
        print(f"{'=' * 60}\n")
        run_study(ModelClass, study_name, exp, rng, datamodule)


if __name__ == "__main__":
    main()
