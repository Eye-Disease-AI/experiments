from common_config import EXPERIMENT_NAME, SEED
from run_study import run_study
from convnext import ConvNext
from data import MyDataModule
from mlflow_setup import Experiment
from seed import RNG

TRY=2

MODELS = [
    (ConvNext,    f"{EXPERIMENT_NAME}/convnext-search_{TRY}"),
]

rng = RNG()
rng.set_seed(SEED)
exp = Experiment(EXPERIMENT_NAME)

datamodule = MyDataModule(rng)
datamodule.prepare_data()

for ModelClass, study_name in MODELS:
    print(f"\n{'='*60}")
    print(f"Model: {ModelClass.__name__}, Study: {study_name}")
    print(f"{'='*60}\n")
    run_study(ModelClass, study_name, exp, rng, datamodule)