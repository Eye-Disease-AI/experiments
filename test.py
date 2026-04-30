from common_config import *
from run_study import run_study
from convnext import ConvNext
from data import MyDataModule
from mlflow_setup import Experiment
from seed import RNG
EXPERIMENT_NAME="test"
STUDY_NAME = f"{EXPERIMENT_NAME}/lr-search5"

rng = RNG()
rng.set_seed(SEED)
exp = Experiment(EXPERIMENT_NAME)

datamodule = MyDataModule(rng)
datamodule.prepare_data()

EPOCHS = 5
MAX_TRIALS = 3
run_study(ConvNext, STUDY_NAME, exp, rng, datamodule)