from experiments.nuclear_cataract.common_config import *
from experiments.nuclear_cataract.run_study import run_study
from experiments.nuclear_cataract.models.convnext_vit import ConvNextViT
from experiments.nuclear_cataract.data import MyDataModule
from lib.mlflow_setup import Experiment
from lib.seed import RNG
EXPERIMENT_NAME="test"
STUDY_NAME = f"{EXPERIMENT_NAME}/lr-search5"

rng = RNG()
rng.set_seed(SEED)
exp = Experiment(EXPERIMENT_NAME)

datamodule = MyDataModule(rng)
datamodule.prepare_data()

EPOCHS = 5
MAX_TRIALS = 3
run_study(ConvNextViT, STUDY_NAME, exp, rng, datamodule)