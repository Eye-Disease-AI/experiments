from experiments.nuclear_cataract.common_config import *
from experiments.nuclear_cataract.run_study import run_study
from experiments.nuclear_cataract.models.convnext import ConvNext
from experiments.nuclear_cataract.data import MyDataModule
from lib.mlflow_setup import Experiment
from lib.seed import RNG

rng = RNG()
rng.set_seed(SEED)
exp = Experiment(EXPERIMENT_NAME)

datamodule = MyDataModule(rng)
datamodule.prepare_data()

run_study(ConvNext, STUDY_NAME, exp, rng, datamodule)