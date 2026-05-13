from dataset.hard_policy import HardPolicy
from experiment.common_config import EXPERIMENT_NAME, SEED
from experiment.data import MyDataModule
from experiment.models.convnext import ConvNext
from experiment.models.swin import Swin
from experiment.models.vit import ViT
from experiment.run_study import run_study
from lib.mlflow_setup import Experiment
from lib.reproducibility import RNG, get_git_sha
SHA = get_git_sha()

TRY = 2
MODELS = [
    (ConvNext, f"{EXPERIMENT_NAME}/convnext-search_{TRY}_{SHA}")
]

rng = RNG()
rng.set_seed(SEED)
exp = Experiment(EXPERIMENT_NAME)

datamodule = MyDataModule(rng, hard_policy=HardPolicy.DOMINATE)
datamodule.prepare_data()

for ModelClass, study_name in MODELS:
    print(f"\n{'=' * 60}")
    print(f"Model: {ModelClass.__name__}, Study: {study_name}")
    print(f"{'=' * 60}\n")
    run_study(ModelClass, study_name, exp, rng, datamodule)
