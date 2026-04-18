from experiments.nuclear_cataract.common_config import EXPERIMENT_NAME, SEED
from experiments.nuclear_cataract.run_study import run_study
from experiments.nuclear_cataract.models.convnext import ConvNext
from experiments.nuclear_cataract.models.vit import ViT
from experiments.nuclear_cataract.models.swin import Swin
from experiments.nuclear_cataract.models.convnext_vit import ConvNextViT
from experiments.nuclear_cataract.data import MyDataModule
from lib.mlflow_setup import Experiment
from lib.seed import RNG

TRY=1

MODELS = [
    (ConvNext,    f"{EXPERIMENT_NAME}/convnext-search_{TRY}"),
    (ViT,         f"{EXPERIMENT_NAME}/vit-search_{TRY}"),
    (Swin,        f"{EXPERIMENT_NAME}/swin-search_{TRY}"),
    (ConvNextViT, f"{EXPERIMENT_NAME}/convnext-vit-search_{TRY}"),
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