import ast
import importlib
import pkgutil

import mlflow
import experiment.common_config
import experiment.models
import lightning as L
import numpy as np
import optuna

from dataset.hard_policy import HardPolicy
from dataset.loader import NuclearCataractDataset, NuclearCataractSubset
from experiment.common_config import EPOCHS, EXPERIMENT_NAME, OPTUNA_METRIC
from experiment.models.convnext import ConvNext
from experiment.data import MyDataModule, SubsetTransformer
from experiment.models.base import ModelBase
from experiment.objective import create_trainer
from experiment.run_study import CONFIG_PARAMS
from lib.mlflow_setup import Experiment
from lib.reproducibility import RNG, get_git_sha
from optuna.exceptions import OptunaError
import sys

for _, _name, _ in pkgutil.iter_modules(experiment.models.__path__):
    importlib.import_module(f"experiment.models.{_name}")
MODEL_CLASSES = {cls.__name__: cls for cls in ModelBase.__subclasses__()}

SHA = get_git_sha()
STUDY_TO_VERIFY = f"{EXPERIMENT_NAME}/convnext-search_{SHA}"
if __name__ == "__main__":
    if len(sys.argv)>1:
        STUDY_TO_VERIFY = sys.argv[1]

MODE = "kfold"  # "kfold" or "seeds"
K = 5
SEEDS = list(range(10))


def parse_logged(v):
    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return v


exp = Experiment(EXPERIMENT_NAME)
study = optuna.load_study(study_name=STUDY_TO_VERIFY, storage=exp.storage)

logged_config = study.user_attrs.get("config") or {}
ModelClass = ConvNext
print(f"model_class={ModelClass.__name__}")

config_changed = {
    k: (logged_config.get(k), CONFIG_PARAMS.get(k))
    for k in set(logged_config) | set(CONFIG_PARAMS)
    if logged_config.get(k) != CONFIG_PARAMS.get(k)
}
if config_changed:
    print("WARNING: common_config changed since training:")
    for k, (logged, current) in config_changed.items():
        print(f"  {k}: logged={logged}  current={current}")

SEED = 2137
GPU_PRECISION = experiment.common_config.GPU_PRECISION
OPTUNA_DIRECTION = experiment.common_config.OPTUNA_DIRECTION
HARD_POLICY = HardPolicy.DOMINATE
PRE_ROT_SIZE = 275

best_params = study.best_params
_agg = min if OPTUNA_DIRECTION == "min" else max
print(f"best_params={best_params} hard_policy={HARD_POLICY.name}")


class _BestMetricCallback(L.Callback):
    def __init__(self):
        self.best = None

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        v = trainer.callback_metrics.get(OPTUNA_METRIC)
        if v is not None:
            v = v.item()
            self.best = _agg(v, self.best) if self.best is not None else v


def train_and_validate(dm: MyDataModule, train_sub: NuclearCataractSubset, val_sub: NuclearCataractSubset, seed, best_params, run):
    L.seed_everything(seed, workers=True)
    dm.train_set = SubsetTransformer(train_sub, transform=dm.transform)
    dm.val_set = SubsetTransformer(val_sub, transform=dm.val_transform)
    dm.train_class_weights = train_sub.class_weights()
    dm.batch_size = best_params["batch_size"]
    params = {k: v for k, v in best_params.items() if k != "batch_size"}
    model = ModelClass(
        dm.dataset.n_classes, **params, class_weights=dm.train_class_weights
    )
    best_cb = _BestMetricCallback()
    trainer = create_trainer(
        run, max_epochs=EPOCHS, precision=GPU_PRECISION, callbacks=[best_cb]
    )
    trainer.fit(model, datamodule=dm)
    return {OPTUNA_METRIC: best_cb.best}


def summarize(results, name):
    print(f"\n=== Summary across {len(results)} {name} ===")
    for k in results[0]:
        vals = np.array([r[k] for r in results])
        print(
            f"  {k}: mean={vals.mean():.4f}  std={vals.std():.4f}  "
            f"min={vals.min():.4f}  max={vals.max():.4f}"
        )


rng = RNG()
rng.set_seed(SEED)

datamodule = MyDataModule(
    rng, batch_size=best_params["batch_size"], hard_policy=HARD_POLICY, cache=False
)
datamodule.setup(stage="fit")

results = []
validation_run_name = f"{STUDY_TO_VERIFY}/validation_{MODE}"

with mlflow.start_run(run_name=validation_run_name) as parent_run:
    mlflow.set_tag("study_name", STUDY_TO_VERIFY)
    mlflow.set_tag("model_class", ModelClass.__name__)
    mlflow.set_tag("mode", MODE)
    mlflow.log_params({
        **best_params,
        "hard_policy": HARD_POLICY.name,
        "seed": SEED,
        **({"K": K} if MODE == "kfold" else {"n_seeds": len(SEEDS)}),
    })

    if MODE == "kfold":
        dataset = NuclearCataractDataset(
            NuclearCataractDataset.KFoldCVMode(K),
            cache_size=PRE_ROT_SIZE,
            hard_policy=HARD_POLICY,
        )
        for i in range(K):
            train_sub = dataset.fold_train_set(i)
            val_sub = dataset.fold_val_set(i)
            print(f"\n--- Fold {i + 1}/{K}  train={len(train_sub)} val={len(val_sub)} ---")
            with mlflow.start_run(run_name=f"fold-{i}", nested=True) as child_run:
                mlflow.set_tag("fold", i)
                m = train_and_validate(datamodule, train_sub, val_sub, SEED, best_params, child_run)
                mlflow.log_metrics(m)
            results.append(m)
        summarize(results, "folds")
    elif MODE == "seeds":
        dataset = NuclearCataractDataset(
            NuclearCataractDataset.TrainValMode(0.8, 0.2),
            cache_size=PRE_ROT_SIZE,
            hard_policy=HARD_POLICY,
        )
        train_sub = dataset.train_set()
        val_sub = dataset.val_set()
        for i, s in enumerate(SEEDS):
            print(f"\n--- Seed {s} ({i + 1}/{len(SEEDS)}) ---")
            with mlflow.start_run(run_name=f"seed-{s}", nested=True) as child_run:
                mlflow.set_tag("seed", s)
                m = train_and_validate(datamodule, train_sub, val_sub, s, best_params, child_run)
                mlflow.log_metrics(m)
            results.append(m)
        summarize(results, "seeds")
    else:
        raise ValueError(f"unknown MODE: {MODE}")

    summary_metrics = {}
    for k in results[0]:
        vals = np.array([r[k] for r in results])
        summary_metrics[f"mean_{k}"] = float(vals.mean())
        summary_metrics[f"std_{k}"] = float(vals.std())
    mlflow.log_metrics(summary_metrics)
