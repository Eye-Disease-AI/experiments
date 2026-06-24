import ast
import importlib
import argparse
import pkgutil
import re
from dataset.hard_policy import HardPolicy
import mlflow
import numpy as np
import optuna

from dataset.loader import NuclearCataractDataset, NuclearCataractSubset
from experiments.experiment.best_snapshot import BestSnapshotCallback
from experiments.experiment import common_config
from experiments.experiment.datamodules.nuclear_cataract_datamodule import (
    _SubsetTransformer,
    NuclearCataractDataModule,
)
from experiments.experiment.objective import create_trainer
from experiments.experiment.run_study import CONFIG_PARAMS
from experiments.lib.mlflow_setup import Experiment
from experiments.lib.reproducibility import get_git_sha, global_seed_rng
import experiments.experiment.models


"""
TODO - rewrite
Do we need it?
Nope, we don't even use it anywhere.
"""


def parse_logged(v):
    try:
        return ast.literal_eval(v)
    except ValueError, SyntaxError:
        return v


"""
TODO - rewrite
I don't think we need this anymore. We can load whole configs
from mlflow now. We do log json and we can then load it to
construct complete study config.
"""


def coerce_config(current, v):
    """
    current: The config value which we'd like to override,
        in common_config
    v: a string value of the config saved to mlflow which
        we'd like to convert into the original type

    mlflow saves all parameters as strings.
    Need to parse them to their original types
    """
    # For string do nothing
    if isinstance(current, str):
        return v
    # Most types can be interpreted using ast
    elif isinstance(current, (int, float, list, dict, tuple)):
        return ast.literal_eval(v)
    # Classes need to have their name extracted and be imported
    elif v.startswith("<class"):
        class_re = re.compile(r"<class '([\w.]+)'>")
        m = class_re.fullmatch(v)
        if not m:
            raise Exception(f"coerce_config: Failed to parse class: {v}")
        mod, _, name = m.group(1).rpartition(".")
        return getattr(importlib.import_module(mod), name)
    else:
        raise Exception("coerce_config: Unsupported type")


"""
TODO - rewrite
We are not guaranteed to have Nuclear Cataract datamodule anymore.
Moreover we can have multiple modules for a given study class, so
I think it could be the best that the study class implemented this logic.
"""


def datamodule_change_subsets(
    dm: NuclearCataractDataModule,
    train_sub: NuclearCataractSubset,
    val_sub: NuclearCataractSubset,
):
    dm.train_set = _SubsetTransformer(train_sub, transform=dm.transform)
    dm.val_set = _SubsetTransformer(val_sub, transform=dm.val_transform)
    dm.train_class_weights = train_sub.class_weights()
    return dm


"""
TODO - rewrite
Hmm... here again we depend on the fact we have one specific study.
So this is study specific too, so it should probably be in the study class.
I am thinking what if we divided studies into two kinds:

* KFoldCValidatable
* SeedValidatable

and let child classes decide if they want to implement the required methods.
I don't know if Python has interfaces, but these could also be ABC or inherit
Study, which is ABC.

Question: What is BestSnapshotCallback for?
"""


def train_and_validate(
    ModelClass, dm: NuclearCataractDataModule, best_params, run, snapshot_prefix=None
):
    params = {k: v for k, v in best_params.items() if k != "batch_size"}
    model = ModelClass(
        dm.dataset.n_classes, **params, class_weights=dm.train_class_weights
    )
    if snapshot_prefix is not None:
        best_cb = BestSnapshotCallback(snapshot_prefix)
    else:
        best_cb = BestSnapshotCallback()
    trainer = create_trainer(
        run,
        max_epochs=common_config.EPOCHS,
        precision=common_config.GPU_PRECISION,
        callbacks=[best_cb],
    )
    trainer.fit(model, datamodule=dm)
    return best_cb.best_metrics


def summarize(results, name):
    print(f"\n=== Summary across {len(results)} {name} ===")
    for k in sorted(results[0]):
        vals = np.array([r[k] for r in results])
        print(
            f"  {k}: mean={vals.mean():.4f}  std={vals.std():.4f}  "
            f"min={vals.min():.4f}  max={vals.max():.4f}"
        )


def load_study_hparams(study):
    # Loading configs
    logged_config = study.user_attrs.get("config") or {}
    config_changed = {
        k: (logged_config.get(k), CONFIG_PARAMS.get(k))
        for k in set(logged_config) | set(CONFIG_PARAMS)
        if logged_config.get(k) != CONFIG_PARAMS.get(k)
    }
    if config_changed:
        print("WARNING: common_config changed since training:")
        for k, (logged, current) in config_changed.items():
            print(f"  {k}: logged={logged}  current={current}")
    # Replace common_config values with the logged ones
    # Required until we pass all hyperparameters instead of using globals
    for k, v in logged_config.items():
        if not hasattr(common_config, k):
            continue
        setattr(common_config, k, coerce_config(getattr(common_config, k), v))

    ModelClass = getattr(experiments.experiment.models, study.user_attrs["model_class"])
    best_params = study.best_params
    _agg = min if common_config.OPTUNA_DIRECTION == "min" else max
    best_epoch = _agg(
        study.best_trial.intermediate_values,
        key=study.best_trial.intermediate_values.get,  # pyright: ignore[reportArgumentType]
    )  # type: ignore
    hard_policy: str = str(study.user_attrs.get("hard_policy"))
    print(
        f"best_params={best_params} best_epoch={best_epoch} hard_policy={hard_policy}"
    )
    return ModelClass, best_params, best_epoch, hard_policy


def run_kfold_validation(
    ModelClass, dm, best_params, hard_policy, seed, validation_study_name, K
):
    results = []
    dataset = NuclearCataractDataset(
        NuclearCataractDataset.KFoldCVMode(K),
        cache_size=common_config.CACHE_SIZE,
        hard_policy=HardPolicy[hard_policy],
    )
    for i in range(K):
        train_sub = dataset.fold_train_set(i)
        val_sub = dataset.fold_val_set(i)
        print(f"\n--- Fold {i + 1}/{K}  train={len(train_sub)} val={len(val_sub)} ---")
        with mlflow.start_run(run_name=f"fold-{i}", nested=True) as child_run:
            mlflow.set_tag("fold", i)
            mlflow.set_tag("optuna_study", validation_study_name)
            mlflow.set_tag("validation_sample", "true")
            global_seed_rng(seed)
            dm = datamodule_change_subsets(dm, train_sub, val_sub)
            m = train_and_validate(ModelClass, dm, best_params, child_run)
        results.append(m)
    return results


def run_seeds_validation(
    ModelClass, dm, best_params, hard_policy, seed, validation_study_name, num_seeds
):
    results = []
    dataset = NuclearCataractDataset(
        NuclearCataractDataset.TrainValMode(0.8, 0.2),
        cache_size=common_config.CACHE_SIZE,
        hard_policy=HardPolicy[hard_policy],
    )
    train_sub = dataset.train_set()
    val_sub = dataset.val_set()
    seeds = list(range(num_seeds))
    for i, s in enumerate(seeds):
        print(f"\n--- Seed {s} ({i + 1}/{len(seeds)}) ---")
        with mlflow.start_run(run_name=f"seed-{s}", nested=True) as child_run:
            mlflow.set_tag("seed", s)
            mlflow.set_tag("optuna_study", validation_study_name)
            mlflow.set_tag("validation_sample", "true")
            global_seed_rng(s)
            dm = datamodule_change_subsets(dm, train_sub, val_sub)
            m = train_and_validate(ModelClass, dm, best_params, child_run)
        results.append(m)
    return results


def main():
    for _, _name, _ in pkgutil.iter_modules(experiments.experiment.models.__path__):
        importlib.import_module(f"experiments.experiment.models.{_name}")
    SHA = get_git_sha()

    parser = argparse.ArgumentParser()
    parser.add_argument("study_to_verify", type=str)
    parser.add_argument("--mode", type=str, choices=["kfold", "seeds"], default="kfold")
    parser.add_argument("-K", type=int, default=5)
    parser.add_argument("--num_seeds", type=int, default=10)
    args = parser.parse_args()
    validation_study_name = f"{args.study_to_verify}/validation_{args.mode}_" + SHA

    exp = Experiment(common_config.EXPERIMENT_NAME)
    study = optuna.load_study(study_name=args.study_to_verify, storage=exp.storage)

    ModelClass, best_params, best_epoch, hard_policy = load_study_hparams(study)

    # Training init
    seed = global_seed_rng(common_config.SEED)
    datamodule = NuclearCataractDataModule(
        seed, batch_size=best_params["batch_size"], hard_policy=HardPolicy[hard_policy]
    )
    datamodule.setup(stage="fit")

    with mlflow.start_run(run_name=validation_study_name):
        mlflow.set_tag("study_name", args.study_to_verify)
        mlflow.set_tag("model_class", ModelClass.__name__)
        mlflow.set_tag("mode", args.mode)
        mlflow.log_params(
            {
                **best_params,
                "hard_policy": hard_policy,
                "seed": common_config.SEED,
                **(
                    {"K": args.K}
                    if args.mode == "kfold"
                    else {"n_seeds": args.num_seeds}
                ),
            }
        )
        mlflow.log_metric("best_epoch", best_epoch)
        best_trial_run_id = study.best_trial.user_attrs.get("mlflow_run_id")
        if best_trial_run_id:
            best_run = exp.client.get_run(best_trial_run_id)
            for k, v in best_run.data.metrics.items():
                if k.startswith("best_val_"):
                    mlflow.log_metric(k, v)

        with mlflow.start_run(run_name="retrain", nested=True) as retrain_run:
            mlflow.set_tag("optuna_study", validation_study_name)
            train_and_validate(
                ModelClass,
                datamodule,
                best_params,
                retrain_run,
                snapshot_prefix="retrain_",
            )

        if args.mode == "kfold":
            results = run_kfold_validation(
                ModelClass,
                datamodule,
                best_params,
                hard_policy,
                seed,
                validation_study_name,
                args.K,
            )
        elif args.mode == "seeds":
            results = run_seeds_validation(
                ModelClass,
                datamodule,
                best_params,
                hard_policy,
                seed,
                validation_study_name,
                args.num_seeds,
            )
        else:
            raise ValueError(f"unknown MODE: {args.mode}")
        summarize(results, args.mode)


if __name__ == "__main__":
    main()
