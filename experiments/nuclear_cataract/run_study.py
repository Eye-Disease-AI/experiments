import optuna
import mlflow
import matplotlib.pyplot as plt
import lightning as L
from optuna.trial import TrialState
from optuna.study import MaxTrialsCallback
from lightning.pytorch.loggers import MLFlowLogger
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_param_importances,
)
from lib.mlflow_setup import Experiment, save_model
from lib.seed import RNG
from .objective import objective
from .common_config import SEED, EPOCHS, MAX_TRIALS, MSE_THRESHOLD


def run_study(ModelClass, study_name: str, exp: Experiment, rng: RNG, datamodule: L.LightningDataModule):
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=exp.storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )

    n_finished = sum(1 for t in study.trials if t.state in (TrialState.COMPLETE, TrialState.PRUNED))
    print(f"Study '{study_name}': {n_finished}/{MAX_TRIALS} finished trials.")

    parent_run_id = study.user_attrs.get("mlflow_parent_run_id")
    if parent_run_id:
        parent_run = mlflow.start_run(run_id=parent_run_id)
    else:
        parent_run = mlflow.start_run(run_name=study_name)
        study.set_user_attr("mlflow_parent_run_id", parent_run.info.run_id)
        mlflow.set_tag("optuna_study", study_name)
        mlflow.set_tag("model", ModelClass.__name__)

    if n_finished >= MAX_TRIALS:
        print(f"Study already complete — best: {study.best_params}, val_loss: {study.best_value:.4f}")
        mlflow.end_run()
        return

    study.optimize(
        lambda trial: objective(datamodule, rng, exp, trial, ModelClass),
        n_trials=MAX_TRIALS * 2,
        callbacks=[
            MaxTrialsCallback(MAX_TRIALS, states=(TrialState.COMPLETE, TrialState.PRUNED))
        ],
    )

    best_trial = study.best_trial
    mlflow.log_params(best_trial.params)
    mlflow.log_metric("best_val_loss", best_trial.value)
    mlflow.set_tag("best_trial_number", best_trial.number)

    best_params = study.best_params
    rng.set_seed(SEED)
    datamodule.batch_size = best_params["batch_size"]
    datamodule.setup(stage="fit")
    model_params = {k: v for k, v in best_params.items() if k != "batch_size"}
    best_model = ModelClass(datamodule.dataset.n_classes, **model_params)

    trainer = L.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=MLFlowLogger(run_id=parent_run.info.run_id, tracking_uri=mlflow.get_tracking_uri()),
    )
    trainer.fit(best_model, datamodule=datamodule)

    mlflow.log_metric("retrain_val_loss", trainer.callback_metrics["val_loss"].item())
    save_model(best_model)

    fig = plot_optimization_history(study)
    mlflow.log_figure(fig.figure, "optimization_history.png")
    plt.close()

    fig = plot_param_importances(study)
    mlflow.log_figure(fig.figure, "param_importances.png")
    plt.close()

    mlflow.end_run()