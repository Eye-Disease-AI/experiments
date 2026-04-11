

import matplotlib.pyplot as plt
import optuna
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_param_importances,
)

# Our utils and config
from experiments.sinus_prediction_test.common_config import *
from lib.seed import RNG
rng = RNG()
rng.enable_determinism(SEED)

from lib.mlflow_setup import Experiment
from experiments.sinus_prediction_test.objective import objective
from experiments.sinus_prediction_test.data import *

# Setup
exp = Experiment(EXPERIMENT_NAME)
exp.set_study(optuna.create_study(
    study_name=STUDY_NAME,
    direction="minimize",
    storage=exp.srv.storage,
    load_if_exists=True,
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
))

# Custom optuna callback example
# Stop if we reach an MSE threshold
def stop_on_threshold(study, trial):
    if study.best_value <= MSE_THRESHOLD:
        study.stop()


# CORE: this will run the experiment
# It will sync with other runners as well as continue from an interrupted run
# If the objective is already met (MAX_TRIALS or other callbacks)
n_finished = sum(1 for t in exp.study.trials if t.state in (TrialState.COMPLETE, TrialState.PRUNED))
print(f"Study '{STUDY_NAME}' has {n_finished}/{MAX_TRIALS} finished trials.")

with mlflow.start_run(run_name=STUDY_NAME) as parent_run:
    mlflow.set_tag("optuna_study", STUDY_NAME)

    if n_finished < MAX_TRIALS:
        exp.study.optimize(
            lambda trial: objective(exp, trial),
            n_trials=None,
            callbacks=[
                MaxTrialsCallback(MAX_TRIALS, states=(TrialState.COMPLETE, TrialState.PRUNED)),
                stop_on_threshold,
            ],
        )
    else:
        print(f"Study already complete, skipping optimization.")

    # Point to the best trial's existing run
    best_trial = exp.study.best_trial
    best_run_id = best_trial.user_attrs["mlflow_run_id"]

    mlflow.log_params(best_trial.params)
    mlflow.log_metric("best_mse", best_trial.value)
    mlflow.set_tag("best_trial_number", best_trial.number)
    mlflow.set_tag("best_trial_run_id", best_run_id)

    # Register the best model from the existing trial run in Model Registry
    model_version = mlflow.register_model(
        model_uri=f"runs:/{best_run_id}/best_model",
        name=STUDY_NAME,
    )
    mlflow.set_tag("registered_model_name", model_version.name)
    mlflow.set_tag("registered_model_version", model_version.version)

    # Study-level plots
    fig = plot_optimization_history(exp.study)
    mlflow.log_figure(fig.figure, "optimization_history.png")
    plt.close()

    fig = plot_param_importances(exp.study)
    mlflow.log_figure(fig.figure, "param_importances.png")
    plt.close()

print(f"Best params: {exp.study.best_params}")
print(f"Best value: {exp.study.best_value}")
