

import matplotlib.pyplot as plt
import optuna
import mlflow
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

from lib.mlflow_setup import Experiment, save_model
from experiments.sinus_prediction_test.objective import objective
from experiments.sinus_prediction_test.data import *
from experiments.sinus_prediction_test.model import MLP

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

n_finished = sum(1 for t in exp.study.trials if t.state in (TrialState.COMPLETE, TrialState.PRUNED))
try:
    threshold_met = exp.study.best_value <= MSE_THRESHOLD   # throws if no best_value was set yet
except ValueError:
    threshold_met = False
print(f"Study '{STUDY_NAME}' has {n_finished}/{MAX_TRIALS} finished trials.")

# Persist the parent run across restarts so all trials group under the same run
parent_run_id = exp.study.user_attrs.get("mlflow_parent_run_id")
if parent_run_id:
    parent_run = mlflow.start_run(run_id=parent_run_id)
else:
    parent_run = mlflow.start_run(run_name=STUDY_NAME)
    exp.study.set_user_attr("mlflow_parent_run_id", parent_run.info.run_id)
    mlflow.set_tag("optuna_study", STUDY_NAME)


if n_finished > MAX_TRIALS or threshold_met:
    print(f"Study already complete, skipping.")
    print(f"Best params: {exp.study.best_params}")
    print(f"Best value: {exp.study.best_value}")
else:
    print("Running the experiment")

    # It will sync with other runners as well as continue from an interrupted run
    # If the objective is already met (MAX_TRIALS or other callbacks)
    exp.study.optimize(
        lambda trial: objective(exp, trial),
        n_trials=MAX_TRIALS * 2,  # safeguard; MaxTrialsCallback is the real stop condition
        callbacks=[
            MaxTrialsCallback(MAX_TRIALS, states=(TrialState.COMPLETE, TrialState.PRUNED)),
            stop_on_threshold,
        ],
    )

    with mlflow.start_run(run_id=parent_run.info.run_id, nested=True):
        print("restored run" , parent_run.info.run_id)
        # Point to the best trial's existing run
        best_trial = exp.study.best_trial
        best_run_id = best_trial.user_attrs["mlflow_run_id"]

        mlflow.log_params(best_trial.params)
        mlflow.log_metric("best_mse", best_trial.value)
        mlflow.set_tag("best_trial_number", best_trial.number)
        mlflow.set_tag("best_trial_run_id", best_run_id)

        # retrain best model
        best_lr = exp.study.best_params["lr"]
        torch.manual_seed(SEED)
        best_model = MLP()
        best_optimizer = OPTIMIZER(best_model.parameters(), lr=best_lr)
        loss_fn_best = LOSS_FN()
        for _ in range(EPOCHS):
            best_optimizer.zero_grad()
            loss_fn_best(best_model(X), y).backward()
            best_optimizer.step()
        best_model.eval()
        save_model(best_model)

        # Study-level plots
        fig = plot_optimization_history(exp.study)
        mlflow.log_figure(fig.figure, "optimization_history.png")
        plt.close()

        fig = plot_param_importances(exp.study)
        mlflow.log_figure(fig.figure, "param_importances.png")
        plt.close()


