

import matplotlib.pyplot as plt
from mlflow import data
import optuna
import mlflow
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_param_importances,
)

# Our utils and config
from experiments.nuclear_cataract.common_config import *
from lib.seed import RNG
rng = RNG()
rng.set_seed(SEED)

from lib.mlflow_setup import Experiment, save_model
from experiments.nuclear_cataract.objective import objective
from experiments.nuclear_cataract.data import MyDataset, MyDataModule
from experiments.nuclear_cataract.model import Model

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
    datamodule = MyDataModule(rng)
    datamodule.prepare_data()

    # It will sync with other runners as well as continue from an interrupted run
    # If the objective is already met (MAX_TRIALS or other callbacks)
    exp.study.optimize(
        lambda trial: objective(datamodule, rng, exp, trial),
        n_trials=MAX_TRIALS * 2,  # safeguard; MaxTrialsCallback is the real stop condition
        callbacks=[
            MaxTrialsCallback(MAX_TRIALS, states=(TrialState.COMPLETE, TrialState.PRUNED)),
            stop_on_threshold,
        ],
    )

    #with mlflow.start_run(run_id=parent_run.info.run_id, nested=True):
    print("Repeating best run: " , parent_run.info.run_id)
    # Point to the best trial's existing run
    best_trial = exp.study.best_trial

    mlflow.log_params(best_trial.params)
    mlflow.log_metric("best_mse", best_trial.value)
    mlflow.set_tag("best_trial_number", best_trial.number)

    # retrain best model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_lr = exp.study.best_params["lr"]
    torch.manual_seed(SEED)
    datamodule.setup(stage="fit")
    best_model = Model(datamodule.dataset.n_classes).to(device)
    best_optimizer = OPTIMIZER(best_model.parameters(), lr=best_lr)
    loss_fn_best = LOSS_FN()
    train_loader = datamodule.train_dataloader()
    
    # Train
    for _ in range(EPOCHS):
        best_model.train()
        for X_batch, y_batch in train_loader:
            best_optimizer.zero_grad()
            loss_fn_best(best_model(X_batch.to(device)), y_batch.to(device)).backward()
            best_optimizer.step()

    # Eval
    val_dataloader = datamodule.val_dataloader()
    best_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_dataloader:
            val_loss += loss_fn_best(best_model(X_batch.to(device)), y_batch.to(device)).item()
    val_loss /= len(val_dataloader)

    mlflow.log_metric("retrain_test_mse", val_loss)
    save_model(best_model)


    # Study-level plots
    fig = plot_optimization_history(exp.study)
    mlflow.log_figure(fig.figure, "optimization_history.png")
    plt.close()

    fig = plot_param_importances(exp.study)
    mlflow.log_figure(fig.figure, "param_importances.png")
    plt.close()


