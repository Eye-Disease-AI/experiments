
import os
import time
import mlflow
import matplotlib.pyplot as plt
import optuna
from tqdm import tqdm

from experiments.sinus_prediction_test.common_config import *
from experiments.sinus_prediction_test.data import *
from lib.mlflow_setup import Experiment
from model import MLP


# Objective, here we define all the hyperparams to search for
def objective(exp: Experiment, trial: optuna.trial.Trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    model = MLP()
    optimizer = OPTIMIZER(model.parameters(), lr=lr)
    loss_fn = LOSS_FN()

    run_start = int(time.time() * 1000)
    buffer = []
    def flush_buffer():
        if buffer:
            exp.srv.client.log_batch(
                run.info.run_id,
                metrics=[mlflow.entities.Metric("loss", l, timestamp=run_start, step=e) for e, l in buffer],
            )
            buffer.clear()

    with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True) as run:
        mlflow.set_tag("optuna_study", STUDY_NAME)
        mlflow.log_params({"lr": lr, "seed": SEED})

        for epoch in tqdm(range(EPOCHS)):
            optimizer.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            optimizer.step()
            buffer.append((epoch, loss.cpu().detach()))

            if len(buffer) >= LOG_EVERY_N_EPOCHS:
                flush_buffer()

            trial.report(loss.cpu().detach(), epoch)
            
            # Optuna might decide that this run is worthless, e.g. worse than the current median
            if trial.should_prune():
                mlflow.set_tag("pruned", "true")
                flush_buffer()
                raise optuna.TrialPruned()

        flush_buffer()

        with torch.no_grad():
            mse = loss_fn(model(X), y).item()
            y_pred = model(X).squeeze().numpy()

        exp.srv.client.log_batch(
            run.info.run_id,
            metrics=[mlflow.entities.Metric("mse", mse, timestamp=run_start, step=0)],
        )

        # prediction plot
        x_np = X.squeeze().numpy()
        fig, ax = plt.subplots()
        ax.plot(x_np, y.squeeze().numpy(), label="sin(x)", linewidth=2)
        ax.plot(x_np, y_pred, label=f"predicted (mse={mse:.4f})", linestyle="--")
        ax.legend()
        ax.set_title(f"trial-{trial.number}  lr={lr:.2e}")
        mlflow.log_figure(fig, "prediction.png")
        plt.close(fig)

        exp.srv.save_model(model)
        trial.set_user_attr("mlflow_run_id", run.info.run_id)

    return mse