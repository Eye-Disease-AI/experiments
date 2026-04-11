import time
from lightning import LightningDataModule
import mlflow
import matplotlib.pyplot as plt
import optuna
from tqdm import tqdm

from lib.mlflow_setup import Experiment
from .common_config import *
from .data import MyDataset, MyDataModule
from .model import Model
from lib.seed import RNG

# Objective, here we define all the hyperparams to search for
def objective(datamodule: LightningDataModule, rng: RNG, exp: Experiment, trial: optuna.trial.Trial):
    # Seed each trial from the start to make them independent
    rng.set_seed(SEED)

    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datamodule.setup(stage="fit")
    model = Model(datamodule.dataset.n_classes).to(device)
    train_loader = datamodule.train_dataloader()
    val_loader   = datamodule.val_dataloader()
    optimizer = OPTIMIZER(model.parameters(), lr=lr)
    loss_fn = LOSS_FN()

    with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True) as run:
        mlflow.set_tag("optuna_study", STUDY_NAME)
        mlflow.log_params({"lr": lr, "seed": SEED, "model": str(model)})
        
        buffer = []
        def flush_buffer():
            if buffer:
                exp.srv.client.log_batch(
                    run.info.run_id,
                    metrics=[mlflow.entities.Metric("val_loss", l, timestamp=ts, step=e) for e, l, ts in buffer],
                )
                buffer.clear()

        for epoch in tqdm(range(EPOCHS)):
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in tqdm(train_loader, leave=False):
                optimizer.zero_grad()
                loss = loss_fn(model(X_batch.to(device)), y_batch.to(device))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    val_loss += loss_fn(model(X_batch.to(device)), y_batch.to(device)).item()
            val_loss /= len(val_loader)

            buffer.append((epoch, val_loss, int(time.time() * 1000)))

            if len(buffer) >= LOG_EVERY_N_EPOCHS:
                flush_buffer()

            trial.report(val_loss, epoch)

            # Optuna might decide that this run is worthless, e.g. worse than the current median
            if trial.should_prune():
                mlflow.set_tag("pruned", "true")
                flush_buffer()
                raise optuna.TrialPruned()

        flush_buffer()

    return val_loss