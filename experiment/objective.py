import time

import lightning as L
import mlflow
import optuna
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger

from experiment.common_config import (
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    GPU_PRECISION,
    LOG_EVERY_N_EPOCHS,
    OPTUNA_DIRECTION,
    OPTUNA_METRIC,
    SEED,
)
from lib.log_silencer import stop_logs
from lib.mlflow_setup import Experiment
from lib.seed import RNG

stop_logs()


class OptunaMLflowCallback(Callback):
    """Reports val_loss to Optuna each epoch and handles pruning + batched MLflow logging."""

    def __init__(
        self,
        trial: optuna.trial.Trial,
        exp: Experiment,
        run_id: str,
        log_every_n_epochs: int,
    ):
        self.trial = trial
        self.exp = exp
        self.run_id = run_id
        self.log_every_n_epochs = log_every_n_epochs
        self.buffer = []
        self.best_value = None

    def _flush(self):
        if self.buffer:
            self.exp.client.log_batch(
                self.run_id,
                metrics=[
                    mlflow.entities.Metric(OPTUNA_METRIC, val, ts, e)
                    for e, val, ts in self.buffer
                ],  # pyright: ignore
            )
            self.buffer.clear()

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        metric_val = trainer.callback_metrics.get(OPTUNA_METRIC)
        if metric_val is None:
            return
        epoch = trainer.current_epoch
        self.buffer.append((epoch, metric_val.item(), int(time.time() * 1000)))

        if len(self.buffer) >= self.log_every_n_epochs:
            self._flush()

        v = metric_val.item()
        agg = min if OPTUNA_DIRECTION == "min" else max
        self.best_value = agg(v, self.best_value) if self.best_value is not None else v

        self.trial.report(v, epoch)
        if self.trial.should_prune():
            self._flush()
            raise optuna.TrialPruned()

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        self._flush()

def objective(
    datamodule: L.LightningDataModule,
    rng: RNG,
    exp: Experiment,
    trial: optuna.trial.Trial,
    ModelClass,
):
    rng.set_seed(SEED)

    lr = trial.suggest_float("lr", 1e-5, 1e-5, log=True)
    weight_decay = 1e-8#trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    dropout = 0.5#trial.suggest_float("dropout", 0.0, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [32])

    datamodule.batch_size = batch_size  # pyright: ignore
    datamodule.setup(stage="fit")
    class_weights = datamodule.train_class_weights  # pyright: ignore
    model = ModelClass(
        n_classes=datamodule.dataset.n_classes,
        lr=lr,
        weight_decay=weight_decay,
        dropout=dropout,
        class_weights=class_weights,
    )  # pyright: ignore

    with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True) as run:
        trial.set_user_attr("mlflow_run_id", run.info.run_id)
        mlflow.set_tag("optuna_study", exp.study.study_name)  # pyright: ignore
        mlflow.set_tag("optuna_trial", trial.number)
        mlflow.log_params(
            {
                "lr": lr,
                "weight_decay": weight_decay,
                "dropout": dropout,
                "batch_size": batch_size,
                "seed": SEED,
                "model": str(model),
            }
        )

        mlf_logger = MLFlowLogger(
            run_id=run.info.run_id, tracking_uri=mlflow.get_tracking_uri()
        )
        pruning_cb = OptunaMLflowCallback(
            trial=trial,
            exp=exp,
            run_id=run.info.run_id,
            log_every_n_epochs=LOG_EVERY_N_EPOCHS,
        )
        early_stop_cb = EarlyStopping(
            monitor=OPTUNA_METRIC,
            patience=EARLY_STOPPING_PATIENCE,
            mode=OPTUNA_DIRECTION,
        )

        trainer = L.Trainer(
            max_epochs=EPOCHS,
            accelerator="auto",
            logger=mlf_logger,
            callbacks=[pruning_cb, early_stop_cb],
            enable_progress_bar=True,
            enable_model_summary=False,
            enable_checkpointing=False,
            log_every_n_steps=1,
            precision=GPU_PRECISION,
            deterministic=True,
        )

        trainer.fit(model, datamodule=datamodule)

        return pruning_cb.best_value
