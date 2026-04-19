import time
import optuna
import mlflow
import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import Callback, EarlyStopping
from lib.seed import RNG
from lib.mlflow_setup import Experiment
from .models.convnext import ConvNext
from .common_config import SEED, EPOCHS, LOG_EVERY_N_EPOCHS, GPU_PRECISION, EARLY_STOPPING_PATIENCE, OPTUNA_METRIC, OPTUNA_DIRECTION
from log_silencer import stop_logs
stop_logs()


class OptunaMLflowCallback(Callback):
    """Reports val_loss to Optuna each epoch and handles pruning + batched MLflow logging."""
    def __init__(self, trial: optuna.trial.Trial, exp: Experiment, run_id: str, log_every_n_epochs: int):
        self.trial = trial
        self.exp = exp
        self.run_id = run_id
        self.log_every_n_epochs = log_every_n_epochs
        self.buffer = []

    def _flush(self):
        if self.buffer:
            self.exp.client.log_batch(
                self.run_id,
                metrics=[mlflow.entities.Metric(OPTUNA_METRIC, l, ts, e) for e, l, ts in self.buffer],
            )
            self.buffer.clear()

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningDataModule):
        metric_val = trainer.callback_metrics.get(OPTUNA_METRIC)
        if metric_val is None:
            return
        epoch = trainer.current_epoch
        self.buffer.append((epoch, metric_val.item(), int(time.time() * 1000)))

        if len(self.buffer) >= self.log_every_n_epochs:
            self._flush()

        self.trial.report(metric_val.item(), epoch)
        if self.trial.should_prune():
            self._flush()
            raise optuna.TrialPruned()

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningDataModule):
        self._flush()


def objective(datamodule: L.LightningDataModule, rng: RNG, exp: Experiment, trial: optuna.trial.Trial, ModelClass):
    rng.set_seed(SEED)

    lr           = trial.suggest_float("lr",           1e-4, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    dropout      = trial.suggest_float("dropout",      0.0,  0.5)
    batch_size   = trial.suggest_categorical("batch_size", [128])

    datamodule.batch_size = batch_size
    datamodule.setup(stage="fit")
    model = ModelClass(n_classes=datamodule.dataset.n_classes, lr=lr, weight_decay=weight_decay, dropout=dropout)

    with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True) as run:
        mlflow.set_tag("optuna_study", exp.study.study_name)
        mlflow.set_tag("optuna_trial", trial.number)
        mlflow.log_params({"lr": lr, "weight_decay": weight_decay, "dropout": dropout, "batch_size": batch_size, "seed": SEED, "model": str(model)})

        mlf_logger = MLFlowLogger(run_id=run.info.run_id, tracking_uri=mlflow.get_tracking_uri())
        pruning_cb = OptunaMLflowCallback(
            trial=trial,
            exp=exp,
            run_id=run.info.run_id,
            log_every_n_epochs=LOG_EVERY_N_EPOCHS,
        )
        early_stop_cb = EarlyStopping(monitor=OPTUNA_METRIC, patience=EARLY_STOPPING_PATIENCE, mode=OPTUNA_DIRECTION)

        trainer = L.Trainer(
            max_epochs=EPOCHS,
            accelerator="auto",
            logger=mlf_logger,
            callbacks=[pruning_cb, early_stop_cb],
            enable_progress_bar=True,
            enable_model_summary=False,
            log_every_n_steps=1,
            precision=GPU_PRECISION,
        )

        trainer.fit(model, datamodule=datamodule)

    agg = min if OPTUNA_DIRECTION == "min" else max
    return agg(trial.intermediate_values.values())