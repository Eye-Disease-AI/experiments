import time

from experiments.experiment.datamodules.default_nuclear_cataract import DataModule
import lightning as L
import mlflow
import optuna
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger

from experiments.experiment.best_snapshot import BestSnapshotCallback
from experiments.experiment import common_config
from experiments.lib.log_silencer import stop_logs
from experiments.lib.mlflow_setup import Experiment
from experiments.lib.reproducibility import global_seed_rng

stop_logs()


def create_trainer(run, optuna_callback=None, **kwargs):
    mlf_logger = MLFlowLogger(
        run_id=run.info.run_id, tracking_uri=mlflow.get_tracking_uri()
    )
    callbacks: list[Callback] = list(kwargs.pop("callbacks", []))

    if optuna_callback is not None:
        callbacks.append(optuna_callback)

    if common_config.USE_EARLY_STOPPING:
        callbacks.append(
            EarlyStopping(
                monitor=common_config.OPTUNA_METRIC,
                patience=common_config.EARLY_STOPPING_PATIENCE,
                mode=common_config.OPTUNA_DIRECTION,
            )
        )
    if common_config.USE_FREEZING:
        callbacks.append(
            BackboneFreezeCallback(
                monitor=common_config.OPTUNA_METRIC,
                patience=common_config.BACKBONE_UNFREEZE_PATIENCE,
                mode=common_config.OPTUNA_DIRECTION,
            )
        )
    defaults = dict(
        max_epochs=common_config.EPOCHS,
        accelerator="auto",
        logger=mlf_logger,
        callbacks=callbacks,
        enable_progress_bar=True,
        enable_model_summary=False,
        enable_checkpointing=False,  # checkpointing required if logging models
        log_every_n_steps=1,
        precision=common_config.GPU_PRECISION,
        deterministic=True,
    )
    defaults.update(kwargs)
    return L.Trainer(**defaults)  # pyright: ignore


class OptunaMLflowCallback(Callback):
    """Reports val_loss to Optuna each epoch and handles pruning + batched MLflow logging."""

    def __init__(
        self,
        trial: optuna.trial.Trial,
        exp: Experiment,
        run_id: str,
        log_every_n_epochs: int,
        optuna_metric: str,
        optuna_direction: str,
    ):
        self.trial = trial
        self.exp = exp
        self.run_id = run_id
        self.log_every_n_epochs = log_every_n_epochs
        self.buffer = []
        self.best_value = None
        self.optuna_metric = optuna_metric
        self.optuna_direction = optuna_direction

    def _flush(self):
        if self.buffer:
            self.exp.client.log_batch(
                self.run_id,
                metrics=[
                    mlflow.entities.Metric(self.optuna_metric, val, ts, e)  # type: ignore
                    for e, val, ts in self.buffer
                ],  # pyright: ignore
            )
            self.buffer.clear()

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if trainer.sanity_checking:
            return
        metric_val = trainer.callback_metrics.get(self.optuna_metric)
        if metric_val is None:
            return
        epoch = trainer.current_epoch
        self.buffer.append((epoch, metric_val.item(), int(time.time() * 1000)))

        if len(self.buffer) >= self.log_every_n_epochs:
            self._flush()

        v = metric_val.item()
        agg = min if self.optuna_direction == "min" else max
        self.best_value = agg(v, self.best_value) if self.best_value is not None else v

        self.trial.report(v, epoch)
        if self.trial.should_prune():
            self._flush()
            raise optuna.TrialPruned()

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        self._flush()


class BackboneFreezeCallback(Callback):
    """Freezes backbone at start; unfreezes it once the metric stops improving."""

    def __init__(self, monitor: str, patience: int, mode: str):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self._best = None
        self._wait = 0
        self._unfrozen = False

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        for m in pl_module.backbone_modules():  # pyright: ignore
            for p in m.parameters():
                p.requires_grad = False

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if trainer.sanity_checking:
            return
        if self._unfrozen:
            return
        val = trainer.callback_metrics.get(self.monitor)
        if val is None:
            return
        v = val.item()
        if common_config.BACKBONE_UNFREEZE_PATIENCE <= 0:
            if self._wait == common_config.BACKBONE_UNFREEZE_EPOCHS:
                self.unfreeze(trainer, pl_module)
            self._wait += 1
        else:
            improved = self._best is None or (
                v < self._best if self.mode == "min" else v > self._best
            )
            print(improved, self._wait)
            if improved:
                self._best = v
                self._wait = 0
            else:
                self._wait += 1
                if self._wait >= self.patience:
                    self.unfreeze(trainer, pl_module)

    def unfreeze(self, trainer: L.Trainer, pl_module: L.LightningModule):
        for m in pl_module.backbone_modules():  # pyright: ignore
            for p in m.parameters():
                p.requires_grad = True
            self._unfrozen = True
        print("### UNFREEZING")
        # reset early stopping patience after unfreezing
        for cb in trainer.callbacks:  # pyright: ignore
            if isinstance(cb, EarlyStopping):
                cb.wait_count = 0
                torch_inf = torch.tensor(torch.inf)
                cb.best_score = torch_inf if cb.monitor_op == torch.lt else -torch_inf


def objective(
    datamodule: DataModule,
    seed: int,
    exp: Experiment,
    trial: optuna.trial.Trial,
    optuna_metric: str,
    optuna_direction: str,
    ModelClass,
):
    global_seed_rng(seed)

    lr = trial.suggest_float("lr", 1e-5, 5e-5, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 5e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.3)
    batch_size = trial.suggest_categorical("batch_size", [64])

    datamodule.batch_size = batch_size  # pyright: ignore
    datamodule.setup(stage="fit")
    class_weights = datamodule.train_class_weights  # pyright: ignore
    model = ModelClass(
        n_classes=datamodule.n_classes,
        lr=lr,
        weight_decay=weight_decay,
        dropout=dropout,
        class_weights=class_weights,
    )  # pyright: ignore

    with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True) as run:
        trial.set_user_attr("mlflow_run_id", run.info.run_id)
        mlflow.set_tag("optuna_study", exp.study.study_name)  # pyright: ignore
        mlflow.set_tag("optuna_trial", trial.number)
        mlflow.set_tag("validation_sample", "true")
        mlflow.log_params(
            {
                "lr": lr,
                "weight_decay": weight_decay,
                "dropout": dropout,
                "batch_size": batch_size,
                "seed": common_config.SEED,
                "model": str(model),
            }
        )
        optuna_callback = OptunaMLflowCallback(
            trial=trial,
            exp=exp,
            run_id=run.info.run_id,
            log_every_n_epochs=common_config.LOG_EVERY_N_EPOCHS,
            optuna_metric=optuna_metric,
            optuna_direction=optuna_direction,
        )
        best_snapshot = BestSnapshotCallback()
        trainer = create_trainer(
            run, optuna_callback=optuna_callback, callbacks=[best_snapshot]
        )

        trainer.fit(model, datamodule=datamodule)

        return optuna_callback.best_value
