import lightning as L
from typing import Literal
import optuna
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import torch
from experiments.lib.mlflow_setup import Experiment
import time
import mlflow


class BestSnapshotCallback(L.Callback):
    def __init__(self, metric: str, direction: Literal["min", "max"], prefix="best_"):
        self.prefix = prefix
        self.best_metrics = {}
        self.optimised_metric = metric
        self.best_epoch = None
        self._best_target = None
        self._agg = max if direction == "max" else min

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        v = trainer.callback_metrics.get(self.optimised_metric)
        v = v.item()
        improved = (
            self._best_target is None
            or self._agg(v, self._best_target) != self._best_target
        )
        if not improved:
            return
        self._best_target = v
        self.best_epoch = trainer.current_epoch
        snap = {}
        for k, val in trainer.callback_metrics.items():
            if not k.startswith("val_"):
                continue
            snap[k] = val.item()
        self.best_metrics = snap

    def on_fit_end(self, trainer, pl_module):
        mlflow.log_metrics(
            {f"{self.prefix}{k}": v for k, v in self.best_metrics.items()}
        )


class OptunaMLflowCallback(L.Callback):
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


class BackboneFreezeCallback(L.Callback):
    """
    Freezes backbone at start;
    Unfreezes if:
    - for_epochs epochs pass - if patience is None
    - if no improvement for patience epochs - if for_epochs is None
    """

    def __init__(
        self,
        num_epochs: int,
        monitor: str = "val_loss",
        direction: Literal["min", "max"] = "min",
        mode: Literal["const_epochs", "patience"] = "const_epochs",
    ):
        self.monitor = monitor
        self.num_epochs = num_epochs
        self.direction = direction
        self.mode = mode
        self._best = None
        self._wait = 0
        self._unfrozen = False

        if self.mode == "const_epochs":
            self.routine = self.const_epochs_routine
        elif self.mode == "patience":
            self.routine = self.patience_routine

    def const_epochs_routine(
        self, val, trainer: L.Trainer, pl_module: L.LightningModule
    ):
        if self._wait == self.num_epochs:
            self.unfreeze(trainer, pl_module)
        self._wait += 1

    def patience_routine(self, val, trainer: L.Trainer, pl_module: L.LightningModule):
        improved = self._best is None or (
            val < self._best if self.mode == "min" else val > self._best
        )
        print(improved, self._wait)
        if improved:
            self._best = val
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                self.unfreeze(trainer, pl_module)

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        for m in pl_module.backbone_modules():
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
        self.routine(val.item(), trainer, pl_module)

    def unfreeze(self, trainer: L.Trainer, pl_module: L.LightningModule):
        for m in pl_module.backbone_modules():
            for p in m.parameters():
                p.requires_grad = True
            self._unfrozen = True
        print("### UNFREEZING")
        # reset early stopping patience after unfreezing
        for cb in trainer.callbacks:
            if isinstance(cb, EarlyStopping):
                cb.wait_count = 0
                torch_inf = torch.tensor(torch.inf)
                cb.best_score = torch_inf if cb.monitor_op == torch.lt else -torch_inf
