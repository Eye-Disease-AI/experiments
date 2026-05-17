import lightning as L
import mlflow

from experiment.common_config import OPTUNA_DIRECTION, OPTUNA_METRIC


class BestSnapshotCallback(L.Callback):
    def __init__(self, prefix="best_"):
        self.prefix = prefix
        self.best_metrics = {}
        self.best_epoch = None
        self._best_target = None
        self._agg = max if OPTUNA_DIRECTION == "max" else min

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        v = trainer.callback_metrics.get(OPTUNA_METRIC)
        if v is None:
            return
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
            if not hasattr(val, "item"):
                continue
            snap[k] = val.item()
        self.best_metrics = snap

    def on_fit_end(self, trainer, pl_module):
        if not self.best_metrics:
            return
        mlflow.log_metrics(
            {f"{self.prefix}{k}": v for k, v in self.best_metrics.items()}
        )
