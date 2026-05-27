import lightning as L
import mlflow

from experiments.experiment import common_config


class BestSnapshotCallback(L.Callback):
    def __init__(self, prefix="best_"):
        self.prefix = prefix
        self.best_metrics = {}
        self.best_epoch = None
        self._best_target = None
        self._agg = max if common_config.OPTUNA_DIRECTION == "max" else min

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        v = trainer.callback_metrics.get(common_config.OPTUNA_METRIC)
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
