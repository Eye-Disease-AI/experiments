import lightning as L
import mlflow
import optuna
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger

from experiments.experiment.objective import BackboneFreezeCallback, OptunaMLflowCallback
from experiments.experiment.data import DataModule
from experiments.experiment.models.convnext import ConvNext
from experiments.experiment.studies.base_study import BaseStudy


class ConvNextStudy(BaseStudy):
    base_name = "convnext-study"

    def __init__(
        self,
        experiment_name: str,
        seed: int,
        max_trials: int,
        data_module: DataModule,
        max_epochs: int,
    ):
        super().__init__(
            experiment_name=experiment_name,
            seed=seed,
            base_name=self.base_name,
            data_module=data_module,
            max_trials=max_trials,
        )
        self.max_epochs = max_epochs

    def _suggest_params(self, trial: optuna.Trial) -> dict:
        return {
            "lr": trial.suggest_float("lr", 1e-6, 1e-4, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-8, 5e-2, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "batch_size": trial.suggest_categorical("batch_size", [64]),
        }

    def _configure_data_module(self, params: dict) -> None:
        # TODO: Is this best way to do that?
        # I would say it is better to manually create dataset and reuse it to create dataloaders.
        # In other words: just pass dataset to the datamodule class, so it will reuse caches.
        self._data_module.set_batch_size(params["batch_size"])

    def _train(
        self, params: dict, run: mlflow.ActiveRun, optuna_callback: OptunaMLflowCallback
    ) -> None:
        model = ConvNext(
            n_classes=self._data_module.get_n_classes(),
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            dropout=params["dropout"],
            class_weights=self._data_module.get_train_class_weights(),
        )

        callbacks: list[L.Callback] = [optuna_callback]

        mlf_logger = MLFlowLogger(
            run_id=run.info.run_id,
            tracking_uri=mlflow.get_tracking_uri(),
            prefix="ConvNext",
        )

        early_stopping_patience = 5
        backbone_unfreeze_patience = 5
        use_early_stopping = True
        use_freezing = False

        if use_early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=early_stopping_patience,
                    mode="min",
                )
            )

        if use_freezing:
            callbacks.append(
                BackboneFreezeCallback(
                    monitor="val_loss",
                    patience=backbone_unfreeze_patience,
                    mode="min",
                )
            )

        trainer = L.Trainer(
            max_epochs=self.max_epochs,
            accelerator="auto",
            logger=mlf_logger,
            callbacks=callbacks,
            enable_progress_bar=True,
            enable_model_summary=False,
            enable_checkpointing=False,
            log_every_n_steps=1,
            precision="bf16-mixed",
            deterministic=True,
        )

        trainer.fit(model, datamodule=self._data_module)
