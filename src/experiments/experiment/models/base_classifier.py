from dataclasses import dataclass
from importlib.metadata import PackagePath
import os
import tempfile

import lightning as L
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.optim import AdamW
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAUROC,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

from experiments.lib.config_serializing import (
    ClassConfig,
    deserialize_class,
    serialize_class,
)


@dataclass(frozen=True, kw_only=True)
class BaseClassifierModelConfig(ClassConfig):
    optimizer: PackagePath = serialize_class(AdamW)
    loss_fn: PackagePath = serialize_class(nn.CrossEntropyLoss)
    scheduler: PackagePath | None = None # serialize_class(torch.optim.lr_scheduler.CosineAnnealingLR)
    scheduler_max_t: int = 40
    scheduler_min_lr: float = 1e-6
    backbone_lr_factor: float = 1
    n_classes: int = 2
    class_weights: list[float] | None = None
    learning_rate: float = 5e-5
    weight_decay: float | None = 1e-6


class BaseClassifierModel(L.LightningModule):
    def __init__(self, config: BaseClassifierModelConfig):
        super().__init__()
        self.config = config
        self.loss_fn = deserialize_class(self.config.loss_fn)(self.config.class_weights)
        self.optimizer = deserialize_class(self.config.optimizer)
        self.scheduler = deserialize_class(self.config.scheduler)
        self.val_metrics = MetricCollection(
            {
                "val_precision": MulticlassPrecision(
                    num_classes=self.config.n_classes, average="macro"
                ),
                "val_recall": MulticlassRecall(
                    num_classes=self.config.n_classes, average="macro"
                ),
                "val_auroc": MulticlassAUROC(
                    num_classes=self.config.n_classes, average="macro"
                ),
                "val_f1": MulticlassF1Score(
                    num_classes=self.config.n_classes, average="macro"
                ),
            }
        )
        self._val_probs: list[torch.Tensor] = []
        self._val_targets: list[torch.Tensor] = []

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        probs = torch.softmax(logits, dim=1)
        acc = (probs.argmax(dim=1) == y).float().mean()
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)
        self._val_probs.append(probs.detach().cpu())
        self._val_targets.append(y.detach().cpu())

    def on_validation_epoch_end(self):
        all_probs = torch.cat(self._val_probs).to(self.device)  # [N, n_classes]
        all_targets = torch.cat(self._val_targets).to(self.device)  # [N]
        self._val_probs.clear()
        self._val_targets.clear()

        self.log_dict(self.val_metrics(all_probs, all_targets), prog_bar=True)
        self.val_metrics.reset()

        self._last_all_probs = all_probs.cpu()
        self._last_all_targets = all_targets.cpu()

    def on_train_end(self):
        if self._last_all_probs is None:
            return
        all_probs = self._last_all_probs.to(self.device)
        all_targets = self._last_all_targets.to(self.device)

        cm = MulticlassConfusionMatrix(num_classes=self.config.n_classes).to(
            self.device
        )
        cm_matrix = cm(all_probs, all_targets).cpu().numpy()
        fig, ax = plt.subplots(
            figsize=(self.config.n_classes * 2, self.config.n_classes * 2)
        )
        sns.heatmap(cm_matrix, annot=True, fmt="d", ax=ax, cmap="Blues")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        all_preds = all_probs.argmax(dim=1)
        df = pd.DataFrame(
            {
                "true_label": all_targets.cpu().numpy(),
                "predicted_label": all_preds.cpu().numpy(),
                **{
                    f"prob_class_{i}": all_probs[:, i].cpu().numpy()
                    for i in range(all_probs.shape[1])
                },
            }
        )
        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w", prefix="val_predictions_"
        ) as f:
            df.to_csv(f, index=True, index_label="sample_idx")
            tmppath = f.name
        try:
            mlflow.log_artifact(tmppath, artifact_path="val_predictions")
        finally:
            os.unlink(tmppath)

    def backbone_modules(self) -> list[nn.Module]:
        return []

    def configure_optimizers(self):
        backbone_ids = {id(p) for m in self.backbone_modules() for p in m.parameters()}
        backbone_params = [p for p in self.parameters() if id(p) in backbone_ids]
        head_params = [p for p in self.parameters() if id(p) not in backbone_ids]
        optimizer = self.optimizer(
            [
                {"params": head_params, "lr": self.config.learning_rate},
                {
                    "params": backbone_params,
                    "lr": self.config.learning_rate * self.config.backbone_lr_factor,
                },
            ],
            weight_decay=self.config.weight_decay,
            amsgrad=True,
        )

        if self.scheduler:
            scheduler = self.scheduler(
                optimizer,
                eta_min=self.config.scheduler_min_lr,
                T_max=self.config.scheduler_max_t,
            )
            return ([optimizer], [scheduler])

        return optimizer
