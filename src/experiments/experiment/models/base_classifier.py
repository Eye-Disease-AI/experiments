from dataclasses import dataclass

import lightning as L
import torch
import torch.nn as nn
from torch.optim import AdamW
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAUROC,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

from experiments.lib.config_serializing import (
    ClassConfig,
    OptunaOptimised,
    deserialize_class,
    must_serialize_class,
)


@dataclass(frozen=True, kw_only=True)
class BaseClassifierModelConfig(ClassConfig):
    optimizer: str = must_serialize_class(AdamW)
    loss_fn: str = must_serialize_class(nn.CrossEntropyLoss)
    scheduler: str | None = (
        None  # serialize_class(torch.optim.lr_scheduler.CosineAnnealingLR)
    )
    scheduler_max_t: int = 40
    scheduler_min_lr: float = 1e-6
    backbone_lr_factor: float = 1
    n_classes: int = 2
    class_weights: list[float] | None = None
    learning_rate: float | OptunaOptimised = 5e-5
    weight_decay: float | None | OptunaOptimised = 1e-6


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
