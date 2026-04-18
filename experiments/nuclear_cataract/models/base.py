import os
import tempfile
import torch
import lightning as L
import mlflow
import pandas as pd
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassAUROC


class ModelBase(L.LightningModule):
    def __init__(self, n_classes: int):
        super().__init__()
        self.val_metrics = MetricCollection({
            "val_precision": MulticlassPrecision(num_classes=n_classes, average='macro'),
            "val_recall":    MulticlassRecall(num_classes=n_classes, average='macro'),
            "val_auroc":     MulticlassAUROC(num_classes=n_classes, average='macro'),
        })
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
        all_probs = torch.cat(self._val_probs).to(self.device)     # [N, n_classes]
        all_targets = torch.cat(self._val_targets).to(self.device) # [N]
        self._val_probs.clear()
        self._val_targets.clear()

        self.log_dict(self.val_metrics(all_probs, all_targets), prog_bar=True)
        self.val_metrics.reset()

        # Save per-sample predictions as CSV artifact on the last epoch only
        if self.current_epoch == self.trainer.max_epochs - 1:
            all_preds = all_probs.argmax(dim=1)
            df = pd.DataFrame({
                "true_label": all_targets.cpu().numpy(),
                "predicted_label": all_preds.cpu().numpy(),
                **{f"prob_class_{i}": all_probs[:, i].cpu().numpy() for i in range(all_probs.shape[1])},
            })
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", prefix="val_predictions_") as f:
                df.to_csv(f, index=True, index_label="sample_idx")
                tmppath = f.name
            try:
                mlflow.log_artifact(tmppath, artifact_path="val_predictions")
            finally:
                os.unlink(tmppath)

    def configure_optimizers(self):
        trainable = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(trainable, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)