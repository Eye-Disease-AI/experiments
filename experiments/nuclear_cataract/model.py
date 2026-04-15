import torch
import torch.nn as nn
import torchvision
import lightning as L

class Model(L.LightningModule):
    def __init__(self, n_classes, lr=1e03):
        super().__init__()
        self.save_hyperparameters()
        self.model = torchvision.models.convnext.convnext_base(weights='IMAGENET1K_V1')
        new_clf = list(self.model.classifier.children())[:-1]
        new_clf.append(nn.LazyLinear(n_classes))
        new_clf.append(nn.Dropout(0.2))
        self.model.classifier = nn.Sequential(*new_clf)
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
