import lightning as L
import torch.nn as nn
from torchgan.losses import MinimaxDiscriminatorLoss, MinimaxGeneratorLoss
from torchgan.models import ConditionalGANDiscriminator, ConditionalGANGenerator
from torchgan.trainer import Trainer


class GAN(L.LightningModule):
    def __init__(self, n_classes: int):
        super().__init__()
        self.generator = ConditionalGANGenerator(
            num_classes=n_classes,
            out_size=512,
            out_channels=3,
        )
        self.discriminator = ConditionalGANDiscriminator(
            num_classes=n_classes,
            in_size=512,
            in_channels=3,
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def on_validation_epoch_end(self):
        pass

    def on_train_end(self):
        pass

    def backbone_modules(self) -> list[nn.Module]:
        return []

    def configure_optimizers(self):
        pass
