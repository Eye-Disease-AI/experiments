from dataclasses import dataclass

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from experiments.experiment.studies.study import override
from experiments.lib.config_serializing import ClassConfig, OptunaOptimised, resolved

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor  # type: ignore
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor  # type: ignore


@dataclass(frozen=True, kw_only=True)
class ACGANModuleConfig(ClassConfig):
    n_classes: int
    latent_dim: int | OptunaOptimised
    img_size: int
    num_channels: int

    @override
    @staticmethod
    def get_configured_class():
        return ACGANModule


class ACGANModule(L.LightningModule):
    def __init__(self, config: ACGANModuleConfig):
        super().__init__()
        self.config = config
        # Lightning won't know how to optimize GAN
        self.automatic_optimization = False
        self.adversarial_loss = torch.nn.BCELoss()
        self.auxiliary_loss = torch.nn.CrossEntropyLoss()
        self.generator = Generator(
            config.n_classes,
            resolved(config.latent_dim),
            config.img_size,
            config.num_channels,
        )
        self.discriminator = Discriminator(
            config.n_classes,
            config.img_size,
            config.num_channels,
        )
        self.generator.apply(self.weights_init_normal)
        self.discriminator.apply(self.weights_init_normal)

    def training_step(self, batch, batch_idx):
        optimizer_G, optimizer_D = self.optimizers()  # type: ignore
        imgs, labels = batch
        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(
            FloatTensor(
                np.random.normal(0, 1, (batch_size, resolved(self.config.latent_dim)))
            )
        )
        gen_labels = Variable(
            LongTensor(np.random.randint(0, self.config.n_classes, batch_size))
        )

        # Generate a batch of images
        gen_imgs = self.generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity, pred_label = self.discriminator(gen_imgs)
        g_loss = 0.5 * (
            self.adversarial_loss(validity, valid)
            + self.auxiliary_loss(pred_label, gen_labels)
        )
        self.log("g_loss", g_loss, prog_bar=True)

        self.manual_backward(g_loss)
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        real_pred, real_aux = self.discriminator(real_imgs)
        d_real_loss = (
            self.adversarial_loss(real_pred, valid)
            + self.auxiliary_loss(real_aux, labels)
        ) / 2

        # Loss for fake images
        fake_pred, fake_aux = self.discriminator(gen_imgs.detach())
        d_fake_loss = (
            self.adversarial_loss(fake_pred, fake)
            + self.auxiliary_loss(fake_aux, gen_labels)
        ) / 2

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2
        self.log("d_loss", d_loss, prog_bar=True)

        # Calculate discriminator accuracy
        pred = np.concatenate(
            [real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0
        )
        gt = np.concatenate(
            [labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0
        )
        d_acc = np.mean(np.argmax(pred, axis=1) == gt)
        self.log("d_acc", d_acc, prog_bar=True)

        self.manual_backward(d_loss)
        optimizer_D.step()

    def validation_step(self, batch, batch_idx):
        pass

    def on_validation_epoch_end(self):
        pass

    def configure_optimizers(self):
        optimizer_G = torch.optim.Adam(self.generator.parameters())
        optimizer_D = torch.optim.Adam(self.discriminator.parameters())
        return optimizer_G, optimizer_D

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.shape[0]
        gen_labels = Variable(
            LongTensor(np.random.randint(0, self.config.n_classes, batch_size))
        )
        return self.generator(z, gen_labels)

    def sample_image(self, n_row):
        """Saves a grid of generated digits ranging from 0 to n_classes"""
        # Sample noise
        z = Variable(
            FloatTensor(
                np.random.normal(0, 1, (n_row**2, resolved(self.config.latent_dim)))
            )
        )
        # Get labels ranging from 0 to n_classes for n rows
        labels = np.array([num for _ in range(n_row) for num in range(n_row)])
        labels = Variable(LongTensor(labels))
        return self.generator(z, labels)

    def weights_init_normal(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm2d") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(
        self, n_classes: int, latent_dim: int, img_size: int, num_channels: int
    ):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(n_classes, latent_dim)

        self.init_size = img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size**2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, num_channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, n_classes: int, img_size: int, num_channels: int):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(num_channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2**4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size**2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(
            nn.Linear(128 * ds_size**2, n_classes), nn.Softmax()
        )

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label
