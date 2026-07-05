import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
from dataset.hard_policy import HardPolicy
from dataset.loader import NuclearCataractDataset
from torch.autograd import Variable
from torchvision.utils import save_image

from experiments.experiment.datamodules.nuclear_cataract_datamodule import (
    _SubsetTransformer,
)

ARTIFACTS_DIR_PATH = Path(__file__).resolve().parent.parent.parent.parent / "artifacts"
IMAGES_DIR_PATH = ARTIFACTS_DIR_PATH / Path("debug_gan_acgan", "images")

os.makedirs(IMAGES_DIR_PATH, exist_ok=True)
cuda = True if torch.cuda.is_available() else False
# cuda = False
img_size = 256
dataset = NuclearCataractDataset(
    NuclearCataractDataset.TrainValMode(0.8, 0.2),
    cache_size=img_size,
    hard_policy=HardPolicy.DOMINATE,
)
train_set = dataset.train_set()
latent_dim = 100
num_channels = 3
batch_size = 64
lr = 0.0002
b1 = 0.5
b2 = 0.999
n_epochs = 5000
sample_interval = 50


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(dataset.n_classes, latent_dim)

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
    def __init__(self):
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
            nn.Linear(128 * ds_size**2, dataset.n_classes), nn.Softmax()
        )

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label


adversarial_loss = torch.nn.BCELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()

generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()

generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)


dataloader = torch.utils.data.DataLoader(
    _SubsetTransformer(
        train_set,
        transform=transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float, scale=True),
                transforms.CenterCrop(img_size),
                # transforms.Resize((img_size, img_size)),
            ]
        ),
    ),
    batch_size=batch_size,
    drop_last=True,
    shuffle=True,
)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor  # type: ignore
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor  # type: ignore


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row**2, latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    save_image(
        gen_imgs.data,
        IMAGES_DIR_PATH / f"{batches_done}.png",
        nrow=n_row,
        normalize=True,
    )


# ----------
#  Training
# ----------

for epoch in range(n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):
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
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
        gen_labels = Variable(
            LongTensor(np.random.randint(0, dataset.n_classes, batch_size))
        )

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity, pred_label = discriminator(gen_imgs)
        g_loss = 0.5 * (
            adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels)
        )

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        real_pred, real_aux = discriminator(real_imgs)
        d_real_loss = (
            adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)
        ) / 2

        # Loss for fake images
        fake_pred, fake_aux = discriminator(gen_imgs.detach())
        d_fake_loss = (
            adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)
        ) / 2

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        # Calculate discriminator accuracy
        pred = np.concatenate(
            [real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0
        )
        gt = np.concatenate(
            [labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0
        )
        d_acc = np.mean(np.argmax(pred, axis=1) == gt)

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]"
            % (
                epoch,
                n_epochs,
                i,
                len(dataloader),
                d_loss.item(),
                100 * d_acc,
                g_loss.item(),
            )
        )
        batches_done = epoch * len(dataloader) + i
        if batches_done % sample_interval == 0:
            sample_image(n_row=2, batches_done=batches_done)
