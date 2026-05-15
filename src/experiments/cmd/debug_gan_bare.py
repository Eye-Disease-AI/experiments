import sys

import matplotlib.pyplot as plt
import torch
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from torchgan.losses import MinimaxGeneratorLoss
from torchgan.losses.minimax import MinimaxDiscriminatorLoss
from torchgan.models.conditional import (
    ConditionalGANDiscriminator,
    ConditionalGANGenerator,
)
from torchvision.transforms import v2
from tqdm import tqdm

from dataset.loader import NuclearCataractDataset

device = "cpu"
dataset = NuclearCataractDataset(NuclearCataractDataset.TrainValMode(0.8, 0.2))
train_set = dataset.train_set()

generator_encoding_dim = 100
generator = ConditionalGANGenerator(
    encoding_dims=generator_encoding_dim,
    num_classes=2,
    out_size=32,
    out_channels=3,
)
generator.to(device)

discriminator = ConditionalGANDiscriminator(
    num_classes=2,
    in_size=32,
    in_channels=3,
)
discriminator.to(device)

# img = gen_x.squeeze().permute(1, 2, 0)
# plt.imshow(img.detach().numpy())
# plt.show()

gen_loss_fn = MinimaxGeneratorLoss()
gen_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
dis_loss_fn = MinimaxDiscriminatorLoss()
dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
resize = v2.Resize(size=(32, 32), antialias=True)

batch_size = 32
data_loader = DataLoader(train_set, batch_size=batch_size)

z = torch.randn(batch_size, 100)
fake_x = generator(z, torch.zeros(batch_size))
print(fake_x)

for epoch in range(10):
    print("Starting epoch", epoch)
    generator.train()
    discriminator.train()

    for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        x, y = data
        x = resize(x)
        x = x.to(device)
        y = y.to(device)

        dis_optimizer.zero_grad()

        z = torch.randn(batch_size, 100).type_as(x)
        fake_x = generator(z, y)

        dis_real_outputs = discriminator(x, y)
        dis_fake_outputs = discriminator(fake_x, y)

        dis_loss = dis_loss_fn(dis_real_outputs, dis_fake_outputs.detach())
        dis_loss.backward()
        dis_optimizer.step()

        print("After dis pass")

        generator.zero_grad()
        fake_x = generator(z, y)
        dis_fake_outputs = discriminator(fake_x, y)
        gen_loss = gen_loss_fn(dis_fake_outputs)
        gen_loss.backward()
        gen_optimizer.step()

        print("Gen loss", gen_loss)
        print("Dis loss", dis_loss)

    generator.eval()
    with torch.no_grad():
        gen_inputs = generator.sampler(1, device)
        gen_x = generator(*gen_inputs)
        print(gen_x.shape)
        img = gen_x.squeeze().permute(1, 2, 0)
        plt.imshow(img.detach().numpy())
        plt.show()
