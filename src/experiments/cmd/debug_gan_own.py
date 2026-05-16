import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from dataset.loader import NuclearCataractDataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm


class PrintShape(nn.Module):
    def __init__(self, name=""):
        super().__init__()
        self.name = name

    def forward(self, x):
        print(f"Shape after {self.name}: {x.shape}")
        return x


def show_img(img):
    img = img.permute(1, 2, 0)
    plt.imshow(img.detach().numpy())
    plt.savefig("gan.png")
    # plt.show()


def generate_fake_single(generator, noise_dims: int, label: torch.Tensor):
    z = create_noise(1, noise_dims, label)
    return generator(z).squeeze(0)


def get_generator(device: str, noise_dims: int = 100, gen_dims: int = 8, num_channels: int = 3):
    generator = nn.Sequential(
        nn.ConvTranspose2d(noise_dims, gen_dims * 8, 4, 1, 0, bias=False),
        nn.BatchNorm2d(gen_dims * 8),
        nn.Tanh(),
        nn.ConvTranspose2d(gen_dims * 8, gen_dims * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(gen_dims * 4),
        nn.Tanh(),
        nn.ConvTranspose2d(gen_dims * 4, gen_dims * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(gen_dims * 2),
        nn.Tanh(),
        nn.ConvTranspose2d(gen_dims * 2, gen_dims, 4, 2, 1, bias=False),
        nn.BatchNorm2d(gen_dims),
        nn.Tanh(),
        nn.ConvTranspose2d(gen_dims, gen_dims, 4, 2, 1, bias=False),
        nn.Tanh(),
        nn.ConvTranspose2d(gen_dims, num_channels, 3, 1, 1, bias=False),
        nn.Sigmoid(),
        PrintShape("Generated shape")
    )
    generator.to(device)
    return generator


def get_discriminator(device: str, base_hidden_dim: int = 64, num_channels: int = 4):
    discriminator = nn.Sequential(
        nn.Conv2d(num_channels, base_hidden_dim, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(base_hidden_dim, base_hidden_dim * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(base_hidden_dim * 2),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(base_hidden_dim * 2, base_hidden_dim * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(base_hidden_dim * 4),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(base_hidden_dim * 4, base_hidden_dim * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(base_hidden_dim * 8),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(base_hidden_dim * 8, 2, 4, 1, 0, bias=False),
    )
    discriminator.to(device)
    return discriminator

# Accepts batched input mat, which must be 4-dimensional tensor
# with following dimensions: (B, C, H, W).
# labels parameter need to container B number of labels.
def add_label_channel(mat: torch.Tensor, labels: torch.Tensor):
    batch_size = mat.shape[0]
    img_size = (mat.shape[2], mat.shape[3])
    labels = labels[-1, None, None, None].expand(batch_size, 1, *img_size)
    return torch.cat((mat, labels), dim=1)

# Creates noise of shape: (B, noise_dims)
def create_noise(batch_size: int, noise_dims: int, labels: torch.Tensor):
    result = torch.rand(batch_size, noise_dims - 1)
    labels = labels.unsqueeze(1)
    result = torch.cat((result, labels), dim=1)
    result = result.unsqueeze(2)
    result = result.unsqueeze(3)
    return result


def train():
    device = "cpu"
    dataset = NuclearCataractDataset(NuclearCataractDataset.TrainValMode(0.8, 0.2))
    train_set = dataset.train_set()

    noise_dims=100
    generator = get_generator(device, noise_dims=noise_dims)
    discriminator = get_discriminator(device)

    gen_loss_fn = nn.BCEWithLogitsLoss()
    gen_optimizer = torch.optim.AdamW(generator.parameters(), lr=0.0006)
    dis_loss_fn = nn.BCEWithLogitsLoss()
    dis_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=0.0002)
    img_size = (64, 64)
    prepare_transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float, scale=True),
            v2.Resize(size=img_size, antialias=True),
        ]
    )

    batch_size = 16
    data_loader = DataLoader(train_set, batch_size=batch_size, drop_last=True)

    skip_dis_train = False
    skip_gen_train = False
    skip_dis_train_until = -1
    skip_gen_train_until = -1
    max_dis_skip_count = 0
    max_gen_skip_count = 0

    for epoch in range(100):
        print("Starting epoch", epoch)

        for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
            generator.train()
            discriminator.train()

            x, y = data
            x = prepare_transforms(x)
            x = x.to(device)
            y = y.to(device)

            gen_loss = None
            dis_fake_loss = None
            dis_real_loss = None

            if not skip_gen_train:
                z = create_noise(batch_size, noise_dims, y)
                fake_x = generator(z)
                fake_x = add_label_channel(fake_x, y)

                dis_fake_outputs = discriminator(fake_x).squeeze()
                dis_target_outputs = nn.functional.one_hot(torch.ones(batch_size, dtype=torch.int64), 2).to(dtype=torch.float)

                gen_optimizer.zero_grad()
                gen_loss = gen_loss_fn(dis_fake_outputs, dis_target_outputs)
                gen_loss.backward()
                gen_optimizer.step()

                if gen_loss < skip_dis_train_until or max_dis_skip_count >= 2:
                    skip_dis_train = False
                    max_dis_skip_count = 0
            else:
                max_gen_skip_count += 1
                print("Skipping generator training")

            if not skip_dis_train:
                z = create_noise(batch_size, noise_dims, y)
                fake_x = generator(z)
                fake_x = add_label_channel(fake_x, y)
                real_x = add_label_channel(x, y)

                dis_fake_outputs = discriminator(fake_x).squeeze()
                print(dis_fake_outputs.shape)
                dis_target_labels = torch.zeros(dis_fake_outputs.shape[0]).to(dtype=torch.int64)
                dis_target_outputs = nn.functional.one_hot(dis_target_labels, 2).to(dtype=torch.float)
                print(dis_target_outputs.shape)

                dis_optimizer.zero_grad()
                dis_fake_loss = dis_loss_fn(dis_fake_outputs, dis_target_outputs)
                dis_fake_loss.backward()
                dis_optimizer.step()

                dis_optimizer.zero_grad()

                dis_real_outputs = discriminator(real_x).squeeze()
                dis_target_labels = torch.ones(dis_real_outputs.shape[0]).to(
                    dtype=torch.int64
                )
                dis_target_outputs = nn.functional.one_hot(dis_target_labels, 2).to(
                    dtype=torch.float
                )
                dis_real_loss = dis_loss_fn(dis_real_outputs, dis_target_outputs)
                dis_real_loss.backward()
                dis_optimizer.step()

                if (
                    dis_real_loss + dis_fake_loss
                ) < skip_gen_train_until or max_gen_skip_count >= 2:
                    skip_gen_train = False
                    max_gen_skip_count = 0
            else:
                max_dis_skip_count += 1
                print("Skipping discriminator training")

            if gen_loss:
                print("Generator loss", gen_loss)

            if dis_fake_loss and dis_real_loss:
                print("Discriminator fake loss", dis_fake_loss)
                print("Discriminator real loss", dis_real_loss)

            if gen_loss and dis_real_loss and dis_fake_loss:
                skip_dis_train = gen_loss > (dis_real_loss + dis_fake_loss)
                skip_dis_train_until = dis_real_loss + dis_fake_loss

                skip_gen_train = (dis_real_loss + dis_fake_loss) > 4 * gen_loss
                skip_gen_train_until = 4 * gen_loss

            with torch.no_grad():
                generator.eval()
                fake_single = generate_fake_single(generator, noise_dims, torch.Tensor([0]))
                show_img(fake_single)

if __name__ == "__main__":
    train()