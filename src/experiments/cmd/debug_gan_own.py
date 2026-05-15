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


def generate_fake_single(generator, generator_noise_dim, label: torch.Tensor):
    z_noises = torch.rand(1, 3, *generator_noise_dim[1:])
    z = add_label_channel(z_noises, label)
    return generator(z).squeeze(0)


def get_generator(device: str):
    generator = nn.Sequential(
        nn.ConvTranspose2d(4, 16, 4, 1, 0, bias=False),
        nn.LazyBatchNorm2d(),
        nn.ReLU(True),
        nn.ConvTranspose2d(16, 8, 4, 2, 1, bias=False),
        nn.LazyBatchNorm2d(),
        nn.ReLU(True),
        nn.ConvTranspose2d(8, 3, 8, 2, 1, bias=False),
        nn.LazyBatchNorm2d(),
        nn.ReLU(True),
        nn.Tanh(),
    )
    generator.to(device)
    return generator


def get_discriminator(device: str):
    discriminator = nn.Sequential(
        nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.LeakyReLU(),
        nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.LeakyReLU(),
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.LeakyReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.LeakyReLU(),
        PrintShape("after last pool"),
        nn.Flatten(start_dim=1, end_dim=3),
        PrintShape("after flatten"),
        nn.Linear(in_features=256, out_features=256),
        nn.LeakyReLU(),
        nn.Linear(in_features=256, out_features=128),
        nn.LeakyReLU(),
        nn.Linear(in_features=128, out_features=2),
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

def train():
    device = "cpu"
    dataset = NuclearCataractDataset(NuclearCataractDataset.TrainValMode(0.8, 0.2))
    train_set = dataset.train_set()

    generator_noise_dim = (4, 4, 4)

    generator = get_generator(device)
    discriminator = get_discriminator(device)

    gen_loss_fn = nn.BCEWithLogitsLoss()
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
    dis_loss_fn = nn.BCEWithLogitsLoss()
    dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0005)
    img_size = (32, 32)
    prepare_transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float, scale=True),
            v2.Resize(size=img_size, antialias=True),
        ]
    )

    batch_size = 32
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

            z_labels = y
            z_labels = z_labels[-1, None, None, None].expand(
                batch_size, 1, *generator_noise_dim[1:]
            )
            z_noises = torch.rand(batch_size, 3, *generator_noise_dim[1:])
            z = torch.cat((z_noises, z_labels), dim=1)

            gen_loss = None
            dis_fake_loss = None
            dis_real_loss = None

            if not skip_gen_train:
                fake_x = generator(z)
                fake_x = add_label_channel(fake_x, y)
                real_x = add_label_channel(x, y)

                dis_fake_outputs = discriminator(fake_x)
                dis_target_outputs = nn.functional.one_hot(y, 2).to(dtype=torch.float)

                gen_optimizer.zero_grad()
                gen_loss = gen_loss_fn(dis_fake_outputs, dis_target_outputs)
                gen_loss.backward()
                gen_optimizer.step()

                if gen_loss < skip_dis_train_until or max_dis_skip_count >= 10:
                    skip_dis_train = False
                    max_dis_skip_count = 0
            else:
                max_gen_skip_count += 1
                print("Skipping generator training")

            if not skip_dis_train:
                fake_x = generator(z)
                fake_x = add_label_channel(fake_x, y)

                dis_fake_outputs = discriminator(fake_x)
                dis_target_labels = torch.zeros(dis_fake_outputs.shape[0]).to(
                    dtype=torch.int64
                )
                dis_target_outputs = nn.functional.one_hot(dis_target_labels, 2).to(
                    dtype=torch.float
                )

                dis_optimizer.zero_grad()
                dis_fake_loss = dis_loss_fn(dis_fake_outputs, dis_target_outputs)
                dis_fake_loss.backward()
                dis_optimizer.step()

                dis_optimizer.zero_grad()

                dis_real_outputs = discriminator(real_x)
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
                ) < skip_gen_train_until or max_gen_skip_count >= 10:
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
                fake_single = generate_fake_single(generator, generator_noise_dim, torch.Tensor([0]))
                show_img(fake_single)

if __name__ == "__main__":
    train()