from pathlib import Path

import matplotlib.pyplot as plt
import torch

from experiments.experiment.augmentors.tacgan_augmentor import TacganAugmentorConfig
from experiments.experiment.datamodules.nuclear_cataract_datamodule import (
    NuclearCataractDataModule,
    NuclearCataractDataModuleConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def draw_grid(datamodule: NuclearCataractDataModule, title: str):
    indices = torch.randperm(len(datamodule.train_set))[:9].tolist()
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    fig.suptitle(title)
    for ax, i in zip(axes.flat, indices):
        img, label, *_ = datamodule.train_set[i]
        ax.imshow(img.permute(1, 2, 0))
        ax.set_title(f"idx={i} label={int(label)}", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(REPO_ROOT / f"sample_datamodule_{title.lower()}.png", dpi=150)
    plt.close(fig)


def main():
    # To see augmented dataset stats
    nc = NuclearCataractDataModule(
        NuclearCataractDataModuleConfig(
            augmentor_config=TacganAugmentorConfig.known_config_tac1(),
            n_augment=100,
            cas_mode=True,
        ),
    )
    nc.setup("train")
    draw_grid(nc, "Fake")

    # To see non augmented dataset stats
    nc = NuclearCataractDataModule(NuclearCataractDataModuleConfig())
    nc.setup("train")
    draw_grid(nc, "Real")


if __name__ == "__main__":
    main()
