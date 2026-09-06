from pathlib import Path
from dataclasses import replace

import torch
from torchvision.utils import save_image

from experiments.experiment.augmentors.tacgan_augmentor import TacganAugmentorConfig
from experiments.experiment.augmentors.stylegan_augmentor import StyleganAugmentorConfig
from experiments.experiment.datamodules.nuclear_cataract_datamodule import (
    NuclearCataractDataModule,
    NuclearCataractDataModuleConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
SAMPLES_DIR = ARTIFACTS_DIR / "samples"


# TODO: Move this to the paper repo
# def draw_grid(datamodule: NuclearCataractDataModule, title: str):
#     indices = torch.randperm(len(datamodule.train_set))[:9].tolist()
#     fig, axes = plt.subplots(3, 3, figsize=(6, 6))
#     fig.suptitle(title)
#     for ax, i in zip(axes.flat, indices):
#         img, label, *_ = datamodule.train_set[i]
#         ax.imshow(img.permute(1, 2, 0))
#         ax.set_title(f"idx={i} label={int(label)}", fontsize=8)
#         ax.axis("off")
#     plt.tight_layout()
#     plt.savefig(REPO_ROOT / f"sample_datamodule_{title.lower()}.png", dpi=150)
#     plt.close(fig)


def generate_samples(
    datamodule: NuclearCataractDataModule,
    per_class_samples: int,
    generation_name: str = "reals",
    seed: int = 188872,
):
    torch.manual_seed(seed)

    dest_dir = SAMPLES_DIR / datamodule.__class__.__name__
    dest_dir = dest_dir / generation_name
    dest_dir = dest_dir / datamodule.config.augment_kind

    if dest_dir.exists():
        print(f"{dest_dir} already exists, skipping {generation_name} generation...")
        return

    assert datamodule.config.cas_mode ^ (not datamodule.config.augmentor_config)
    datamodule.setup("train")

    indices = torch.randperm(len(datamodule.train_set))
    n_classes = datamodule.n_classes
    remaining = [per_class_samples] * n_classes

    while sum(remaining) != 0:
        img, label = datamodule.train_set[indices[0]]

        if remaining[label] != 0:
            class_dest_dir = dest_dir / f"class_{label}"
            class_dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = class_dest_dir / f"{indices[0]}.png"
            print(f"Saving sample to {dest_path}...")
            save_image(img, dest_path)
            remaining[label] -= 1

        if len(indices) != 0:
            indices = indices[1:]
        else:
            raise RuntimeError("Dataset not big enough")


def combination(original, changes):
    result = []

    for change in changes:
        result.append(replace(original, **change))

    return result


def with_name(name: str, configs: list[NuclearCataractDataModuleConfig]):
    return [(name, cfg) for cfg in configs]


def main():
    sampled_configs = [
        *with_name(
            "reals",
            combination(
                NuclearCataractDataModuleConfig(),
                [{"augment_kind": "R"}, {"augment_kind": "RA"}],
            ),
        ),
        *with_name(
            "TACGAN",
            combination(
                NuclearCataractDataModuleConfig(
                    augmentor_config=TacganAugmentorConfig.known_config_tac3(),
                    n_augment=1000,
                    cas_mode=True,  # CAS mode moves real samples away from the training set
                ),
                [{"augment_kind": "RG"}, {"augment_kind": "RAG"}],
            ),
        ),
        *with_name(
            "StyleGAN3_T",
            combination(
                NuclearCataractDataModuleConfig(
                    augmentor_config=StyleganAugmentorConfig.known_config_t(),
                    n_augment=1000,
                    cas_mode=True,  # CAS mode moves real samples away from the training set
                ),
                [{"augment_kind": "RG"}, {"augment_kind": "RAG"}],
            ),
        ),
        *with_name(
            "StyleGAN3_R",
            combination(
                NuclearCataractDataModuleConfig(
                    augmentor_config=StyleganAugmentorConfig.known_config_r(),
                    n_augment=1000,
                    cas_mode=True,  # CAS mode moves real samples away from the training set
                ),
                [{"augment_kind": "RG"}, {"augment_kind": "RAG"}],
            ),
        ),
    ]

    for cfg_name, cfg in sampled_configs:
        nc = NuclearCataractDataModule(cfg)
        generate_samples(nc, 16, cfg_name)


if __name__ == "__main__":
    main()
