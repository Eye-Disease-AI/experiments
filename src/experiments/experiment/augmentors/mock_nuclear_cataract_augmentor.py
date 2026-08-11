import os
import subprocess
from dataclasses import dataclass
from tempfile import TemporaryDirectory
from typing import override

import torch
from torchvision.io import read_image

from experiments.experiment.augmentors.augmentor import Augmentor, AugmentorConfig


@dataclass(frozen=True, kw_only=True)
class MockNuclearCataractAugmentorConfig(AugmentorConfig):
    image_size: int = 256

    @staticmethod
    def get_configured_class():
        return MockNuclearCataractAugmentor


class MockNuclearCataractAugmentor(Augmentor):
    def __init__(self, config: MockNuclearCataractAugmentorConfig):
        self.config = config

    @override
    def generate(self, seeds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        super().generate(seeds, labels)
        num_images = labels.shape[0]
        assert num_images != 0
        generated_images = []
        img_size_str = f"{self.config.image_size}x{self.config.image_size}"
        img_bytes = 3 * self.config.image_size * self.config.image_size

        with TemporaryDirectory() as tmp_dir:
            for _ in range(num_images):
                noise_path = os.path.join(tmp_dir, "noise.png")
                subprocess.run(
                    f"head -c $(({img_bytes})) /dev/urandom | magick -size {img_size_str} -depth 8 rgb:- {noise_path}",
                    shell=True,
                    check=True,
                )
                img = read_image(noise_path)
                generated_images.append(img)

        return torch.stack(generated_images)
