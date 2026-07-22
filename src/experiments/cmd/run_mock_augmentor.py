import torch

from experiments.experiment.augmentors.mock_nuclear_cataract_augmentor import (
    MockNuclearCataractAugmentor,
    MockNuclearCataractAugmentorConfig,
)

if __name__ == "__main__":
    augm = MockNuclearCataractAugmentor(MockNuclearCataractAugmentorConfig())
    # Returns (B, C, H, W)
    print(augm.generate(torch.Tensor([1, 2, 3])).shape)
