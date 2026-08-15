import torch

from experiments.experiment.augmentors.augmentor import Augmentor
from experiments.experiment.augmentors.common import create_augmentor

if __name__ == "__main__":
    aug: Augmentor = create_augmentor("tacgan-tac3").build()
    a = aug.generate(torch.Tensor([1, 2, 3]), torch.Tensor([0, 1, 0]))
    b = aug.generate(torch.Tensor([1, 2, 3]), torch.Tensor([0, 1, 0]))

    # Cast to avoid number underflow or overflow
    distance = a.to(torch.int16) - b.to(torch.int16).abs()
    print(
        f"Max distance between generations with the same seeds and labels: {distance.max()}"
    )

    if not torch.equal(a, b):
        raise RuntimeError("TACGAN generation is non-deterministic")
