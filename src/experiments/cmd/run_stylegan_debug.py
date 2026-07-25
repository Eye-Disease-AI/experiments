import torch

from experiments.experiment.augmentors.stylegan_augmentor import (
    StyleganAugmentor,
    StyleganAugmentorConfig,
)

if __name__ == "__main__":
    # StyleganAugmentor.upload_results_to_mlflow(
    #     Path(
    #         "/home/maciek/Desktop/masters/experiments3/packages/stylegan3/training-runs/00030-stylegan3-r-ncsg3-gpus1-batch12-gamma2"
    #     )
    # )

    c: StyleganAugmentor = StyleganAugmentorConfig.known_config_r().build()
    random_labels = torch.randint(0, 2, [10])
    gens = c.generate(random_labels)
    print(gens.shape)
