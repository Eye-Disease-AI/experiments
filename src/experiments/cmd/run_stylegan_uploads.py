import argparse
from pathlib import Path

from experiments.experiment.augmentors.stylegan_augmentor import (
    StyleganAugmentor,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-r", type=bool)
    parser.add_argument("--only-t", type=bool)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    upload_r = not args.only_t
    upload_t = not args.only_r

    if upload_r:
        print("Uploading stylegan t")
        StyleganAugmentor.upload_results_to_mlflow(
            Path(
                "/home/maciek/Desktop/masters/experiments3/packages/stylegan3/training-runs/00030-stylegan3-r-ncsg3-gpus1-batch12-gamma2"
            )
        )

    if upload_t:
        print("Uploading stylegan r")
        StyleganAugmentor.upload_results_to_mlflow(
            Path(
                "/home/maciek/Desktop/masters/experiments3/packages/stylegan3/training-runs/00029-stylegan3-t-ncsg3-gpus1-batch12-gamma2"
            )
        )

    # c: StyleganAugmentor = StyleganAugmentorConfig.known_config_r().build()
    # random_labels = torch.randint(0, 2, [10])
    # gens = c.generate(random_labels)
    # print(gens.shape)
    # plt.imshow(gens[0].permute(1, 2, 0))
    # plt.savefig("probe.png")
