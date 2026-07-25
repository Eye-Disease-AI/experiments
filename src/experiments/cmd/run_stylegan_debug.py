from pathlib import Path

from experiments.experiment.augmentors.stylegan_augmentor import (
    StyleganAugmentor,
    StyleganAugmentorConfig,
)

if __name__ == "__main__":
    sa = StyleganAugmentor(StyleganAugmentorConfig())
    sa.upload_results_to_mlflow(
        Path(
            "/home/maciek/Desktop/masters/experiments3/packages/stylegan3/training-runs/00030-stylegan3-r-ncsg3-gpus1-batch12-gamma2"
        )
    )
