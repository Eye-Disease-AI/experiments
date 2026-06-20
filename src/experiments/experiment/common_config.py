import torch

EXPERIMENT_NAME = "restored-baseline-search"
# AUROC graphs are generally smoother than ACC or F1.
# This makes tracking it more stable like when using loss,
# but with AUROC we optimise a meaningful metric, contrary to using loss.

# calculated from nuclear_cataract dataset
NORMALIZE = False
IMAGE_SIZE = 224
AUGM_ROT_ANGLE = 15

LOG_EVERY_N_EPOCHS = 1

torch.set_float32_matmul_precision("medium")  # to use tensor cores on newer gpus
