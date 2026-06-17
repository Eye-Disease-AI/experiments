import torch
import math

MAX_TRIALS = 100
EPOCHS = 100

QUICK_DEV_TEST = False
if QUICK_DEV_TEST:
    MAX_TRIALS = 1
    EPOCHS = 1

SEED = 2137
EXPERIMENT_NAME = "restored-baseline-search"
# AUROC graphs are generally smoother than ACC or F1.
# This makes tracking it more stable like when using loss,
# but with AUROC we optimise a meaningful metric, contrary to using loss.
OPTUNA_METRIC = "val_auroc"
OPTUNA_DIRECTION = "max"  # "min" or "max"

USE_SCHEDULER = False
SCHEDULER_MAX_T = 40
SCHEDULER_MIN_LR = 1e-6

CLASS_WEIGHTS = False

USE_EARLY_STOPPING = False
EARLY_STOPPING_PATIENCE = 15

USE_FREEZING = False
BACKBONE_UNFREEZE_PATIENCE = -1
BACKBONE_UNFREEZE_EPOCHS = 5
BACKBONE_LR_FACTOR = 1


# calculated from nuclear_cataract dataset
NORMALIZE = False
IMAGE_SIZE = 224
AUGM_ROT_ANGLE = 15
CACHE_SIZE = int(
    math.ceil(
        IMAGE_SIZE
        * (
            math.sin(math.radians(AUGM_ROT_ANGLE))
            + math.cos(math.radians(AUGM_ROT_ANGLE))
        )
    )
)
NORMALIZE_MEAN = [0.229015, 0.1663, 0.106812]
NORMALIZE_STD = [0.281245, 0.243682, 0.220464]

LOG_EVERY_N_EPOCHS = 1
OPTIMIZER = torch.optim.AdamW

GPU_PRECISION = "bf16-mixed"
torch.set_float32_matmul_precision("medium")  # to use tensor cores on newer gpus
