import torch

SEED = 2137
EXPERIMENT_NAME = "restored-baseline-search"
MAX_TRIALS = 100
OPTUNA_METRIC = "val_auroc"
OPTUNA_DIRECTION = "max"  # "min" or "max"

EPOCHS = 100

USE_SCHEDULER=False
SCHEDULER_MAX_T=40
SCHEDULER_MIN_LR=1e-6

CLASS_WEIGHTS = False

USE_EARLY_STOPPING=False
EARLY_STOPPING_PATIENCE = 15

USE_FREEZING=False
BACKBONE_UNFREEZE_PATIENCE = -1
BACKBONE_UNFREEZE_EPOCHS=5
BACKBONE_LR_FACTOR = 1


# calculated from nuclear_cataract dataset
NORMALIZE = False
NORMALIZE_MEAN = [0.229015, 0.1663, 0.106812]
NORMALIZE_STD = [0.281245, 0.243682, 0.220464]

LOG_EVERY_N_EPOCHS = 1
OPTIMIZER = torch.optim.AdamW

GPU_PRECISION = "bf16-mixed"
torch.set_float32_matmul_precision("medium")  # to use tensor cores on newer gpus
