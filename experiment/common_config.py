import torch

SEED = 2137
EXPERIMENT_NAME = "restore-cosine-scheduler-test"

EPOCHS = 100
SCHEDULER_MAX_T=40
SCHEDULER_MIN_LR=1e-6
USE_SCHEDULER=False

MAX_TRIALS = 20
USE_EARLY_STOPPING=False
EARLY_STOPPING_PATIENCE = 15
BACKBONE_UNFREEZE_PATIENCE = 5
USE_FREEZING=False
BACKBONE_LR_FACTOR = 1
OPTUNA_METRIC = "val_loss"
OPTUNA_DIRECTION = "min"  # "min" or "max"

# calculated from nuclear_cataract dataset
NORMALIZE_MEAN = [0.229015, 0.1663, 0.106812]
NORMALIZE_STD = [0.281245, 0.243682, 0.220464]
MSE_THRESHOLD = 0.0001
LOG_EVERY_N_EPOCHS = 1
GPU_PRECISION = "bf16-mixed"
OPTIMIZER = torch.optim.AdamW

torch.set_float32_matmul_precision("medium")  # to use tensor cores on newer gpus
