import torch

SEED = 2137
EXPERIMENT_NAME = "nuclear-cataract-baseline_v5"

EPOCHS = 100
MAX_TRIALS = 20
EARLY_STOPPING_PATIENCE = 15
BACKBONE_UNFREEZE_PATIENCE = 5
BACKBONE_LR_FACTOR = 0.1
OPTUNA_METRIC = "val_f1"
OPTUNA_DIRECTION = "max"  # "min" or "max"

# calculated from nuclear_cataract dataset
NORMALIZE_MEAN = [0.229015, 0.1663, 0.106812]
NORMALIZE_STD = [0.281245, 0.243682, 0.220464]
MSE_THRESHOLD = 0.0001
LOG_EVERY_N_EPOCHS = 1
GPU_PRECISION = "bf16-mixed"
OPTIMIZER = torch.optim.AdamW

torch.set_float32_matmul_precision("medium")  # to use tensor cores on newer gpus
