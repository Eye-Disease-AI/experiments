import torch
import torch.nn as nn

SEED = 2137
EXPERIMENT_NAME = "nuclear-cataract-baseline_v2"

EPOCHS = 100
MAX_TRIALS = 20
EARLY_STOPPING_PATIENCE = 15
OPTUNA_METRIC = "val_auroc"
OPTUNA_DIRECTION = "max"  # "min" or "max"
MSE_THRESHOLD = 0.0001
LOG_EVERY_N_EPOCHS = 1
GPU_PRECISION = 'bf16-mixed'
OPTIMIZER= torch.optim.AdamW
LOSS_FN= nn.CrossEntropyLoss

torch.set_float32_matmul_precision('medium')  # to use tensor cores on newer gpus
