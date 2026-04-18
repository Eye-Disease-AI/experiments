import torch
import torch.nn as nn
import os

SEED = 2137
EXPERIMENT_NAME = "nuclear-cataract-test"

EPOCHS = 100
MAX_TRIALS = 20
MSE_THRESHOLD = 0.0001
LOG_EVERY_N_EPOCHS = 1
OPTIMIZER= torch.optim.AdamW
LOSS_FN= nn.CrossEntropyLoss


# Disable annoying and harmless logs
import logging
import warnings
from optuna.exceptions import ExperimentalWarning
logging.getLogger("mlflow.utils.uv_utils").setLevel(logging.WARNING)
logging.getLogger("mlflow.utils.environment").setLevel(logging.WARNING)
logging.getLogger("mlflow.pytorch").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=ExperimentalWarning)
os.environ["LT_DISABLE_TIPS"] = "1"

torch.set_float32_matmul_precision('medium')  # to use tensor cores on newer gpus