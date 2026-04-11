import torch
import torch.nn as nn

SEED = 2137
EXPERIMENT_NAME = "nuclear-cataract-test"
STUDY_NAME = f"{EXPERIMENT_NAME}/lr-search3" # Multiple studies can be run per experiment, eg. optimizing different things

EPOCHS = 1
MAX_TRIALS = 2
MSE_THRESHOLD = 0.0001
LOG_EVERY_N_EPOCHS = 1 # API calls are expensive, so its better to batch them
OPTIMIZER= torch.optim.Adam
LOSS_FN= nn.CrossEntropyLoss


# Disable annoying and harmless logs
import logging
import warnings
from optuna.exceptions import ExperimentalWarning
logging.getLogger("mlflow.utils.uv_utils").setLevel(logging.WARNING)
logging.getLogger("mlflow.utils.environment").setLevel(logging.WARNING)
logging.getLogger("mlflow.pytorch").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=ExperimentalWarning)
