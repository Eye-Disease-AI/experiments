import os
import random

import numpy
import torch
import lightning as L

class RNG:
    def __init__(self):
        pass

    def is_deterministic(self):
        return self.seed is not None

    def set_seed(self, seed):
        self.seed = seed
        L.seed_everything(seed)

    def disable_determinism(self):
        self.seed = None
        L.seed_everything(None)
